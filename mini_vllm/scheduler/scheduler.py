from __future__ import annotations

import time
from typing import Dict, List

from ..cache.manager import BlockManager
from ..config import Config
from ..sequence.sequence import Sequence
from ..sequence.sequence_group import RequestQueue, SequenceGroup
from ..sequence.status import Status
from .schedule_result import ScheduleResult


class Scheduler:
    """Continuous-batching scheduler with chunked prefill and decode-first.

    At each engine step::

        1. **Finish** — check every running group; sequences that hit
           ``max_tokens`` are marked FINISHED; groups whose every sequence
           is done move to the finished pool.
        2. **Categorize** — split remaining running groups into decode
           (status=RUNNING) and prefill-continue (status=PREFILL).
        3. **Decode-first budget** — deduct decode tokens, then compute
           remaining budget for prefill.
        4. **Chunked-prefill continue** — advance prefill cursor for
           groups that are mid-prefill.
        5. **Admit new** — admit waiting groups with chunked budget.
        6. **Token counts & debug_reason**.
    """

    def __init__(self, config: Config, block_manager: BlockManager, queue: RequestQueue) -> None:
        self._config = config
        self._block_manager = block_manager
        self._queue = queue

    def schedule(self) -> ScheduleResult:
        """Run one scheduling iteration — called once per engine step."""
        result = ScheduleResult()

        chunk_size = self._config.max_prefill_chunk_size

        # ------------------------------------------------------------------
        # Phase 1: Finish check — running groups
        # ------------------------------------------------------------------
        keep_running_groups: List[SequenceGroup] = []

        for sg in self._queue.running:
            for seq in sg.get_unfinished_seqs():
                if seq.num_generated_tokens >= seq.sampling_params.max_tokens:
                    seq.status = Status.FINISHED
                    seq.finish_time = time.time()
                    self._block_manager.free(seq.seq_id)

            if sg.is_finished:
                self._queue.mark_finished(sg)
                result.finished_groups.append(sg)
            else:
                keep_running_groups.append(sg)

        # ------------------------------------------------------------------
        # Phase 2: Categorize running groups
        # ------------------------------------------------------------------
        decode_groups: List[SequenceGroup] = []
        prefill_continue_groups: List[SequenceGroup] = []

        for sg in keep_running_groups:
            seq = sg.get_unfinished_seqs()[0]
            if seq.status == Status.PREFILL:
                prefill_continue_groups.append(sg)
            else:
                decode_groups.append(sg)

        # ------------------------------------------------------------------
        # Phase 3: Decode-first budget
        # ------------------------------------------------------------------
        remaining_token_budget = self._config.max_num_batched_tokens
        remaining_seq_budget = self._config.max_num_seqs

        # Deduct decode first (1 token per decode sequence)
        for sg in decode_groups:
            n = len(sg.get_unfinished_seqs())
            remaining_token_budget -= n
            remaining_seq_budget -= n
        result.scheduled_decode_groups = list(decode_groups)

        # Prefill budget = remaining tokens, capped by max_num_prefill_tokens
        prefill_budget = min(remaining_token_budget, self._config.max_num_prefill_tokens)
        num_prefill_tokens = 0

        # ------------------------------------------------------------------
        # Phase 4: Chunked-prefill continue
        # ------------------------------------------------------------------
        for sg in prefill_continue_groups:
            if remaining_seq_budget <= 0:
                result.ignored_groups.append(sg)
                result.ignored_reasons[sg.request_id] = "MAX_NUM_SEQS_LIMIT"
                continue

            seq = sg.get_unfinished_seqs()[0]
            # prefill_cursor already accounts for cached prefix (set at admit time)
            remaining_prompt = len(seq.prompt_token_ids) - seq.prefill_cursor
            this_chunk = min(remaining_prompt, chunk_size)

            if this_chunk > prefill_budget:
                result.ignored_groups.append(sg)
                result.ignored_reasons[sg.request_id] = "WAITING_FOR_NEXT_STEP"
                continue

            result.scheduled_prefill_groups.append(sg)
            prefill_budget -= this_chunk
            remaining_token_budget -= this_chunk
            remaining_seq_budget -= 1
            num_prefill_tokens += this_chunk
            result.num_uncached_prefill_tokens += this_chunk

        # ------------------------------------------------------------------
        # Phase 5: Admit new waiting groups (with prefix cache awareness)
        # ------------------------------------------------------------------
        for sg in list(self._queue.waiting):
            if remaining_seq_budget <= 0:
                result.ignored_groups.append(sg)
                result.ignored_reasons[sg.request_id] = "MAX_NUM_SEQS_LIMIT"
                continue

            prompt_len = len(sg.prompt_token_ids)

            # Read-only probe: how many prompt tokens are already cached?
            probe = self._block_manager.probe_prefix_cache(sg.prompt_token_ids)
            uncached_tokens = prompt_len - probe.cached_token_count

            if self._config.chunked_prefill_enabled:
                this_chunk = min(uncached_tokens, chunk_size)
            else:
                this_chunk = uncached_tokens

            if this_chunk > prefill_budget:
                # Rejection: the uncached portion alone is too long to fit
                if uncached_tokens > self._config.max_num_batched_tokens:
                    self._queue.mark_rejected(sg)
                    result.rejected_groups.append(sg)
                else:
                    result.ignored_groups.append(sg)
                    result.ignored_reasons[sg.request_id] = "NO_TOKEN_BUDGET"
                continue

            seq_id = f"{sg.request_id}-seq-0"
            seq = sg.create_sequence(seq_id)
            # allocate_for_seq does the real attach: increment_ref, add_shared_block, etc.
            self._block_manager.allocate_for_seq(seq)

            seq.status = Status.PREFILL
            # Prefill cursor starts after the cached prefix (not at 0).
            # The executor's prefill loop starts from this position and only
            # processes uncached tokens.
            seq.prefill_cursor = probe.cached_token_count

            self._queue.mark_running(sg)
            result.scheduled_prefill_groups.append(sg)

            prefill_budget -= this_chunk
            remaining_token_budget -= this_chunk
            remaining_seq_budget -= 1
            num_prefill_tokens += this_chunk
            result.cached_token_count += probe.cached_token_count
            result.num_uncached_prefill_tokens += this_chunk
            result.matched_block_count += probe.matched_block_count

        # ------------------------------------------------------------------
        # Phase 6: Token counts & debug_reason
        # ------------------------------------------------------------------
        num_decode = 0
        for sg in decode_groups:
            num_decode += len(sg.get_unfinished_seqs())

        result.num_prefill_tokens = num_prefill_tokens
        result.num_decode_tokens = num_decode
        result.num_batched_tokens = num_prefill_tokens + num_decode
        result.token_budget_remaining = remaining_token_budget
        result.debug_reason = self._build_debug_reason(result)

        return result

    def block_manager_stats(self) -> dict:
        """Expose block manager stats for metrics reporting."""
        return self._block_manager.stats()

    @staticmethod
    def _build_debug_reason(result: ScheduleResult) -> str:
        parts = []
        if result.scheduled_prefill_groups:
            p = [sg.request_id for sg in result.scheduled_prefill_groups]
            if result.cached_token_count > 0:
                parts.append(
                    f"prefill({result.num_uncached_prefill_tokens}t "
                    f"+{result.cached_token_count}cached): {', '.join(p)}"
                )
            else:
                parts.append(f"prefill({result.num_prefill_tokens}t): {', '.join(p)}")
        if result.scheduled_decode_groups:
            d = [sg.request_id for sg in result.scheduled_decode_groups]
            parts.append(f"decode({result.num_decode_tokens}t): {', '.join(d)}")
        if result.ignored_groups:
            i = [sg.request_id for sg in result.ignored_groups]
            parts.append(f"ignored: {', '.join(i)}")
        if result.finished_groups:
            f = [sg.request_id for sg in result.finished_groups]
            parts.append(f"done: {', '.join(f)}")
        return " | ".join(parts) if parts else "idle"
