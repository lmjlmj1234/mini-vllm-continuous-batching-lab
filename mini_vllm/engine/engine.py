from __future__ import annotations

import time
from typing import Dict, Optional

from ..cache.allocator import BlockAllocator
from ..cache.manager import BlockManager
from ..config import Config
from ..executor.base import Executor
from ..scheduler.scheduler import Scheduler
from ..scheduler.schedule_result import ScheduleResult
from ..sequence.sequence_group import RequestQueue, SequenceGroup
from ..sequence.sampling_params import SamplingParams
from ..sequence.status import Status
from .engine_core import EngineCore


class LLMEngine:
    """Public-facing engine API — analogous to vLLM's ``LLMEngine``.

    ``LLMEngine`` provides the user-facing interface::

        LLMEngine → EngineCore → Scheduler → BlockManager
                           ↓               → BlockAllocator
                       Executor (fake or Qwen)

    The executor type is selected via ``Config.executor_type``
    (``"fake"`` or ``"qwen"``).  The scheduler and memory manager
    are completely model-agnostic.

    Responsibilities:
    - ``add_request()``: create a ``SequenceGroup`` and enqueue it
    - ``run_until_done()`` / ``step()``: drive the engine loop
    - ``get_outputs()``: collect final generation results
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()
        self._queue = RequestQueue()
        self._seq_group_count = 0

        # Build the BlockAllocator and BlockManager first
        allocator = BlockAllocator(
            num_blocks=self._config.num_gpu_blocks,
        )
        self._block_manager = BlockManager(self._config.block_size, allocator)

        # Create worker → executor
        self._worker = self._create_worker()
        self._executor = self._worker.get_executor()
        # Wire block_manager for on-demand allocation (ensure_block)
        self._executor._block_manager = self._block_manager  # type: ignore[attr-defined]

        # Wire BlockAllocator callbacks → executor
        allocator.set_callbacks(
            on_allocate=self._executor.prepare_block,
            on_free=self._executor.release_block,
        )
        self._scheduler = Scheduler(self._config, self._block_manager, self._queue)
        self._engine_core = EngineCore(self._scheduler, self._executor)

        self._outputs: Dict[str, str] = {}

    def _create_worker(self):
        """Factory: instantiate the configured worker type."""
        if self._config.executor_type == "fake":
            from ..worker.fake_worker import FakeWorker
            return FakeWorker(self._config)
        else:
            from ..worker.qwen_worker import QwenWorker
            return QwenWorker(self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 16,
    ) -> str:
        """Add a new request to the engine and return its request ID."""
        request_id = f"req-{self._seq_group_count:04d}"
        self._seq_group_count += 1

        prompt_token_ids = self._executor.tokenize(prompt)
        sampling_params = SamplingParams(max_tokens=max_new_tokens)
        sg = SequenceGroup(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=time.time(),
        )
        self._queue.add(sg)
        return request_id

    def step(self) -> ScheduleResult:
        """Run one scheduling-and-execution step.

        Returns the ``ScheduleResult`` for inspection / testing.
        """
        result = self._engine_core.step()

        # Print step summary
        if self._config.print_step_events:
            self._print_step(result)
        if self._config.memory_trace:
            self._print_memory_trace(result)

        # Capture finished outputs
        for sg in result.finished_groups:
            for seq in sg.seqs:
                if seq.status == Status.FINISHED:
                    text = self._executor.detokenize(seq.output_token_ids)
                    self._outputs[seq.group_id] = text

        return result

    def run_until_done(self) -> Dict[str, str]:
        """Keep calling ``step()`` until all requests have finished.

        Returns ``{request_id: output_text}``.
        """
        while True:
            result = self.step()
            if self._queue.num_waiting == 0 and self._queue.num_running == 0:
                break
        return self.get_outputs()

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request by ID, freeing all resources."""
        return self._engine_core.cancel_request(request_id)

    def get_outputs(self) -> Dict[str, str]:
        return dict(self._outputs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _print_step(self, result: ScheduleResult) -> None:
        s = self._engine_core.step_count
        cfg = self._config

        # Waiting
        wait_parts = []
        for sg in self._queue.waiting:
            plen = len(sg.prompt_token_ids)
            wait_parts.append(f"{sg.request_id}(prompt={plen})")

        # Running
        run_parts = []
        for sg in self._queue.running:
            seq = sg.get_unfinished_seqs()
            st = seq[0].status.name if seq else "?"
            cursor = seq[0].prefill_cursor if seq and st == "PREFILL" else "-"
            nt = seq[0].num_generated_tokens if seq else 0
            run_parts.append(f"{sg.request_id}({st},cursor={cursor},gen={nt})")

        # Scheduled prefill (from scheduler result, pre-execution view)
        pre_parts = [sg.request_id for sg in result.scheduled_prefill_groups]

        # Scheduled decode
        dec_parts = [sg.request_id for sg in result.scheduled_decode_groups]

        # Ignored
        ign_parts = []
        for sg in result.ignored_groups:
            reason = result.ignored_reasons.get(sg.request_id, "?")
            ign_parts.append(f"{sg.request_id}({reason})")

        # Finished
        fin_parts = [sg.request_id for sg in result.finished_groups]

        stats = self._block_manager.stats()
        kv_stats = self._executor.get_kv_stats()

        print(f"  [step {s}]")
        print(f"    waiting:                  [{', '.join(wait_parts) or '—'}]")
        print(f"    running:                  [{', '.join(run_parts) or '—'}]")
        if result.cached_token_count > 0:
            print(f"    scheduled prefill:        [{', '.join(pre_parts) or '—'}]  "
                  f"prefill_tokens={result.num_uncached_prefill_tokens} "
                  f"(+{result.cached_token_count}cached)")
        else:
            print(f"    scheduled prefill:        [{', '.join(pre_parts) or '—'}]  "
                  f"prefill_tokens={result.num_prefill_tokens}")
        print(f"    scheduled decode:         [{', '.join(dec_parts) or '—'}]  "
              f"decode_tokens={result.num_decode_tokens}")
        print(f"    token budget remaining:   {result.token_budget_remaining}/{cfg.max_num_batched_tokens}")
        print(f"    KV blocks allocated:      {stats['used_blocks']}/{stats['total_blocks']}  "
              f"(slot_capacity={kv_stats['kv_slot_capacity']})")
        print(f"    KV tokens written:        {kv_stats['kv_tokens_written']}  "
              f"(actual data in cache)")
        if ign_parts:
            print(f"    ignored:                  [{', '.join(ign_parts)}]")
        if result.rejected_groups:
            rej_ids = [sg.request_id for sg in result.rejected_groups]
            print(f"    rejected:                 [{', '.join(rej_ids)}]")
        if fin_parts:
            print(f"    finished:                 [{', '.join(fin_parts)}]")

    def _print_memory_trace(self, result: ScheduleResult) -> None:
        s = self._engine_core.step_count
        mgr = self._block_manager
        allocator = mgr._allocator

        free_ids = allocator.dump_free_list()
        used_ids = allocator.dump_used_list()
        print(f"    +- BlockAllocator free list [step {s}]")
        print(f"    |  free blocks:  {free_ids}")
        print(f"    |  used blocks:  {used_ids}")

        tables = mgr.dump_tables()
        for seq_id, mapping in tables.items():
            logical_map = ", ".join(
                f"L{m['logical']}->P{m['physical']}" for m in mapping
            )
            print(f"    |  [{seq_id}] BlockTable: {logical_map}")

        allocated = mgr.get_trace_allocated()
        freed = mgr.get_trace_freed()
        for seq_id, pids in allocated.items():
            print(f"    |  [{seq_id}] ALLOC: blocks {pids}")
        for seq_id, pids in freed.items():
            print(f"    |  [{seq_id}] FREE:  blocks {pids}")

        block_size = self._config.block_size
        for sg in self._queue.running:
            for seq in sg.get_unfinished_seqs():
                allocated_count = len(seq.block_table)
                # On-demand: allocated blocks == ceil(kv_tokens_written / block_size)
                # (actually == blocks needed for current data, no waste)
                total_lifetime = len(seq.prompt_token_ids) + seq.sampling_params.max_tokens
                blocks_if_eager = (total_lifetime + block_size - 1) // block_size
                print(f"    |  [{seq.seq_id}] "
                      f"blocks={allocated_count} (would be {blocks_if_eager} with eager), "
                      f"saved={blocks_if_eager - allocated_count}")

        print(f"    +-")
        mgr.clear_trace_events()

    @property
    def config(self) -> Config:
        return self._config

    @property
    def queue(self) -> RequestQueue:
        return self._queue

    @property
    def block_manager(self) -> BlockManager:
        return self._block_manager

    @property
    def executor(self) -> Executor:
        return self._executor

    @property
    def engine_core(self) -> EngineCore:
        return self._engine_core
