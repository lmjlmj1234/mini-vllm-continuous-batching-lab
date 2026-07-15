from __future__ import annotations

import time
from typing import List, Optional

import torch

from ..cache.manager import BlockManager
from ..config import Config
from ..executor.base import Executor
from ..model_runner.base import ModelRunnerOutput
from ..engine.input_builder import ModelInputBuilder
from ..scheduler.scheduler import Scheduler
from ..scheduler.schedule_result import ScheduleResult
from ..sequence.sequence import Sequence
from ..sequence.status import Status
from .metrics import MetricsCollector
from .stage_profiler import StageProfiler


class EngineCore:
    """Inner engine loop — analogous to vLLM's ``EngineCore``.

    ``EngineCore`` owns the scheduler and the model executor.  It runs
    one step at a time::

        EngineCore.step()
            ↓
        result = scheduler.schedule()
        executor.prefill(prefill_seqs)
        executor.decode(decode_seqs)
        executor.cleanup(finished_seqs)
            ↓
        result (with finished/output tracking)
    """

    def __init__(
        self,
        scheduler: Scheduler,
        executor: Executor,
        block_manager: Optional['BlockManager'] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        profiler: Optional[StageProfiler] = None,
    ) -> None:
        self._scheduler = scheduler
        self._executor = executor
        self._config: Config = scheduler._config
        self._block_manager = block_manager
        self._metrics = metrics_collector or MetricsCollector(
            block_manager=block_manager
        )
        self._profiler = profiler or StageProfiler()
        self._step_count = 0
        # Wire up scheduler trace from config
        self._scheduler.enable_trace(
            getattr(self._config, 'trace_enabled', False)
        )
        # Phase 1.5: ModelInputBuilder
        device = torch.device(self._config.device) if hasattr(self._config, 'device') else None
        self._input_builder = ModelInputBuilder(
            block_manager=block_manager,  # type: ignore[arg-type]
            config=self._config,
            device=device,
        )

    def step(self) -> ScheduleResult:
        """Run one scheduling-and-execution step.

        Returns the ``ScheduleResult`` for inspection / testing.
        """
        step_start = time.time()
        with self._profiler.record("engine_step_total"):
            self._step_count += 1
            self._profiler.increment_steps()

            # Check for timeouts before scheduling
            self._check_timeouts()

            # --- Schedule ---
            sched_start = time.time()
            with self._profiler.record("scheduler_step"):
                result = self._scheduler.schedule()
            sched_latency = time.time() - sched_start

            # Record request queue waiting for newly admitted sequences
            for sg in result.scheduled_prefill_groups:
                for seq in sg.get_unfinished_seqs():
                    if seq.first_scheduled_time is not None:
                        continue
                    seq.first_scheduled_time = time.time()
                    waiting_s = seq.first_scheduled_time - seq.arrival_time
                    self._profiler.record_raw("request_queue_waiting", waiting_s)
                    self._profiler.increment_requests()

            # --- Prefill & Decode (with combined executor_forward) ---
            only_prefill_seqs: List[Sequence] = []
            for sg in result.scheduled_prefill_groups:
                for seq in sg.get_unfinished_seqs():
                    if seq.status == Status.PREFILL:
                        only_prefill_seqs.append(seq)

            decode_seqs: List[Sequence] = []
            for sg in result.scheduled_decode_groups:
                decode_seqs.extend(sg.get_unfinished_seqs())

            has_prefill = bool(only_prefill_seqs)
            has_decode = bool(decode_seqs)

            if has_prefill or has_decode:
                # Ensure blocks before building ModelInput
                # (allocation mirrors what the old executor.prefill()/decode()
                #  did via ensure_block — blocks must exist for slot mapping)
                bm = self._block_manager
                if has_prefill and bm is not None:
                    for seq in only_prefill_seqs:
                        chunk_end = min(
                            seq.prefill_cursor + self._config.max_prefill_chunk_size,
                            len(seq.prompt_token_ids),
                        )
                        for pos in range(seq.prefill_cursor, chunk_end):
                            bm.ensure_block(seq, pos)
                if has_decode and bm is not None:
                    for seq in decode_seqs:
                        # Ensure block at the position where the new token's
                        # KV will be written: cached_len_before.
                        # Decode invariant: num_generated_tokens >= 1
                        pos = len(seq.prompt_token_ids) + seq.num_generated_tokens - 1
                        bm.ensure_block(seq, pos)

                with self._profiler.record("executor_forward"):
                    model_input = self._input_builder.build(
                        prefill_seqs=only_prefill_seqs,
                        decode_seqs=decode_seqs,
                    )
                    model_output = self._executor.execute(model_input)
                    self._apply_model_output(
                        model_output, only_prefill_seqs, decode_seqs,
                    )

                    # Advance prefill cursor for all prefilling sequences
                    chunk_size = self._config.max_prefill_chunk_size
                    for seq in only_prefill_seqs:
                        end = min(
                            seq.prefill_cursor + chunk_size,
                            len(seq.prompt_token_ids),
                        )
                        seq.prefill_cursor = end

            # --- Cleanup finished sequences ---
            for sg in result.finished_groups:
                for seq in sg.seqs:
                    self._executor.cleanup_sequence(seq.seq_id)
                    self._metrics.register_sequence(seq)

            # --- Record step metrics ---
            step_wall = time.time() - step_start
            with self._profiler.record("metrics_update"):
                bm_stats = self._scheduler.block_manager_stats()
                total_blocks = bm_stats["total_blocks"]
                used_blocks = bm_stats["used_blocks"]
                effective_batch_size = (
                    len(result.scheduled_decode_groups)
                    + len(result.scheduled_prefill_groups)
                )
                running_count = self._scheduler._queue.num_running
                waiting_count = self._scheduler._queue.num_waiting
                self._metrics.record_step(
                    result, sched_latency, step_wall, total_blocks, used_blocks,
                    effective_batch_size=effective_batch_size,
                    running_count=running_count,
                    waiting_count=waiting_count,
                )

        return result

    # ------------------------------------------------------------------
    # Apply model output (Phase 1.5)
    # ------------------------------------------------------------------

    def _apply_model_output(
        self,
        output: ModelRunnerOutput,
        prefill_seqs: List[Sequence],
        decode_seqs: List[Sequence],
    ) -> None:
        """Write sampled tokens back to the correct Sequence objects.

        Maps ``sampled_sequence_ids`` from the output back to the
        source ``Sequence`` by ``seq_id``.
        """
        # Build seq_id → Sequence lookup
        seq_map: dict = {}
        for seq in prefill_seqs:
            seq_map[seq.seq_id] = seq
        for seq in decode_seqs:
            seq_map[seq.seq_id] = seq

        now = time.time()
        for token_id, seq_id in zip(
            output.sampled_token_ids, output.sampled_sequence_ids
        ):
            seq = seq_map.get(seq_id)
            if seq is None:
                continue

            if seq.status == Status.PREFILL:
                # Completing prefill — first output token
                seq.output_token_ids = [token_id]
                seq.num_generated_tokens = 1
                seq.first_token_time = now
                seq.status = Status.RUNNING
            else:
                # Decode or already-running prefill — append token
                seq.output_token_ids.append(token_id)
                seq.num_generated_tokens += 1

    @property
    def metrics_collector(self) -> MetricsCollector:
        return self._metrics

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Cancel / Timeout
    # ------------------------------------------------------------------

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a running or waiting request and free its resources.

        Returns True if the request was found and cancelled.
        """
        sg = self._scheduler._queue.get_by_id(request_id)
        if sg is None:
            return False

        for seq in sg.seqs:
            if not seq.finished:
                seq.status = Status.CANCELLED
                seq.finish_time = time.time()
                self._scheduler._block_manager.free(seq.seq_id)
                self._executor.cleanup_sequence(seq.seq_id)
                self._metrics.register_sequence(seq)

        # Count cancellation even if no sequences (unscheduled request)
        self._metrics.count_cancelled()

        # Remove from queue
        if request_id in self._scheduler._queue._running:
            self._scheduler._queue._running.pop(request_id)
        elif request_id in self._scheduler._queue._waiting:
            self._scheduler._queue._waiting.pop(request_id)
        if request_id not in self._scheduler._queue._finished:
            self._scheduler._queue._finished[request_id] = sg
        return True

    def _check_timeouts(self) -> None:
        """Find and cancel requests exceeding timeout threshold."""
        now = time.time()
        timeout = self._scheduler._config.request_timeout_s
        to_cancel: List[str] = []
        for sg in self._scheduler._queue.running:
            if now - sg.arrival_time > timeout:
                to_cancel.append((sg.request_id, "running"))
        for sg in self._scheduler._queue.waiting:
            if now - sg.arrival_time > timeout:
                to_cancel.append((sg.request_id, "waiting"))
        for rid, pool_name in to_cancel:
            sg = self._scheduler._queue.get_by_id(rid)
            if sg is None:
                continue
            # Cancel sequences if any were created (scheduled requests)
            for seq in sg.seqs:
                if not seq.finished:
                    seq.status = Status.TIMEOUT
                    seq.finish_time = now
                    self._scheduler._block_manager.free(seq.seq_id)
                    self._executor.cleanup_sequence(seq.seq_id)
                    self._metrics.register_sequence(seq)
            # Count timeout even for groups with no sequences (unscheduled)
            self._metrics.count_timeout()
            if pool_name == "running":
                self._scheduler._queue._running.pop(rid)
            else:
                self._scheduler._queue._waiting.pop(rid)
            if rid not in self._scheduler._queue._finished:
                self._scheduler._queue._finished[rid] = sg
