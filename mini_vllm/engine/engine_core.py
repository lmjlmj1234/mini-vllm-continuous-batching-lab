from __future__ import annotations

import time
from typing import List, Optional

from ..executor.base import Executor
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
        metrics_collector: Optional[MetricsCollector] = None,
        profiler: Optional[StageProfiler] = None,
    ) -> None:
        self._scheduler = scheduler
        self._executor = executor
        self._metrics = metrics_collector or MetricsCollector()
        self._profiler = profiler or StageProfiler()
        self._step_count = 0

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
                with self._profiler.record("executor_forward"):
                    if has_prefill:
                        with self._profiler.record("prefill"):
                            self._executor.prefill(only_prefill_seqs)
                        for seq in only_prefill_seqs:
                            if seq.is_prefill_finished:
                                # first_token_time is set AFTER prefill completes,
                                # marking when the first output token was produced.
                                # This is more accurate than capturing time before
                                # prefill (which would under-report TTFT).
                                seq.first_token_time = time.time()
                    if has_decode:
                        with self._profiler.record("decode"):
                            self._executor.decode(decode_seqs)

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
                self._metrics.record_step(
                    result, sched_latency, step_wall, total_blocks, used_blocks
                )

        return result

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
