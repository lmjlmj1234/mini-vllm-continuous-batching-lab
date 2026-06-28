from __future__ import annotations

import time
from typing import List, Optional

from ..executor.base import Executor
from ..scheduler.scheduler import Scheduler
from ..scheduler.schedule_result import ScheduleResult
from ..sequence.sequence import Sequence
from ..sequence.status import Status
from .metrics import MetricsCollector


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
    ) -> None:
        self._scheduler = scheduler
        self._executor = executor
        self._metrics = metrics_collector or MetricsCollector()
        self._step_count = 0

    def step(self) -> ScheduleResult:
        """Run one scheduling-and-execution step.

        Returns the ``ScheduleResult`` for inspection / testing.
        """
        step_start = time.time()
        self._step_count += 1

        # --- Schedule (timed) ---
        sched_start = time.time()
        result = self._scheduler.schedule()
        sched_latency = time.time() - sched_start

        # --- Prefill ---
        only_prefill_seqs: List[Sequence] = []
        for sg in result.scheduled_prefill_groups:
            for seq in sg.get_unfinished_seqs():
                if seq.status == Status.PREFILL:
                    only_prefill_seqs.append(seq)

        if only_prefill_seqs:
            t = time.time()
            self._executor.prefill(only_prefill_seqs)
            for seq in only_prefill_seqs:
                if seq.is_prefill_finished:
                    seq.first_token_time = t

        # --- Decode ---
        decode_seqs: List[Sequence] = []
        for sg in result.scheduled_decode_groups:
            decode_seqs.extend(sg.get_unfinished_seqs())

        if decode_seqs:
            self._executor.decode(decode_seqs)

        # --- Cleanup finished sequences ---
        for sg in result.finished_groups:
            for seq in sg.seqs:
                self._executor.cleanup_sequence(seq.seq_id)
                self._metrics.register_sequence(seq)

        # --- Record step metrics ---
        step_wall = time.time() - step_start
        bm_stats = self._scheduler.block_manager_stats()
        total_blocks = bm_stats["total_blocks"]
        used_blocks = bm_stats["used_blocks"]
        self._metrics.record_step(result, sched_latency, step_wall, total_blocks, used_blocks)

        return result

    @property
    def metrics_collector(self) -> MetricsCollector:
        return self._metrics

    @property
    def step_count(self) -> int:
        return self._step_count
