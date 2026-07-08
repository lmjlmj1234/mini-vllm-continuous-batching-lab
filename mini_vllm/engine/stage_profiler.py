from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, List, Optional


_STAGES = [
    "request_queue_waiting",
    "scheduler_step",
    "kv_cache_allocation",
    "prefix_cache_lookup",
    "prefill",
    "decode",
    "executor_forward",
    "kv_cache_release",
    "metrics_update",
    "engine_step_total",
]

_STAGE_LABELS = {
    "request_queue_waiting": "request_queue_waiting",
    "scheduler_step": "scheduler_step",
    "kv_cache_allocation": "kv_cache_allocation",
    "prefix_cache_lookup": "prefix_cache_lookup",
    "prefill": "prefill",
    "decode": "decode",
    "executor_forward": "executor_forward",
    "kv_cache_release": "kv_cache_release",
    "metrics_update": "metrics_update",
    "engine_step_total": "engine_step_total",
}


class StageProfiler:
    """Lightweight stage-level profiler for LLM serving request breakdown.

    Records wall-clock durations for individual stages of the engine step
    loop and produces a summary report showing count, total, average, max,
    and percentage-of-total for each stage.

    Usage::

        profiler = StageProfiler()

        # Context-manager style (preferred)
        with profiler.record("scheduler_step"):
            scheduler.schedule()

        # Start/end style
        profiler.start("prefill")
        executor.prefill(seqs)
        profiler.end("prefill")

        # Report
        profiler.print_report()
        report = profiler.report()
    """

    def __init__(self) -> None:
        self._records: Dict[str, List[float]] = {s: [] for s in _STAGES}
        self._stack: Dict[str, float] = {}
        self._total_requests: int = 0
        self._total_engine_steps: int = 0

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @contextmanager
    def record(self, stage: str) -> None:
        """Context manager: time *stage* and record the duration in seconds."""
        start = time.time()
        try:
            yield
        finally:
            self._records.setdefault(stage, []).append(time.time() - start)

    def start(self, stage: str) -> None:
        """Begin timing *stage*."""
        self._stack[stage] = time.time()

    def end(self, stage: str) -> None:
        """End timing *stage* and record the duration in seconds."""
        start = self._stack.pop(stage, None)
        if start is not None:
            self._records.setdefault(stage, []).append(time.time() - start)

    def record_raw(self, stage: str, duration_s: float) -> None:
        """Record a duration *in seconds* for *stage* directly."""
        self._records.setdefault(stage, []).append(duration_s)

    def increment_requests(self, count: int = 1) -> None:
        self._total_requests += count

    def increment_steps(self, count: int = 1) -> None:
        self._total_engine_steps += count

    def reset(self) -> None:
        """Clear all recorded data."""
        for s in _STAGES:
            self._records[s].clear()
        self._records.clear()
        self._stack.clear()
        self._total_requests = 0
        self._total_engine_steps = 0

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> dict:
        """Compute statistics for every stage and return as a dict.

        Returns::

            {
                "stages": {
                    "scheduler_step": {
                        "count": 10, "total_ms": 5.2, "avg_ms": 0.52,
                        "max_ms": 1.1, "percent_of_total": 12.3,
                    },
                    ...
                },
                "total_profiled_ms": 42.0,
                "total_requests": 4,
                "total_engine_steps": 10,
            }

        Stages with zero observations are omitted from the per-stage dict.
        """
        stages: dict = {}
        all_durations: List[float] = []
        for stage in _STAGES:
            durations = self._records.get(stage, [])
            if not durations:
                continue
            all_durations.extend(durations)
            total_s = sum(durations)
            total_ms = total_s * 1000
            avg_ms = (total_s / len(durations)) * 1000
            max_ms = max(durations) * 1000
            stages[stage] = {
                "count": len(durations),
                "total_ms": round(total_ms, 2),
                "avg_ms": round(avg_ms, 4),
                "max_ms": round(max_ms, 4),
            }

        # Compute percentages based on engine_step_total if available,
        # otherwise based on total of all recorded stage times.
        total_profiled_s = sum(sum(v) for v in self._records.values())
        total_profiled_ms = total_profiled_s * 1000

        engine_total = self._records.get("engine_step_total", [])
        base_total_s = sum(engine_total) if engine_total else total_profiled_s

        for stage, info in stages.items():
            if base_total_s > 0:
                total_for_stage_s = sum(self._records.get(stage, []))
                pct = (total_for_stage_s / base_total_s) * 100
            else:
                pct = 0.0
            info["percent_of_total"] = round(pct, 1)

        # Sort stages by total_ms descending
        sorted_stages = dict(
            sorted(stages.items(), key=lambda x: x[1]["total_ms"], reverse=True)
        )

        return {
            "stages": sorted_stages,
            "total_profiled_ms": round(total_profiled_ms, 2),
            "total_requests": self._total_requests,
            "total_engine_steps": self._total_engine_steps,
        }

    def print_report(self) -> None:
        """Print a formatted stage breakdown table."""
        r = self.report()
        stages = r["stages"]

        if not stages:
            print("  (no profiling data recorded)")
            return

        print("\n  Stage Breakdown")
        print("  " + "-" * 75)
        header = f"  {'stage':<26s} {'count':>6s}  {'total_ms':>9s}  {'avg_ms':>8s}  {'max_ms':>8s}  {'pct':>5s}"
        print(header)
        print("  " + "-" * 75)
        for stage, info in stages.items():
            name = _STAGE_LABELS.get(stage, stage)
            print(
                f"  {name:<26s} {info['count']:>6d}  "
                f"{info['total_ms']:>9.2f}  {info['avg_ms']:>8.4f}  "
                f"{info['max_ms']:>8.4f}  {info['percent_of_total']:>5.1f}%"
            )
        print("  " + "-" * 75)
        print(f"  {'Total profiled time':36s} {r['total_profiled_ms']:>9.2f} ms")
        print(f"  {'Total requests':36s} {r['total_requests']:>9d}")
        print(f"  {'Total engine steps':36s} {r['total_engine_steps']:>9d}")

        # --- Bottleneck hint ---
        self._print_bottleneck_hint(stages, r)

    # ------------------------------------------------------------------
    # Bottleneck hints
    # ------------------------------------------------------------------

    @staticmethod
    def _print_bottleneck_hint(stages: dict, report: dict) -> None:
        """Print a simple bottleneck hint based on stage breakdown."""
        if not stages:
            return

        sorted_by_pct = sorted(
            stages.items(), key=lambda x: x[1]["percent_of_total"], reverse=True
        )
        top_stage, top_info = sorted_by_pct[0]
        top_pct = top_info["percent_of_total"]

        if top_pct < 30:
            return  # no dominant stage

        hints = {
            "request_queue_waiting": (
                "Request queue waiting dominates ({pct}%). "
                "This may indicate queueing / admission control / batch capacity "
                "is the bottleneck. Consider increasing max_num_seqs or checking "
                "token budget constraints."
            ),
            "scheduler_step": (
                "Scheduler step dominates ({pct}%). "
                "CPU scheduler overhead may be high. "
                "Consider reviewing scheduling algorithms for efficiency."
            ),
            "executor_forward": (
                "Executor forward dominates ({pct}%). "
                "Most time is spent in model execution "
                "(or the fake executor simulation)."
            ),
            "prefill": (
                "Prefill dominates ({pct}%). "
                "Long prompt processing may be the bottleneck. "
                "Consider chunked prefill or reducing prompt lengths."
            ),
            "decode": (
                "Decode dominates ({pct}%). "
                "Decode / KV cache read may be the bottleneck. "
                "Consider larger batch sizes or KV cache optimization."
            ),
            "kv_cache_allocation": (
                "KV cache allocation dominates ({pct}%). "
                "Block allocation overhead may be significant. "
                "Consider larger block sizes or caching allocator state."
            ),
            "prefix_cache_lookup": (
                "Prefix cache lookup dominates ({pct}%). "
                "Hash computation or cache probing may be slow for long prompts. "
                "Consider optimising the hash function or cache data structure."
            ),
            "kv_cache_release": (
                "KV cache release dominates ({pct}%). "
                "Block free overhead is relatively high. "
                "Consider deferred or batched freeing."
            ),
            "metrics_update": (
                "Metrics update dominates ({pct}%). "
                "Metrics collection overhead is relatively high. "
                "Consider reducing metric computation frequency."
            ),
        }

        if top_stage in hints:
            print()
            print(f"  Bottleneck hint: {hints[top_stage].format(pct=top_pct)}")
