"""Metrics endpoint — exposes runtime state as JSON.

Protects: Operational visibility — without metrics, you're flying blind.

This is NOT Prometheus — it's a simple ``GET /metrics`` returning JSON.
The purpose is to expose exactly the numbers that matter for LLM serving
in a format that can be scraped by any monitoring system.

Key metric: block_utilization
-------------------------------
When this approaches 100%, the server will soon reject requests with
BLOCK_EXHAUSTED.  This is the LLM equivalent of "disk full."

Key metric: prefix_cache_hit_rate
-----------------------------------
When this drops, prefill compute increases — KV cache eviction policy
may need tuning, or request patterns have shifted.
"""

import json
from mini_vllm.engine.engine import LLMEngine


class MetricsEndpoint:
    """Generates the /metrics JSON response from engine state."""

    def __init__(
        self,
        engine: LLMEngine,
        stream_manager: object,
        rate_limiter: object,
        admission_control: object,
    ) -> None:
        self._engine = engine
        self._stream_manager = stream_manager
        self._rate_limiter = rate_limiter
        self._admission_control = admission_control

    def render(self) -> str:
        """Return a JSON string of current metrics."""
        bm = self._engine.block_manager
        alloc = bm._allocator
        queue = self._engine.queue
        mc = self._engine.engine_core.metrics_collector
        base = mc.report()

        total_blocks = alloc.num_total_blocks
        used_blocks = alloc.num_used_blocks
        block_util = round(used_blocks / total_blocks * 100, 1) if total_blocks > 0 else 0.0

        data = {
            "total_requests": queue.total,
            "running_requests": queue.num_running,
            "waiting_requests": queue.num_waiting,
            "finished_requests": queue.num_finished,
            "rejected_requests": base.get("rejected_requests", 0),
            "cancelled_requests": base.get("cancelled_requests", 0),
            "timeout_requests": base.get("timeout_requests", 0),
            "active_streams": self._stream_manager.active_count if hasattr(self._stream_manager, "active_count") else 0,
            "rpm_rejected": base.get("rpm_rejected", 0),
            "tpm_rejected": base.get("tpm_rejected", 0),
            "ttft": base.get("avg_ttft_ms", 0.0),
            "tpot": base.get("avg_tpot_ms", 0.0),
            "throughput": base.get("throughput_tok_per_sec", 0.0),
            "block_utilization": block_util,
            "prefix_cache_hit_rate": base.get("prefix_cache_hit_rate", 0.0),
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
