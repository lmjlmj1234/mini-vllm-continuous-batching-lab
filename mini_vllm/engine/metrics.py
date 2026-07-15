from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..scheduler.schedule_result import ScheduleResult
from ..sequence.sequence import Sequence
from ..sequence.status import Status

if TYPE_CHECKING:
    from ..cache.manager import BlockManager


def _percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (p / 100.0) * (len(sorted_vals) - 1)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


class MetricsCollector:
    """Collect and report performance metrics for an engine run.

    Design principle: one central collector, not log lines scattered
    across modules.  Each metric answers a specific question about the
    runtime architecture.

    Metrics tracked:

      **TTFT** (Time To First Token): first_token_time - arrival_time
          Measures scheduler + prefill latency.  Reported as avg/P50/P95 in ms.

      **TPOT** (Time Per Output Token, a.k.a. inter-token latency):
          (finish_time - first_token_time) / max(num_output_tokens - 1, 1)
          Measures average decode latency between successive output tokens.
          Sequences with only 1 output token are excluded (no inter-token gap
          to measure).  Reported as avg/P50/P95 in ms.

      **E2E latency**: finish_time - arrival_time, per completed request.

      **Throughput** (request_throughput_rps & token_throughput_tps):
          completed_requests / total_elapsed_time  (req/s)
          total_output_tokens / total_elapsed_time  (tok/s)
          Only FINISHED sequences (not cancelled/timeout) are counted.

      **KV utilisation**:  peak / total physical blocks in use
      **Block utilisation**:  tokens packed per allocated block
      **Scheduler latency**:  overhead of scheduler.schedule() per step

      **Effective batch size**: mean/max running requests per step.

    Formulas:
        TTFT  = first_token_time - arrival_time
        TPOT  = (finish_time - first_token_time) / max(num_output_tokens - 1, 1)
               (single-token outputs excluded)
        req/s = completed_requests / (last_finish_time - first_arrival_time)
        tok/s = total_output_tokens / (last_finish_time - first_arrival_time)
    """

    def __init__(self, block_manager: Optional[BlockManager] = None) -> None:
        self._block_manager = block_manager
        self._step_times: List[float] = []  # wall-clock per step (ms)
        self._scheduler_times: List[float] = []  # scheduler.schedule() (ms)
        self._timeline_prefill: List[int] = []
        self._timeline_decode: List[int] = []
        self._timeline_cached: List[int] = []
        self._timeline_blocks: List[int] = []
        self._timeline_total_blocks: int = 0
        self._finished_seqs: List[Sequence] = []

        # Per-step batch size tracking
        self._effective_batch_sizes: List[int] = []
        self._running_request_counts: List[int] = []
        self._waiting_request_counts: List[int] = []

        # Per-request raw data for detailed reporting
        self._per_request_data: List[Dict[str, Any]] = []

        # Serving-layer counters
        self._total_requests: int = 0
        self._rejected_requests: int = 0
        self._cancelled_requests: int = 0
        self._timeout_requests: int = 0
        self._rpm_rejected: int = 0
        self._tpm_rejected: int = 0

    # ------------------------------------------------------------------
    # Recording – called by EngineCore after each step
    # ------------------------------------------------------------------

    def record_step(
        self,
        result: ScheduleResult,
        scheduler_latency: float,
        step_wall_time: float,
        total_blocks: int,
        used_blocks: int,
        effective_batch_size: int = 0,
        running_count: int = 0,
        waiting_count: int = 0,
    ) -> None:
        """Record data for one engine step."""
        self._scheduler_times.append(scheduler_latency * 1000)  # ms
        self._step_times.append(step_wall_time * 1000)
        self._timeline_prefill.append(result.num_prefill_tokens)
        self._timeline_cached.append(result.cached_token_count)
        self._timeline_decode.append(result.num_decode_tokens)
        self._timeline_blocks.append(used_blocks)
        self._timeline_total_blocks = total_blocks
        self._effective_batch_sizes.append(effective_batch_size)
        self._running_request_counts.append(running_count)
        self._waiting_request_counts.append(waiting_count)

    def register_sequence(self, seq: Sequence) -> None:
        """Register a finished sequence for final metrics."""
        self._finished_seqs.append(seq)
        # Record per-request data
        if seq.arrival_time is not None and seq.finish_time is not None:
            e2e = seq.finish_time - seq.arrival_time
            ttft = None
            tpot = None
            if seq.first_token_time is not None:
                ttft = seq.first_token_time - seq.arrival_time
            if (
                seq.num_output_tokens > 1
                and seq.first_token_time is not None
                and seq.finish_time is not None
            ):
                decode_time = seq.finish_time - seq.first_token_time
                tpot = decode_time / (seq.num_output_tokens - 1)
            self._per_request_data.append({
                "seq_id": seq.seq_id,
                "status": seq.status.name,
                "prompt_length": seq.prompt_length,
                "num_output_tokens": seq.num_output_tokens,
                "ttft_s": ttft,
                "tpot_s": tpot,
                "e2e_s": e2e,
            })

    # ------------------------------------------------------------------
    # Serving-layer counters
    # ------------------------------------------------------------------

    def count_request(self) -> None:
        self._total_requests += 1

    def count_rejected(self) -> None:
        self._rejected_requests += 1

    def count_cancelled(self) -> None:
        self._cancelled_requests += 1

    def count_timeout(self) -> None:
        self._timeout_requests += 1

    def count_rpm_rejected(self) -> None:
        self._rpm_rejected += 1

    def count_tpm_rejected(self) -> None:
        self._tpm_rejected += 1

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def per_request_data(self) -> List[Dict[str, Any]]:
        """Return detailed per-request data."""
        return list(self._per_request_data)

    def report(self, include_per_request: bool = False) -> dict:
        """Generate a comprehensive benchmark report dict.

        All timing values are in milliseconds unless noted.
        """
        finished = [s for s in self._finished_seqs if s.status == Status.FINISHED]
        total_sequences = len(finished)
        total_steps = len(self._step_times)

        # --- TTFT (Time to First Token) ---
        ttft_values = []
        for s in finished:
            if s.first_token_time is not None and s.arrival_time is not None:
                ttft = s.first_token_time - s.arrival_time
                ttft_values.append(ttft)

        avg_ttft_ms = (sum(ttft_values) / len(ttft_values) * 1000) if ttft_values else 0.0
        max_ttft_ms = max(ttft_values) * 1000 if ttft_values else 0.0
        min_ttft_ms = min(ttft_values) * 1000 if ttft_values else 0.0
        p50_ttft_ms = _percentile([t * 1000 for t in ttft_values], 50) if ttft_values else 0.0
        p95_ttft_ms = _percentile([t * 1000 for t in ttft_values], 95) if ttft_values else 0.0
        std_ttft_ms = statistics.stdev([t * 1000 for t in ttft_values]) if len(ttft_values) > 1 else 0.0

        # --- TPOT (Time Per Output Token / inter-token latency) ---
        tpot_values = []
        for s in finished:
            if (
                s.num_output_tokens > 1  # need at least one inter-token gap
                and s.first_token_time is not None
                and s.finish_time is not None
            ):
                decode_time = s.finish_time - s.first_token_time
                tpot_values.append(decode_time / (s.num_output_tokens - 1))

        avg_tpot_ms = (sum(tpot_values) / len(tpot_values) * 1000) if tpot_values else 0.0
        max_tpot_ms = max(tpot_values) * 1000 if tpot_values else 0.0
        min_tpot_ms = min(tpot_values) * 1000 if tpot_values else 0.0
        p50_tpot_ms = _percentile([t * 1000 for t in tpot_values], 50) if tpot_values else 0.0
        p95_tpot_ms = _percentile([t * 1000 for t in tpot_values], 95) if tpot_values else 0.0
        std_tpot_ms = statistics.stdev([t * 1000 for t in tpot_values]) if len(tpot_values) > 1 else 0.0

        # --- E2E latency ---
        e2e_values = []
        for s in finished:
            if s.arrival_time is not None and s.finish_time is not None:
                e2e_values.append(s.finish_time - s.arrival_time)

        avg_e2e_ms = (sum(e2e_values) / len(e2e_values) * 1000) if e2e_values else 0.0
        max_e2e_ms = max(e2e_values) * 1000 if e2e_values else 0.0
        min_e2e_ms = min(e2e_values) * 1000 if e2e_values else 0.0
        p50_e2e_ms = _percentile([t * 1000 for t in e2e_values], 50) if e2e_values else 0.0
        p95_e2e_ms = _percentile([t * 1000 for t in e2e_values], 95) if e2e_values else 0.0

        # --- End-to-end time (wall-clock elapsed) ---
        #
        # total_time_s:  wall-clock from earliest arrival to latest finish.
        #     This correctly captures idle gaps from staggered arrivals.
        #     req/s = N / total_time_s gives the *workload-level* throughput.
        #
        # active_time_s:  sum of step wall-times (engine actively processing).
        #     Excludes idle gaps (no requests in system).
        #     req/s = N / active_time_s gives the *system-level* throughput
        #     under continuous load.
        #
        arrivals = [s.arrival_time for s in finished if s.arrival_time is not None]
        finishes = [s.finish_time for s in finished if s.finish_time is not None]
        total_time_s = max(finishes) - min(arrivals) if arrivals and finishes else 0.0
        active_time_s = sum(self._step_times) / 1000  # step_times stored in ms

        total_output_tokens = sum(s.num_output_tokens for s in finished)
        total_prompt_tokens = sum(s.prompt_length for s in finished)

        # --- Throughput ---
        throughput_req = total_sequences / total_time_s if total_time_s > 0 else 0.0
        throughput_tok = total_output_tokens / total_time_s if total_time_s > 0 else 0.0
        active_throughput_req = total_sequences / active_time_s if active_time_s > 0 else 0.0
        active_throughput_tok = total_output_tokens / active_time_s if active_time_s > 0 else 0.0

        # --- KV block utilization ---
        total_blocks = self._timeline_total_blocks
        if self._timeline_blocks and total_blocks > 0:
            peak_blocks = max(self._timeline_blocks)
            avg_blocks = sum(self._timeline_blocks) / len(self._timeline_blocks)
            kv_util_peak_pct = (peak_blocks / total_blocks) * 100
            kv_util_avg_pct = (avg_blocks / total_blocks) * 100
        else:
            peak_blocks = 0
            kv_util_peak_pct = 0.0
            kv_util_avg_pct = 0.0

        # --- Per-sequence block utilisation ---
        block_util_values = []
        for s in finished:
            if self._block_manager is not None:
                total_blocks_for_seq = len(self._block_manager.get_block_table(s.seq_id))
            else:
                total_blocks_for_seq = 0
            total_tokens = s.prompt_length + s.num_output_tokens
            if total_blocks_for_seq > 0:
                util = total_tokens / total_blocks_for_seq
                block_util_values.append(util)

        avg_block_util = (
            sum(block_util_values) / len(block_util_values) if block_util_values else 0.0
        )

        # --- Scheduler latency ---
        avg_sched_ms = (
            sum(self._scheduler_times) / len(self._scheduler_times)
            if self._scheduler_times
            else 0.0
        )
        max_sched_ms = max(self._scheduler_times) if self._scheduler_times else 0.0

        # --- Process latency (step wall time) ---
        avg_step_ms = (
            sum(self._step_times) / len(self._step_times) if self._step_times else 0.0
        )
        total_elapsed_ms = sum(self._step_times)

        # --- Scheduling stats ---
        total_scheduler_steps = len(self._step_times)
        mean_effective_batch = (
            sum(self._effective_batch_sizes) / len(self._effective_batch_sizes)
            if self._effective_batch_sizes else 0.0
        )
        max_effective_batch = max(self._effective_batch_sizes) if self._effective_batch_sizes else 0
        mean_running = (
            sum(self._running_request_counts) / len(self._running_request_counts)
            if self._running_request_counts else 0.0
        )
        peak_running = max(self._running_request_counts) if self._running_request_counts else 0
        peak_waiting = max(self._waiting_request_counts) if self._waiting_request_counts else 0

        # --- Total generated tokens ---
        total_generated_tokens = sum(s.num_output_tokens for s in finished)

        result = {
            # Request counts
            "total_requests": total_sequences,
            "total_steps": total_steps,
            # TTFT
            "avg_ttft_ms": round(avg_ttft_ms, 2),
            "min_ttft_ms": round(min_ttft_ms, 2),
            "max_ttft_ms": round(max_ttft_ms, 2),
            "p50_ttft_ms": round(p50_ttft_ms, 2),
            "p95_ttft_ms": round(p95_ttft_ms, 2),
            "std_ttft_ms": round(std_ttft_ms, 2),
            # TPOT
            "avg_tpot_ms": round(avg_tpot_ms, 2),
            "min_tpot_ms": round(min_tpot_ms, 2),
            "max_tpot_ms": round(max_tpot_ms, 2),
            "p50_tpot_ms": round(p50_tpot_ms, 2),
            "p95_tpot_ms": round(p95_tpot_ms, 2),
            "std_tpot_ms": round(std_tpot_ms, 2),
            # E2E latency
            "avg_e2e_ms": round(avg_e2e_ms, 2),
            "min_e2e_ms": round(min_e2e_ms, 2),
            "max_e2e_ms": round(max_e2e_ms, 2),
            "p50_e2e_ms": round(p50_e2e_ms, 2),
            "p95_e2e_ms": round(p95_e2e_ms, 2),
            "std_e2e_ms": round(statistics.stdev([v * 1000 for v in e2e_values]), 2)
            if len(e2e_values) > 1 else 0.0,
            # Throughput (wall-clock = workload-level)
            "throughput_req_per_sec": round(throughput_req, 2),
            "throughput_tok_per_sec": round(throughput_tok, 2),
            # Throughput (active = system-level, excluding idle gaps)
            "active_throughput_req_per_sec": round(active_throughput_req, 2),
            "active_throughput_tok_per_sec": round(active_throughput_tok, 2),
            "total_generated_tokens": total_generated_tokens,
            "total_output_tokens": total_output_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_time_seconds": round(total_time_s, 3),
            "active_time_seconds": round(active_time_s, 3),
            # KV utilisation
            "kv_total_blocks": total_blocks,
            "kv_peak_blocks": peak_blocks,
            "kv_util_peak_pct": round(kv_util_peak_pct, 1),
            "kv_util_avg_pct": round(kv_util_avg_pct, 1),
            # Block utilisation
            "avg_block_util_tokens_per_block": round(avg_block_util, 2),
            # Prefix cache
            "total_cached_tokens": sum(self._timeline_cached),
            "prefix_cache_hit_rate": (
                round(sum(self._timeline_cached) / total_prompt_tokens * 100, 1)
                if total_prompt_tokens > 0 else 0.0
            ),
            # Serving-layer
            "total_requests_serving": self._total_requests,
            "rejected_requests": self._rejected_requests,
            "cancelled_requests": self._cancelled_requests,
            "timeout_requests": self._timeout_requests,
            "rpm_rejected": self._rpm_rejected,
            "tpm_rejected": self._tpm_rejected,
            # Scheduler
            "avg_scheduler_latency_ms": round(avg_sched_ms, 4),
            "max_scheduler_latency_ms": round(max_sched_ms, 4),
            "avg_step_latency_ms": round(avg_step_ms, 4),
            "total_elapsed_ms": round(total_elapsed_ms, 2),
            # Scheduling stats
            "total_scheduler_steps": total_scheduler_steps,
            "mean_effective_batch_size": round(mean_effective_batch, 2),
            "max_effective_batch_size": max_effective_batch,
            "mean_running_requests": round(mean_running, 2),
            "peak_running_requests": peak_running,
            "peak_waiting_requests": peak_waiting,
        }

        if include_per_request:
            result["per_request"] = self._per_request_data

        return result

    def print_report(self, report: Optional[dict] = None) -> None:
        """Print a human-readable benchmark report."""
        r = report or self.report()

        title = "Benchmark Report"
        sep = "=" * len(title)
        print(f"\n{sep}\n{title}\n{sep}\n")

        print(f"  Requests:              {r['total_requests']}  "
              f"(prompt={r['total_prompt_tokens']} tok, "
              f"output={r['total_output_tokens']} tok)")
        print(f"  Steps:                 {r['total_steps']}")
        print(f"  Total time (wall):        {r['total_time_seconds']}s  "
              f"(active: {r['active_time_seconds']}s)\n")

        print(f"  TTFT (avg/P50/P95):      {r['avg_ttft_ms']} / {r['p50_ttft_ms']} / "
              f"{r['p95_ttft_ms']} ms")
        print(f"  TPOT (avg/P50/P95):      {r['avg_tpot_ms']} / {r['p50_tpot_ms']} / "
              f"{r['p95_tpot_ms']} ms")
        print(f"  E2E (avg/P50/P95):       {r['avg_e2e_ms']} / {r['p50_e2e_ms']} / "
              f"{r['p95_e2e_ms']} ms\n")

        print(f"  Throughput (wall):        {r['throughput_req_per_sec']} req/s,  "
              f"{r['throughput_tok_per_sec']} tok/s")
        print(f"  Throughput (active):      {r['active_throughput_req_per_sec']} req/s,  "
              f"{r['active_throughput_tok_per_sec']} tok/s\n")

        print(f"  Effective batch (mean/max):  {r['mean_effective_batch_size']} / "
              f"{r['max_effective_batch_size']}")
        print(f"  Running requests (mean/peak): {r['mean_running_requests']} / "
              f"{r['peak_running_requests']}")
        print(f"  Peak waiting:                {r['peak_waiting_requests']}\n")

        print(f"  KV blocks (peak/avg):  {r['kv_peak_blocks']} / "
              f"{r['kv_util_avg_pct']}%  (of {r['kv_total_blocks']} total)")
        print(f"  Block utilisation:     {r['avg_block_util_tokens_per_block']} "
              f"tokens/block")
        if r['total_cached_tokens'] > 0:
            print(f"  Prefix cache:          {r['total_cached_tokens']} tokens cached  "
                  f"(hit rate: {r['prefix_cache_hit_rate']}% of prompt tokens)\n")
        else:
            print()

        print(f"  Scheduler latency:     {r['avg_scheduler_latency_ms']} ms avg,  "
              f"{r['max_scheduler_latency_ms']} ms max")
        print(f"  Step latency:          {r['avg_step_latency_ms']} ms avg,  "
              f"{r['total_elapsed_ms']} ms total\n")
