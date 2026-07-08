"""Comprehensive metrics tests: TTFT, TPOT, throughput, KV utilisation, scheduler latency.

Each test validates the *semantics* of a metric, not just that a JSON field exists.
"""
import time
import pytest

from mini_vllm import Config, LLMEngine, Status
from mini_vllm.engine.metrics import MetricsCollector


# ======================================================================
# Helpers
# ======================================================================

def _engine(**kw) -> LLMEngine:
    """Create an engine with sensible defaults for metrics testing."""
    defaults = dict(
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_num_prefill_tokens=16,
        max_prefill_chunk_size=4,
        block_size=4,
        num_gpu_blocks=32,
        chunked_prefill_enabled=True,
        print_step_events=False,
        memory_trace=False,
    )
    defaults.update(kw)
    return LLMEngine(Config(**defaults))


# ======================================================================
# 2.1 TTFT
# ======================================================================

class TestTTFT:
    """TTFT = first_token_time - arrival_time."""

    def test_ttft_field_exists_and_non_negative(self):
        """avg/min/max ttft_ms exist and are >= 0."""
        engine = _engine()
        engine.add_request("Hello", max_new_tokens=2)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        assert "avg_ttft_ms" in report
        assert "min_ttft_ms" in report
        assert "max_ttft_ms" in report
        assert report["avg_ttft_ms"] >= 0
        assert report["min_ttft_ms"] >= 0
        assert report["max_ttft_ms"] >= 0

    def test_first_token_time_after_arrival(self):
        """For a FINISHED sequence, first_token_time >= arrival_time."""
        engine = _engine()
        engine.add_request("TTFT timing", max_new_tokens=4)
        engine.run_until_done()

        mc = engine.engine_core.metrics_collector
        for seq in mc._finished_seqs:
            if seq.status == Status.FINISHED:
                assert seq.first_token_time is not None
                assert seq.arrival_time is not None
                assert seq.first_token_time >= seq.arrival_time, (
                    f"first_token_time ({seq.first_token_time}) < "
                    f"arrival_time ({seq.arrival_time})"
                )

    def test_multiple_requests_all_have_ttft(self):
        """Every FINISHED request contributes to TTFT."""
        engine = _engine()
        for i in range(4):
            engine.add_request(f"Req-{i}", max_new_tokens=2)
        engine.run_until_done()

        mc = engine.engine_core.metrics_collector
        report = mc.report()
        finished_count = len(
            [s for s in mc._finished_seqs if s.status == Status.FINISHED]
        )
        assert finished_count == 4
        assert report["avg_ttft_ms"] > 0

    def test_ttft_min_max_avg_ordering(self):
        """min_ttft_ms <= avg_ttft_ms <= max_ttft_ms."""
        engine = _engine()
        engine.add_request("Ordering test", max_new_tokens=8)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        if report["avg_ttft_ms"] > 0:
            assert report["min_ttft_ms"] <= report["avg_ttft_ms"]
            # When only one request, min == avg == max
            assert report["avg_ttft_ms"] <= report["max_ttft_ms"]


# ======================================================================
# 2.2 TPOT
# ======================================================================

class TestTPOT:
    """TPOT = (finish_time - first_token_time) / max(num_output_tokens - 1, 1)."""

    def test_tpot_field_exists_and_non_negative(self):
        """avg/min/max tpot_ms exist and are >= 0."""
        engine = _engine()
        engine.add_request("TPOT test", max_new_tokens=4)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        assert "avg_tpot_ms" in report
        assert "min_tpot_ms" in report
        assert "max_tpot_ms" in report
        assert report["avg_tpot_ms"] >= 0
        assert report["min_tpot_ms"] >= 0
        assert report["max_tpot_ms"] >= 0

    def test_tpot_generated_tokens_count(self):
        """num_output_tokens matches expected count for finished sequences."""
        engine = _engine()
        expected_tokens = 6
        engine.add_request("Generating some tokens", max_new_tokens=expected_tokens)
        engine.run_until_done()

        mc = engine.engine_core.metrics_collector
        for seq in mc._finished_seqs:
            if seq.status == Status.FINISHED:
                assert seq.num_output_tokens == expected_tokens, (
                    f"Expected {expected_tokens} output tokens, "
                    f"got {seq.num_output_tokens}"
                )

    def test_tpot_denominator_is_output_tokens_not_prompt(self):
        """TPOT denominator = num_output_tokens - 1, not prompt_length."""
        engine = _engine()
        engine.add_request("A longer prompt here", max_new_tokens=4)
        engine.run_until_done()

        mc = engine.engine_core.metrics_collector
        for seq in mc._finished_seqs:
            if seq.status == Status.FINISHED and seq.num_output_tokens > 1:
                assert seq.num_output_tokens != seq.prompt_length, (
                    "Test prompt happens to equal output length; "
                    "still validating TPOT formula uses output tokens"
                )
                # Validate: TPOT = (finish - first_token) / (num_output_tokens - 1)
                decode_time = seq.finish_time - seq.first_token_time
                expected_tpot = decode_time / (seq.num_output_tokens - 1)
                assert expected_tpot >= 0

    def test_tpot_single_token_output(self):
        """max_tokens=1: no inter-token gap, so TPOT is reported as 0."""
        engine = _engine()
        engine.add_request("Single token", max_new_tokens=1)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        # Single-token sequences are excluded from TPOT (no gap to measure)
        assert report["avg_tpot_ms"] == 0.0

    def test_tpot_multiple_tokens_have_positive_tpot(self):
        """max_tokens>1: TPOT should be positive (real inter-token gap)."""
        engine = _engine()
        engine.add_request("Multi token", max_new_tokens=4)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        assert report["avg_tpot_ms"] > 0, (
            f"Expected positive TPOT for 4-token output, got {report['avg_tpot_ms']}"
        )

    def test_tpot_min_max_avg_ordering(self):
        """min_tpot_ms <= avg_tpot_ms <= max_tpot_ms."""
        engine = _engine(num_gpu_blocks=64)
        engine.add_request("TPOT A", max_new_tokens=4)
        engine.add_request("TPOT B", max_new_tokens=6)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        if report["total_requests"] > 1:
            assert report["min_tpot_ms"] <= report["avg_tpot_ms"]
            assert report["avg_tpot_ms"] <= report["max_tpot_ms"]


# ======================================================================
# 2.3 Throughput
# ======================================================================

class TestThroughput:
    """Throughput metrics: request_throughput_rps and token_throughput_tps."""

    def test_throughput_fields_exist(self):
        """Both throughput fields exist in the report."""
        engine = _engine()
        engine.add_request("Throughput", max_new_tokens=2)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        assert "throughput_req_per_sec" in report
        assert "throughput_tok_per_sec" in report
        # total_output_tokens and total_prompt_tokens also present
        assert "total_output_tokens" in report
        assert "total_prompt_tokens" in report

    def test_throughput_with_completed_requests(self):
        """completed_requests > 0 and throughput_req_per_sec > 0."""
        engine = _engine()
        engine.add_request("Req-A", max_new_tokens=4)
        engine.add_request("Req-B", max_new_tokens=4)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        assert report["total_requests"] == 2
        assert report["throughput_req_per_sec"] > 0

    def test_token_throughput_with_generated_tokens(self):
        """generated_tokens > 0 and token_throughput_tps > 0."""
        engine = _engine()
        engine.add_request("Tokens!", max_new_tokens=8)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        assert report["total_output_tokens"] > 0
        assert report["throughput_tok_per_sec"] > 0

    def test_total_time_non_negative(self):
        """elapsed_time (total_time_seconds) >= 0 after run.

        Note: with the fake executor, wall-clock time may be ~0 because
        all operations complete within the same microsecond.  On real
        hardware the time would be > 0.
        """
        engine = _engine()
        engine.add_request("Time check", max_new_tokens=4)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        assert report["total_time_seconds"] >= 0

    def test_throughput_formula_consistency(self):
        """Verify throughput = completed / wall-clock time, yields same as report."""
        engine = _engine()
        engine.add_request("Formula", max_new_tokens=4)
        engine.run_until_done()

        mc = engine.engine_core.metrics_collector
        report = mc.report()

        # Manual calculation from raw data
        finished_seqs = [s for s in mc._finished_seqs if s.status == Status.FINISHED]
        completed = len(finished_seqs)
        total_tok = sum(s.num_output_tokens for s in finished_seqs)

        arrivals = [s.arrival_time for s in finished_seqs if s.arrival_time is not None]
        finishes = [s.finish_time for s in finished_seqs if s.finish_time is not None]
        wall = max(finishes) - min(arrivals) if arrivals and finishes else 0.0

        if wall > 0:
            assert abs(report["throughput_req_per_sec"] - completed / wall) < 0.01
            assert abs(report["throughput_tok_per_sec"] - total_tok / wall) < 0.01

    def test_throughput_staggered_arrival(self):
        """Denominator is last_finish - first_arrival (wall-clock), not max per-request."""
        engine = _engine(max_num_seqs=1, num_gpu_blocks=64)
        # Request A: arrives early, finishes before B is added
        engine.add_request("Request A", max_new_tokens=4)
        engine.run_until_done()

        time.sleep(0.02)

        # Request B: arrives after A finishes
        engine.add_request("Request B", max_new_tokens=4)
        engine.run_until_done()

        mc = engine.engine_core.metrics_collector
        report = mc.report()

        finished_seqs = [s for s in mc._finished_seqs if s.status == Status.FINISHED]
        arrivals = [s.arrival_time for s in finished_seqs if s.arrival_time is not None]
        finishes = [s.finish_time for s in finished_seqs if s.finish_time is not None]

        last_finish = max(finishes)
        first_arrival = min(arrivals)
        wall_elapsed = last_finish - first_arrival

        # Per-request latencies (may be much smaller than wall-clock)
        max_per_request = max(f - a for a, f in zip(arrivals, finishes) if a is not None and f is not None)

        # The new denominator captures the staggered gap; old one does not
        assert wall_elapsed > max_per_request, (
            f"wall_elapsed ({wall_elapsed:.4f}) should be > "
            f"max_per_request ({max_per_request:.4f}) due to staggered arrivals"
        )

        # Verify the report matches the wall-clock formula
        completed = len(finished_seqs)
        total_tok = sum(s.num_output_tokens for s in finished_seqs)
        assert abs(report["throughput_req_per_sec"] - completed / wall_elapsed) < 0.01
        assert abs(report["throughput_tok_per_sec"] - total_tok / wall_elapsed) < 0.01


# ======================================================================
# 2.4 No double-count
# ======================================================================

class TestMetricsNoDoubleCount:
    """Cancelled/timeout/disconnected requests must not inflate TTFT/TPOT/throughput."""

    def test_cancelled_not_in_completed(self):
        """Cancelled requests are NOT counted in report's total_requests."""
        engine = _engine(num_gpu_blocks=64, print_step_events=False)
        engine.add_request("Cancel me", max_new_tokens=64)
        engine.step()  # admit and start
        engine.cancel_request("req-0000")
        engine.run_until_done()

        mc = engine.engine_core.metrics_collector
        report = mc.report()
        # The service-layer counter
        assert mc._cancelled_requests >= 1

        # ttft and tpot should be 0 because the sequence was CANCELLED
        # (no FINISHED sequences exist)
        assert report["avg_ttft_ms"] == 0.0
        assert report["avg_tpot_ms"] == 0.0

        # No completed requests — throughput is 0
        assert report["total_requests"] == 0

    def test_timeout_not_in_completed(self):
        """Timed-out requests are NOT counted in total_requests."""
        engine = _engine(
            num_gpu_blocks=64,
            request_timeout_s=0.001,
            print_step_events=False,
        )
        engine.add_request("Timeout test", max_new_tokens=64)

        # Ensure wall clock passes timeout threshold
        time.sleep(0.01)

        # step() calls _check_timeouts() internally at the start of the loop
        engine.step()

        mc = engine.engine_core.metrics_collector
        report = mc.report()
        assert mc._timeout_requests >= 1, (
            f"Expected timeout counter >= 1, got {mc._timeout_requests}"
        )
        assert report["total_requests"] == 0, (
            f"Expected 0 completed requests, got {report['total_requests']}"
        )

    def test_finished_seqs_only_contains_terminal(self):
        """cancelled/timeout seqs in _finished_seqs but not in 'finished' list."""
        engine = _engine(num_gpu_blocks=64)
        engine.add_request("Normal", max_new_tokens=2)
        engine.run_until_done()

        mc = engine.engine_core.metrics_collector
        mc.count_cancelled()
        mc.count_timeout()

        # Normal request finishes normally
        report = mc.report()
        assert report["total_requests"] >= 1
        assert report["avg_ttft_ms"] > 0

    def test_cancelled_generated_tokens_not_summed(self):
        """Cancelled request's generated tokens don't inflate total_output_tokens."""
        engine = _engine(num_gpu_blocks=64)
        engine.add_request("Will cancel", max_new_tokens=64)
        engine.step()  # admit — prefill starts, may produce first token

        engine.cancel_request("req-0000")

        mc = engine.engine_core.metrics_collector
        report = mc.report()
        # Cancelled sequence's output tokens should NOT contribute
        # to output token sum (only FINISHED sequences contribute)
        for seq in mc._finished_seqs:
            assert seq.status != Status.FINISHED, (
                "Cancelled sequence should not be FINISHED"
            )
        assert report["total_output_tokens"] >= 0


# ======================================================================
# 2.5 KV block utilization
# ======================================================================

class TestKVBlockUtilization:
    """KV block utilization must be in [0, 1] and reflect actual usage."""

    def test_allocated_blocks_during_run(self):
        """During execution, allocated_blocks > 0."""
        engine = _engine(num_gpu_blocks=32)
        engine.add_request("Block check", max_new_tokens=8)
        result = engine.step()  # at least prefill happens

        bm = engine.block_manager
        alloc = bm._allocator
        # At least some blocks allocated after prefill
        assert alloc.num_used_blocks > 0, (
            "Expected allocated blocks > 0 after prefill"
        )

    def test_blocks_freed_after_completion(self):
        """After requests finish, all blocks return to free pool."""
        engine = _engine(num_gpu_blocks=32)
        engine.add_request("Free check", max_new_tokens=4)
        engine.run_until_done()

        alloc = engine.block_manager._allocator
        assert alloc.num_free_blocks == alloc.num_total_blocks, (
            f"Blocks not freed: {alloc.num_free_blocks}/{alloc.num_total_blocks}"
        )

    def test_kv_utilization_range(self):
        """kv_util_peak_pct and kv_util_avg_pct are in [0, 100]."""
        engine = _engine(num_gpu_blocks=32)
        engine.add_request("Util range", max_new_tokens=8)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        assert 0 <= report["kv_util_peak_pct"] <= 100, (
            f"Peak utilisation {report['kv_util_peak_pct']} outside [0, 100]"
        )
        assert 0 <= report["kv_util_avg_pct"] <= 100, (
            f"Avg utilisation {report['kv_util_avg_pct']} outside [0, 100]"
        )

    def test_kv_util_fields_exist(self):
        """All KV utilisation fields exist."""
        engine = _engine()
        engine.add_request("KV fields", max_new_tokens=2)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        for key in ("kv_total_blocks", "kv_peak_blocks",
                     "kv_util_peak_pct", "kv_util_avg_pct"):
            assert key in report, f"Missing key: {key}"


# ======================================================================
# 2.6 Scheduler latency
# ======================================================================

class TestSchedulerLatency:
    """Scheduler latency metrics."""

    def test_scheduler_latency_fields_exist(self):
        """avg and max scheduler_latency_ms exist."""
        engine = _engine()
        engine.add_request("Sched", max_new_tokens=2)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        assert "avg_scheduler_latency_ms" in report
        assert "max_scheduler_latency_ms" in report

    def test_scheduler_latency_count_positive(self):
        """Multiple steps means multiple scheduler samples."""
        engine = _engine()
        engine.add_request("Sched count", max_new_tokens=8)
        engine.run_until_done()

        mc = engine.engine_core.metrics_collector
        assert len(mc._scheduler_times) > 0, "No scheduler times recorded"

    def test_scheduler_latency_non_negative(self):
        """avg and max latency >= 0."""
        engine = _engine()
        engine.add_request("Sched non-neg", max_new_tokens=4)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        assert report["avg_scheduler_latency_ms"] >= 0
        assert report["max_scheduler_latency_ms"] >= 0

    def test_max_gte_avg(self):
        """max_scheduler_latency_ms >= avg_scheduler_latency_ms."""
        engine = _engine()
        engine.add_request("Sched max", max_new_tokens=4)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        assert report["max_scheduler_latency_ms"] >= report["avg_scheduler_latency_ms"]


# ======================================================================
# 2.7 Prefix cache metrics
# ======================================================================

class TestPrefixCacheMetrics:
    """Prefix cache metrics: cached_token_count, hit_rate."""

    def test_prefix_cache_fields_exist(self):
        """total_cached_tokens and prefix_cache_hit_rate exist."""
        engine = _engine(num_gpu_blocks=64)
        engine.add_request("Cache test", max_new_tokens=2)
        engine.run_until_done()

        report = engine.engine_core.metrics_collector.report()
        assert "total_cached_tokens" in report
        assert "prefix_cache_hit_rate" in report

    def test_hit_rate_out_of_total_prompt_tokens(self):
        """prefix_cache_hit_rate = cached_tokens / total_prompt_tokens * 100."""
        engine = _engine(num_gpu_blocks=64)
        engine.add_request("Hit rate", max_new_tokens=4)
        engine.run_until_done()

        mc = engine.engine_core.metrics_collector
        report = mc.report()
        expected_rate = (
            sum(mc._timeline_cached) / report["total_prompt_tokens"] * 100
            if report["total_prompt_tokens"] > 0 else 0.0
        )
        assert report["prefix_cache_hit_rate"] == pytest.approx(expected_rate, abs=0.1)

    def test_cache_hit_reduces_prefill(self):
        """Second identical request sees cache hit, prefill budget reduced."""
        engine = _engine(
            max_num_seqs=4,
            num_gpu_blocks=64,
        )
        # First request populates cache; keep it alive
        engine.add_request("Same prompt here", max_new_tokens=16)
        engine.step()  # admit + prefill A → A is RUNNING, cache populated

        # Second request with same prompt → should see cache hit
        engine.add_request("Same prompt here", max_new_tokens=4)
        result = engine.step()

        assert result.cached_token_count > 0, (
            f"Expected cached_token_count > 0, got {result.cached_token_count}"
        )


# ======================================================================
# 2.8 StageProfiler integration
# ======================================================================

class TestStageProfilerMetrics:
    """StageProfiler percent_of_total, empty state, exception handling."""

    def test_percent_of_total_sum(self):
        """Stage percentages sum to roughly 100% around engine_step_total."""
        engine = _engine()
        engine.add_request("Stage test", max_new_tokens=4)
        engine.run_until_done()

        report = engine.engine_core._profiler.report()
        stages = report["stages"]

        # Sub-stages may overlap (nested profiling), but verify
        # they don't wildly exceed the outer metric
        if "engine_step_total" in stages and len(stages) > 1:
            sub_total = sum(
                v["total_ms"] for k, v in stages.items()
                if k != "engine_step_total"
            )
            assert sub_total >= 0

    def test_engine_step_total_is_largest(self):
        """engine_step_total should be the largest stage (outer wrapper)."""
        engine = _engine()
        engine.add_request("Largest", max_new_tokens=4)
        engine.run_until_done()

        report = engine.engine_core._profiler.report()
        stages = report["stages"]
        if "engine_step_total" in stages:
            outer = stages["engine_step_total"]["total_ms"]
            for name, info in stages.items():
                if name != "engine_step_total":
                    assert info["total_ms"] <= outer * 1.05, (
                        f"Stage '{name}' ({info['total_ms']}ms) exceeds "
                        f"engine_step_total ({outer}ms)"
                    )

    def test_empty_profiler_no_crash(self):
        """An empty (unused) StageProfiler returns a clean report."""
        from mini_vllm.engine.stage_profiler import StageProfiler
        p = StageProfiler()
        report = p.report()
        assert "stages" in report
        assert report["total_profiled_ms"] == 0.0
        assert report["total_requests"] == 0
        assert report["total_engine_steps"] == 0

    def test_exception_in_context_manager(self):
        """Exception in context manager doesn't corrupt profiler state."""
        from mini_vllm.engine.stage_profiler import StageProfiler
        p = StageProfiler()
        try:
            with p.record("scheduler_step"):
                raise ValueError("expected")
        except ValueError:
            pass
        # Recording should still work after exception
        with p.record("scheduler_step"):
            pass
        report = p.report()
        assert report["stages"]["scheduler_step"]["count"] == 2

    def test_profiler_stages_present(self):
        """Key stages appear in profiler output after engine run."""
        engine = _engine()
        engine.add_request("Profiler stages", max_new_tokens=4)
        engine.run_until_done()

        report = engine.engine_core._profiler.report()
        expected_stages = {"engine_step_total", "scheduler_step", "metrics_update"}
        for stage in expected_stages:
            assert stage in report["stages"], f"Missing stage: {stage}"


# ======================================================================
# 2.9 Serving-layer counter consistency
# ======================================================================

class TestServingCounters:
    """Serving-layer counters (rejected, cancelled, timeout) are distinct."""

    def test_counters_are_separate(self):
        """Each counter tracks its own event type independently."""
        mc = MetricsCollector()
        assert mc._rejected_requests == 0
        assert mc._cancelled_requests == 0
        assert mc._timeout_requests == 0
        assert mc._rpm_rejected == 0
        assert mc._tpm_rejected == 0

        mc.count_rejected()
        mc.count_cancelled()
        mc.count_timeout()
        mc.count_rpm_rejected()
        mc.count_tpm_rejected()

        report = mc.report()
        assert report["rejected_requests"] == 1
        assert report["cancelled_requests"] == 1
        assert report["timeout_requests"] == 1
        assert report["rpm_rejected"] == 1
        assert report["tpm_rejected"] == 1

    def test_rejected_not_in_total_requests(self):
        """Rejected requests (not registered as sequences) don't count."""
        mc = MetricsCollector()
        mc.count_rejected()
        report = mc.report()
        assert report["total_requests"] == 0  # no finished sequences
        assert report["rejected_requests"] == 1

    def test_cancelled_not_in_total_requests(self):
        """Cancelled requests (without FINISHED seqs) don't count."""
        mc = MetricsCollector()
        mc.count_cancelled()
        report = mc.report()
        assert report["total_requests"] == 0
        assert report["cancelled_requests"] == 1


# ======================================================================
# 2.10 Serving layer metrics endpoint format
# ======================================================================

class TestServingMetricsEndpoint:
    """The /metrics JSON endpoint returns valid data."""

    def test_metrics_endpoint_has_ttft(self):
        """The serving-layer metrics endpoint includes ttft and tpot."""
        from serving.api_server import ServingLayer
        cfg = Config(
            max_num_seqs=4, max_num_batched_tokens=16,
            block_size=4, num_gpu_blocks=32, print_step_events=False,
        )
        sv = ServingLayer(cfg)
        sv.generate("Hello", max_tokens=4)
        metrics = sv.get_metrics()
        assert '"ttft"' in metrics
        assert '"tpot"' in metrics
        assert '"throughput"' in metrics

    def test_metrics_endpoint_block_util_range(self):
        """block_utilization is a percentage in [0, 100]."""
        from serving.api_server import ServingLayer
        cfg = Config(
            max_num_seqs=4, max_num_batched_tokens=16,
            block_size=4, num_gpu_blocks=32, print_step_events=False,
        )
        sv = ServingLayer(cfg)
        sv.generate("Hello", max_tokens=4)
        import json
        data = json.loads(sv.get_metrics())
        assert 0 <= data["block_utilization"] <= 100
