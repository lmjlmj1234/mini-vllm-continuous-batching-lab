"""Tests for the StageProfiler."""

import time

import pytest

from mini_vllm.engine.stage_profiler import StageProfiler


class TestStageProfiler:
    """Unit tests for StageProfiler core functionality."""

    def test_record_single_stage(self) -> None:
        p = StageProfiler()
        with p.record("prefill"):
            time.sleep(0.001)
        report = p.report()
        assert "prefill" in report["stages"]
        assert report["stages"]["prefill"]["count"] == 1
        # total_ms should be >= 1ms
        assert report["stages"]["prefill"]["total_ms"] > 0

    def test_stats_count_total_avg_max(self) -> None:
        p = StageProfiler()
        # Record three synthetic durations via record_raw
        p.record_raw("decode", 0.010)
        p.record_raw("decode", 0.020)
        p.record_raw("decode", 0.030)
        report = p.report()
        stage = report["stages"]["decode"]
        assert stage["count"] == 3
        assert stage["total_ms"] == 60.0  # 10 + 20 + 30 ms
        assert stage["avg_ms"] == 20.0  # 60 / 3
        assert stage["max_ms"] == 30.0

    def test_context_manager_exception_handling(self) -> None:
        p = StageProfiler()
        try:
            with p.record("scheduler_step"):
                raise ValueError("test error")
        except ValueError:
            pass
        # Recording should still work even after exception
        with p.record("scheduler_step"):
            pass
        report = p.report()
        assert report["stages"]["scheduler_step"]["count"] == 2

    def test_empty_report_no_error(self) -> None:
        p = StageProfiler()
        report = p.report()
        assert report["stages"] == {}
        assert report["total_profiled_ms"] == 0.0
        assert report["total_requests"] == 0
        assert report["total_engine_steps"] == 0

    def test_reset_clears_data(self) -> None:
        p = StageProfiler()
        with p.record("prefill"):
            time.sleep(0.001)
        p.increment_requests(5)
        p.increment_steps(3)
        assert len(p.report()["stages"]) > 0
        p.reset()
        report = p.report()
        assert report["stages"] == {}
        assert report["total_requests"] == 0
        assert report["total_engine_steps"] == 0

    def test_start_end_api(self) -> None:
        p = StageProfiler()
        p.start("kv_cache_allocation")
        time.sleep(0.001)
        p.end("kv_cache_allocation")
        report = p.report()
        assert "kv_cache_allocation" in report["stages"]
        assert report["stages"]["kv_cache_allocation"]["count"] == 1

    def test_start_without_end_ignored(self) -> None:
        p = StageProfiler()
        p.start("prefix_cache_lookup")
        # Missing end() — should not create a record
        report = p.report()
        assert "prefix_cache_lookup" not in report["stages"]

    def test_percent_of_total(self) -> None:
        p = StageProfiler()
        p.record_raw("decode", 0.030)
        p.record_raw("engine_step_total", 0.100)
        report = p.report()
        # Should compute % relative to engine_step_total
        assert report["stages"]["decode"]["percent_of_total"] == 30.0

    def test_increment_requests_and_steps(self) -> None:
        p = StageProfiler()
        p.increment_requests(3)
        p.increment_steps(7)
        report = p.report()
        assert report["total_requests"] == 3
        assert report["total_engine_steps"] == 7

    def test_multiple_stages_independent(self) -> None:
        p = StageProfiler()
        p.record_raw("scheduler_step", 0.005)
        p.record_raw("prefill", 0.015)
        p.record_raw("decode", 0.025)
        report = p.report()
        assert set(report["stages"].keys()) == {"scheduler_step", "prefill", "decode"}

    def test_engine_core_integration(self) -> None:
        """Integration test: StageProfiler wired through LLMEngine."""
        from mini_vllm import Config, LLMEngine

        cfg = Config(
            max_num_seqs=4,
            max_num_batched_tokens=16,
            block_size=4,
            num_gpu_blocks=16,
            print_step_events=False,
        )
        engine = LLMEngine(cfg)
        engine.add_request("Hello", max_new_tokens=2)
        engine.add_request("World", max_new_tokens=3)
        engine.run_until_done()

        profiler = engine.profiler
        report = profiler.report()
        # Should have recorded at least engine_step_total
        assert "engine_step_total" in report["stages"]
        assert report["total_engine_steps"] > 0
        # Should have recorded request_queue_waiting for both requests
        assert report["total_requests"] == 2

    def test_engine_core_integration_many_steps(self) -> None:
        """More steps should produce more profiler records."""
        from mini_vllm import Config, LLMEngine

        cfg = Config(
            max_num_seqs=4,
            max_num_batched_tokens=32,
            block_size=4,
            num_gpu_blocks=32,
            max_prefill_chunk_size=4,
            print_step_events=False,
        )
        engine = LLMEngine(cfg)
        engine.add_request("Test A", max_new_tokens=8)
        engine.add_request("Test B", max_new_tokens=12)
        engine.run_until_done()

        profiler = engine.profiler
        report = profiler.report()
        assert "scheduler_step" in report["stages"]
        assert "executor_forward" in report["stages"]
        assert report["stages"]["scheduler_step"]["count"] == report["total_engine_steps"]

    def test_cancel_does_not_crash_profiler(self) -> None:
        """Cancelling a request should not break profiler state."""
        from mini_vllm import Config, LLMEngine

        cfg = Config(
            max_num_seqs=2,
            max_num_batched_tokens=8,
            block_size=4,
            num_gpu_blocks=16,
            print_step_events=False,
        )
        engine = LLMEngine(cfg)
        rid = engine.add_request("Cancel me", max_new_tokens=10)
        engine.step()
        # Cancel mid-flight
        engine.cancel_request(rid)
        engine.run_until_done()

        report = engine.profiler.report()
        assert report["total_engine_steps"] > 0  # no crash
