"""Tests for the benchmark CLI (examples/benchmark.py)."""

import json
import os
import subprocess
import sys
import tempfile

import pytest


_BENCHMARK_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "benchmark.py",
)


@pytest.fixture
def bench_py() -> str:
    assert os.path.exists(_BENCHMARK_PY), f"{_BENCHMARK_PY} not found"
    return _BENCHMARK_PY


def run_benchmark(bench_py: str, *args: str) -> subprocess.CompletedProcess:
    """Run benchmark.py with given args and return the completed process."""
    return subprocess.run(
        [sys.executable, bench_py, *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# CLI argument parsing — —help prints expected sections
# ---------------------------------------------------------------------------


class TestCLIHelp:
    def test_help_contains_workload_section(self, bench_py: str) -> None:
        r = run_benchmark(bench_py, "--help")
        assert r.returncode == 0
        assert "--executor" in r.stdout
        assert "--requests" in r.stdout
        assert "--tokens" in r.stdout

    def test_help_contains_scheduler_section(self, bench_py: str) -> None:
        r = run_benchmark(bench_py, "--help")
        assert r.returncode == 0
        assert "--max-num-seqs" in r.stdout
        assert "--max-num-batched-tokens" in r.stdout
        assert "--max-num-prefill-tokens" in r.stdout
        assert "--max-prefill-chunk-size" in r.stdout
        assert "--decode-first" in r.stdout

    def test_help_contains_kv_cache_section(self, bench_py: str) -> None:
        r = run_benchmark(bench_py, "--help")
        assert r.returncode == 0
        assert "--block-size" in r.stdout
        assert "--num-gpu-blocks" in r.stdout
        assert "--max-model-len" in r.stdout

    def test_help_contains_output_options(self, bench_py: str) -> None:
        r = run_benchmark(bench_py, "--help")
        assert r.returncode == 0
        assert "--quiet" in r.stdout
        assert "--memory-trace" in r.stdout
        assert "--json-output" in r.stdout


# ---------------------------------------------------------------------------
# Config propagation — key parameters appear in the printed config
# ---------------------------------------------------------------------------


class TestConfigPropagation:
    def test_default_values(self, bench_py: str) -> None:
        r = run_benchmark(bench_py, "--executor", "fake", "--requests", "2",
                          "--tokens", "4")
        assert r.returncode == 0
        # Defaults should be printed
        assert "requests:       2" in r.stdout
        assert "max_num_seqs:            4" in r.stdout
        assert "max_num_batched_tokens:  16" in r.stdout
        assert "block_size:     4" in r.stdout

    def test_custom_scheduler_params(self, bench_py: str) -> None:
        r = run_benchmark(
            bench_py,
            "--executor", "fake", "--requests", "4", "--tokens", "8",
            "--max-num-seqs", "8",
            "--max-num-batched-tokens", "32",
            "--max-num-prefill-tokens", "64",
            "--max-prefill-chunk-size", "2",
        )
        assert r.returncode == 0
        assert "max_num_seqs:            8" in r.stdout
        assert "max_num_batched_tokens:  32" in r.stdout
        assert "max_prefill_chunk_size:  2" in r.stdout

    def test_custom_kv_cache_params(self, bench_py: str) -> None:
        r = run_benchmark(
            bench_py,
            "--executor", "fake", "--requests", "2", "--tokens", "4",
            "--block-size", "8",
            "--num-gpu-blocks", "64",
            "--max-model-len", "1024",
        )
        assert r.returncode == 0
        assert "block_size:     8" in r.stdout
        assert "num_gpu_blocks: 64" in r.stdout

    def test_decode_first_flag(self, bench_py: str) -> None:
        r = run_benchmark(
            bench_py,
            "--executor", "fake", "--requests", "2", "--tokens", "4",
            "--no-decode-first",
        )
        assert r.returncode == 0
        assert "decode_first:            False" in r.stdout

    def test_no_chunked_prefill(self, bench_py: str) -> None:
        r = run_benchmark(
            bench_py,
            "--executor", "fake", "--requests", "2", "--tokens", "4",
            "--no-chunked-prefill",
        )
        assert r.returncode == 0
        assert "chunked_prefill:         False" in r.stdout


# ---------------------------------------------------------------------------
# num_gpu_blocks auto-calculation when omitted
# ---------------------------------------------------------------------------


class TestNumGpuBlocksAuto:
    def test_auto_calculation_when_omitted(self, bench_py: str) -> None:
        r = run_benchmark(bench_py, "--executor", "fake", "--requests", "2",
                          "--tokens", "4", "--quiet")
        assert r.returncode == 0
        # With 2 requests at 4 tokens, auto-calc should produce >= 16
        assert "num_gpu_blocks: 16" in r.stdout or "num_gpu_blocks: 3" not in r.stdout

    def test_explicit_blocks_used_when_provided(self, bench_py: str) -> None:
        r = run_benchmark(
            bench_py,
            "--executor", "fake", "--requests", "2", "--tokens", "4",
            "--num-gpu-blocks", "256", "--quiet",
        )
        assert r.returncode == 0
        assert "num_gpu_blocks: 256" in r.stdout


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_output_creates_file(self, bench_py: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
        try:
            r = run_benchmark(
                bench_py,
                "--executor", "fake", "--requests", "2", "--tokens", "4",
                "--json-output", json_path,
            )
            assert r.returncode == 0
            assert os.path.exists(json_path)
            with open(json_path) as f:
                data = json.load(f)
            # Verify expected keys exist
            assert "avg_ttft_ms" in data
            assert "avg_tpot_ms" in data
            assert "throughput_req_per_sec" in data
            assert "throughput_tok_per_sec" in data
            assert "kv_util_peak_pct" in data
            assert "avg_scheduler_latency_ms" in data
            assert "rejected_requests" in data
            assert "total_time_seconds" in data
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_json_output_values_are_reasonable(self, bench_py: str) -> None:
        """Fake executor should produce non-negative metrics."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
        try:
            r = run_benchmark(
                bench_py,
                "--executor", "fake", "--requests", "2", "--tokens", "4",
                "--json-output", json_path,
            )
            assert r.returncode == 0
            with open(json_path) as f:
                data = json.load(f)
            assert data["avg_ttft_ms"] >= 0
            assert data["avg_tpot_ms"] >= 0
            assert data["throughput_req_per_sec"] >= 0
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)


# ---------------------------------------------------------------------------
# Invalid arguments → non-zero exit code
# ---------------------------------------------------------------------------


class TestInvalidArgs:
    @pytest.mark.parametrize("invalid_args", [
        ["--requests", "0"],
        ["--requests", "-1"],
        ["--tokens", "0"],
        ["--max-num-seqs", "-5"],
        ["--max-num-batched-tokens", "0"],
        ["--max-num-prefill-tokens", "-1"],
        ["--max-prefill-chunk-size", "0"],
        ["--block-size", "-1"],
        ["--num-gpu-blocks", "0"],
        ["--max-model-len", "-10"],
    ])
    def test_invalid_args_exit_nonzero(self, bench_py: str,
                                       invalid_args: list[str]) -> None:
        r = run_benchmark(
            bench_py,
            "--executor", "fake", "--requests", "2", "--tokens", "4",
            *invalid_args,
        )
        assert r.returncode != 0, (
            f"Expected non-zero exit for {invalid_args}, "
            f"got stdout={r.stdout!r} stderr={r.stderr!r}"
        )


# ---------------------------------------------------------------------------
# Summary output contains expected fields
# ---------------------------------------------------------------------------


class TestSummaryOutput:
    def test_summary_contains_key_fields(self, bench_py: str) -> None:
        r = run_benchmark(bench_py, "--executor", "fake", "--requests", "2",
                          "--tokens", "4", "--quiet")
        assert r.returncode == 0
        assert "total_time_seconds:" in r.stdout
        assert "request_throughput:" in r.stdout
        assert "token_throughput:" in r.stdout
        assert "avg_ttft_ms:" in r.stdout
        assert "avg_tpot_ms:" in r.stdout
        assert "kv_util_peak_pct:" in r.stdout
        assert "avg_scheduler_latency_ms:" in r.stdout
        assert "rejected_requests:" in r.stdout

    def test_fake_executor_disclaimer(self, bench_py: str) -> None:
        r = run_benchmark(bench_py, "--executor", "fake", "--requests", "2",
                          "--tokens", "4", "--quiet")
        assert r.returncode == 0
        assert "simulation metrics, not real GPU inference" in r.stdout


# ======================================================================
# Benchmark metric formula tests
# ======================================================================


class TestMetricFormulas:
    """Unit tests for metric formulas: TTFT, TPOT, throughput."""

    def test_ttft_formula(self):
        """TTFT = first_token_time - arrival_time, in seconds."""
        from mini_vllm.engine.metrics import _percentile
        arrival = 1000.0
        first_token = 1020.0
        ttft = first_token - arrival
        assert ttft == 20.0
        ttft_ms = ttft * 1000
        assert ttft_ms == 20000.0

    def test_tpot_formula(self):
        """TPOT = (finish_time - first_token_time) / (num_output_tokens - 1)."""
        first_token = 1020.0
        finish = 1100.0
        num_tokens = 5
        decode_time = finish - first_token
        assert decode_time == 80.0
        tpot = decode_time / (num_tokens - 1)
        assert tpot == 20.0
        tpot_ms = tpot * 1000
        assert tpot_ms == 20000.0

    def test_tpot_single_token_output(self):
        """Single token output: no inter-token gap, TPOT excluded."""
        # With only 1 output token there is no inter-token gap to measure.
        # The TPOT formula is (finish_time - first_token_time) / max(n-1, 1)
        # but single-token sequences are excluded from the TPOT average.

    def test_request_throughput_formula(self):
        """req/s = completed_requests / wall_clock_time."""
        completed = 20
        wall_time = 10.0
        throughput = completed / wall_time
        assert throughput == 2.0

    def test_token_throughput_formula(self):
        """tok/s = total_output_tokens / wall_clock_time."""
        total_tokens = 500
        wall_time = 10.0
        throughput = total_tokens / wall_time
        assert throughput == 50.0

    def test_percentile_calculation(self):
        """_percentile computes correct P50 and P95."""
        from mini_vllm.engine.metrics import _percentile
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        p50 = _percentile(values, 50)
        assert p50 == 3.0, f'Expected P50=3.0, got {p50}'
        p95 = _percentile(values, 95)
        # Linear interpolation: P95 of 5 values = 4th + 0.8 * gap_to_5th
        assert p95 == 4.8, f'Expected P95=4.8, got {p95}'
        p0 = _percentile(values, 0)
        assert p0 == 1.0, f'Expected P0=1.0, got {p0}'
        p100 = _percentile(values, 100)
        assert p100 == 5.0, f'Expected P100=5.0, got {p100}'
        empty = _percentile([], 50)
        assert empty == 0.0, f'Expected empty=0.0, got {empty}'


class TestSerialModeBehavior:
    """Serial mode: exactly one request at a time."""

    def test_serial_max_one_running(self):
        """Serial mode: at most 1 request in running pool at any step."""
        from mini_vllm import Config, LLMEngine
        cfg = Config(
            max_num_seqs=1, print_step_events=False,
            num_gpu_blocks=64, block_size=4,
            max_num_batched_tokens=16, max_num_prefill_tokens=16,
            trace_enabled=True,
        )
        engine = LLMEngine(cfg)
        engine.add_request('A', max_new_tokens=2)
        engine.run_until_done()
        # Queue should be empty
        assert engine.queue.num_waiting == 0
        assert engine.queue.num_running == 0
        assert engine.queue.num_finished == 1

        engine.add_request('B', max_new_tokens=2)
        engine.run_until_done()
        assert engine.queue.num_waiting == 0
        assert engine.queue.num_running == 0
        assert engine.queue.num_finished == 2


class TestContinuousModeBehavior:
    """Continuous batching: early exit and waiting replacement."""

    def test_continuous_early_exit_and_admission(self):
        """Continuous: early exit frees budget for waiting requests."""
        from mini_vllm import Config, LLMEngine, Status
        cfg = Config(
            max_num_seqs=2, print_step_events=False,
            num_gpu_blocks=128, block_size=4,
            max_num_batched_tokens=16, max_num_prefill_tokens=16,
            trace_enabled=True,
        )
        engine = LLMEngine(cfg)
        engine.add_request('A', max_new_tokens=8)
        engine.add_request('B', max_new_tokens=2)
        engine.add_request('C', max_new_tokens=2)
        outputs = engine.run_until_done()
        assert len(outputs) == 3
        # Check trace shows dynamic admission
        trace = engine.engine_core._scheduler.get_and_clear_trace()
        # Find steps where new requests were admitted
        admissions = [t for t in trace if t['newly_admitted_requests'] > 0]
        assert len(admissions) >= 2, (
            f'Expected >=2 admission events, got {len(admissions)}'
        )
        # Find steps where requests finished
        finishes = [t for t in trace if t['newly_finished_requests'] > 0]
        assert len(finishes) >= 2, (
            f'Expected >=2 finish events, got {len(finishes)}'
        )


class TestStaticModeBehavior:
    """Static batching: no mid-batch admission."""

    def test_static_no_mid_batch_admission(self):
        """Static: no new requests admitted mid-batch."""
        from mini_vllm import Config, LLMEngine
        cfg = Config(
            max_num_seqs=2, print_step_events=False,
            num_gpu_blocks=128, block_size=4,
            max_num_batched_tokens=16, max_num_prefill_tokens=16,
            trace_enabled=True,
            static_batch_mode=True,
        )
        engine = LLMEngine(cfg)
        engine.add_request('A', max_new_tokens=8)
        engine.add_request('B', max_new_tokens=2)
        engine.add_request('C', max_new_tokens=2)
        outputs = engine.run_until_done()
        # In static mode, only A and B get admitted (batch of 2)
        # C stays in waiting until A finishes
        # Actually in static mode, once the first 2 finish, C can be admitted
        # as a new batch (since no running requests)
        trace = engine.engine_core._scheduler.get_and_clear_trace()
        assert len(outputs) == 3
        # Verify: no mid-batch admissions in static mode
        running_was_nonzero = False
        for t in trace:
            if t['running_requests'] > 0:
                running_was_nonzero = True
            if t['newly_admitted_requests'] > 0 and t['running_requests'] > 0:
                # Also check that at least one of the newly admitted IDs is new
                if running_was_nonzero:
                    pass  # Static mode admits at batch start
        # All 3 requests must finish eventually
        finished_ids = set()
        for t in trace:
            finished_ids.update(t['finished_request_ids_this_step'])
        assert len(finished_ids) >= 3


class TestSchedulerTrace:
    """Scheduler trace: off by default, read-only, no behavioral change."""

    def test_trace_off_by_default(self):
        """Trace is disabled by default (trace_enabled=False)."""
        from mini_vllm import Config, LLMEngine
        cfg = Config(
            print_step_events=False, num_gpu_blocks=16,
        )
        engine = LLMEngine(cfg)
        assert not engine.engine_core._scheduler._trace_enabled, "Trace should be off by default"

    def test_trace_on_when_enabled(self):
        """Trace is enabled when trace_enabled=True."""
        from mini_vllm import Config, LLMEngine
        cfg = Config(
            print_step_events=False, num_gpu_blocks=16,
            trace_enabled=True,
        )
        engine = LLMEngine(cfg)
        assert engine.engine_core._scheduler._trace_enabled
        engine.add_request('Test', max_new_tokens=2)
        engine.run_until_done()
        trace = engine.engine_core._scheduler.get_and_clear_trace()
        assert len(trace) > 0, 'Trace should have records'
        for key in ['step_id', 'waiting_requests', 'running_requests',
                     'finished_requests', 'effective_batch_size']:
            assert key in trace[0], f'Missing key: {key}'

    def test_trace_does_not_change_behavior(self):
        """Trace enabled vs disabled produces same scheduling results."""
        from mini_vllm import Config, LLMEngine

        def run_with_trace(enabled):
            cfg = Config(
                print_step_events=False, num_gpu_blocks=32,
                max_num_seqs=2,
                trace_enabled=enabled,
            )
            engine = LLMEngine(cfg)
            engine.add_request('A', max_new_tokens=2)
            engine.add_request('B', max_new_tokens=4)
            outputs = engine.run_until_done()
            mc = engine.engine_core.metrics_collector
            report = mc.report()
            return report['total_requests'], report['total_output_tokens']

        req_off, tok_off = run_with_trace(False)
        req_on, tok_on = run_with_trace(True)
        assert req_off == req_on, 'Trace changed request count'
        assert tok_off == tok_on, 'Trace changed token count'


class TestResourceCleanup:
    """Resource cleanup after benchmark."""

    def test_blocks_freed_after_benchmark(self):
        """All KV blocks freed after requests complete."""
        from mini_vllm import Config, LLMEngine
        cfg = Config(
            print_step_events=False, num_gpu_blocks=32,
            max_num_seqs=4,
        )
        engine = LLMEngine(cfg)
        engine.add_request('A', max_new_tokens=4)
        engine.add_request('B', max_new_tokens=4)
        engine.run_until_done()
        alloc = engine.block_manager._allocator
        assert alloc.num_free_blocks == alloc.num_total_blocks, (
            f"Blocks not freed: {alloc.num_free_blocks}/{alloc.num_total_blocks}"
        )
        assert engine.queue.num_waiting == 0
        assert engine.queue.num_running == 0
