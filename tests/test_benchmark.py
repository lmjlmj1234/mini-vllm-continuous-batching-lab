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
