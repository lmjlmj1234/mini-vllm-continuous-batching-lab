#!/usr/bin/env python3
"""Continuous Batching Benchmark -- compare serial, static, and continuous modes.

Usage:

    # Smoke test with fake executor
    python -m benchmarks.continuous_batching \
        --executor fake --modes serial continuous --concurrency 2 --requests 4 --repeats 1

    # Full benchmark with Qwen model
    python -m benchmarks.continuous_batching \
        --executor qwen --modes serial continuous \
        --concurrency 1 2 4 8 --requests 20 --repeats 3 \
        --output-dir benchmark_results

    # With static batching (if available)
    python -m benchmarks.continuous_batching \
        --executor qwen --modes serial static continuous \
        --concurrency 1 2 4 8 --requests 20 --repeats 3
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch

from mini_vllm import Config, LLMEngine, Status
from mini_vllm.sequence.status import Status as SeqStatus


# ======================================================================
# Workload definitions
# ======================================================================

# (name, ratio, min_input_tokens, max_input_tokens, output_tokens)
WORKLOAD = [
    ("short", 0.50, 64, 128, 16),
    ("medium", 0.30, 128, 256, 32),
    ("long", 0.20, 256, 512, 64),
]

_BASE_CHARS: Dict[str, str] = {
    "short": (
        "This is a test prompt for benchmarking continuous batching. "
        "It contains enough text to reach the target token count. "
    ),
    "medium": (
        "Explain the concept of attention mechanisms in transformer "
        "neural network architectures used for natural language "
        "processing and generation tasks. Attention allows the model to "
        "focus on different parts of the input sequence when producing "
        "each output token, enabling it to capture long-range dependencies. "
    ),
    "long": (
        "Provide a detailed explanation of how the transformer attention "
        "mechanism works in large language models for natural language "
        "understanding and generation tasks, including the mathematical "
        "foundations of scaled dot-product attention, multi-head attention, "
        "and how these components are organized in the encoder-decoder "
        "architecture that forms the basis of modern foundation models "
        "like GPT, BERT, and their successors. Describe the query, key, "
        "value computation, the role of positional encodings, and how "
        "the self-attention mechanism enables parallel processing of "
        "input sequences while maintaining contextual relationships. "
    ),
}


def _generate_request_prompts(
    tokenizer,
    num_requests: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a deterministic set of requests with controlled token lengths.

    Each request has: prompt (str), prompt_token_count (target),
    output_length (target), and actual tokenized prompt.
    """
    rng = random.Random(seed)
    requests: List[Dict[str, Any]] = []
    rng.shuffle

    # Assign workload categories
    categories = []
    for i in range(num_requests):
        r = rng.random()
        cum = 0.0
        chosen = None
        for name, ratio, min_in, max_in, out in WORKLOAD:
            cum += ratio
            if r <= cum:
                chosen = (min_in, max_in, out)
                break
        if chosen is None:
            chosen = (64, 128, 16)
        categories.append(chosen)

    for i in range(num_requests):
        min_in, max_in, out_tokens = categories[i]
        target_in = rng.randint(min_in, max_in)

        # Build prompt to approximate target token count
        # Use a known pattern that tokenizes to roughly 2 chars per token
        # for Qwen2.5 tokenizer
        prompt_chars_needed = target_in * 3  # rough estimate
        for name, ratio, min_in_2, max_in_2, out_2 in WORKLOAD:
            if min_in_2 == min_in and max_in_2 == max_in:
                base = _BASE_CHARS[name]
                break
        else:
            base = _BASE_CHARS["short"]

        repeats = max(1, prompt_chars_needed // len(base) + 1)
        prompt = base * repeats

        # Record what we'll actually pass
        # Unique suffix prevents prefix cache block sharing between requests
        # Unique random prefix ensures prefix cache never shares blocks
        unique_prefix = f"<|req_{i:04d}|>" + "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8))
        prompt_with_id = unique_prefix + prompt
        requests.append({
            "prompt": prompt_with_id,
            "target_input_tokens": target_in,
            "output_length": out_tokens,
        })

    # Tokenize all prompts to get actual lengths
    for req in requests:
        tokens = tokenizer.encode(req["prompt"], add_special_tokens=True)
        req["actual_input_tokens"] = len(tokens)
        req["prompt_token_ids"] = tokens

    return requests


# ======================================================================
# A/B Workload generation (homogeneous / ragged overrides)
# ======================================================================

def _generate_ab_request_prompts(
    tokenizer,
    num_requests: int,
    seed: int = 42,
    homogeneous_ctx: Optional[int] = None,
    homogeneous_out: Optional[int] = None,
    ragged_ctx: bool = False,
    ragged_out: bool = False,
) -> List[Dict[str, Any]]:
    """Generate requests for A/B experiment with homogeneous/ragged modes.

    Args:
        homogeneous_ctx: If set, all requests have this exact context length.
        homogeneous_out: If set, all requests have this exact output length.
        ragged_ctx: Wide uniform distribution of context lengths (32-1024).
        ragged_out: Wide uniform distribution of output lengths (8-128).
    """
    rng = random.Random(seed)

    if homogeneous_ctx is not None and homogeneous_out is not None:
        # Fully homogeneous: all requests identical
        prompt_chars_needed = homogeneous_ctx * 3
        base = _BASE_CHARS["medium"]
        repeats = max(1, prompt_chars_needed // len(base) + 1)
        prompt = base * repeats
        requests = [
            {"prompt": f"<|req_{j:04d}|>" + prompt, "target_input_tokens": homogeneous_ctx,
             "output_length": homogeneous_out}
            for j in range(num_requests)
        ]
    elif homogeneous_ctx is not None and ragged_out:
        # Fixed context, varying output
        prompt_chars_needed = homogeneous_ctx * 3
        base = _BASE_CHARS["medium"]
        repeats = max(1, prompt_chars_needed // len(base) + 1)
        prompt = base * repeats
        requests = [
            {"prompt": f"<|req_{j:04d}|>" + prompt, "target_input_tokens": homogeneous_ctx,
             "output_length": rng.randint(8, 128)}
            for j in range(num_requests)
        ]
    elif homogeneous_ctx is not None:
        # Fixed context, default output distribution
        prompt_chars_needed = homogeneous_ctx * 3
        base = _BASE_CHARS["medium"]
        repeats = max(1, prompt_chars_needed // len(base) + 1)
        prompt = base * repeats
        requests = [
            {"prompt": prompt + f" Request {j}", "target_input_tokens": homogeneous_ctx,
             "output_length": rng.randint(8, 64)}
            for j in range(num_requests)
        ]
    elif ragged_ctx:
        # Wide uniform distribution of context lengths
        requests = []
        for i in range(num_requests):
            target_in = rng.randint(32, 1024)
            out_len = homogeneous_out or rng.randint(8, 128) if ragged_out else rng.randint(16, 64)
            prompt_chars_needed = target_in * 3
            base = _BASE_CHARS["medium"]
            repeats = max(1, prompt_chars_needed // len(base) + 1)
            prompt = base * repeats
            requests.append({
                "prompt": f"<|req_{i:04d}|>" + prompt, "target_input_tokens": target_in,
                "output_length": out_len,
            })
    elif ragged_out:
        # Default context distribution, wide output lengths
        requests = []
        for i in range(num_requests):
            target_in = rng.randint(64, 512)
            out_len = rng.randint(8, 128)
            prompt_chars_needed = target_in * 3
            base = _BASE_CHARS["medium"]
            repeats = max(1, prompt_chars_needed // len(base) + 1)
            prompt = base * repeats
            requests.append({
                "prompt": f"<|req_{i:04d}|>" + prompt, "target_input_tokens": target_in,
                "output_length": out_len,
            })
    else:
        # Use standard workload distribution
        return _generate_request_prompts(tokenizer, num_requests, seed=seed)

    # Tokenize all prompts to get actual lengths
    for req in requests:
        tokens = tokenizer.encode(req["prompt"], add_special_tokens=True)
        req["actual_input_tokens"] = len(tokens)
        req["prompt_token_ids"] = tokens

    return requests


# ======================================================================
# A/B Workload generation (homogeneous / ragged overrides)
# ======================================================================

def _generate_ab_request_prompts(
    tokenizer,
    num_requests: int,
    seed: int = 42,
    homogeneous_ctx: Optional[int] = None,
    homogeneous_out: Optional[int] = None,
    ragged_ctx: bool = False,
    ragged_out: bool = False,
) -> List[Dict[str, Any]]:
    """Generate requests for A/B experiment with homogeneous/ragged modes."""
    rng = random.Random(seed)

    if homogeneous_ctx is not None and homogeneous_out is not None:
        prompt_chars_needed = homogeneous_ctx * 3
        base = _BASE_CHARS["medium"]
        repeats = max(1, prompt_chars_needed // len(base) + 1)
        prompt = base * repeats
        requests = [
            {"prompt": prompt, "target_input_tokens": homogeneous_ctx,
             "output_length": homogeneous_out}
            for _ in range(num_requests)
        ]
    elif homogeneous_ctx is not None:
        prompt_chars_needed = homogeneous_ctx * 3
        base = _BASE_CHARS["medium"]
        repeats = max(1, prompt_chars_needed // len(base) + 1)
        prompt = base * repeats
        requests = [
            {"prompt": prompt, "target_input_tokens": homogeneous_ctx,
             "output_length": rng.randint(8, 64)}
            for _ in range(num_requests)
        ]
    elif ragged_ctx:
        requests = []
        for _ in range(num_requests):
            target_in = rng.randint(32, 1024)
            out_len = homogeneous_out or (rng.randint(8, 128) if ragged_out else rng.randint(16, 64))
            prompt_chars_needed = target_in * 3
            base = _BASE_CHARS["medium"]
            repeats = max(1, prompt_chars_needed // len(base) + 1)
            prompt = base * repeats
            requests.append({
                "prompt": prompt, "target_input_tokens": target_in,
                "output_length": out_len,
            })
    elif ragged_out:
        requests = []
        for _ in range(num_requests):
            target_in = rng.randint(64, 512)
            out_len = rng.randint(8, 128)
            prompt_chars_needed = target_in * 3
            base = _BASE_CHARS["medium"]
            repeats = max(1, prompt_chars_needed // len(base) + 1)
            prompt = base * repeats
            requests.append({
                "prompt": prompt, "target_input_tokens": target_in,
                "output_length": out_len,
            })
    else:
        return _generate_request_prompts(tokenizer, num_requests, seed=seed)

    for req in requests:
        tokens = tokenizer.encode(req["prompt"], add_special_tokens=True)
        req["actual_input_tokens"] = len(tokens)
        req["prompt_token_ids"] = tokens

    return requests


# ======================================================================
# Mode runners
# ======================================================================


def _build_config(mode: str, concurrency: int, **overrides) -> Config:
    """Build Config for the given mode and concurrency level."""
    params = dict(
        executor_type="qwen",
        max_num_seqs=concurrency,
        max_num_batched_tokens=8192,
        max_num_prefill_tokens=2048,
        max_prefill_chunk_size=128,
        block_size=16,
        chunked_prefill_enabled=True,
        decode_first=True,
        print_step_events=False,
        memory_trace=False,
        trace_enabled=True,
    )
    if mode == "serial":
        params["max_num_seqs"] = 1
        params["static_batch_mode"] = False
    elif mode == "static":
        params["static_batch_mode"] = True
    else:  # continuous
        params["static_batch_mode"] = False
    params.update(overrides)
    return Config(**params)


def _run_serial(
    engine: LLMEngine,
    requests: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run requests serially: one at a time, run to completion."""
    all_outputs: Dict[str, str] = {}
    for i, req in enumerate(requests):
        rid_prefix = f"ser-{i:04d}"
        prompt = req["prompt"]
        max_tokens = req["output_length"]
        rid = engine.add_request(prompt, max_new_tokens=max_tokens)
        outputs = engine.run_until_done()
        all_outputs.update(outputs)
    return all_outputs


def _run_static_batch(
    engine: LLMEngine,
    requests: List[Dict[str, Any]],
    concurrency: int,
) -> Dict[str, Any]:
    """Run requests in static batches. Each batch runs to completion
    and no new requests are admitted mid-batch."""
    all_outputs: Dict[str, str] = {}
    batch_idx = 0
    for i in range(0, len(requests), concurrency):
        batch = requests[i:i + concurrency]
        for j, req in enumerate(batch):
            rid_prefix = f"st-{batch_idx:04d}-{j:04d}"
            prompt = req["prompt"]
            max_tokens = req["output_length"]
            rid = engine.add_request(prompt, max_new_tokens=max_tokens)

        # Run this batch to completion (static mode prevents new admission)
        outputs = engine.run_until_done()
        all_outputs.update(outputs)
        batch_idx += 1
    return all_outputs


def _run_continuous(
    engine: LLMEngine,
    requests: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run all requests with continuous batching: add all upfront,
    scheduler admits as budget allows."""
    for i, req in enumerate(requests):
        rid_prefix = f"con-{i:04d}"
        prompt = req["prompt"]
        max_tokens = req["output_length"]
        engine.add_request(prompt, max_new_tokens=max_tokens)

    outputs = engine.run_until_done()
    return outputs


def _compute_metadata() -> Dict[str, Any]:
    """Collect environment metadata."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": "",
        "git_dirty": False,
        "gpu": "",
        "cuda_version": "",
        "pytorch_version": "",
        "transformers_version": "",
        "python_version": sys.version.split()[0],
    }
    try:
        import torch
        meta["pytorch_version"] = torch.__version__
        meta["cuda_version"] = torch.version.cuda or ""
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import transformers
        meta["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=_project_root,
        )
        if result.returncode == 0:
            meta["git_commit"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
            cwd=_project_root,
        )
        meta["git_dirty"] = bool(result.stdout.strip())
    except Exception:
        pass
    return meta


def _stats_summary(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    mean = sum(sorted_v) / n
    median = sorted_v[n // 2] if n % 2 == 1 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    std = (sum((v - mean) ** 2 for v in sorted_v) / (n - 1)) ** 0.5 if n > 1 else 0.0
    p50 = median
    k95 = max(0, min(n - 1, int(0.95 * n)))
    p95 = sorted_v[k95]
    return {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(std, 4),
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "min": round(sorted_v[0], 4),
        "max": round(sorted_v[-1], 4),
    }


def _save_trace(scheduler, output_dir: str, mode: str, concurrency: int, repeat: int) -> None:
    """Save scheduler trace records to a JSONL file."""
    records = scheduler.get_and_clear_trace()
    if not records:
        return
    filename = f"scheduler_trace_{mode}_c{concurrency}_r{repeat}.jsonl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

def run_benchmark_mode(
    mode: str,
    concurrency: int,
    requests_data: List[Dict[str, Any]],
    output_dir: str,
    model_path: str,
    num_gpu_blocks: int,
    repeat: int,
) -> Dict[str, Any]:
    """Run one mode-concurrency-repeat combination and return results."""
    config = _build_config(
        mode=mode,
        concurrency=concurrency,
        model_path=model_path,
        num_gpu_blocks=num_gpu_blocks,
        request_timeout_s=120.0,
        enable_prefix_caching=False,
    )
    engine = LLMEngine(config)
    start_time = time.time()
    try:
        if mode == "serial":
            all_outputs = _run_serial(engine, requests_data)
        elif mode == "static":
            all_outputs = _run_static_batch(engine, requests_data, concurrency)
        else:
            all_outputs = _run_continuous(engine, requests_data)
    except Exception as e:
        print(f"  ERROR in {mode} concurrency={concurrency}: {e}")
        traceback.print_exc()
        return {"error": str(e), "mode": mode, "concurrency": concurrency}
    wall_time = time.time() - start_time
    report = engine.engine_core.metrics_collector.report(include_per_request=True)
    report["benchmark_wall_time_s"] = round(wall_time, 3)
    _save_trace(engine.engine_core._scheduler, output_dir, mode, concurrency, repeat)
    finished_seqs = [s for s in engine.engine_core.metrics_collector._finished_seqs
                     if s.status == Status.FINISHED]
    report["num_success"] = len(finished_seqs)
    report["num_failed"] = len(engine.engine_core.metrics_collector._finished_seqs) - len(finished_seqs)
    report["num_cancelled"] = report.get("cancelled_requests", 0)
    report["num_timeout"] = report.get("timeout_requests", 0)
    report["all_outputs_ok"] = all(len(v) > 0 for v in all_outputs.values()) if all_outputs else False
    return report

def run_warmup(
    concurrency: int,
    requests_data: List[Dict[str, Any]],
    model_path: str,
    num_gpu_blocks: int,
) -> None:
    """Run a warmup (continuous mode) to stabilize GPU state."""
    config = _build_config(
        mode="continuous",
        concurrency=concurrency,
        model_path=model_path,
        num_gpu_blocks=num_gpu_blocks,
        request_timeout_s=120.0,
        enable_prefix_caching=False,
    )
    engine = LLMEngine(config)
    try:
        _run_continuous(engine, requests_data)
    except Exception:
        pass
    print(f"  Warmup done (concurrency={concurrency})")

def run_smoke_test(
    model_path: str,
    num_gpu_blocks: int,
    output_dir: str,
    executor_type: str = "qwen",
) -> bool:
    """Run smoke test: 4 requests, concurrency=2, all modes once."""
    print("\n" + "=" * 60)
    print("SMOKE TEST")
    print("=" * 60)
    print()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    test_requests = 4
    requests_data = _generate_request_prompts(tokenizer, test_requests, seed=42)
    print(f"  Requests: {test_requests}, Concurrency: 2")
    print(f"  Input tokens: {[r['actual_input_tokens'] for r in requests_data]}")
    print(f"  Output tokens: {[r['output_length'] for r in requests_data]}")
    print()
    all_ok = True
    for mode in ["serial", "continuous"]:
        print(f"  Running {mode} mode...")
        result = run_benchmark_mode(
            mode=mode, concurrency=2,
            requests_data=requests_data,
            output_dir=output_dir,
            model_path=model_path,
            num_gpu_blocks=num_gpu_blocks,
            repeat=0,
        )
        if "error" in result:
            print(f"  SMOKE TEST FAILED for {mode}: {result['error']}")
            all_ok = False
        else:
            print(f"  {mode}: {result['total_requests']} req, "
                  f"{result['throughput_req_per_sec']} req/s, "
                  f"TTFT P50={result.get('p50_ttft_ms', 'N/A')}ms")
        print()
    status_str = "PASSED" if all_ok else "FAILED"
    print(f"  Smoke test {status_str}")
    return all_ok

def _save_results(reports, metadata, output_dir):
    """Save all results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save environment metadata
    env_path = os.path.join(output_dir, "environment.json")
    with open(env_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Environment saved to {env_path}")

    # Save raw results
    raw_path = os.path.join(output_dir, "raw_results.json")
    with open(raw_path, 'w') as f:
        json.dump(reports, f, indent=2)
    print(f"  Raw results saved to {raw_path}")

    # Save summary CSV
    csv_path = os.path.join(output_dir, "summary.csv")
    csv_fields = [
        "mode", "concurrency", "repeat",
        "total_requests", "num_success", "num_failed", "num_cancelled", "num_timeout",
        "total_steps",
        "throughput_req_per_sec", "throughput_tok_per_sec",
        "total_generated_tokens", "total_time_seconds", "active_time_seconds",
        "avg_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "std_ttft_ms",
        "avg_tpot_ms", "p50_tpot_ms", "p95_tpot_ms", "std_tpot_ms",
        "avg_e2e_ms", "p50_e2e_ms", "p95_e2e_ms", "std_e2e_ms",
        "mean_effective_batch_size", "max_effective_batch_size",
        "mean_running_requests", "peak_running_requests", "peak_waiting_requests",
        "kv_util_peak_pct", "avg_scheduler_latency_ms",
        "total_prompt_tokens", "total_output_tokens",
        "benchmark_wall_time_s", "all_outputs_ok",
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        for report in reports:
            if 'error' not in report:
                writer.writerow(report)
    print(f"  Summary CSV saved to {csv_path}")

    # Save summary Markdown
    _save_markdown_summary(reports, metadata, output_dir)

    # Generate plots if matplotlib available
    try:
        _generate_plots(reports, metadata, output_dir)
    except ImportError:
        print("  matplotlib not available, skipping plots")
    except Exception as e:
        print(f"  Plot generation failed: {e}")

def _save_markdown_summary(reports, metadata, output_dir):
    """Generate a Markdown summary of benchmark results."""
    md_path = os.path.join(output_dir, "summary.md")
    model_name = os.path.basename(metadata.get('model_path', 'unknown'))
    gpu_name = metadata.get('gpu', 'unknown')

    with open(md_path, "w") as f:
        f.write("# Continuous Batching Benchmark Summary\n\n")
        f.write(f"- **Model**: {model_name}\n")
        f.write(f"- **GPU**: {gpu_name}\n")
        f.write(f"- **PyTorch**: {metadata.get('pytorch_version', '?')}\n")
        f.write(f"- **dtype**: float16\n")
        f.write(f"- **Git commit**: {metadata.get('git_commit', '?')}\n")
        f.write(f"- **Working tree dirty**: {metadata.get('git_dirty', '?')}\n")
        f.write(f"- **Timestamp**: {metadata.get('timestamp', '?')}\n")
        f.write("\n---\n\n")

        # Group by mode and concurrency
        grouped: Dict[Tuple[str, int], List[Dict]] = {}
        for r in reports:
            if 'error' in r:
                continue
            key = (r["mode"], r["concurrency"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)

        for mode in ["serial", "static", "continuous"]:
            mode_reports = {k: v for k, v in grouped.items() if k[0] == mode}
            if not mode_reports:
                continue
            f.write(f"## Mode: {mode.upper()}\n\n")
            f.write("| Concurrency | Req/s | Tok/s | TTFT P50 (ms) | TTFT P95 (ms) | TPOT P50 (ms) | TPOT P95 (ms) | E2E P50 (ms) | E2E P95 (ms) | Mean Batch |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for concurrency in sorted(set(k[1] for k in mode_reports)):
                reports_for_c = mode_reports.get((mode, concurrency), [])
                if not reports_for_c:
                    continue
                avg_req_s = sum(r['throughput_req_per_sec'] for r in reports_for_c) / len(reports_for_c)
                avg_tok_s = sum(r['throughput_tok_per_sec'] for r in reports_for_c) / len(reports_for_c)
                avg_ttft50 = sum(r.get('p50_ttft_ms', 0) for r in reports_for_c) / len(reports_for_c)
                avg_ttft95 = sum(r.get('p95_ttft_ms', 0) for r in reports_for_c) / len(reports_for_c)
                avg_tpot50 = sum(r.get('p50_tpot_ms', 0) for r in reports_for_c) / len(reports_for_c)
                avg_tpot95 = sum(r.get('p95_tpot_ms', 0) for r in reports_for_c) / len(reports_for_c)
                avg_e2e50 = sum(r.get('p50_e2e_ms', 0) for r in reports_for_c) / len(reports_for_c)
                avg_e2e95 = sum(r.get('p95_e2e_ms', 0) for r in reports_for_c) / len(reports_for_c)
                avg_batch = sum(r.get('mean_effective_batch_size', 0) for r in reports_for_c) / len(reports_for_c)
                f.write(
                    f"| {concurrency} | {avg_req_s:.2f} | {avg_tok_s:.1f} |"
                    f" {avg_ttft50:.1f} | {avg_ttft95:.1f} |"
                    f" {avg_tpot50:.2f} | {avg_tpot95:.2f} |"
                    f" {avg_e2e50:.1f} | {avg_e2e95:.1f} | {avg_batch:.1f} |\n"
                )
            f.write("\n")


def _generate_plots(reports, metadata, output_dir):
    """Generate benchmark plots using matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Filter valid reports
    valid = [r for r in reports if 'error' not in r]
    if not valid:
        return

    # Group by mode
    by_mode = {}
    for r in valid:
        by_mode.setdefault(r['mode'], []).append(r)

    model_name = os.path.basename(metadata.get('model_path', 'unknown'))

    # 1. Concurrency vs Request Throughput
    plt.figure(figsize=(8, 5))
    for mode, color in [('serial', 'gray'), ('static', 'blue'), ('continuous', 'green')]:
        if mode not in by_mode:
            continue
        conc_levels = sorted(set(r['concurrency'] for r in by_mode[mode]))
        means = []
        for c in conc_levels:
            vals = [r['throughput_req_per_sec'] for r in by_mode[mode] if r['concurrency'] == c]
            means.append(sum(vals) / len(vals) if vals else 0)
        plt.plot(conc_levels, means, marker='o', label=mode, color=color, linewidth=2)
    plt.xlabel('Concurrency')
    plt.ylabel('Request Throughput (req/s)')
    plt.title(f'Request Throughput vs Concurrency\n{model_name}, float16')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'throughput_req.png'), dpi=150)
    plt.close()

    # 2. Concurrency vs Token Throughput
    plt.figure(figsize=(8, 5))
    for mode, color in [('serial', 'gray'), ('static', 'blue'), ('continuous', 'green')]:
        if mode not in by_mode:
            continue
        conc_levels = sorted(set(r['concurrency'] for r in by_mode[mode]))
        means = []
        for c in conc_levels:
            vals = [r['throughput_tok_per_sec'] for r in by_mode[mode] if r['concurrency'] == c]
            means.append(sum(vals) / len(vals) if vals else 0)
        plt.plot(conc_levels, means, marker='s', label=mode, color=color, linewidth=2)
    plt.xlabel('Concurrency')
    plt.ylabel('Token Throughput (tok/s)')
    plt.title(f'Token Throughput vs Concurrency\n{model_name}, float16')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'throughput_tok.png'), dpi=150)
    plt.close()

    # 3. P95 TTFT / P95 TPOT comparison (at highest concurrency)
    plt.figure(figsize=(10, 5))
    max_conc = max(r['concurrency'] for r in valid)
    modes_found = []
    ttft95_vals = []
    tpot95_vals = []
    colors = []
    for mode, color in [('serial', 'gray'), ('static', 'blue'), ('continuous', 'green')]:
        if mode not in by_mode:
            continue
        at_max = [r for r in by_mode[mode] if r['concurrency'] == max_conc]
        if not at_max:
            continue
        avg_ttft95 = sum(r.get('p95_ttft_ms', 0) for r in at_max) / len(at_max)
        avg_tpot95 = sum(r.get('p95_tpot_ms', 0) for r in at_max) / len(at_max)
        modes_found.append(mode)
        ttft95_vals.append(avg_ttft95)
        tpot95_vals.append(avg_tpot95)
        colors.append(color)

    x = range(len(modes_found))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar([i - width/2 for i in x], ttft95_vals, width, label='P95 TTFT', color='orange')
    bars2 = ax.bar([i + width/2 for i in x], tpot95_vals, width, label='P95 TPOT', color='purple')
    ax.set_xlabel('Mode')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'P95 Latency Comparison (concurrency={max_conc})\n{model_name}, float16')
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes_found)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'latency_comparison.png'), dpi=150)
    plt.close()

    print(f"  Plots saved to {plots_dir}")

def _build_ab_config(
    concurrency: int,
    model_path: str,
    num_gpu_blocks: int,
    attention_backend: str,
    **overrides,
) -> "Config":
    """Build Config for paged executor with the given attention backend."""
    from mini_vllm import Config
    params = dict(
        executor_type="paged",
        attention_backend=attention_backend,
        max_num_seqs=concurrency,
        max_num_batched_tokens=8192,
        max_num_prefill_tokens=2048,
        max_prefill_chunk_size=128,
        block_size=16,
        chunked_prefill_enabled=True,
        decode_first=True,
        print_step_events=False,
        memory_trace=False,
        trace_enabled=True,
        model_path=model_path,
        num_gpu_blocks=num_gpu_blocks,
        request_timeout_s=120.0,
        enable_prefix_caching=False,
    )
    params.update(overrides)
    return Config(**params)


def _extract_output_token_ids(engine):
    """Extract per-request output token IDs from a finished engine."""
    from mini_vllm.sequence.status import Status as SeqStatus
    token_ids = {}
    for rid, sg in engine._queue._finished.items():
        for seq in sg.seqs:
            if seq.status == SeqStatus.FINISHED:
                token_ids[rid] = list(seq.output_token_ids)
    return token_ids


def _run_ab_single_backend(
    concurrency: int,
    requests_data,
    model_path: str,
    num_gpu_blocks: int,
    attention_backend: str,
    label: str = "",
) -> dict:
    """Run continuous batching with the specified attention backend."""
    import time
    import traceback
    import torch
    from mini_vllm import LLMEngine, Status

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    config = _build_ab_config(
        concurrency=concurrency,
        model_path=model_path,
        num_gpu_blocks=num_gpu_blocks,
        attention_backend=attention_backend,
    )
    engine = LLMEngine(config)
    start_time = time.time()
    try:
        for i, req in enumerate(requests_data):
            engine.add_request(req["prompt"], max_new_tokens=req["output_length"])
        engine.run_until_done()
    except Exception as e:
        print(f"  ERROR in {label} concurrency={concurrency}: {e}")
        traceback.print_exc()
        del engine
        _cleanup_gpu()
        return {"error": str(e), "backend": attention_backend, "concurrency": concurrency}

    wall_time = time.time() - start_time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
    else:
        peak_memory = 0

    report = engine.engine_core.metrics_collector.report(include_per_request=True)
    report["benchmark_wall_time_s"] = round(wall_time, 3)
    report["peak_gpu_memory_bytes"] = peak_memory
    report["backend"] = attention_backend
    report["concurrency"] = concurrency

    finished_seqs = [s for s in engine.engine_core.metrics_collector._finished_seqs
                     if s.status == Status.FINISHED]
    report["num_success"] = len(finished_seqs)
    report["num_failed"] = len(engine.engine_core.metrics_collector._finished_seqs) - len(finished_seqs)

    token_ids = _extract_output_token_ids(engine)
    report["_output_token_ids"] = token_ids

    # Explicitly drop engine reference before cleanup to ensure GPU memory
    # is freed (the local variable must be gone before gc.collect runs).
    del engine
    _cleanup_gpu()
    return report


def _create_ab_result_dir(base_dir: str = "benchmark_results"):
    """Create timestamped result directory with git hash."""
    import os, time, subprocess
    ts = time.strftime("%Y%m%d_%H%M%S")
    git_hash = ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_hash = "_" + result.stdout.strip()
    except Exception:
        pass
    dirname = f"{ts}{git_hash}"
    result_dir = os.path.join(base_dir, "continuous_batching_backend_ab", dirname)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir, dirname


def _compare_correctness(
    ref_token_ids, triton_token_ids,
    ref_request_ids, triton_request_ids,
):
    """Compare output token sequences between reference and triton."""
    results = {
        "total_ref_requests": len(ref_token_ids),
        "total_triton_requests": len(triton_token_ids),
        "exact_matches": 0,
        "partial_matches": 0,
        "total_mismatches": 0,
        "details": [],
    }
    for rid in ref_request_ids:
        ref_tokens = ref_token_ids.get(rid, [])
        tri_tokens = triton_token_ids.get(rid, [])
        if len(ref_tokens) == len(tri_tokens) and ref_tokens == tri_tokens:
            results["exact_matches"] += 1
        elif ref_tokens and tri_tokens:
            min_len = min(len(ref_tokens), len(tri_tokens))
            first_diff = next(
                (i for i in range(min_len) if ref_tokens[i] != tri_tokens[i]),
                None,
            )
            results["details"].append({
                "request_id": rid,
                "ref_tokens": len(ref_tokens),
                "triton_tokens": len(tri_tokens),
                "first_diff_pos": first_diff,
                "ref_at_diff": int(ref_tokens[first_diff]) if first_diff is not None else None,
                "tri_at_diff": int(tri_tokens[first_diff]) if first_diff is not None else None,
            })
            if first_diff is None:
                results["partial_matches"] += 1
            else:
                results["total_mismatches"] += 1
    results["correctness_pass"] = results["exact_matches"] == results["total_ref_requests"]
    return results


def _compute_ab_derived_metrics(ref_report, triton_report):
    """Compute derived speedup metrics between backends."""
    ref_rps = ref_report.get("throughput_req_per_sec", 0.0)
    tri_rps = triton_report.get("throughput_req_per_sec", 0.0)
    ref_tps = ref_report.get("throughput_tok_per_sec", 0.0)
    tri_tps = triton_report.get("throughput_tok_per_sec", 0.0)
    ref_tpot = ref_report.get("avg_tpot_ms", 0.0)
    tri_tpot = triton_report.get("avg_tpot_ms", 0.0)
    return {
        "concurrency": ref_report.get("concurrency", triton_report.get("concurrency", 0)),
        "throughput_speedup_req": round(tri_rps / ref_rps, 4) if ref_rps > 0 else 0.0,
        "throughput_speedup_tok": round(tri_tps / ref_tps, 4) if ref_tps > 0 else 0.0,
        "throughput_uplift_pct": round((tri_rps - ref_rps) / ref_rps * 100, 2) if ref_rps > 0 else 0.0,
        "tpot_reduction_pct": round((ref_tpot - tri_tpot) / ref_tpot * 100, 2) if ref_tpot > 0 else 0.0,
        "ref_peak_gpu_mem_mb": round(ref_report.get("peak_gpu_memory_bytes", 0) / 1024 / 1024, 1),
        "triton_peak_gpu_mem_mb": round(triton_report.get("peak_gpu_memory_bytes", 0) / 1024 / 1024, 1),
        "ref_wall_time_s": ref_report.get("benchmark_wall_time_s", 0.0),
        "triton_wall_time_s": triton_report.get("benchmark_wall_time_s", 0.0),
        "ref_avg_tpot_ms": ref_tpot,
        "triton_avg_tpot_ms": tri_tpot,
        "ref_p50_tpot_ms": ref_report.get("p50_tpot_ms", 0.0),
        "triton_p50_tpot_ms": triton_report.get("p50_tpot_ms", 0.0),
        "ref_avg_ttft_ms": ref_report.get("avg_ttft_ms", 0.0),
        "triton_avg_ttft_ms": triton_report.get("avg_ttft_ms", 0.0),
        "ref_avg_batch": ref_report.get("mean_effective_batch_size", 0.0),
        "triton_avg_batch": triton_report.get("mean_effective_batch_size", 0.0),
    }




def _save_ab_results(all_pairs, derived_metrics_list, correctness_list,
                     environment_data, result_dir, requests_data):
    """Save all A/B experiment results to the result directory."""
    import os, json, csv

    # 1. raw_results.json - clean reports (remove _output_token_ids)
    clean_pairs = []
    for pair in all_pairs:
        cp = {
            "concurrency": pair["concurrency"],
            "order": pair.get("order", ""),
            "repeat": pair.get("repeat", 0),
            "reference": {k: v for k, v in pair["reference"].items() if k != "_output_token_ids"},
            "triton": {k: v for k, v in pair["triton"].items() if k != "_output_token_ids"},
            "derived_metrics": pair.get("derived_metrics", {}),
            "correctness": pair.get("correctness", {}),
        }
        clean_pairs.append(cp)
    with open(os.path.join(result_dir, "raw_results.json"), "w") as f:
        json.dump(clean_pairs, f, indent=2, default=str)

    # 2. environment.json
    with open(os.path.join(result_dir, "environment.json"), "w") as f:
        json.dump(environment_data, f, indent=2, default=str)

    # 3. correctness.json
    with open(os.path.join(result_dir, "correctness.json"), "w") as f:
        json.dump(correctness_list, f, indent=2, default=str)

    # 4. summary.csv
    csv_fields = [
        "concurrency", "order",
        "ref_throughput_req", "tri_throughput_req", "speedup_req",
        "ref_throughput_tok", "tri_throughput_tok", "speedup_tok",
        "throughput_uplift_pct", "tpot_reduction_pct",
        "ref_avg_tpot_ms", "tri_avg_tpot_ms",
        "ref_p50_tpot_ms", "tri_p50_tpot_ms",
        "ref_avg_ttft_ms", "tri_avg_ttft_ms",
        "ref_avg_e2e_ms", "tri_avg_e2e_ms",
        "ref_peak_gpu_mem_mb", "tri_peak_gpu_mem_mb",
        "ref_wall_time_s", "tri_wall_time_s",
        "ref_avg_batch", "tri_avg_batch",
        "correctness_pass", "correctness_exact",
    ]
    csv_path = os.path.join(result_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for dm in derived_metrics_list:
            correctness = dm.get("correctness", {})
            writer.writerow({
                "concurrency": dm.get("concurrency", ""),
                "order": dm.get("order", ""),
                "ref_throughput_req": dm.get("ref_throughput_req", ""),
                "tri_throughput_req": dm.get("tri_throughput_req", ""),
                "speedup_req": dm.get("throughput_speedup_req", ""),
                "ref_throughput_tok": dm.get("ref_throughput_tok", ""),
                "tri_throughput_tok": dm.get("tri_throughput_tok", ""),
                "speedup_tok": dm.get("throughput_speedup_tok", ""),
                "throughput_uplift_pct": dm.get("throughput_uplift_pct", ""),
                "tpot_reduction_pct": dm.get("tpot_reduction_pct", ""),
                "ref_avg_tpot_ms": dm.get("ref_avg_tpot_ms", ""),
                "tri_avg_tpot_ms": dm.get("triton_avg_tpot_ms", ""),
                "ref_p50_tpot_ms": dm.get("ref_p50_tpot_ms", ""),
                "tri_p50_tpot_ms": dm.get("triton_p50_tpot_ms", ""),
                "ref_avg_ttft_ms": dm.get("ref_avg_ttft_ms", ""),
                "tri_avg_ttft_ms": dm.get("triton_avg_ttft_ms", ""),
                "ref_avg_e2e_ms": dm.get("ref_avg_e2e_ms", ""),
                "tri_avg_e2e_ms": dm.get("triton_avg_e2e_ms", ""),
                "ref_peak_gpu_mem_mb": dm.get("ref_peak_gpu_mem_mb", ""),
                "tri_peak_gpu_mem_mb": dm.get("triton_peak_gpu_mem_mb", ""),
                "ref_wall_time_s": dm.get("ref_wall_time_s", ""),
                "tri_wall_time_s": dm.get("triton_wall_time_s", ""),
                "ref_avg_batch": dm.get("ref_avg_batch", ""),
                "tri_avg_batch": dm.get("triton_avg_batch", ""),
                "correctness_pass": correctness.get("correctness_pass", False),
                "correctness_exact": correctness.get("exact_matches", 0),
            })
    print(f"  Summary CSV saved to {csv_path}")

    # 5. summary.md
    _save_ab_markdown(all_pairs, derived_metrics_list, correctness_list,
                      environment_data, result_dir)

    # 6. commands.sh
    _save_ab_commands(result_dir)

    # 7. git_status.txt
    try:
        import subprocess
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        with open(os.path.join(result_dir, "git_status.txt"), "w") as f:
            f.write(result.stdout)
    except Exception:
        pass

    # 8. Plots
    try:
        _generate_ab_plots(derived_metrics_list, environment_data, result_dir)
    except ImportError:
        print("  matplotlib not available, skipping plots")
    except Exception as e:
        print(f"  Plot generation failed: {e}")


def _save_ab_commands(result_dir):
    """Save the shell command used to run the experiment."""
    import sys, os
    with open(os.path.join(result_dir, "commands.sh"), "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(f"# A/B Experiment - {os.path.basename(result_dir)}\n")
        f.write(f"# Command: {' '.join(sys.argv)}\n")
        f.write(f"# Timestamp: {__import__('time').strftime('%Y-%m-%dT%H:%M:%S')}\n")
        f.write("set -x\n")
        f.write(f"{' '.join(sys.argv)}\n")
    os.chmod(os.path.join(result_dir, "commands.sh"), 0o755)


def _save_ab_markdown(all_pairs, derived_metrics_list, correctness_list,
                      env, result_dir):
    """Generate Markdown summary for A/B experiment."""
    import os
    md_path = os.path.join(result_dir, "summary.md")
    model_name = os.path.basename(env.get("model_path", "unknown"))
    gpu_name = env.get("gpu", "unknown")

    with open(md_path, "w") as f:
        f.write("# Continuous Batching Backend A/B Experiment\n\n")
        f.write("Reference Attention vs Triton Paged Decode Attention\n\n")
        f.write(f"- **Model**: {model_name}\n")
        f.write(f"- **GPU**: {gpu_name}\n")
        f.write(f"- **PyTorch**: {env.get('pytorch_version', '?')}\n")
        f.write(f"- **dtype**: float16\n")
        f.write(f"- **Git commit**: {env.get('git_commit', '?')}\n")
        f.write(f"- **Working tree dirty**: {env.get('git_dirty', '?')}\n")
        f.write(f"- **Timestamp**: {env.get('timestamp', '?')}\n")
        f.write(f"- **Requests per run**: {env.get('num_requests', '?')}\n")
        f.write(f"- **GPU blocks**: {env.get('num_gpu_blocks', '?')}\n\n")
        f.write("---\n\n")

        # Overview table
        f.write("## Throughput Comparison\n\n")
        f.write("| Concurrency | Order | Ref Req/s | Tri Req/s | Speedup | "
                "Ref Tok/s | Tri Tok/s | Speedup | Uplift % | "
                "TPOT Reduction % | Correct |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for dm in derived_metrics_list:
            c = dm.get("correctness", {})
            f.write(
                f"| {dm.get('concurrency', '?')} "
                f"| {dm.get('order', '?')} "
                f"| {dm.get('ref_throughput_req', 0):.2f} "
                f"| {dm.get('tri_throughput_req', 0):.2f} "
                f"| {dm.get('throughput_speedup_req', 0):.2f}x "
                f"| {dm.get('ref_throughput_tok', 0):.1f} "
                f"| {dm.get('tri_throughput_tok', 0):.1f} "
                f"| {dm.get('throughput_speedup_tok', 0):.2f}x "
                f"| {dm.get('throughput_uplift_pct', 0):.1f}% "
                f"| {dm.get('tpot_reduction_pct', 0):.1f}% "
                f"| {'PASS' if c.get('correctness_pass') else 'FAIL'}"
                f" ({c.get('exact_matches', 0)}/{c.get('total_ref_requests', 0)}) |\n"
            )
        f.write("\n")

        # Latency table
        f.write("## Latency Comparison\n\n")
        f.write("| Concurrency | Ref TPOT50 | Tri TPOT50 | Ref TPOT95 | Tri TPOT95 | "
                "Ref TTFT50 | Tri TTFT50 | Ref E2E50 | Tri E2E50 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for dm in derived_metrics_list:
            f.write(
                f"| {dm.get('concurrency', '?')} "
                f"| {dm.get('ref_p50_tpot_ms', 0):.2f} "
                f"| {dm.get('triton_p50_tpot_ms', 0):.2f} "
                f"| {dm.get('ref_p95_tpot_ms', 0):.2f} "
                f"| {dm.get('triton_p95_tpot_ms', 0):.2f} "
                f"| {dm.get('ref_avg_ttft_ms', 0):.1f} "
                f"| {dm.get('triton_avg_ttft_ms', 0):.1f} "
                f"| {dm.get('ref_avg_e2e_ms', 0):.1f} "
                f"| {dm.get('triton_avg_e2e_ms', 0):.1f} |\n"
            )
        f.write("\n")

        # Memory table
        f.write("## GPU Memory\n\n")
        f.write("| Concurrency | Ref Peak (MB) | Tri Peak (MB) |\n")
        f.write("|---|---|---|\n")
        for dm in derived_metrics_list:
            f.write(
                f"| {dm.get('concurrency', '?')} "
                f"| {dm.get('ref_peak_gpu_mem_mb', 0):.1f} "
                f"| {dm.get('triton_peak_gpu_mem_mb', 0):.1f} |\n"
            )
        f.write("\n")

        # Correctness details
        f.write("## Correctness\n\n")
        all_pass = all(c.get("correctness_pass", False) for c in correctness_list)
        f.write(f"**Overall Correctness: {'PASS' if all_pass else 'FAIL'}**\n\n")
        for i, c in enumerate(correctness_list):
            f.write(f"### Run {i + 1} (concurrency={c.get('concurrency', '?')}, order={c.get('order', '?')})\n\n")
            f.write(f"- Exact matches: {c.get('exact_matches', 0)}/{c.get('total_ref_requests', 0)}\n")
            f.write(f"- Mismatches: {c.get('total_mismatches', 0)}\n")
            details = c.get("details", [])
            if details:
                f.write("- Mismatch details:\n")
                for d in details[:10]:
                    f.write(f"  - {d['request_id']}: ref={d['ref_tokens']}tok vs tri={d['triton_tokens']}tok")
                    if d.get("first_diff_pos") is not None:
                        f.write(f", first diff at pos {d['first_diff_pos']}")
                    f.write("\n")
                if len(details) > 10:
                    f.write(f"  - ... and {len(details) - 10} more\n")
            f.write("\n")

        # Workload details
        f.write("## Workload\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Total requests | {env.get('num_requests', '?')} |\n")
        f.write(f"| Concurrency levels | {env.get('concurrency_levels', '?')} |\n")
        f.write(f"| Repeats | {env.get('repeats', 1)} |\n")
        f.write(f"| Workload | {env.get('workload_desc', 'default')} |\n")
        f.write("\n")

        print(f"  Summary MD saved to {md_path}")


def _generate_ab_plots(derived_metrics_list, env, result_dir):
    """Generate A/B comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    plots_dir = os.path.join(result_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    model_name = os.path.basename(env.get("model_path", "unknown"))

    if not derived_metrics_list:
        return

    conc_levels = sorted(set(dm.get("concurrency", 0) for dm in derived_metrics_list))

    # 1. Throughput comparison (req/s)
    plt.figure(figsize=(8, 5))
    ref_rps = []
    tri_rps = []
    for c in conc_levels:
        vals_ref = [dm.get("ref_throughput_req", 0) for dm in derived_metrics_list if dm.get("concurrency") == c]
        vals_tri = [dm.get("tri_throughput_req", 0) for dm in derived_metrics_list if dm.get("concurrency") == c]
        ref_rps.append(sum(vals_ref) / len(vals_ref) if vals_ref else 0)
        tri_rps.append(sum(vals_tri) / len(vals_tri) if vals_tri else 0)
    x = range(len(conc_levels))
    width = 0.35
    plt.bar([i - width/2 for i in x], ref_rps, width, label="Reference (SDPA)", color="orange", alpha=0.8)
    plt.bar([i + width/2 for i in x], tri_rps, width, label="Triton Paged Decode", color="steelblue", alpha=0.8)
    plt.xlabel("Concurrency")
    plt.ylabel("Throughput (req/s)")
    plt.title(f"Request Throughput: Reference vs Triton\n{model_name}, float16")
    plt.xticks(list(x), [str(c) for c in conc_levels])
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "throughput_comparison.png"), dpi=150)
    plt.close()

    # 2. TPOT comparison
    plt.figure(figsize=(8, 5))
    ref_tpot = []
    tri_tpot = []
    for c in conc_levels:
        vals_ref = [dm.get("ref_avg_tpot_ms", 0) for dm in derived_metrics_list if dm.get("concurrency") == c]
        vals_tri = [dm.get("triton_avg_tpot_ms", 0) for dm in derived_metrics_list if dm.get("concurrency") == c]
        ref_tpot.append(sum(vals_ref) / len(vals_ref) if vals_ref else 0)
        tri_tpot.append(sum(vals_tri) / len(vals_tri) if vals_tri else 0)
    plt.bar([i - width/2 for i in x], ref_tpot, width, label="Reference (SDPA)", color="orange", alpha=0.8)
    plt.bar([i + width/2 for i in x], tri_tpot, width, label="Triton Paged Decode", color="steelblue", alpha=0.8)
    plt.xlabel("Concurrency")
    plt.ylabel("Avg TPOT (ms)")
    plt.title(f"TPOT: Reference vs Triton\n{model_name}, float16")
    plt.xticks(list(x), [str(c) for c in conc_levels])
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "tpot_comparison.png"), dpi=150)
    plt.close()

    # 3. Speedup heatmap (as bar chart)
    plt.figure(figsize=(8, 5))
    speedups = []
    for c in conc_levels:
        vals = [dm.get("throughput_speedup_req", 0) for dm in derived_metrics_list if dm.get("concurrency") == c]
        speedups.append(sum(vals) / len(vals) if vals else 1.0)
    colors_speedup = ["green" if s >= 1.0 else "red" for s in speedups]
    plt.bar(list(x), speedups, color=colors_speedup, alpha=0.8)
    plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Concurrency")
    plt.ylabel("Speedup (Triton / Reference)")
    plt.title(f"Throughput Speedup vs Concurrency\n{model_name}, float16")
    plt.xticks(list(x), [str(c) for c in conc_levels])
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "speedup_comparison.png"), dpi=150)
    plt.close()

    # 4. GPU memory comparison
    plt.figure(figsize=(8, 5))
    ref_mem = []
    tri_mem = []
    for c in conc_levels:
        vals_ref = [dm.get("ref_peak_gpu_mem_mb", 0) for dm in derived_metrics_list if dm.get("concurrency") == c]
        vals_tri = [dm.get("triton_peak_gpu_mem_mb", 0) for dm in derived_metrics_list if dm.get("concurrency") == c]
        ref_mem.append(sum(vals_ref) / len(vals_ref) if vals_ref else 0)
        tri_mem.append(sum(vals_tri) / len(vals_tri) if vals_tri else 0)
    plt.bar([i - width/2 for i in x], ref_mem, width, label="Reference", color="orange", alpha=0.8)
    plt.bar([i + width/2 for i in x], tri_mem, width, label="Triton", color="steelblue", alpha=0.8)
    plt.xlabel("Concurrency")
    plt.ylabel("Peak GPU Memory (MB)")
    plt.title(f"GPU Memory: Reference vs Triton\n{model_name}, float16")
    plt.xticks(list(x), [str(c) for c in conc_levels])
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "memory_comparison.png"), dpi=150)
    plt.close()

    print(f"  Plots saved to {plots_dir}")



def _cleanup_gpu(engine=None):
    """Delete engine and free GPU memory."""
    import gc
    import torch
    if engine is not None:
        del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()




def _run_ab_experiment(args):
    """Run the full A/B experiment: Reference vs Triton attention backends."""
    import os, json, torch
    from transformers import AutoTokenizer

    model_path = args.model_path or _auto_model_path()
    if not model_path or not os.path.isdir(model_path):
        print("ERROR: Valid model path is required for A/B experiment.")
        print("Use --model-path to specify the local model directory.")
        sys.exit(1)
    print(f"  Model path: {model_path}")

    num_gpu_blocks = args.num_gpu_blocks or _calc_gpu_blocks(model_path)
    print(f"  KV cache blocks: {num_gpu_blocks}")

    result_dir, dirname = _create_ab_result_dir(args.output_dir)
    print(f"  Result directory: {result_dir}")

    env = _compute_metadata()
    env["model_path"] = model_path
    env["dtype"] = "float16"
    env["num_gpu_blocks"] = num_gpu_blocks
    env["seed"] = 42
    env["args"] = vars(args)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    num_requests = args.requests
    env["num_requests"] = num_requests
    env["concurrency_levels"] = args.concurrency
    env["repeats"] = args.repeats

    if args.homogeneous_ctx:
        if args.homogeneous_out:
            env["workload_desc"] = "homogeneous_ctx={}_out={}".format(args.homogeneous_ctx, args.homogeneous_out)
        else:
            env["workload_desc"] = "homogeneous_ctx={}".format(args.homogeneous_ctx)
    elif args.ragged_ctx:
        env["workload_desc"] = "ragged_ctx"
    elif args.ragged_out:
        env["workload_desc"] = "ragged_out"
    else:
        env["workload_desc"] = "default_mixed"

    requests_data = _generate_ab_request_prompts(
        tokenizer, num_requests, seed=42,
        homogeneous_ctx=args.homogeneous_ctx,
        homogeneous_out=args.homogeneous_out,
        ragged_ctx=args.ragged_ctx,
        ragged_out=args.ragged_out,
    )
    print(f"\n  Generated {len(requests_data)} requests")
    avg_in = sum(r["actual_input_tokens"] for r in requests_data) / len(requests_data)
    avg_out = sum(r["output_length"] for r in requests_data) / len(requests_data)
    print(f"  Avg input tokens: {avg_in:.0f}")
    print(f"  Avg output tokens: {avg_out:.0f}")
    print(f"  Workload: {env['workload_desc']}")

    concurrency_levels = sorted(args.concurrency)
    all_pairs = []
    derived_metrics_list = []
    correctness_list = []

    for order_idx, concurrency in enumerate(concurrency_levels):
        # Free GPU from previous iteration's engines
        _cleanup_gpu()

        print("\n" + "=" * 60)
        print("CONCURRENCY = {}".format(concurrency))
        print("=" * 60)

        if order_idx % 2 == 0:
            first_backend, second_backend = "reference", "triton"
            first_label, second_label = "Reference (1st)", "Triton (2nd)"
            order_str = "ref_first"
        else:
            first_backend, second_backend = "triton", "reference"
            first_label, second_label = "Triton (1st)", "Reference (2nd)"
            order_str = "tri_first"

        print(f"\n  Order: {first_label} -> {second_label}\n")

        # Warmup with first backend
        print(f"  Warmup...", end=" ", flush=True)
        warmup_config = _build_ab_config(
            concurrency=concurrency,
            model_path=model_path,
            num_gpu_blocks=num_gpu_blocks,
            attention_backend=first_backend,
        )
        try:
            warmup_engine = LLMEngine(warmup_config)
            wu_data = requests_data[:min(4, len(requests_data))]
            for req in wu_data:
                warmup_engine.add_request(req["prompt"], max_new_tokens=req["output_length"])
            warmup_engine.run_until_done()
            print("done")
        except Exception as e:
            print(f"warmup failed ({e}), continuing")
        finally:
            _cleanup_gpu()

        for rep in range(args.repeats):
            print(f"\n  Repeat {rep + 1}/{args.repeats}")

            # First backend
            print(f"    Running {first_label}...", end=" ", flush=True)
            result1 = _run_ab_single_backend(
                concurrency=concurrency,
                requests_data=requests_data,
                model_path=model_path,
                num_gpu_blocks=num_gpu_blocks,
                attention_backend=first_backend,
                label=first_label,
            )
            if "error" in result1:
                print(f"ERROR: {result1['error']}")
                _cleanup_gpu()
                continue

            rps1 = result1["throughput_req_per_sec"]
            tpot1 = result1.get("p50_tpot_ms", 0)
            print(f"{result1['total_requests']} req, {rps1:.2f} req/s, TPOT P50={tpot1:.2f}ms")

            # Free GPU memory before loading second backend
            _cleanup_gpu()

            # Second backend
            print(f"    Running {second_label}...", end=" ", flush=True)
            result2 = _run_ab_single_backend(
                concurrency=concurrency,
                requests_data=requests_data,
                model_path=model_path,
                num_gpu_blocks=num_gpu_blocks,
                attention_backend=second_backend,
                label=second_label,
            )
            if "error" in result2:
                print(f"ERROR: {result2['error']}")
                _cleanup_gpu()
                continue

            rps2 = result2["throughput_req_per_sec"]
            tpot2 = result2.get("p50_tpot_ms", 0)
            print(f"{result2['total_requests']} req, {rps2:.2f} req/s, TPOT P50={tpot2:.2f}ms")

            if first_backend == "reference":
                ref_result, tri_result = result1, result2
            else:
                ref_result, tri_result = result2, result1

            # Correctness
            ref_token_ids = ref_result.get("_output_token_ids", {})
            tri_token_ids = tri_result.get("_output_token_ids", {})
            correctness = _compare_correctness(
                ref_token_ids, tri_token_ids,
                list(ref_token_ids.keys()), list(tri_token_ids.keys()),
            )
            correctness["concurrency"] = concurrency
            correctness["order"] = order_str
            correctness_list.append(correctness)
            cpass = "PASS" if correctness["correctness_pass"] else "FAIL"
            print(f"    Correctness: {cpass} ({correctness['exact_matches']}/{correctness['total_ref_requests']})")

            # Derived metrics
            derived = _compute_ab_derived_metrics(ref_result, tri_result)
            derived["order"] = order_str
            derived["correctness"] = correctness
            derived["ref_throughput_req"] = ref_result.get("throughput_req_per_sec", 0)
            derived["tri_throughput_req"] = tri_result.get("throughput_req_per_sec", 0)
            derived["ref_throughput_tok"] = ref_result.get("throughput_tok_per_sec", 0)
            derived["tri_throughput_tok"] = tri_result.get("throughput_tok_per_sec", 0)
            derived["ref_avg_e2e_ms"] = ref_result.get("avg_e2e_ms", 0)
            derived["triton_avg_e2e_ms"] = tri_result.get("avg_e2e_ms", 0)
            derived["ref_p95_tpot_ms"] = ref_result.get("p95_tpot_ms", 0)
            derived["triton_p95_tpot_ms"] = tri_result.get("p95_tpot_ms", 0)
            derived_metrics_list.append(derived)

            all_pairs.append({
                "concurrency": concurrency,
                "order": order_str,
                "repeat": rep,
                "reference": ref_result,
                "triton": tri_result,
            })

            speedup = derived["throughput_speedup_req"]
            tpot_red = derived["tpot_reduction_pct"]
            print(f"    Speedup: {speedup:.2f}x req/s, {tpot_red:.1f}% TPOT reduction")

    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60 + "\n")
    _save_ab_results(all_pairs, derived_metrics_list, correctness_list,
                     env, result_dir, requests_data)

    print("\n" + "=" * 60)
    print("A/B EXPERIMENT SUMMARY")
    print("=" * 60 + "\n")
    hdr = "{:<8} {:<14} {:<12} {:<12} {:<10} {:<12} {:<10}".format(
        "Conc", "Order", "Ref Req/s", "Tri Req/s", "Speedup", "TPOT Red%", "Correct")
    print(hdr)
    print("-" * 78)
    for dm in derived_metrics_list:
        c = dm.get("correctness", {})
        cstr = "PASS" if c.get("correctness_pass") else "FAIL"
        print("{:<8} {:<14} {:<12.2f} {:<12.2f} {:<10.2f}x {:<12.1f} {:<10}".format(
            dm.get('concurrency', 0), dm.get('order', ''),
            dm.get('ref_throughput_req', 0), dm.get('tri_throughput_req', 0),
            dm.get('throughput_speedup_req', 0), dm.get('tpot_reduction_pct', 0),
            cstr))

    print(f"\nResults saved to: {result_dir}")
    print("Done!")



def build_parser():
    parser = argparse.ArgumentParser(description="Continuous Batching Benchmark")
    parser.add_argument("--executor", choices=["fake", "qwen"], default="qwen",
                        help="Executor type (default: qwen)")
    parser.add_argument("--model-path", type=str, default="",
                        help="Local model path (auto-detected if empty)")
    parser.add_argument("--modes", nargs="+",
                        choices=["serial", "static", "continuous"],
                        default=["serial", "continuous"],
                        help="Benchmark modes to run (default: serial continuous)")
    parser.add_argument("--concurrency", nargs="+", type=int,
                        default=[1, 2, 4, 8],
                        help="Concurrency levels (default: 1 2 4 8)")
    parser.add_argument("--requests", type=int, default=20,
                        help="Number of requests per run (default: 20)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of repeats per config (default: 3)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Output directory (default: benchmark_results)")
    parser.add_argument("--num-gpu-blocks", type=int, default=None,
                        help="KV cache blocks (auto-calc if omitted)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run quick smoke test before full benchmark")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip GPU warmup before benchmark")
    # A/B experiment flags
    parser.add_argument("--ab-test", action="store_true",
                        help="Run Reference vs Triton attention backend A/B comparison")
    parser.add_argument("--attention-backend", type=str, default=None,
                        choices=["reference", "triton"],
                        help="Single attention backend mode (override for manual runs)")
    parser.add_argument("--homogeneous-ctx", type=int, default=None,
                        help="All requests have this exact context token length")
    parser.add_argument("--homogeneous-out", type=int, default=None,
                        help="All requests have this exact output token length")
    parser.add_argument("--ragged-ctx", action="store_true",
                        help="Wide uniform distribution of context lengths (32-1024)")
    parser.add_argument("--ragged-out", action="store_true",
                        help="Wide uniform distribution of output lengths (8-128)")
    return parser




def _auto_model_path():
    """Auto-detect local model path based on platform."""
    candidates = [
        "/mnt/e/vllm_awq_qwen_exp/models/Qwen2.5-0.5B-Instruct",
        "E:\\vllm_awq_qwen_exp\\models\\Qwen2.5-0.5B-Instruct",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return ""


def _calc_gpu_blocks(model_path: str, block_size: int = 16) -> int:
    """Calculate a safe number of KV cache blocks from available GPU memory."""
    try:
        import torch
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        # Estimate KV block size: 2 (K+V) * 2 bytes * num_layers * kv_heads * head_dim * block_size
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
        num_layers = cfg.num_hidden_layers
        kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        bytes_per_token = 2 * 2 * num_layers * kv_heads * head_dim  # float16
        bytes_per_block = bytes_per_token * block_size
        # Use 70% of free memory for KV cache
        usable = int(free_bytes * 0.70)
        num_blocks = usable // bytes_per_block
        return max(128, min(num_blocks, 16384))
    except Exception:
        return 4096


def main():
    parser = build_parser()
    args = parser.parse_args()

    # A/B experiment dispatch
    if args.ab_test:
        _run_ab_experiment(args)
        return

    # Single backend mode for manual testing
    if args.attention_backend:
        print(f"\n  Using paged executor with attention backend: {args.attention_backend}\n")
        args.executor = "paged"

    if args.executor == "fake":
        print("\n  Using fake executor (no GPU needed, simulation only)")
        print("  Results are NOT real inference metrics!\n")
    else:
        print("\n  Using Qwen executor with GPU")

    # Resolve model path
    model_path = args.model_path or _auto_model_path()
    if args.executor == "qwen" and not model_path:
        print("ERROR: No model path specified and auto-detection failed.")
        print("Use --model-path to specify the local model directory.")
        sys.exit(1)
    if args.executor == "qwen":
        if not os.path.isdir(model_path):
            print(f"ERROR: Model path does not exist: {model_path}")
            sys.exit(1)
        print(f"  Model path: {model_path}")

    # Calculate GPU blocks
    num_gpu_blocks = args.num_gpu_blocks
    if num_gpu_blocks is None and args.executor == "qwen":
        num_gpu_blocks = _calc_gpu_blocks(model_path)
    elif num_gpu_blocks is None:
        num_gpu_blocks = 512
    print(f"  KV cache blocks: {num_gpu_blocks}")

    # Create output dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Collect environment metadata
    metadata = _compute_metadata()
    metadata["model_path"] = model_path
    metadata["dtype"] = "float16"
    metadata["num_gpu_blocks"] = num_gpu_blocks
    metadata["seed"] = 42
    metadata["args"] = vars(args)

    # Smoke test
    if args.smoke_test:
        # For smoke test, use a smaller model if available or the main one
        smoke_path = model_path
        smoke_blocks = min(num_gpu_blocks, 1024) if args.executor == "qwen" else 128
        ok = run_smoke_test(smoke_path, smoke_blocks, output_dir, args.executor)
        if not ok:
            print("SMOKE TEST FAILED. Aborting.")
            sys.exit(1)

    # Generate requests
    print("\n" + "=" * 60)
    print("GENERATING REQUESTS")
    print("=" * 60)
    print()
    from transformers import AutoTokenizer
    if args.executor == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    else:
        # Use a minimal tokenizer for fake executor
        tokenizer = type('obj', (object,), {'encode': lambda self, s, **kw: list(range(min(100, len(s))))})()

    requests_data = _generate_request_prompts(
        tokenizer, args.requests, seed=42
    )
    print(f"  Generated {len(requests_data)} requests")
    avg_in = sum(r['actual_input_tokens'] for r in requests_data) / len(requests_data)
    avg_out = sum(r['output_length'] for r in requests_data) / len(requests_data)
    print(f"  Avg input tokens: {avg_in:.0f}")
    print(f"  Avg output tokens: {avg_out:.0f}")
    print(f"  Total prompt tokens: {sum(r['actual_input_tokens'] for r in requests_data)}")
    print(f"  Total output tokens: {sum(r['output_length'] for r in requests_data)}")
    print()

    # Collect all results
    all_reports = []
    concurrency_levels = sorted(args.concurrency)

    for concurrency in concurrency_levels:
        # Warmup
        if not args.skip_warmup and args.executor == "qwen":
            print(f"--- Warmup (concurrency={concurrency}) ---")
            warmup_data = requests_data[:min(4, len(requests_data))]
            run_warmup(concurrency, warmup_data, model_path, num_gpu_blocks)

        for mode in args.modes:
            print(f"--- Mode: {mode}, Concurrency: {concurrency} ---")
            for rep in range(args.repeats):
                print(f"  Repeat {rep + 1}/{args.repeats}...", end=" ", flush=True)
                result = run_benchmark_mode(
                    mode=mode, concurrency=concurrency,
                    requests_data=requests_data,
                    output_dir=output_dir,
                    model_path=model_path,
                    num_gpu_blocks=num_gpu_blocks,
                    repeat=rep,
                )
                if "error" in result:
                    print(f"ERROR: {result['error']}")
                else:
                    result["mode"] = mode
                    result["concurrency"] = concurrency
                    result["repeat"] = rep
                    all_reports.append(result)
                    print(f"{result['total_requests']} req, "
                          f"{result['throughput_req_per_sec']:.2f} req/s, "
                          f"TTFT P50={result.get('p50_ttft_ms', 0):.1f}ms")
            print()

    # Save all results
    print("=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    print()
    _save_results(all_reports, metadata, output_dir)

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Mode':<12} {'Conc':<6} {'Req/s':<10} {'Tok/s':<12} {'TTFT50':<10} {'TTFT95':<10} {'TPOT50':<10} {'TPOT95':<10} {'Batch':<8}")
    print("-" * 88)
    by_mode_conc = {}
    for r in all_reports:
        key = (r['mode'], r['concurrency'])
        if key not in by_mode_conc:
            by_mode_conc[key] = []
        by_mode_conc[key].append(r)
    for (mode, conc), reports in sorted(by_mode_conc.items()):
        avg_req = sum(r['throughput_req_per_sec'] for r in reports) / len(reports)
        avg_tok = sum(r['throughput_tok_per_sec'] for r in reports) / len(reports)
        avg_ttft50 = sum(r.get('p50_ttft_ms', 0) for r in reports) / len(reports)
        avg_ttft95 = sum(r.get('p95_ttft_ms', 0) for r in reports) / len(reports)
        avg_tpot50 = sum(r.get('p50_tpot_ms', 0) for r in reports) / len(reports)
        avg_tpot95 = sum(r.get('p95_tpot_ms', 0) for r in reports) / len(reports)
        avg_batch = sum(r.get('mean_effective_batch_size', 0) for r in reports) / len(reports)
        print(f"{mode:<12} {conc:<6} {avg_req:<10.2f} {avg_tok:<12.1f} {avg_ttft50:<10.1f} {avg_ttft95:<10.1f} {avg_tpot50:<10.2f} {avg_tpot95:<10.2f} {avg_batch:<8.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
