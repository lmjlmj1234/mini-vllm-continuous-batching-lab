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
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mini_vllm import Config, LLMEngine, Status


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
        requests.append({
            "prompt": prompt,
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
