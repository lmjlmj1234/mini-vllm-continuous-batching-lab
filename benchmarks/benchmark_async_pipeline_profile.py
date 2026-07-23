#!/usr/bin/env python3
"""Async Pipeline Profiling Script for mini-vLLM.

Measures fine-grained per-decode-step stage timing to assess async
scheduling and pipeline overlap potential.

Usage:
    python benchmarks/benchmark_async_pipeline_profile.py

Output:
    benchmark_results/async_pipeline_profile.json
    benchmark_results/async_pipeline_profile.csv
    Console summary with P50/P95/P99 stage latencies.
"""

import argparse
import csv
import json
import os
import sys
import time as time_module
from typing import Dict, List, Optional

import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mini_vllm import Config, LLMEngine
from mini_vllm.sequence.status import Status


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SHORT_PROMPT = "Hello, world!"
MEDIUM_PROMPT = "What is the capital of France? Explain your reasoning."
LONG_PROMPT = (
    "Machine learning is a subset of artificial intelligence that involves "
    "the use of statistical techniques to enable machines to improve with experience. "
    "Deep learning is a further subset that uses neural networks with many layers. "
    "Transformer models have revolutionized natural language processing "
    "by introducing the self-attention mechanism."
)


# ---------------------------------------------------------------------------
# Fine-grained profiler wrapper
# ---------------------------------------------------------------------------
class StepProfiler:
    """Records per-step timing for individual engine phases.

    Uses CUDA events for GPU time and high-precision monotonic clock for CPU.
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._records: Dict[str, List[float]] = {}
        self._stack: Dict[str, float] = {}
        self._cuda_stack: Dict[str, List] = {}
        self._step_count = 0

    def reset(self):
        self._records.clear()
        self._stack.clear()
        self._cuda_stack.clear()
        self._step_count = 0

    def start_cpu(self, stage: str):
        self._stack[stage] = time_module.perf_counter()

    def end_cpu(self, stage: str):
        start = self._stack.pop(stage, None)
        if start is not None:
            self._records.setdefault(stage, []).append(
                (time_module.perf_counter() - start) * 1000  # ms
            )

    def start_cuda(self, stage: str):
        """Start a GPU-timed region using CUDA events."""
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        self._cuda_stack[stage] = [ev_start]

    def end_cuda(self, stage: str):
        evs = self._cuda_stack.pop(stage, None)
        if evs is not None:
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_end.record()
            torch.cuda.synchronize()
            elapsed_ms = evs[0].elapsed_time(ev_end)
            self._records.setdefault(stage, []).append(elapsed_ms)

    def record_cuda_gap(self, stage: str, ev_start, ev_end, add_sync=True):
        """Record time between two pre-existing CUDA events."""
        if add_sync:
            torch.cuda.synchronize()
        elapsed_ms = ev_start.elapsed_time(ev_end)
        self._records.setdefault(stage, []).append(elapsed_ms)

    @property
    def stage_names(self) -> List[str]:
        return sorted(self._records.keys())

    def summary(self) -> Dict[str, Dict]:
        result = {}
        for stage, times in self._records.items():
            if not times:
                continue
            sorted_t = sorted(times)
            n = len(sorted_t)
            result[stage] = {
                "count": n,
                "total_ms": round(sum(times), 3),
                "avg_ms": round(sum(times) / n, 4),
                "min_ms": round(min(times), 4),
                "max_ms": round(max(times), 4),
                "p50_ms": round(sorted_t[n // 2], 4),
                "p95_ms": round(sorted_t[int(n * 0.95)], 4),
                "p99_ms": round(sorted_t[int(n * 0.99)], 4),
            }
        return result


# ---------------------------------------------------------------------------
# Profiling scenarios
# ---------------------------------------------------------------------------

SCENARIOS = []


def register_scenario(name, requests, prompt, max_tokens,
                      max_num_seqs, max_num_batched_tokens,
                      description=""):
    SCENARIOS.append({
        "name": name,
        "requests": requests,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "description": description,
    })


# Scenario 1: Single request decode
register_scenario(
    "decode_1r_short", 1, SHORT_PROMPT, 64,
    4, 64, "1 request, pure decode"
)
register_scenario(
    "decode_1r_long", 1, LONG_PROMPT, 128,
    4, 256, "1 request, long prompt+decode"
)

# Scenario 2: Multiple concurrent requests
register_scenario(
    "decode_2r_short", 2, SHORT_PROMPT, 64,
    4, 128, "2 concurrent requests"
)
register_scenario(
    "decode_4r_short", 4, SHORT_PROMPT, 64,
    4, 256, "4 concurrent requests"
)
register_scenario(
    "decode_8r_short", 8, SHORT_PROMPT, 64,
    8, 512, "8 concurrent requests"
)

# Scenario 3: Mixed prefill + decode
register_scenario(
    "mixed_4r_staggered", 4, MEDIUM_PROMPT, 64,
    4, 256, "4 requests with staggered arrivals"
)

# Scenario 4: Long context
register_scenario(
    "longctx_1r_512", 1, SHORT_PROMPT, 512,
    4, 256, "1 request, 512 decode tokens"
)

# Scenario 5: Request completion/cancel stress
register_scenario(
    "completion_8r", 8, SHORT_PROMPT, 32,
    8, 512, "8 requests completing at different times"
)


# ---------------------------------------------------------------------------
# Profiling harness
# ---------------------------------------------------------------------------

def profile_scenario(scenario: dict, quiet: bool = True) -> dict:
    """Run one profiling scenario and return results."""
    device = torch.device("cuda")

    config = Config(
        executor_type="paged",
        model_path="/mnt/e/vllm_awq_qwen_exp/models/Qwen2.5-0.5B-Instruct",
        max_num_seqs=scenario["max_num_seqs"],
        max_num_batched_tokens=scenario["max_num_batched_tokens"],
        max_num_prefill_tokens=scenario["max_num_batched_tokens"],
        max_prefill_chunk_size=scenario["max_num_batched_tokens"],
        block_size=16,
        chunked_prefill_enabled=True,
        decode_first=True,
        print_step_events=not quiet,
        memory_trace=False,
        attention_backend="triton",
        peak_runtime_estimate=0,
        request_timeout_s=300,
    )

    engine = LLMEngine(config)
    engine_core = engine.engine_core
    profiler = StepProfiler(device)

    # Reset existing profiler
    engine_core._profiler.reset()

    # Build a custom EngineCore.step wrapper for fine-grained timing
    original_step = engine_core.step

    def timed_step():
        """Wrapped engine_core.step with per-stage CUDA + CPU timing."""
        nonlocal profiler

        # 1. Scheduler (CPU only)
        profiler.start_cpu("scheduler")
        result = engine_core._scheduler.schedule()
        profiler.end_cpu("scheduler")

        # Record waiting time for newly admitted sequences
        for sg in result.scheduled_prefill_groups:
            for seq in sg.get_unfinished_seqs():
                if seq.first_scheduled_time is not None:
                    continue
                seq.first_scheduled_time = time_module.time()
                waiting_s = seq.first_scheduled_time - seq.arrival_time
                engine_core._profiler.record_raw(
                    "request_queue_waiting", waiting_s
                )
                engine_core._profiler.increment_requests()

        # 2. Separate prefill and decode seqs
        only_prefill_seqs = []
        for sg in result.scheduled_prefill_groups:
            for seq in sg.get_unfinished_seqs():
                if seq.status == Status.PREFILL:
                    only_prefill_seqs.append(seq)

        decode_seqs = []
        for sg in result.scheduled_decode_groups:
            decode_seqs.extend(sg.get_unfinished_seqs())

        has_prefill = bool(only_prefill_seqs)
        has_decode = bool(decode_seqs)

        if has_prefill or has_decode:
            # 3. KV block ensure (CPU)
            profiler.start_cpu("kv_ensure_blocks")
            bm = engine_core._block_manager
            if has_prefill and bm is not None:
                for seq in only_prefill_seqs:
                    chunk_end = min(
                        seq.prefill_cursor + engine_core._config.max_prefill_chunk_size,
                        len(seq.prompt_token_ids),
                    )
                    for pos in range(seq.prefill_cursor, chunk_end):
                        bm.ensure_block(seq, pos)
            if has_decode and bm is not None:
                for seq in decode_seqs:
                    pos = len(seq.prompt_token_ids) + seq.num_generated_tokens - 1
                    bm.ensure_block(seq, pos)
            profiler.end_cpu("kv_ensure_blocks")

            # CUDA sync before timing GPU
            torch.cuda.synchronize()

            # 4. Model input building (CPU + H2D)
            profiler.start_cpu("model_input_build")
            model_input = engine_core._input_builder.build(
                prefill_seqs=only_prefill_seqs,
                decode_seqs=decode_seqs,
            )
            profiler.end_cpu("model_input_build")

            torch.cuda.synchronize()

            # 5. Executor forward (GPU) with CUDA events
            # Record the CUDA event before executor starts
            profiler.start_cpu("executor_forward_cpu_wall")
            ev_fwd_start = torch.cuda.Event(enable_timing=True)
            ev_fwd_start.record()

            model_output = engine_core._executor.execute(model_input)

            ev_fwd_end = torch.cuda.Event(enable_timing=True)
            ev_fwd_end.record()
            torch.cuda.synchronize()

            # GPU time for executor_forward
            gpu_ms = ev_fwd_start.elapsed_time(ev_fwd_end)
            profiler._records.setdefault("executor_forward_gpu", []).append(gpu_ms)
            profiler.end_cpu("executor_forward_cpu_wall")

            # 6. Apply model output (CPU: sampled tokens → sequence state)
            profiler.start_cpu("apply_output")
            engine_core._apply_model_output(
                model_output, only_prefill_seqs, decode_seqs,
            )
            # Advance prefill cursor
            chunk_size = engine_core._config.max_prefill_chunk_size
            for seq in only_prefill_seqs:
                end = min(
                    seq.prefill_cursor + chunk_size,
                    len(seq.prompt_token_ids),
                )
                seq.prefill_cursor = end
            profiler.end_cpu("apply_output")

        # 7. Cleanup finished sequences
        profiler.start_cpu("cleanup_finished")
        for sg in result.finished_groups:
            for seq in sg.seqs:
                engine_core._executor.cleanup_sequence(seq.seq_id)
                engine_core._metrics.register_sequence(seq)
        profiler.end_cpu("cleanup_finished")

        # 8. Metrics update
        profiler.start_cpu("metrics_update")
        bm_stats = engine_core._scheduler.block_manager_stats()
        total_blocks = bm_stats["total_blocks"]
        used_blocks = bm_stats["used_blocks"]
        effective_batch_size = (
            len(result.scheduled_decode_groups)
            + len(result.scheduled_prefill_groups)
        )
        running_count = engine_core._scheduler._queue.num_running
        waiting_count = engine_core._scheduler._queue.num_waiting
        engine_core._metrics.record_step(
            result,
            time_module.time() - time_module.time(),
            time_module.time() - time_module.time(),
            total_blocks, used_blocks,
            effective_batch_size=effective_batch_size,
            running_count=running_count,
            waiting_count=waiting_count,
        )
        profiler.end_cpu("metrics_update")

        engine_core._step_count += 1

        return result

    # Hook our timed_step
    engine_core.step = timed_step  # type: ignore[assignment]

    # --- Add requests ---
    num_requests = scenario["requests"]
    prompt = scenario["prompt"]
    max_tokens = scenario["max_tokens"]

    for i in range(num_requests):
        engine.add_request(prompt, max_new_tokens=max_tokens)

    # --- Run to completion ---
    step_times = []
    while engine.queue.num_waiting > 0 or engine.queue.num_running > 0:
        t0 = time_module.perf_counter()
        engine.step()
        dt = (time_module.perf_counter() - t0) * 1000
        step_times.append(dt)

    # Restore original step
    engine_core.step = original_step

    # --- Collect results ---
    stage_summary = profiler.summary()
    metrics_report = engine_core.metrics_collector.report()

    # Calculate CPU gap between GPU executions
    gpu_times = profiler._records.get("executor_forward_gpu", [])
    gpu_gaps = []
    if len(gpu_times) > 1:
        for i in range(len(gpu_times) - 1):
            # Gap = next step GPU start - this step GPU end
            # This is captured via the CPU total step time minus GPU time
            pass

    # Gap analysis: CPU wall time around GPU execution
    cpu_wall_times = profiler._records.get("executor_forward_cpu_wall", [])
    gap_analysis = {}
    if gpu_times and cpu_wall_times:
        cpu_overheads = []
        gpu_ratios = []
        for g, c in zip(gpu_times, cpu_wall_times):
            cpu_overheads.append(c - g)
            gpu_ratios.append(g / c if c > 0 else 0)
        gap_analysis = {
            "cpu_overhead_avg_ms": round(sum(cpu_overheads) / len(cpu_overheads), 4)
                if cpu_overheads else 0,
            "cpu_overhead_p50_ms": round(sorted(cpu_overheads)[len(cpu_overheads)//2], 4)
                if cpu_overheads else 0,
            "cpu_overhead_p95_ms": round(
                sorted(cpu_overheads)[int(len(cpu_overheads)*0.95)], 4
            ) if cpu_overheads else 0,
            "gpu_ratio_of_step_avg_pct": round(
                sum(gpu_ratios) / len(gpu_ratios) * 100, 1
            ) if gpu_ratios else 0,
        }

    # Full decode step time analysis (CPU gap between successive GPU execs)
    # Each step total = scheduler + kv_ensure + model_input + forward(cpu_wall) + apply_output + cleanup + metrics
    scheduler_times = profiler._records.get("scheduler", [])
    kv_times = profiler._records.get("kv_ensure_blocks", [])
    input_times = profiler._records.get("model_input_build", [])
    forward_cpu_times = profiler._records.get("executor_forward_cpu_wall", [])
    apply_times = profiler._records.get("apply_output", [])
    cleanup_times = profiler._records.get("cleanup_finished", [])
    metrics_times = profiler._records.get("metrics_update", [])

    # Per-step gap: step.total_time - step.gpu_time
    # Approximate by matching indices
    per_step_gaps = []
    max_len = min(
        len(scheduler_times), len(gpu_times),
        len(forward_cpu_times)
    )
    for i in range(max_len):
        # Total non-GPU time for this step
        sched = scheduler_times[i] if i < len(scheduler_times) else 0
        kv = kv_times[i] if i < len(kv_times) else 0
        inp = input_times[i] if i < len(input_times) else 0
        fwd_cpu = forward_cpu_times[i] if i < len(forward_cpu_times) else 0
        apply_t = apply_times[i] if i < len(apply_times) else 0
        clean = cleanup_times[i] if i < len(cleanup_times) else 0
        metr = metrics_times[i] if i < len(metrics_times) else 0
        gpu = gpu_times[i] if i < len(gpu_times) else 0

        step_total = sched + kv + inp + fwd_cpu + apply_t + clean + metr
        cpu_gap = step_total - gpu
        per_step_gaps.append({
            "step": i,
            "step_total_ms": round(step_total, 4),
            "gpu_ms": round(gpu, 4),
            "cpu_gap_ms": round(cpu_gap, 4),
            "scheduler_ms": round(sched, 4),
            "kv_ensure_ms": round(kv, 4),
            "input_build_ms": round(inp, 4),
            "forward_cpu_overhead_ms": round(fwd_cpu - gpu, 4) if fwd_cpu >= gpu else 0,
            "apply_output_ms": round(apply_t, 4),
            "cleanup_ms": round(clean, 4),
            "metrics_ms": round(metr, 4),
        })

    # Summary stats on gaps
    gaps_ms = [g["cpu_gap_ms"] for g in per_step_gaps]
    step_total_ms = [g["step_total_ms"] for g in per_step_gaps]
    sorted_gaps = sorted(gaps_ms)
    sorted_steps = sorted(step_total_ms)

    # Theoretical upper bound on async scheduling benefit
    # If we can fully hide scheduler + input_build + kv_ensure:
    #   hidden = scheduler + kv_ensure + model_input_build (pre-decode steps)
    # But scheduler includes finish checks that depend on GPU output...
    # We'll do the detailed classification in the report script.
    scheduler_total = sum(scheduler_times) if scheduler_times else 0
    kv_total = sum(kv_times) if kv_times else 0
    input_total = sum(input_times) if input_times else 0
    gpu_total = sum(gpu_times) if gpu_times else 0
    apply_total = sum(apply_times) if apply_times else 0

    return {
        "scenario": scenario["name"],
        "description": scenario.get("description", ""),
        "config": {
            "requests": num_requests,
            "max_tokens": max_tokens,
            "prompt_len": len(engine.executor.tokenize(prompt)),
            "max_num_seqs": scenario["max_num_seqs"],
            "max_num_batched_tokens": scenario["max_num_batched_tokens"],
        },
        "metrics_report": metrics_report,
        "stage_summary": stage_summary,
        "gap_analysis": gap_analysis,
        "per_step_gaps": per_step_gaps,
        "summary": {
            "total_decode_steps": len(gpu_times),
            "avg_step_total_ms": round(sum(step_total_ms) / len(step_total_ms), 4)
                if step_total_ms else 0,
            "avg_gpu_ms": round(sum(gpu_times) / len(gpu_times), 4)
                if gpu_times else 0,
            "avg_cpu_gap_ms": round(sum(gaps_ms) / len(gaps_ms), 4)
                if gaps_ms else 0,
            "min_cpu_gap_ms": round(min(gaps_ms), 4) if gaps_ms else 0,
            "max_cpu_gap_ms": round(max(gaps_ms), 4) if gaps_ms else 0,
            "p50_cpu_gap_ms": round(
                sorted_gaps[len(sorted_gaps)//2], 4
            ) if gaps_ms else 0,
            "p95_cpu_gap_ms": round(
                sorted_gaps[int(len(sorted_gaps)*0.95)], 4
            ) if gaps_ms else 0,
            "gpu_pct_of_step": round(
                sum(gpu_times) / sum(step_total_ms) * 100, 1
            ) if step_total_ms else 0,
            "scheduler_pct_of_step": round(
                scheduler_total / max(sum(step_total_ms), 1) * 100, 1
            ),
            "kv_ensure_pct": round(
                kv_total / max(sum(step_total_ms), 1) * 100, 1
            ),
            "input_build_pct": round(
                input_total / max(sum(step_total_ms), 1) * 100, 1
            ),
            "theoretical_hideable_ms": round(
                scheduler_total + input_total + kv_total, 4
            ),
            "theoretical_speedup_pct": round(
                (sum(step_total_ms) / max(
                    sum(step_total_ms) - (scheduler_total + input_total + kv_total), 1
                ) - 1) * 100, 1
            ) if step_total_ms else 0,
            "realistic_hideable_ms": round(
                # Only scheduler_reorder + input_build_pre_gpu can be hidden;
                # finish checks, apply_output cannot; kv_ensure partially
                # Let's be conservative: 50% of scheduler + 70% of input_build
                scheduler_total * 0.5 + input_total * 0.7 + kv_total * 0.3, 4
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Async Pipeline Profiling for mini-vLLM"
    )
    parser.add_argument(
        "--scenarios", type=str, default=None,
        help="Comma-separated scenario names to run (default: all)"
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Warmup steps (default: 2)"
    )
    parser.add_argument(
        "--quiet", action="store_true", default=True,
        help="Suppress step output"
    )
    args = parser.parse_args()

    # Select scenarios
    selected = SCENARIOS
    if args.scenarios:
        names = set(s.strip() for s in args.scenarios.split(","))
        selected = [s for s in SCENARIOS if s["name"] in names]

    if not selected:
        print("No scenarios selected.")
        return

    # Filter out scenarios that are too large for a quick run
    all_results = []
    run_scenarios = [
        s for s in selected
        if s["max_tokens"] <= 128  # Keep runs reasonable
    ]
    # Always include the most informative scenarios
    priority_names = [
        "decode_1r_short", "decode_4r_short", "decode_8r_short",
        "mixed_4r_staggered",
    ]
    priority = [s for s in run_scenarios if s["name"] in priority_names]
    rest = [s for s in run_scenarios if s["name"] not in priority_names]
    ordered = priority + rest

    print("=" * 70)
    print("mini-vLLM Async Pipeline Profiling")
    print("=" * 70)
    print(f"  GPU:           {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  CUDA:          {torch.version.cuda}")
    import triton
    print(f"  Triton:        {triton.__version__}")
    print(f"  Git commit:    6c57ba8")
    print(f"  Scenarios:     {len(ordered)} ({', '.join(s['name'] for s in ordered)})")
    print()

    for scenario in ordered:
        print(f"\n{'─' * 60}")
        print(f"Scenario: {scenario['name']}")
        print(f"  {scenario.get('description', '')}")
        print(f"  requests={scenario['requests']}, prompt_len=~{len(scenario['prompt'])}, "
              f"max_tokens={scenario['max_tokens']}")
        print(f"  max_num_seqs={scenario['max_num_seqs']}, "
              f"budget={scenario['max_num_batched_tokens']}")

        try:
            result = profile_scenario(scenario, quiet=args.quiet)
            all_results.append(result)

            s = result["summary"]
            g = result["gap_analysis"]
            print(f"\n  === Timing (ms) ===")
            print(f"  Avg step total:        {s['avg_step_total_ms']:.3f}")
            print(f"  Avg GPU time:          {s['avg_gpu_ms']:.3f}")
            print(f"  Avg CPU gap:           {s['avg_cpu_gap_ms']:.3f}")
            print(f"  P50/P95 CPU gap:       {s['p50_cpu_gap_ms']:.3f} / {s['p95_cpu_gap_ms']:.3f}")
            print(f"  GPU % of step:         {s['gpu_pct_of_step']:.1f}%")
            print(f"  Scheduler %:           {s['scheduler_pct_of_step']:.1f}%")
            print(f"  Input build %:         {s['input_build_pct']:.1f}%")
            print(f"  KV ensure %:           {s['kv_ensure_pct']:.1f}%")
            print(f"\n  === Async Scheduling Potential ===")
            print(f"  Theoretical hideable:  {s['theoretical_hideable_ms']:.3f} ms  "
                  f"(speedup: {s['theoretical_speedup_pct']:.1f}%)")
            print(f"  Realistic hideable:    {s['realistic_hideable_ms']:.3f} ms")
            print(f"  CPU overhead on fwd:   avg={g.get('cpu_overhead_avg_ms', 0):.3f}  "
                  f"P95={g.get('cpu_overhead_p95_ms', 0):.3f}")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # Save results
    output_dir = os.path.join(_project_root, "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "async_pipeline_profile.json")
    with open(json_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": time_module.strftime("%Y-%m-%dT%H:%M:%S"),
                "git_commit": "6c57ba8",
                "gpu": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
                "python_version": sys.version.split()[0],
                "model": "Qwen2.5-0.5B-Instruct",
            },
            "scenarios": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # CSV summary
    csv_path = os.path.join(output_dir, "async_pipeline_profile.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "requests", "max_tokens",
            "decode_steps", "avg_step_ms", "avg_gpu_ms",
            "avg_cpu_gap_ms", "p50_cpu_gap_ms", "p95_cpu_gap_ms",
            "gpu_pct", "scheduler_pct", "input_build_pct", "kv_ensure_pct",
            "theoretical_hideable_ms", "realistic_hideable_ms",
        ])
        for r in all_results:
            s = r["summary"]
            c = r["config"]
            writer.writerow([
                r["scenario"], c["requests"], c["max_tokens"],
                s["total_decode_steps"],
                s["avg_step_total_ms"], s["avg_gpu_ms"],
                s["avg_cpu_gap_ms"], s["p50_cpu_gap_ms"], s["p95_cpu_gap_ms"],
                s["gpu_pct_of_step"], s["scheduler_pct_of_step"],
                s["input_build_pct"], s["kv_ensure_pct"],
                s["theoretical_hideable_ms"], s["realistic_hideable_ms"],
            ])
    print(f"Summary CSV saved to {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
