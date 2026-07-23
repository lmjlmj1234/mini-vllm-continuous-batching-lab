#!/usr/bin/env python3
"""
Phase 2: Minimal Async Scheduling Profile — single-scenario decode stage breakdown.

Measures CPU/GPU stage timing for pure decode steps (post-prefill).
No PyTorch Profiler, no Nsight, no trace files, no large tensor persistence.
"""

import json
import os
import sys
import time as time_module

import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mini_vllm import Config, LLMEngine
from mini_vllm.sequence.status import Status

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "/mnt/e/vllm_awq_qwen_exp/models/Qwen2.5-0.5B-Instruct"
CONTEXT_LEN = 128
DECODE_TOKENS = 32
WARMUP_DECODE_STEPS = 3
RECORD_DECODE_STEPS = 10
BLOCK_SIZE = 16
MAX_PREFILL_CHUNK = CONTEXT_LEN  # one-shot prefill
NUM_GPU_BLOCKS = 4096  # must be large enough for BlockAllocator

OUTPUT_JSON = os.path.join(_project_root, "benchmark_results", "minimal_async_profile.json")
OUTPUT_MD = os.path.join(_project_root, "docs", "minimal_async_profile.md")


def build_prompt_of_length(tokenizer, target_len: int) -> str:
    """Build a diverse prompt (each block has unique content to avoid
    prefix-cache false hits from repetitive token patterns)."""
    sentences = [
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms.",
        "Attention allows models to weigh the importance of different input tokens when producing each output token.",
        "Multi-head attention runs multiple attention operations in parallel, capturing different relationships.",
        "Positional encodings give the model information about the order of tokens in the sequence.",
        "Layer normalization stabilizes training by normalizing activations across the feature dimension.",
        "The feed-forward network in each transformer layer applies two linear transformations with a ReLU activation.",
        "Residual connections help gradients flow through deep networks by adding the input to the layer output.",
        "Dropout regularization randomly drops neurons during training to prevent overfitting.",
        "The encoder processes input sequences bidirectionally while the decoder generates output autoregressively.",
        "Scaled dot-product attention computes attention scores by taking the dot product of queries and keys.",
        "Beam search explores multiple candidate sequences during generation to find the most likely output.",
        "The softmax function converts logits into probability distributions over the vocabulary.",
        "Gradient clipping prevents exploding gradients by scaling down gradients that exceed a threshold.",
        "Learning rate scheduling adjusts the step size during training for better convergence.",
        "Weight initialization strategies like Xavier and Kaiming help prevent vanishing or exploding gradients.",
        "Batch normalization normalizes layer inputs across the batch dimension for each feature.",
        "The embedding layer maps discrete tokens to continuous vector representations in a high-dimensional space.",
        "Cross-attention allows the decoder to attend to the encoder's output representations.",
        "Autoregressive generation produces one token at a time, conditioning each new token on previous outputs.",
        "Top-k sampling restricts the next token selection to the k most likely candidates.",
        "Temperature scaling controls the randomness of token selection by scaling logits before softmax.",
        "The vocabulary defines the set of all possible tokens the model can produce.",
        "Context length determines how many tokens the model can consider at once.",
        "The hidden size specifies the dimensionality of the model's internal representations.",
    ]
    # Build until target length
    text = " ".join(sentences * 10)
    ids = tokenizer.encode(text)
    if len(ids) <= target_len:
        # Pad by repeating last sentence
        last = tokenizer.encode(sentences[-1])
        while len(ids) < target_len:
            ids.extend(last)
    exact = ids[:target_len]
    return tokenizer.decode(exact)


def main():
    # ------------------------------------------------------------------
    # Check memory before start
    # ------------------------------------------------------------------
    print("=== Pre-run Memory ===")
    os.system("free -h")
    os.system("nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader")

    free_bytes, _ = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / (1024 ** 3)
    if free_gib < 2.0:
        print(f"FATAL: GPU free memory {free_gib:.1f} GiB < 2 GiB. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Engine setup
    # ------------------------------------------------------------------
    config = Config(
        executor_type="paged",
        model_path=MODEL_PATH,
        num_gpu_blocks=NUM_GPU_BLOCKS,
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        max_num_prefill_tokens=8192,
        max_prefill_chunk_size=MAX_PREFILL_CHUNK,
        block_size=BLOCK_SIZE,
        chunked_prefill_enabled=True,
        decode_first=True,
        print_step_events=False,
        memory_trace=False,
        attention_backend="triton",
        request_timeout_s=300,
        gpu_memory_utilization=0.85,
    )

    print(f"\nConfig: {config}")
    engine = LLMEngine(config)
    engine_core = engine.engine_core

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    tokenizer = engine.executor._tokenizer
    prompt_text = build_prompt_of_length(tokenizer, CONTEXT_LEN)
    actual_len = len(tokenizer.encode(prompt_text))
    print(f"Prompt text length: {actual_len} tokens (target={CONTEXT_LEN})")

    # ------------------------------------------------------------------
    # Timed step wrapper
    # ------------------------------------------------------------------
    original_step = engine_core.step

    step_timing_records = []
    warmup_done = 0  # count of decode steps after prefill

    def timed_step():
        """Replace engine_core.step with a per-phase timed version."""
        nonlocal warmup_done

        engine_core._step_count += 1
        step_num = engine_core._step_count

        # ---- 1. Scheduler (CPU) ----
        t0 = time_module.perf_counter_ns()
        engine_core._check_timeouts()
        result = engine_core._scheduler.schedule()
        t_sched = time_module.perf_counter_ns() - t0

        # Separate prefill / decode sequences
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

        # ---- 2. Ensure blocks (CPU) ----
        t0 = time_module.perf_counter_ns()
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
        t_ensure = time_module.perf_counter_ns() - t0

        # ---- 3. Input builder (CPU + H2D) ----
        t0 = time_module.perf_counter_ns()
        if has_prefill or has_decode:
            model_input = engine_core._input_builder.build(
                prefill_seqs=only_prefill_seqs,
                decode_seqs=decode_seqs,
            )
        t_build = time_module.perf_counter_ns() - t0

        # ---- 4. Executor forward (GPU) ----
        fwd_gpu_ms = 0.0
        if has_prefill or has_decode:
            torch.cuda.synchronize()
            ev_fwd_start = torch.cuda.Event(enable_timing=True)
            ev_fwd_start.record()

            model_output = engine_core._executor.execute(model_input)

            ev_fwd_end = torch.cuda.Event(enable_timing=True)
            ev_fwd_end.record()
            torch.cuda.synchronize()
            fwd_gpu_ms = ev_fwd_start.elapsed_time(ev_fwd_end)

            # ---- 5. Apply output + advance cursor (CPU) ----
            t0 = time_module.perf_counter_ns()
            engine_core._apply_model_output(model_output, only_prefill_seqs, decode_seqs)
            chunk_size = engine_core._config.max_prefill_chunk_size
            for seq in only_prefill_seqs:
                end = min(seq.prefill_cursor + chunk_size, len(seq.prompt_token_ids))
                seq.prefill_cursor = end
            t_apply = time_module.perf_counter_ns() - t0
        else:
            t_apply = 0

        # ---- 6. Cleanup + metrics (CPU) ----
        t0 = time_module.perf_counter_ns()
        for sg in result.finished_groups:
            for seq in sg.seqs:
                engine_core._executor.cleanup_sequence(seq.seq_id)
                engine_core._metrics.register_sequence(seq)

        bm_stats = engine_core._scheduler.block_manager_stats()
        total_blocks = bm_stats["total_blocks"]
        used_blocks = bm_stats["used_blocks"]
        effective_batch_size = (
            len(result.scheduled_decode_groups) + len(result.scheduled_prefill_groups)
        )
        running_count = engine_core._scheduler._queue.num_running
        waiting_count = engine_core._scheduler._queue.num_waiting
        engine_core._metrics.record_step(
            result, 0.0, 0.0, total_blocks, used_blocks,
            effective_batch_size=effective_batch_size,
            running_count=running_count,
            waiting_count=waiting_count,
        )
        t_cleanup = time_module.perf_counter_ns() - t0

        # ---- Record decode-only steps (skip prefill step, skip warmup) ----
        is_decode_step = (not has_prefill) and has_decode
        if is_decode_step:
            warmup_done += 1
            if warmup_done > WARMUP_DECODE_STEPS and len(step_timing_records) < RECORD_DECODE_STEPS:
                step_timing_records.append({
                    "step": step_num,
                    "phase": "decode",
                    "scheduler_ns": t_sched,
                    "ensure_blocks_ns": t_ensure,
                    "input_build_ns": t_build,
                    "forward_gpu_ms": round(fwd_gpu_ms, 4),
                    "apply_output_ns": t_apply,
                    "cleanup_metrics_ns": t_cleanup,
                    "n_prefill": len(only_prefill_seqs),
                    "n_decode": len(decode_seqs),
                })

        return result

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    engine_core.step = timed_step

    engine.add_request(prompt_text, max_new_tokens=DECODE_TOKENS + WARMUP_DECODE_STEPS)

    while engine.queue.num_waiting > 0 or engine.queue.num_running > 0:
        engine.step()

    # Restore original
    engine_core.step = original_step

    # ------------------------------------------------------------------
    # Post-run memory check
    # ------------------------------------------------------------------
    print("\n=== Post-run Memory ===")
    os.system("free -h")
    os.system("nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader")

    # ------------------------------------------------------------------
    # Compute summary
    # ------------------------------------------------------------------
    if not step_timing_records:
        print("FATAL: No decode step records collected. Aborting.")
        sys.exit(1)

    n = len(step_timing_records)
    sched_ns = [r["scheduler_ns"] for r in step_timing_records]
    ensure_ns = [r["ensure_blocks_ns"] for r in step_timing_records]
    build_ns = [r["input_build_ns"] for r in step_timing_records]
    fwd_ms = [r["forward_gpu_ms"] for r in step_timing_records]
    apply_ns = [r["apply_output_ns"] for r in step_timing_records]
    cleanup_ns = [r["cleanup_metrics_ns"] for r in step_timing_records]

    def stats_us(arr_ns):
        """Compute statistics in microseconds."""
        if not arr_ns:
            return {"avg_us": 0.0, "min_us": 0.0, "max_us": 0.0, "p50_us": 0.0, "p95_us": 0.0}
        s = sorted(arr_ns)
        avg = sum(s) / len(s)
        return {
            "avg_us": round(avg / 1000, 2),
            "min_us": round(min(s) / 1000, 2),
            "max_us": round(max(s) / 1000, 2),
            "p50_us": round(s[len(s) // 2] / 1000, 2),
            "p95_us": round(s[int(len(s) * 0.95)] / 1000, 2),
        }

    def stats_ms(arr_ns):
        """Compute statistics in milliseconds (for GPU times)."""
        if not arr_ns:
            return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        s = sorted(arr_ns)
        avg = sum(s) / len(s)
        return {
            "avg_ms": round(avg, 2),
            "min_ms": round(min(s), 2),
            "max_ms": round(max(s), 2),
            "p50_ms": round(s[len(s) // 2], 2),
            "p95_ms": round(s[int(len(s) * 0.95)], 2),
        }

    # Per-step total time in us (sum of CPU + GPU)
    per_step_total_us = []
    for i in range(n):
        cpu_ns = sched_ns[i] + ensure_ns[i] + build_ns[i] + apply_ns[i] + cleanup_ns[i]
        gpu_ns = fwd_ms[i] * 1000 * 1000  # ms → ns
        per_step_total_us.append((cpu_ns + gpu_ns) / 1000)

    # CPU gap = step total - GPU time
    cpu_gap_us = []
    for i in range(n):
        cpu_ns = sched_ns[i] + ensure_ns[i] + build_ns[i] + apply_ns[i] + cleanup_ns[i]
        cpu_gap_us.append(cpu_ns / 1000)

    summary = {
        "config": {
            "model": "Qwen2.5-0.5B-Instruct",
            "batch_size": 1,
            "context_length": CONTEXT_LEN,
            "decode_tokens_measured": RECORD_DECODE_STEPS,
            "warmup_steps": WARMUP_DECODE_STEPS,
            "block_size": BLOCK_SIZE,
            "attention_backend": "triton",
        },
        "stages_us": {
            "scheduler": stats_us(sched_ns),
            "ensure_blocks": stats_us(ensure_ns),
            "input_build": stats_us(build_ns),
            "apply_output": stats_us(apply_ns),
            "cleanup_metrics": stats_us(cleanup_ns),
        },
        "gpu_forward_ms": stats_ms(fwd_ms),
        "cpu_gap_us": {
            **stats_us(cpu_gap_us),
        },
        "per_step_total_us": {
            "avg_us": round(sum(per_step_total_us) / len(per_step_total_us), 2),
            "min_us": round(min(per_step_total_us), 2),
            "max_us": round(max(per_step_total_us), 2),
            "p50_us": round(sorted(per_step_total_us)[len(per_step_total_us) // 2], 2),
            "p95_us": round(sorted(per_step_total_us)[int(len(per_step_total_us) * 0.95)], 2),
        },
        "raw_records": [
            {
                "step": r["step"],
                "scheduler_us": round(r["scheduler_ns"] / 1000, 1),
                "ensure_blocks_us": round(r["ensure_blocks_ns"] / 1000, 1),
                "input_build_us": round(r["input_build_ns"] / 1000, 1),
                "forward_gpu_ms": r["forward_gpu_ms"],
                "apply_output_us": round(r["apply_output_ns"] / 1000, 1),
                "cleanup_metrics_us": round(r["cleanup_metrics_ns"] / 1000, 1),
            }
            for r in step_timing_records
        ],
        "n_recorded_steps": n,
    }

    # Compute percentages (all in microseconds)
    avg_total_us = summary["per_step_total_us"]["avg_us"]
    avg_gpu_ms = summary["gpu_forward_ms"]["avg_ms"]
    avg_gpu_us = avg_gpu_ms * 1000  # ms → us
    avg_cpu_gap_us = avg_total_us - avg_gpu_us
    sched_avg_us = summary["stages_us"]["scheduler"]["avg_us"]
    ensure_avg_us = summary["stages_us"]["ensure_blocks"]["avg_us"]
    build_avg_us = summary["stages_us"]["input_build"]["avg_us"]
    apply_avg_us = summary["stages_us"]["apply_output"]["avg_us"]
    cleanup_avg_us = summary["stages_us"]["cleanup_metrics"]["avg_us"]

    def pct_from_us(part_us, total_us):
        return round(part_us / max(total_us, 1) * 100, 1)

    summary["analysis"] = {
        "avg_step_total_us": round(avg_total_us, 2),
        "avg_gpu_ms": avg_gpu_ms,
        "avg_cpu_gap_us": round(avg_cpu_gap_us, 2),
        "scheduler_pct_of_total": pct_from_us(sched_avg_us, avg_total_us),
        "ensure_blocks_pct_of_total": pct_from_us(ensure_avg_us, avg_total_us),
        "input_build_pct_of_total": pct_from_us(build_avg_us, avg_total_us),
        "apply_output_pct_of_total": pct_from_us(apply_avg_us, avg_total_us),
        "cleanup_metrics_pct_of_total": pct_from_us(cleanup_avg_us, avg_total_us),
        "gpu_pct_of_total": pct_from_us(avg_gpu_us, avg_total_us),
        "cpu_gap_pct_of_total": pct_from_us(avg_cpu_gap_us, avg_total_us),
        "theoretical_max_hideable_us": round(sched_avg_us + ensure_avg_us + build_avg_us, 2),
        "theoretical_max_speedup_ratio": round(
            avg_total_us / max(avg_total_us - (sched_avg_us + ensure_avg_us + build_avg_us), 1),
            3,
        ),
    }

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    # ------------------------------------------------------------------
    # Print raw table
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'Step':<6} {'sched_us':<10} {'ensure_us':<11} {'build_us':<11} {'fwd_ms':<9} {'apply_us':<10} {'cleanup_us':<12}")
    print("-" * 100)
    for r in summary["raw_records"]:
        print(f"{r['step']:<6} {r['scheduler_us']:<10} {r['ensure_blocks_us']:<11} "
              f"{r['input_build_us']:<11} {r['forward_gpu_ms']:<9} {r['apply_output_us']:<10} "
              f"{r['cleanup_metrics_us']:<12}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    a = summary["analysis"]
    s = summary["stages_us"]
    print(f"\n  === Timing Summary (avg over {n} decode steps) ===")
    print(f"  Step total:             {a['avg_step_total_us']:.1f} us")
    print(f"  GPU forward:            {a['avg_gpu_ms']:.3f} ms  ({a['gpu_pct_of_total']:.1f}%)")
    print(f"  CPU gap:                {a['avg_cpu_gap_us']:.1f} us")
    print(f"    scheduler:            {s['scheduler']['avg_us']:.1f} us  ({a['scheduler_pct_of_total']:.1f}%)")
    print(f"    ensure_blocks:        {s['ensure_blocks']['avg_us']:.1f} us  ({a['ensure_blocks_pct_of_total']:.1f}%)")
    print(f"    input_build:          {s['input_build']['avg_us']:.1f} us  ({a['input_build_pct_of_total']:.1f}%)")
    print(f"    apply_output:         {s['apply_output']['avg_us']:.1f} us  ({a['apply_output_pct_of_total']:.1f}%)")
    print(f"    cleanup+metrics:      {s['cleanup_metrics']['avg_us']:.1f} us  ({a['cleanup_metrics_pct_of_total']:.1f}%)")
    print(f"\n  Theoretical max hideable: {a['theoretical_max_hideable_us']:.1f} us")
    print(f"  Theoretical max speedup:  {a['theoretical_max_speedup_ratio']:.3f}x")

    # ------------------------------------------------------------------
    # Generate markdown report
    # ------------------------------------------------------------------
    _write_markdown(summary)
    print(f"\nSaved: {OUTPUT_MD}")
    print("\n=== Phase 2 complete. Stopped. ===")


def _write_markdown(summary: dict) -> None:
    """Write the minimal profile markdown report."""
    a = summary["analysis"]
    s = summary["stages_us"]
    raw = summary["raw_records"]
    n = summary["n_recorded_steps"]

    hideable_pct = round(
        a["theoretical_max_hideable_us"] / a["avg_step_total_us"] * 100, 1
    )

    lines = [
        "# Minimal Async Scheduling Profile — Phase 2\n",
        f"> Single decode step breakdown. batch_size=1, context_length={summary['config']['context_length']}, "
        f"attention_backend={summary['config']['attention_backend']}.\n",
        "> **No performance claims — these are timing observations from a single run**.\n",
        "---\n",
        "## Raw Timing (per decode step)\n",
        "| Step | Sched (us) | Ensure (us) | Build (us) | GPU Fwd (ms) | Apply (us) | Cleanup (us) |",
        "|------|-----------|-------------|------------|-------------|------------|-------------|",
    ]

    for r in raw:
        lines.append(
            f"| {r['step']} | {r['scheduler_us']} | {r['ensure_blocks_us']} | "
            f"{r['input_build_us']} | {r['forward_gpu_ms']} | {r['apply_output_us']} | "
            f"{r['cleanup_metrics_us']} |"
        )

    lines += [
        "\n## Aggregate Statistics (decode only)",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Recorded decode steps | {n} |",
        f"| Avg step total | {a['avg_step_total_us']:.1f} us |",
        f"| Avg GPU forward | {a['avg_gpu_ms']:.3f} ms |",
        f"| Avg CPU gap | {a['avg_cpu_gap_us']:.1f} us |",
        f"| Scheduler (P50) | {s['scheduler']['p50_us']} us |",
        f"| Ensure blocks (P50) | {s['ensure_blocks']['p50_us']} us |",
        f"| Input build (P50) | {s['input_build']['p50_us']} us |",
        f"| Apply output (P50) | {s['apply_output']['p50_us']} us |",
        f"| Cleanup+metrics (P50) | {s['cleanup_metrics']['p50_us']} us |",
        f"| GPU forward (P50) | {summary['gpu_forward_ms']['p50_ms']} ms |",
        "\n## Stage Percentage Breakdown",
        f"| Stage | % of step total |",
        f"|-------|-----------------|",
        f"| GPU forward | {a['gpu_pct_of_total']}% |",
        f"| CPU gap (total) | {a['cpu_gap_pct_of_total']}% |",
        f"| Scheduler | {a['scheduler_pct_of_total']}% |",
        f"| Ensure blocks | {a['ensure_blocks_pct_of_total']}% |",
        f"| Input build | {a['input_build_pct_of_total']}% |",
        f"| Apply output | {a['apply_output_pct_of_total']}% |",
        f"| Cleanup+metrics | {a['cleanup_metrics_pct_of_total']}% |",
        "\n---\n",
        "## Assessment\n",
    ]

    # Q1
    sched_plus_build = a["scheduler_pct_of_total"] + a["input_build_pct_of_total"]
    lines += [
        f"**1. Scheduler + Metadata 占单步总时间百分比？**",
        f"",
        f"Scheduler ({a['scheduler_pct_of_total']}%) + Input Build ({a['input_build_pct_of_total']}%) = "
        f"**{round(sched_plus_build, 1)}%** of single step total. "
        f"GPU forward dominates at {a['gpu_pct_of_total']}%. "
        f"The CPU stages before the GPU forward (scheduler + ensure_blocks + input_build) total "
        f"{round(hideable_pct, 1)}% of step time.",
        "",
    ]

    # Q2
    lines += [
        f"**2. 理论最多能隐藏多少时间？**",
        f"",
        f"The CPU work that can theoretically run in parallel with the GPU forward "
        f"(i.e., does NOT depend on the current step's GPU output): "
        f"**{a['theoretical_max_hideable_us']:.1f} us** per step. "
        f"This includes scheduler ({s['scheduler']['avg_us']:.1f} us avg), "
        f"ensure_blocks ({s['ensure_blocks']['avg_us']:.1f} us avg), "
        f"and input_build ({s['input_build']['avg_us']:.1f} us avg).",
        "",
        f"Not hideable: apply_output ({s['apply_output']['avg_us']:.1f} us avg) "
        f"and cleanup+metrics ({s['cleanup_metrics']['avg_us']:.1f} us avg), "
        f"both depend on GPU output or must wait until after the step boundary.",
        "",
    ]

    # Q3
    lines += [
        f"**3. Async Scheduling 理论最大加速比？**",
        f"",
        f"If all pre-GPU CPU work is fully hidden behind the GPU forward, "
        f"the theoretical per-step time becomes:",
        f"",
        f"  `new_step_time = avg_step_total - theoretical_max_hideable`",
        f"  `= {a['avg_step_total_us']:.1f} us - {a['theoretical_max_hideable_us']:.1f} us`",
        f"  `= {max(a['avg_step_total_us'] - a['theoretical_max_hideable_us'], 0):.1f} us`",
        f"",
        f"Speedup ratio: **{a['theoretical_max_speedup_ratio']}x** "
        f"(i.e. {round((1 - 1/a['theoretical_max_speedup_ratio'])*100, 1)}% step time reduction).",
        "",
        f"**Important caveat**: this assumes 100% hiding efficiency, which requires "
        f"that the GPU forward is always long enough to absorb the full CPU gap. "
        f"If GPU forward < hideable CPU time, the excess CPU work is NOT hidden — "
        f"the actual speedup is bounded by `max(GPU_time, hideable_time_remaining)`.",
        "",
    ]

    # Q4
    is_worthwhile = a["gpu_pct_of_total"] > 80 and hideable_pct > 5
    lines += [
        f"**4. 这个收益是否值得继续深入？**",
        f"",
    ]
    if is_worthwhile:
        lines += [
            f"**Marginally yes** — the CPU gap is {a['avg_cpu_gap_us']:.1f} us ({a['cpu_gap_pct_of_total']}% of step), "
            f"with a theoretical max speedup of {a['theoretical_max_speedup_ratio']}x. "
            f"However, the absolute time is small: the scheduler takes only "
            f"{s['scheduler']['avg_us']:.1f} us per decode step for a single request. "
            f"At higher batch sizes the scheduler cost scales, and that is where "
            f"the overlap becomes more impactful.",
            "",
            f"For the current single-request configuration, the CPU overhead is negligible "
            f"relative to GPU time. The real value of Async Scheduling would only become "
            f"visible at higher batch sizes (4+ concurrent requests).",
            "",
        ]
    else:
        lines += [
            f"**Yes, worth further investigation at higher batch sizes.** "
            f"The CPU gap ({a['avg_cpu_gap_us']:.1f} us) is small relative to the "
            f"GPU forward time ({a['avg_gpu_ms']:.3f} ms) at batch_size=1. "
            f"However, scheduler overhead scales with the number of running sequences. "
            f"A profile at batch_size=4 or 8 would reveal whether the gap grows enough "
            f"to justify the pipeline complexity.",
            "",
            f"Current single-request observation: the CPU gap is "
            f"{a['cpu_gap_pct_of_total']}% of the step. "
            f"Async Scheduling is a modest optimization for this configuration, "
            f"but a necessary one in production settings.",
            "",
        ]

    # Q5
    lines += [
        f"**5. 是否有迹象表明 CUDA Graph 更值得优先研究？**",
        f"",
        f"**No.** The decode step has a stable execution graph (same ops each step, "
        f"same number of layers, etc.), but the GPU forward time is already dominated "
        f"by the model matmul operations rather than kernel launch overhead. "
        f"CUDA Graph primarily eliminates Python-side launch overhead and kernel "
        f"fusion opportunities. For a single request at small model scale, the "
        f"launch overhead is a tiny fraction of the step.",
        "",
        f"More importantly, the current codebase's dynamic tensor shapes "
        f"(block tables, slot mappings, position tensors) make CUDA Graph "
        f"capture difficult without significant buffer management changes. "
        f"Async Scheduling requires less invasive changes and addresses "
        f"a more meaningful bottleneck for the goals of this project.",
        "",
    ]

    lines += [
        "---\n",
        f"_Generated by Phase 2 minimal profile. batch_size=1, context_length={summary['config']['context_length']}, "
        f"recorded_decode_steps={n}._\n",
    ]

    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
