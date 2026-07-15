#!/usr/bin/env python3
"""Benchmark: Reference vs Triton backend — single-step decode with Qwen2.5-0.5B.

Compares two attention backends for a single decode step of the full
Qwen2.5-0.5B forward pass:

1. **Reference** (AttentionBackendRef): gather_paged_kv + PyTorch SDPA
2. **Triton** (AttentionBackendGPU): triton_decode_attention

Usage::

    python -m benchmarks.benchmark_decode_e2e

    python -m benchmarks.benchmark_decode_e2e \\
        --ctx-lengths 16 32 64 --warmup 5 --repeats 50 --quiet
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mini_vllm.attention.backend import AttentionBackend
from mini_vllm.model_runner.base import (
    AttentionGroup,
    AttentionMetadata,
    ModelInput,
)
from mini_vllm.model_runner.config_adapter import ConfigAdapter
from mini_vllm.model_runner.qwen_runner import QwenModelRunner


# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/mnt/e/vllm_awq_qwen_exp/models/Qwen2.5-0.5B-Instruct",
)


def _has_model() -> bool:
    if not os.path.isdir(MODEL_PATH):
        return False
    for fname in os.listdir(MODEL_PATH):
        if fname.endswith((".safetensors", ".bin")):
            return True
    return False


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda")


def _make_prompt_ids(length: int) -> List[int]:
    """Deterministic prompt token IDs, avoiding special tokens."""
    return [(100 + i) % 32000 for i in range(length)]


def _create_runner(backend_name: str, block_size: int = 16) -> QwenModelRunner:
    """Create a QwenModelRunner with the specified attention backend."""
    model_config = ConfigAdapter.from_pretrained(MODEL_PATH)
    backend = AttentionBackend.create(model_config, backend=backend_name)
    runner = QwenModelRunner(
        model_path=MODEL_PATH,
        attention_backend=backend,
        config=model_config,
        device=DEVICE,
        block_size=block_size,
        num_gpu_blocks_override=1024,  # Enough for all test configs
    )
    return runner


def _prefill_and_get_decode_input(
    runner: QwenModelRunner,
    prompt_ids: List[int],
    block_size: int,
) -> Tuple[Any, Any, Any]:
    """Run prefill, return (block_table, decode_metadata, decode_input_for_logits).

    Returns block_table, attn_metadata, model_input for the first decode step.
    """
    prompt_len = len(prompt_ids)
    # Need enough blocks for prefill + one decode token
    total_tokens_needed = prompt_len + 1
    num_blocks_needed = (total_tokens_needed + block_size - 1) // block_size
    all_pids = list(range(num_blocks_needed))
    block_table = torch.tensor([all_pids], device=DEVICE)
    pref_slots = torch.tensor(list(range(prompt_len)), device=DEVICE)

    pref_meta = AttentionMetadata(
        groups=[
            AttentionGroup(
                seq_indices=[0],
                attention_type="prefill_gpu",
                cached_len_before=torch.tensor([0], device=DEVICE),
                query_len=torch.tensor([prompt_len], device=DEVICE),
                kv_len_after=torch.tensor([prompt_len], device=DEVICE),
            ),
        ],
        prefill_slot_mapping=pref_slots,
        prefill_block_tables=block_table,
        prefill_positions=torch.tensor(list(range(prompt_len)), device=DEVICE),
        decode_block_tables=torch.zeros((0, num_blocks_needed), dtype=torch.long, device=DEVICE),
        decode_slot_mapping=torch.tensor([], dtype=torch.long, device=DEVICE),
        decode_positions=torch.tensor([], dtype=torch.long, device=DEVICE),
        block_size=block_size,
        num_kv_heads=runner._model_config.num_kv_heads,
        head_dim=runner._model_config.head_dim,
    )

    pref_input = ModelInput(
        input_ids=torch.tensor(prompt_ids, device=DEVICE),
        positions=torch.tensor(list(range(prompt_len)), device=DEVICE),
        slot_mapping=pref_slots,
        attn_metadata=pref_meta,
        sample_token_indices=torch.tensor([prompt_len - 1], device=DEVICE),
    )

    # Run prefill
    with torch.no_grad():
        _ = runner.execute_model(pref_input)

    # Build decode input for the NEXT step (single token decode)
    decode_id = prompt_ids[-1]  # use last prompt token as input (no tokenizer)
    dec_ids = torch.tensor([decode_id], device=DEVICE)
    dec_pos = torch.tensor([prompt_len], device=DEVICE)
    dec_slots = torch.tensor([prompt_len], device=DEVICE)

    dec_meta = AttentionMetadata(
        groups=[
            AttentionGroup(
                seq_indices=[0],
                attention_type="decode_gpu",
                cached_len_before=torch.tensor([prompt_len], device=DEVICE),
                query_len=torch.tensor([1], device=DEVICE),
                kv_len_after=torch.tensor([prompt_len + 1], device=DEVICE),
            ),
        ],
        prefill_slot_mapping=torch.tensor([], dtype=torch.long, device=DEVICE),
        prefill_block_tables=torch.zeros((0, num_blocks_needed), dtype=torch.long, device=DEVICE),
        prefill_positions=torch.tensor([], dtype=torch.long, device=DEVICE),
        decode_block_tables=block_table,
        decode_slot_mapping=dec_slots,
        decode_positions=dec_pos,
        block_size=block_size,
        num_kv_heads=runner._model_config.num_kv_heads,
        head_dim=runner._model_config.head_dim,
    )

    dec_input = ModelInput(
        input_ids=dec_ids,
        positions=dec_pos,
        slot_mapping=dec_slots,
        attn_metadata=dec_meta,
        sample_token_indices=torch.tensor([0], device=DEVICE),
    )

    return block_table, dec_meta, dec_input


# ---------------------------------------------------------------------------
# Single-step decode measurement
# ---------------------------------------------------------------------------


def measure_decode_step(
    runner: QwenModelRunner,
    decode_input: ModelInput,
    warmup: int,
    repeats: int,
) -> Tuple[float, torch.Tensor]:
    """Measure single-step decode latency.

    Returns (latency_us_per_step, logits).
    Logits shape: [1, vocab_size] — the model output before sampling.
    """
    # Warmup (also excludes any JIT compilation for the triton backend)
    with torch.no_grad():
        for _ in range(warmup):
            _ = runner.execute_model(decode_input)
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    logits = None
    with torch.no_grad():
        for _ in range(repeats):
            logits = runner.execute_model(decode_input)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    us_per_step = (elapsed_ms * 1000.0) / repeats

    return us_per_step, logits


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _collect_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": "",
        "gpu": "",
        "cuda_version": "",
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "model": os.path.basename(MODEL_PATH),
    }
    try:
        meta["cuda_version"] = torch.version.cuda or ""
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=_project_root,
        )
        if result.returncode == 0:
            meta["git_commit"] = result.stdout.strip()
    except Exception:
        pass
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark: Reference vs Triton backend — single-step decode with Qwen2.5-0.5B",
    )
    parser.add_argument(
        "--ctx-lengths", nargs="+", type=int,
        default=[16, 32, 64, 128, 256, 512],
        help="Context lengths (prompt tokens before decode) (default: 16 32 64 128 256 512)",
    )
    parser.add_argument(
        "--block-size", type=int, default=16,
        help="KV cache block size (default: 16)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Warmup decode steps (default: 10)",
    )
    parser.add_argument(
        "--repeats", type=int, default=50,
        help="Timed decode steps (default: 50)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file prefix (default: benchmark_results/e2e_results_reproduced)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-config progress output",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required.")
        sys.exit(1)

    if not _has_model():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Set QWEN_MODEL_PATH or check the model directory.")
        sys.exit(1)

    output_prefix = args.output or os.path.join(
        _project_root, "benchmark_results", "e2e_results_reproduced",
    )

    metadata = _collect_metadata()
    metadata["warmup"] = args.warmup
    metadata["repeats"] = args.repeats
    metadata["args"] = vars(args)

    print("=" * 72)
    print("E2E Decode Benchmark — Qwen2.5-0.5B")
    print("=" * 72)
    print(f"  GPU:            {metadata.get('gpu', '?')}")
    print(f"  PyTorch:        {metadata['pytorch_version']}")
    print(f"  CUDA:           {metadata.get('cuda_version', '?')}")
    print(f"  Model:          {metadata['model']}")
    print(f"  block_size:     {args.block_size}")
    print(f"  ctx_lengths:    {args.ctx_lengths}")
    print(f"  warmup:         {args.warmup}")
    print(f"  repeats:        {args.repeats}")
    print()

    results: List[Dict[str, Any]] = []

    print("Loading Reference backend runner...")
    ref_runner = None
    triton_runner = None
    model_config = None

    # Load model once to get config, then create two runners sharing the
    # same model weights (different attention backends).
    # Note: qwen_runner creates separate pools per runner; we need to
    # prefill each separately.
    ref_runner = _create_runner("reference", args.block_size)
    triton_runner = _create_runner("triton", args.block_size)

    # Set common pool size override to match
    model_config = ref_runner._model_config
    num_q_heads = model_config.num_heads
    num_kv_heads = model_config.num_kv_heads
    head_dim = model_config.head_dim

    print(f"  Model dimensions: Q={num_q_heads}, KV={num_kv_heads}, "
          f"head_dim={head_dim}, GQA={num_q_heads // num_kv_heads}")
    print()

    # Reset pool to clear any residual state
    ref_runner.pool.reset()
    triton_runner.pool.reset()

    for ctx_length in args.ctx_lengths:
        if not args.quiet:
            print(f"  ctx={ctx_length} ... ", end="", flush=True)

        prompt_ids = _make_prompt_ids(ctx_length)

        # Prefill Reference
        _, _, ref_dec_input = _prefill_and_get_decode_input(
            ref_runner, prompt_ids, args.block_size,
        )

        # Prefill Triton (separate pool)
        _, _, triton_dec_input = _prefill_and_get_decode_input(
            triton_runner, prompt_ids, args.block_size,
        )

        # Measure Reference decode
        ref_us, ref_logits = measure_decode_step(
            ref_runner, ref_dec_input, args.warmup, args.repeats,
        )

        # Measure Triton decode
        triton_us, triton_logits = measure_decode_step(
            triton_runner, triton_dec_input, args.warmup, args.repeats,
        )

        speedup = ref_us / triton_us if triton_us > 0 else 0.0

        # Correctness check: compare logits
        ref_logits_f = ref_logits.float()
        triton_logits_f = triton_logits.float()
        abs_diff = (triton_logits_f - ref_logits_f).abs()
        max_abs = abs_diff.max().item()
        denom = torch.maximum(ref_logits_f.abs(), triton_logits_f.abs())
        rel_diff = abs_diff / (denom + 1e-10)
        max_rel = rel_diff.max().item()

        # Top-1 token comparison
        ref_top1 = ref_logits.argmax(dim=-1).item()
        triton_top1 = triton_logits.argmax(dim=-1).item()
        top1_match = ref_top1 == triton_top1

        passed = max_abs < 0.5 and top1_match

        result = {
            "ctx_length": ctx_length,
            "block_size": args.block_size,
            "num_kv_blocks": (ctx_length + args.block_size - 1) // args.block_size,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "status": "ok",
            "ref_latency_us": round(ref_us, 2),
            "triton_latency_us": round(triton_us, 2),
            "speedup": round(speedup, 2),
            "passed": bool(passed),
            "max_abs_error": round(max_abs, 6),
            "max_rel_error": round(max_rel, 6),
            "ref_top1_token": int(ref_top1),
            "triton_top1_token": int(triton_top1),
            "top1_match": bool(top1_match),
        }
        results.append(result)

        # Clean pools for next config
        ref_runner.pool.reset()
        triton_runner.pool.reset()

        if not args.quiet:
            print(
                f"ref={ref_us:.1f}μs  "
                f"triton={triton_us:.1f}μs  "
                f"speedup={speedup:.2f}×  "
                f"top1_match={top1_match}"
            )

    # Clean up runners
    del ref_runner
    del triton_runner
    torch.cuda.empty_cache()

    # Save JSON
    output = {
        "metadata": metadata,
        "results": results,
    }
    json_path = f"{output_prefix}.json"
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    # Save CSV
    csv_path = f"{output_prefix}.csv"
    csv_fields = [
        "ctx_length", "block_size", "num_kv_blocks",
        "num_q_heads", "num_kv_heads", "head_dim",
        "status", "ref_latency_us", "triton_latency_us", "speedup",
        "passed", "max_abs_error", "max_rel_error",
        "ref_top1_token", "triton_top1_token", "top1_match",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Results saved to {csv_path}")

    # Summary table
    print()
    print("-" * 80)
    print(f"{'CTX':<6} {'Blocks':<8} {'Ref (μs)':<12} {'Triton (μs)':<14} {'Speedup':<10} {'Top-1':<8}")
    print("-" * 80)
    for r in results:
        if r.get("status") == "ok":
            print(
                f"{r['ctx_length']:<6} {r['num_kv_blocks']:<8} "
                f"{r['ref_latency_us']:<12.2f} {r['triton_latency_us']:<14.2f} "
                f"{r['speedup']:<10.2f}× {str(r['top1_match']):<8}"
            )
    print("-" * 80)

    # Historical comparison
    print()
    print("=" * 72)
    print("Historical Comparison")
    print("=" * 72)
    for ctx in [16, 128, 512]:
        match = [r for r in results
                 if r.get("ctx_length") == ctx and r.get("status") == "ok"]
        if match:
            r = match[0]
            print(f"  ctx={ctx:<6}  ref={r['ref_latency_us']:>8.2f}μs  "
                  f"triton={r['triton_latency_us']:>8.2f}μs  "
                  f"speedup={r['speedup']:>6.2f}×")
    print()


if __name__ == "__main__":
    main()
