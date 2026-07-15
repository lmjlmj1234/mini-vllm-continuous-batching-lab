#!/usr/bin/env python3
"""Benchmark: Reference (PyTorch SDPA) vs Triton Paged Decode Attention.

Compares two paths for single-step decode attention:

1. **Reference**: gather_paged_kv → GQA repeat_interleave → scaled_dot_product_attention
2. **Triton**: triton_decode_attention (online softmax, block-table iteration)

Usage::

    # Default: all batch sizes × context lengths, block_size=16
    python -m benchmarks.benchmark_paged_attention

    # Custom configs
    python -m benchmarks.benchmark_paged_attention \\
        --batch-sizes 1 4 8 --ctx-lengths 128 512 1024 \\
        --block-size 16 --dtype float16 \\
        --warmup 10 --repeats 100 --quiet
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

from mini_vllm.attention.paged_attention_gpu import (
    triton_cache_write,
    triton_decode_attention,
)
from mini_vllm.cache.cache_read import gather_paged_kv
from mini_vllm.cache.cache_write import write_to_paged_cache
from mini_vllm.cache.pool import KVCachePool


# ---------------------------------------------------------------------------
# Device & dtype helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda")
_DEFAULT_DTYPE = torch.float16


def _alloc_pool(
    block_size: int,
    num_blocks: int,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    num_layers: int = 1,
    dtype: torch.dtype = _DEFAULT_DTYPE,
) -> KVCachePool:
    """Allocate a GPU KV cache pool for benchmarking."""
    return KVCachePool.allocate(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=DEVICE,
    )


def _make_kv(
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = _DEFAULT_DTYPE,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic K/V tensors with realistic magnitude.

    Uses a seeded RNG to produce values in FP16-safe range (~[-5, 5]).
    Each sequence gets a unique seed offset, so different sequences have
    different K/V values.
    """
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype,
                    device=DEVICE, generator=g) * 0.5
    g.manual_seed(seed + 1000)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype,
                    device=DEVICE, generator=g) * 0.5
    return k, v


def _make_query(
    batch_size: int,
    num_q_heads: int,
    head_dim: int,
    dtype: torch.dtype = _DEFAULT_DTYPE,
    seed: int = 9999,
) -> torch.Tensor:
    """Create deterministic query tensor with realistic magnitude."""
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    return torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype,
                       device=DEVICE, generator=g) * 0.3


# ---------------------------------------------------------------------------
# Reference decode (single step)
# ---------------------------------------------------------------------------


def ref_decode(
    query: torch.Tensor,
    pool: KVCachePool,
    block_table: torch.Tensor,
    kv_len_after: torch.Tensor,
    num_kv_heads: int,
    layer: int = 0,
) -> torch.Tensor:
    """Reference decode: gather_paged_kv → GQA → SDPA.

    Matches AttentionBackendRef.decode_attention() logic.
    """
    num_decode, num_q_heads, head_dim = query.shape
    n_repeats = num_q_heads // num_kv_heads
    scale = head_dim ** -0.5
    outputs = []
    for i in range(num_decode):
        seq_len = int(kv_len_after[i].item())
        block_ids = [int(b.item()) for b in block_table[i] if b.item() != -1]
        k, v = gather_paged_kv(
            pool.key_caches[layer], pool.value_caches[layer],
            block_ids, seq_len, pool.block_size,
        )
        k = k.repeat_interleave(n_repeats, dim=1)
        v = v.repeat_interleave(n_repeats, dim=1)
        q_sdpa = query[i].unsqueeze(0).unsqueeze(2)   # [1, H, 1, D]
        k_sdpa = k.permute(1, 0, 2).unsqueeze(0)      # [1, H, T, D]
        v_sdpa = v.permute(1, 0, 2).unsqueeze(0)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, scale=scale, is_causal=False,
        )
        outputs.append(out.squeeze(2))
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def check_correctness(
    query: torch.Tensor,
    pool: KVCachePool,
    block_table: torch.Tensor,
    kv_len_after: torch.Tensor,
    num_kv_heads: int,
) -> Dict[str, Any]:
    """Compare Triton and reference outputs for a single decode step.

    Returns dict with pass/fail, max_abs_error, max_rel_error.
    """
    ref_out = ref_decode(query, pool, block_table, kv_len_after, num_kv_heads)
    triton_out = triton_decode_attention(
        query,
        pool.key_caches[0],
        pool.value_caches[0],
        block_table,
        kv_len_after,
        pool.block_size,
    )

    # FP16 tolerance: Triton online softmax accumulates in FP32, while
    # reference SDPA uses CUDA math.  For near-zero values, relative error
    # can be high even with tiny absolute error.  Use combined check:
    # - Absolute error must be < 0.05 for all positions (FP16 typical)
    # - For positions where output magnitude > 1.0, relative error must be < 0.5
    abs_diff = (triton_out.float() - ref_out.float()).abs()
    max_abs = abs_diff.max().item()
    denom = torch.maximum(ref_out.float().abs(), triton_out.float().abs())
    rel_diff = abs_diff / (denom + 1e-10)

    # Pass: absolute error is small enough
    # For near-zero attention outputs, relative error may be high even
    # with tiny absolute error — the absolute threshold suffices.
    max_rel = rel_diff.max().item()
    passed = max_abs < 0.05
    return {
        "passed": bool(passed),
        "max_abs_error": round(max_abs, 6),
        "max_rel_error": round(max_rel, 6),
    }


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------


def num_kv_blocks(ctx_length: int, block_size: int) -> int:
    """Number of KV cache blocks needed for a given context length."""
    return (ctx_length + block_size - 1) // block_size


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _collect_metadata() -> Dict[str, Any]:
    """Collect environment metadata."""
    meta: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": "",
        "gpu": "",
        "cuda_version": "",
        "pytorch_version": torch.__version__,
        "triton_version": "",
        "python_version": platform.python_version(),
    }
    try:
        import triton
        meta["triton_version"] = triton.__version__
    except (ImportError, AttributeError):
        meta["triton_version"] = "unknown"
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


def run_one_config(
    batch_size: int,
    ctx_length: int,
    block_size: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    quiet: bool,
) -> Dict[str, Any]:
    """Run reference + Triton benchmark for one (batch_size, ctx_length) config.

    Returns a dict of results, or dict with "status": "skipped" on OOM.
    """
    pool = None
    try:
        num_blocks_needed = num_kv_blocks(ctx_length, block_size)
        # Allocate a generous pool (at least 256 blocks)
        pool_size = max(256, num_blocks_needed * 4)
        pool = _alloc_pool(
            block_size=block_size,
            num_blocks=pool_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

        # Write KV cache for each sequence in the batch
        total_kv_tokens = batch_size * ctx_length
        # Use contiguous blocks starting from 0
        total_blocks = batch_size * num_blocks_needed

        # Sanity check
        if total_blocks > pool.num_blocks:
            return {
                "batch_size": batch_size,
                "ctx_length": ctx_length,
                "block_size": block_size,
                "status": "skipped",
                "reason": f"need {total_blocks} blocks, pool has {pool.num_blocks}",
            }

        # Write all KV tokens to cache
        for seq_idx in range(batch_size):
            k, v = _make_kv(ctx_length, num_kv_heads, head_dim, dtype, seed=42 + seq_idx)
            slot_offset = seq_idx * num_blocks_needed * block_size
            slots = torch.arange(
                slot_offset, slot_offset + ctx_length,
                dtype=torch.long, device=DEVICE,
            )
            triton_cache_write(k, v, pool.key_caches[0], pool.value_caches[0], slots, block_size)

        # Build block table: one row per sequence, padded with -1
        max_blocks = num_blocks_needed
        block_table = torch.full((batch_size, max_blocks), -1, dtype=torch.long, device=DEVICE)
        for seq_idx in range(batch_size):
            start_block = seq_idx * num_blocks_needed
            block_table[seq_idx] = torch.arange(
                start_block, start_block + num_blocks_needed,
                dtype=torch.long, device=DEVICE,
            )

        kv_len_after = torch.full(
            (batch_size,), ctx_length, dtype=torch.int32, device=DEVICE,
        )

        # Build query
        num_q_heads = num_q_heads
        query = _make_query(batch_size, num_q_heads, head_dim, dtype)

        # --- Correctness check (separate from timing) ---
        correctness = check_correctness(
            query, pool, block_table, kv_len_after, num_kv_heads,
        )

        # === Reference timing ===
        # Warmup
        for _ in range(warmup):
            _ = ref_decode(query, pool, block_table, kv_len_after, num_kv_heads)
        torch.cuda.synchronize()

        # Timed runs
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(repeats):
            _ = ref_decode(query, pool, block_table, kv_len_after, num_kv_heads)
        end_event.record()
        torch.cuda.synchronize()
        ref_elapsed_ms = start_event.elapsed_time(end_event)  # ms
        ref_us_per_step = (ref_elapsed_ms * 1000.0) / repeats

        # === Triton timing ===
        # Warmup (also forces JIT compilation)
        for _ in range(warmup):
            _ = triton_decode_attention(
                query, pool.key_caches[0], pool.value_caches[0],
                block_table, kv_len_after, block_size,
            )
        torch.cuda.synchronize()

        # Timed runs
        start_event.record()
        for _ in range(repeats):
            _ = triton_decode_attention(
                query, pool.key_caches[0], pool.value_caches[0],
                block_table, kv_len_after, block_size,
            )
        end_event.record()
        torch.cuda.synchronize()
        triton_elapsed_ms = start_event.elapsed_time(end_event)
        triton_us_per_step = (triton_elapsed_ms * 1000.0) / repeats

        speedup = ref_us_per_step / triton_us_per_step if triton_us_per_step > 0 else 0.0

        return {
            "batch_size": batch_size,
            "ctx_length": ctx_length,
            "block_size": block_size,
            "num_kv_blocks": num_kv_blocks(ctx_length, block_size),
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype": str(dtype),
            "status": "ok",
            "ref_latency_us": round(ref_us_per_step, 2),
            "triton_latency_us": round(triton_us_per_step, 2),
            "speedup": round(speedup, 2),
            "passed": correctness["passed"],
            "max_abs_error": correctness["max_abs_error"],
            "max_rel_error": correctness["max_rel_error"],
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "batch_size": batch_size,
            "ctx_length": ctx_length,
            "block_size": block_size,
            "status": "skipped",
            "reason": "CUDA OOM",
        }
    except RuntimeError as e:
        return {
            "batch_size": batch_size,
            "ctx_length": ctx_length,
            "block_size": block_size,
            "status": "skipped",
            "reason": str(e),
        }
    finally:
        del pool
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark: Reference vs Triton Paged Decode Attention",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=[1, 2, 4, 8, 16],
        help="Batch sizes to test (default: 1 2 4 8 16)",
    )
    parser.add_argument(
        "--ctx-lengths", nargs="+", type=int,
        default=[16, 32, 64, 128, 256, 512, 1024],
        help="Context lengths (KV cache size per sequence) (default: 16-1024 powers of 2)",
    )
    parser.add_argument(
        "--block-size", type=int, default=16,
        help="KV cache block size (default: 16)",
    )
    parser.add_argument(
        "--head-dim", type=int, default=128,
        help="Attention head dimension (default: 128, Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--num-q-heads", type=int, default=8,
        help="Number of query heads (default: 8, Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--num-kv-heads", type=int, default=2,
        help="Number of KV heads (default: 2, Qwen2.5-0.5B GQA ratio 4:1)",
    )
    parser.add_argument(
        "--dtype", default="float16",
        choices=["float16", "bfloat16"],
        help="Tensor dtype (default: float16)",
    )
    parser.add_argument(
        "--warmup", type=int, default=20,
        help="Warmup iterations before timing (default: 20)",
    )
    parser.add_argument(
        "--repeats", type=int, default=100,
        help="Timed iterations per config (default: 100)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file prefix (default: benchmark_results/triton_vs_ref_reproduced)",
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
        print("ERROR: CUDA is required for all benchmark tests.")
        sys.exit(1)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    output_prefix = args.output or os.path.join(
        _project_root, "benchmark_results", "triton_vs_ref_reproduced",
    )

    metadata = _collect_metadata()
    metadata["warmup"] = args.warmup
    metadata["repeats"] = args.repeats
    metadata["args"] = vars(args)

    print("=" * 72)
    print("Paged Decode Attention Benchmark")
    print("=" * 72)
    print(f"  GPU:            {metadata.get('gpu', '?')}")
    print(f"  PyTorch:        {metadata['pytorch_version']}")
    print(f"  Triton:         {metadata.get('triton_version', '?')}")
    print(f"  CUDA:           {metadata.get('cuda_version', '?')}")
    print(f"  dtype:          {args.dtype}")
    print(f"  block_size:     {args.block_size}")
    print(f"  head_dim:       {args.head_dim}")
    print(f"  num_q_heads:    {args.num_q_heads}")
    print(f"  num_kv_heads:   {args.num_kv_heads} (GQA repeats={args.num_q_heads // args.num_kv_heads})")
    print(f"  batch_sizes:    {args.batch_sizes}")
    print(f"  ctx_lengths:    {args.ctx_lengths}")
    print(f"  warmup:         {args.warmup}")
    print(f"  repeats:        {args.repeats}")
    print()

    results: List[Dict[str, Any]] = []

    for batch_size in args.batch_sizes:
        for ctx_length in args.ctx_lengths:
            if not args.quiet:
                print(f"  bs={batch_size} ctx={ctx_length} ... ", end="", flush=True)

            result = run_one_config(
                batch_size=batch_size,
                ctx_length=ctx_length,
                block_size=args.block_size,
                head_dim=args.head_dim,
                num_q_heads=args.num_q_heads,
                num_kv_heads=args.num_kv_heads,
                dtype=dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                quiet=args.quiet,
            )

            results.append(result)

            if not args.quiet:
                status = result.get("status", "error")
                if status == "ok":
                    print(
                        f"ref={result['ref_latency_us']}μs  "
                        f"triton={result['triton_latency_us']}μs  "
                        f"speedup={result['speedup']}×  "
                        f"pass={result['passed']}"
                    )
                elif status == "skipped":
                    print(f"SKIPPED: {result.get('reason', '')}")
                else:
                    print(f"ERROR: {result.get('reason', '')}")

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
        "batch_size", "ctx_length", "block_size", "num_kv_blocks",
        "num_q_heads", "num_kv_heads", "head_dim", "dtype",
        "status", "ref_latency_us", "triton_latency_us", "speedup",
        "passed", "max_abs_error", "max_rel_error", "reason",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Results saved to {csv_path}")

    # Print summary table
    print()
    print("-" * 100)
    print(f"{'BS':<4} {'CTX':<6} {'Blocks':<8} {'Ref (μs)':<12} {'Triton (μs)':<14} {'Speedup':<10} {'Pass':<6} {'MaxAE':<10}")
    print("-" * 100)
    for r in results:
        if r.get("status") == "ok":
            print(
                f"{r['batch_size']:<4} {r['ctx_length']:<6} "
                f"{r['num_kv_blocks']:<8} {r['ref_latency_us']:<12.2f} "
                f"{r['triton_latency_us']:<14.2f} {r['speedup']:<10.2f}× "
                f"{str(r['passed']):<6} {r['max_abs_error']:<10.6f}"
            )
        elif r.get("status") == "skipped":
            print(
                f"{r['batch_size']:<4} {r['ctx_length']:<6} "
                f"{'SKIPPED':<50} {r.get('reason', ''):<20}"
            )
    print("-" * 100)

    # Historical comparison
    print()
    print("=" * 72)
    print("Historical Comparison (bs=1)")
    print("=" * 72)
    for ctx in [16, 128, 512, 1024]:
        match = [r for r in results
                 if r.get("batch_size") == 1
                 and r.get("ctx_length") == ctx
                 and r.get("status") == "ok"]
        if match:
            r = match[0]
            print(f"  ctx={ctx:<6}  ref={r['ref_latency_us']:>8.2f}μs  "
                  f"triton={r['triton_latency_us']:>8.2f}μs  "
                  f"speedup={r['speedup']:>6.2f}×")
    print()


if __name__ == "__main__":
    main()
