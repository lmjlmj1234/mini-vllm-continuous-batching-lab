# Paged Decode Attention Benchmark

Benchmark scripts for measuring PagedAttention decode kernel performance vs a PyTorch SDPA baseline, at both the kernel level and the full-model level.

## Scripts

| Script | Scope | Measured |
|--------|-------|----------|
| `benchmarks/benchmark_paged_attention.py` | Isolated attention layer | Gather KV → decode attention (Triton vs ref) |
| `benchmarks/benchmark_decode_e2e.py` | Full Qwen2.5-0.5B forward pass | Prefill once, measure single decode step |

## Kernel Benchmark

```
python -m benchmarks.benchmark_paged_attention \
    --batch-sizes 1 2 4 8 \
    --ctx-lengths 16 128 512 1024 \
    --warmup 20 --repeats 100 --quiet
```

Default config: 5 batch sizes × 7 context lengths = 35 configs, block_size=16, head_dim=128, 8 Q heads, 2 KV heads (GQA repeats=4).

Output columns:

- **BS / CTX** — sequences in the batch / prompt tokens before decode
- **Blocks** — KV cache blocks consumed (`ceil(CTX / block_size)`)
- **Ref (μs)** — PyTorch SDPA baseline (CV-style CUDA event timing)
- **Triton (μs)** — `_paged_decode_kernel` Triton kernel (same timing)
- **Speedup** — `Ref / Triton`
- **Pass** — correctness check: `|triton - ref| < 0.05` in FP16
- **MaxAE** — maximum absolute elementwise error

### Default Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| block_size | 16 | Power of 2 |
| head_dim | 128 | Only 64 and 128 supported |
| num_q_heads | 8 | Must be multiple of num_kv_heads |
| num_kv_heads | 2 | GQA repeats = 4 |
| dtype | float16 | |
| warmup | 20 | Steps before timing (excludes JIT) |
| repeats | 100 | Timed iterations |

### Expected Results (GPU: RTX 3060, PyTorch 2.10, CUDA 12.8)

Configurations beyond 256 blocks are skipped when the KV cache pool has only 256 blocks.

**Scalability:** Triton kernel latency grows sub-linearly with context length; ref latency grows linearly, so speedup increases with longer contexts. Higher batch sizes give larger speedups because sequences share the GPU and the Triton kernel's block iteration is compute-bound relative to the gathering overheads.

### Historical Speedups (head_dim=128, num_q_heads=8, num_kv_heads=2)

| Context | Historical | This Run | Note |
|---------|-----------|----------|------|
| 16 | 4.25× | 2.74× | Ref faster in PyTorch 2.10 SDPA |
| 128 | 14.19× | 5.50× | Same ref speedup |
| 512 | — | 7.05× | |
| 1024 | 29.46× | 7.70× | Same ref speedup |

The triton times are consistent across runs; the reference path improved ~2-4× due to PyTorch 2.10 SDPA optimizations.

## E2E Decode Benchmark

```
python -m benchmarks.benchmark_decode_e2e \
    --ctx-lengths 16 128 512 \
    --warmup 10 --repeats 50 --quiet
```

Uses the full Qwen2.5-0.5B model: head_dim=64, 14 Q heads, 2 KV heads (GQA repeats=7). The script creates two `QwenModelRunner` instances sharing the same weights but with different attention backends (`reference` vs `triton`).

Each context length:
1. Prefills both runners with the same prompt
2. Measures a single decode step (CUDA events)
3. Compares output logits: max absolute error, max relative error, top-1 token match

Pass condition: `max_abs < 0.5 AND top1_match`.

## Comparison with Historical Results

### Kernel (bs=1, head_dim=128)

| Context | Historical Ref | New Ref | Historical Triton | New Triton | Historical Speedup | New Speedup |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|
| 16 | 483.94μs | 273.54μs | 113.81μs | 99.93μs | 4.25× | 2.74× |
| 1024 | 15839.38μs | 4116.18μs | 537.60μs | 534.28μs | 29.46× | 7.70× |

Triton times are nearly identical. The reference (PyTorch SDPA) got 2-4× faster, reducing the observed speedup.

### E2E (Qwen2.5-0.5B, head_dim=64)

| Context | Historical Ref | Historical Triton | Historical Speedup |
|---------|:-:|:-:|:-:|
| 16 | 42791.9μs | 29410.7μs | 1.45× |
| 128 | 91904.0μs | 26646.0μs | 3.45× |
| 512 | 310985.7μs | 27577.4μs | 11.28× |

## Notes

- All measurements use `torch.cuda.Event` for GPU timing (CV-style).
- Warmup steps are always excluded from timing.
- The KV cache pool is reset between configurations.
- OOM / RuntimeError configs are reported as "skipped" rather than crashing.
