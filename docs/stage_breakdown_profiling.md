# Stage Breakdown Profiling

A lightweight stage profiler for understanding where
time goes in an LLM serving engine.

## Why Stage Breakdown?

In a production LLM serving system, end-to-end request latency is the
aggregate of many internal stages:

```
request arrives
    ↓
  [queue waiting]       ← time spent in admission / rate limiting
    ↓
  [scheduler step]      ← scheduling decisions, token budget, prefix cache probe
    ↓
  [kv cache allocation] ← allocating physical blocks for new tokens
    ↓
  [executor forward]    ← the actual model inference
    ├─ prefill           ←   prompt processing (KV cache write, first token)
    └─ decode            ←   auto-regressive token generation (KV read + write)
    ↓
  [kv cache release]    ← freeing blocks for finished sequences
    ↓
  [metrics update]      ← collecting step-level metrics
```

Without a breakdown, you only see the end-to-end latency. With breakdown,
you can answer questions like:

- Is the bottleneck in **queueing** (requests waiting too long before admission)?
- Is **scheduler CPU overhead** significant compared to GPU execution?
- Is **KV cache allocation** adding noticeable latency?
- Is **prefill** (prompt processing) or **decode** (token generation) dominant?
- Is **metrics collection** overhead meaningful?

## TTFT / TPOT / Throughput vs. Stages

| Metric | Related Stages | Notes |
|--------|---------------|-------|
| **TTFT** (Time To First Token) | queue_waiting + scheduler + prefix_cache_lookup + kv_cache_allocation + prefill | Prefill dominates for long prompts. Short prompt TTFT is mostly scheduler + queue. |
| **TPOT** (Time Per Output Token) | decode (+ scheduler + kv_cache_allocation per step) | Decode time per token. Scheduler latency is amortized over the batch. |
| **Throughput** (tok/s, req/s) | All stages combined | Bottlenecks in any stage cap throughput. |

## What This Project Can Profile

| Stage | Source | Notes |
|-------|--------|-------|
| `request_queue_waiting` | EngineCore | Time from `arrival_time` to `first_scheduled_time`. |
| `scheduler_step` | EngineCore | Full scheduler.schedule() call. |
| `prefix_cache_lookup` | BlockManager.probe_prefix_cache | Block hash computation + cache probe. |
| `kv_cache_allocation` | BlockManager.ensure_block | Allocator.allocate() for a new block. |
| `prefill` | EngineCore | Executor prefill call. |
| `decode` | EngineCore | Executor decode call. |
| `executor_forward` | EngineCore | Combined prefill + decode (one measurement per engine step). |
| `kv_cache_release` | BlockManager.free | Allocator.free() for finished sequences. |
| `metrics_update` | EngineCore | Metrics collector record_step. |
| `engine_step_total` | EngineCore | Total wall-clock per engine step. |

## What This Project Cannot Accurately Profile

- **GPU kernel-level breakdown** (attention, FFN, sampling, GPU memcpy):
  The project does not use PyTorch profiler or Nsight Systems. Executor
  timing is at the Python-level `executor.prefill()` / `executor.decode()`
  call, not at individual kernel launch granularity.

- **Fine-grained prefill vs decode splitting**: The fake executor runs
  prefill and decode synchronously in the same thread. The StageProfiler
  wraps `prefill()` and `decode()` as separate stages, but the fake executor
  cannot split "token computation" from "KV cache write". The current
  breakdown provides coarse prefill/decode timing, which is useful for
  understanding scheduling overhead but **does not equal GPU kernel-level
  profiling** (e.g., Nsight Systems, PyTorch profiler, or nvbench).

- **PyTorch / CUDA kernel overlap**: When using the Qwen executor with a GPU,
  `torch.no_grad()` model execution includes time for multiple GPU kernel
  launches that may overlap. The profiler treats them as one block.

- **CPU-GPU synchronization overhead**: The profiler does not instrument
  `torch.cuda.synchronize()` because the project doesn't explicitly call it.

- **Tokenizer overhead**: Tokenization is not separately timed. It happens
  during `add_request()` which is outside the engine step loop.

- **request_queue_waiting split into admission + queue**: In the serving layer
  (not this engine test), admission control (rate limiting, reject/accept) is
  separate from queue waiting. This breakdown combines them.

## How to Run

```bash
# Fake executor (fast, no GPU, pure Python)
python examples/demo_stage_breakdown.py --executor fake --requests 16 --tokens 16

# Qwen executor (real Qwen2-0.5B, needs torch + transformers + optional GPU)
python examples/demo_stage_breakdown.py --executor qwen --requests 4 --tokens 16
```

### Example Output

```
$ python examples/demo_stage_breakdown.py --executor fake --requests 8 --tokens 8

======================================================================
Stage Breakdown Profiling
======================================================================

  Experiment Configuration
  --------------------------------------------------
  executor:                fake
  num_requests:            8
  max_new_tokens:          8
  block_size:              4
  max_num_seqs:            4
  max_num_batched_tokens:  16
  max_prefill_chunk_size:  4
  chunked_prefill:         True
  num_gpu_blocks:          32

  Added request req-0000: prompt_len=13
  ...

  Running 8 requests to completion...
  Done in 0.008s

  --------------------------------------------------
  Generated Outputs
  --------------------------------------------------
    req-0000: '~Xm5KIjA'
    ...

  Stage Breakdown
  ---------------------------------------------------------------------------
  stage                        count   total_ms    avg_ms     max_ms     pct
  ---------------------------------------------------------------------------
  scheduler_step                  14      2.10     0.1502    0.2439    37.0%
  engine_step_total               14      5.68     0.4055    0.5911   100.0%
  decode                          12      1.22     0.1015    0.1364    21.5%
  executor_forward                14      1.67     0.1194    0.1833    29.4%
  prefill                          8      0.54     0.0673    0.0797     9.5%
  kv_cache_allocation             30      0.35     0.0116    0.0197     6.2%
  kv_cache_release                 8      0.08     0.0096    0.0138     1.3%
  request_queue_waiting            8      0.06     0.0076    0.0090     1.1%
  metrics_update                  14      0.05     0.0033    0.0050     0.8%
  prefix_cache_lookup              0      0.00     0.0000    0.0000     0.0%
  ---------------------------------------------------------------------------
  Total profiled time                       9.61 ms
  Total requests                             8
  Total engine steps                        14

  Interpretation
  --------------------------------------------------
  This run used the FAKE executor (pure CPU arithmetic).
  ...
```

## Usage Guide

### How to explain this experiment

**"I built a lightweight stage profiler that decomposes the serving request
lifetime into stages: queue waiting, scheduler, KV cache allocation, prefix
cache lookup, prefill, decode, executor forward, KV cache release, and
metrics update."

**"The key insight:** instead of just looking at total latency or even
aggregate TTFT/TPOT, I wanted to know *where* that latency comes from.
Is the scheduler CPU overhead meaningful? Is KV block allocation a
bottleneck? Is prefill or decode dominant?"

**"With this breakdown you can make data-driven decisions:**
- High queue waiting → increase batch capacity or tune admission control
- High scheduler overhead → optimise scheduling algorithm
- High KV allocation time → increase block size or use block-level caching
- High prefill time → chunk prefill or reduce prompt length
- High decode time → larger batch or faster attention (FlashAttention)"

### Key caveats to mention

1. **This is a coarse Python-level profiler**, not a GPU profiler.
2. **It does not replace Nsight Systems** or PyTorch profiler for
   GPU kernel-level optimisation.
3. **With the fake executor, all numbers are CPU overhead** — no real
   inference. Use the Qwen executor for real model timing.
4. **Executor forward includes both prefill and decode plus any
   allocation inside those calls** — it's not a "pure" execution time.

### What to say about the architecture

- Uses Python `time.time()` and context managers — lightweight, no
  external dependencies.
- The StageProfiler is a standalone class with a `record(stage)` context
  manager and `start(stage)` / `end(stage)` methods.
- The profiler is wired into `EngineCore.step()` and `BlockManager`
  operations with zero changes to scheduling or execution logic.
- All profiling is additive — enabling it doesn't change correctness.

## Comparison with Nsight Systems / PyTorch Profiler

| Feature | This Profiler | Nsight Systems | PyTorch Profiler |
|---------|--------------|----------------|-----------------|
| **Purpose** | Serving-stage breakdown | GPU kernel analysis | PyTorch operator profiling |
| **Granularity** | Python function / block | GPU kernel launch | PyTorch op (aten) |
| **Overhead** | < 0.1 ms per record | Low (HW counters) | Moderate |
| **GPU kernels** | No (only Python wall) | Yes (detailed) | Yes |
| **Python-only** | Yes | No | Partial |
| **Dependencies** | None (stdlib) | NVIDIA tools | PyTorch |
| **Best for** | Serving architecture explanation | Low-level GPU work | Model-level optimisation |

**Bottom line:** This profiler is for *serving architecture understanding*
and *architectural demonstrations*. Use Nsight Systems / PyTorch profiler when
you need to optimise GPU kernel performance.
