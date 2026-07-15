# mini-vLLM Continuous Batching Lab

**A minimal, educational re-implementation of vLLM's Continuous Batching engine.**
**一个极简的、用于教学的 vLLM 连续批处理引擎复刻实现。**

## Why Continuous Batching?

Traditional LLM serving processes requests one at a time or in fixed-size
batches.  **Continuous Batching** (also called *in-flight* or *iterative-level*
batching) allows new requests to **preempt** the decode batch at every
iteration.  The result is dramatically higher GPU utilisation and lower latency
for interactive workloads.

vLLM popularised this technique with PagedAttention and a scheduler that
treats each *token generation step* as the scheduling granularity — rather
than waiting for an entire request to finish before scheduling the next one,
new requests merge into the batch on the very next step.

This project breaks down exactly how that works.

这个项目详细拆解了连续批处理的工作原理。

## Phase 1 — Core Continuous Batching Engine

Phase 1 implements the **core engine boundaries** of vLLM's Continuous Batching
architecture without any GPU code.  Everything runs in pure Python.

### What's implemented (Phase 1)

| Module | vLLM counterpart | Responsibility |
|--------|-----------------|----------------|
| `SequenceGroup` | `SequenceGroup` | User request metadata: prompt, sampling params, arrival time |
| `Sequence` | `Sequence` | Token buffers, status, KV block table, generation timing |
| `SamplingParams` | `SamplingParams` | Generation knobs (max_tokens, temperature, top_p, stop strings) |
| `RequestQueue` | (implicit in scheduler) | Four pools (waiting/running/finished/rejected), all store SequenceGroup |
| `BlockTable` | `BlockTable` | Logical → physical block mapping for PagedAttention |
| `BlockAllocator` | `BlockAllocator` | Low-level physical block free-list with reference counting |
| `BlockManager` | `BlockSpaceManager` | Coordinates BlockAllocator, manages per-sequence allocation & free |
| `Scheduler` / `ScheduleResult` | `Scheduler` / `SchedulerOutputs` | Admit waiting groups, build prefill+decode batches, rich result reporting |
| `FakeModelExecutor` | `ModelRunner` | Fake tokenisation, KV cache simulation, prefill & decode with fake logits |
| `LLMEngine` / `EngineCore` | `LLMEngine` / `EngineCore` | Public API (LLMEngine) and inner step loop (EngineCore) |
| `MetricsCollector` | StatLogger / metrics | TTFT, TPOT, throughput, KV utilization stats |
| `Config` | `EngineConfig` | Single place for tuning knobs |

### Architecture layers

```
mini_vllm/
├── config.py                  # Engine-wide configuration
├── sequence/                  # Data model layer
│   ├── status.py              # Status enum
│   ├── sampling_params.py     # SamplingParams dataclass
│   ├── sequence.py            # Sequence (per-generation state)
│   ├── sequence_group.py      # SequenceGroup (user request → sequences)
│   └── request_queue.py       # 4-pool queue (all store SequenceGroup)
├── cache/                     # KV cache layer
│   ├── block.py               # Block dataclass
│   ├── block_table.py         # Logical → physical mapping with shared block tracking
│   ├── allocator.py           # Free-list allocator with ref_count (low-level)
│   ├── manager.py             # Alloc/free per seq, prefix cache integration (high-level)
│   └── prefix_cache.py        # Block-hash based prefix cache (Phase 2)
├── scheduler/                 # Scheduling layer
│   ├── scheduler.py           # Scheduler (6-phase loop, decode-first, token budget)
│   └── schedule_result.py     # ScheduleResult dataclass
├── executor/                  # Model execution layer
│   ├── base.py                # Executor protocol
│   ├── executor.py            # FakeModelExecutor with simulated KV cache
│   └── qwen_executor.py       # Qwen2-0.5B real model executor (Phase 2)
├── model/                     # Model layer
│   └── fake_model.py          # Fake model (mathematical simulation of key/value/logits)
├── engine/                    # Engine layer
│   ├── engine_core.py         # Inner step loop with timeout/cancel support
│   ├── engine.py              # LLMEngine public API
│   ├── metrics.py             # Performance metrics (TTFT, TPOT, throughput)
│   └── stage_profiler.py      # Stage-level latency breakdown profiler
├── worker/                    # Worker layer
│   ├── fake_worker.py         # Fake worker factory
│   └── qwen_worker.py         # Qwen worker factory (Phase 2)
└── serving/                   # Serving extension layer (Phase 3)
    └── ...                    # HTTP/SSE streaming, rate limiting, lifecycle
```

### Key design decisions

- **SequenceGroup / Sequence split** — mirrors vLLM.  The group owns sampling
  params and prompt metadata; each Sequence owns token buffers and KV block
  table.  Queue pools hold groups; `ScheduleResult` reports groups.

- **BlockAllocator → BlockManager → BlockTable** — three-layer KV cache.
  BlockAllocator manages the free-list with reference counting for shared blocks;
  BlockManager coordinates per-sequence allocation and prefix cache integration;
  BlockTable provides logical-to-physical mapping with shared-block tracking.

- **Fake KV cache** — the executor maintains a simulated device-side KV cache
  (`Dict[int, List[int]]`).  Prefill writes prompt tokens to KV; decode reads
  from existing KV to influence the next token.  This makes the fake model
  output genuinely depend on KV cache content.

- **LLMEngine / EngineCore split** — EngineCore owns the scheduler+executor
  and runs the inner step loop.  LLMEngine provides the public API and
  handles output capture and logging.

### What's **not** implemented (project-wide)

- Real GPU kernels (CUDA / PagedAttention CUDA kernel)
- Preemption / swapping (GPU memory pressure handling)
- Speculative decoding
- Multi-node / distributed serving
- Chunked prefill in the vLLM sense (cross-attention masking in a single prefill step)
- Production-grade prefix cache (eviction policy, stale entry GC)

> **Phase 1 说明：** 本阶段实现了 vLLM 连续批处理引擎的核心骨架。所有代码均为纯 Python，不依赖 GPU。
> 核心模块包括：序列层（Sequence/SequenceGroup）、KV Cache 三层架构（Allocator→Manager→BlockTable）、
> 六阶段调度器（含 token budget 和 decode-first 策略）、FakeModelExecutor（纯数学模拟）、
> LLMEngine/EngineCore 引擎拆分、以及 MetricsCollector 指标采集系统。
> 关键设计：SequenceGroup/Sequence 分离、按需分配（on-demand）而非预分配的 KV Cache、
> 假 KV Cache 使 fake logits 真实依赖于缓存内容。
>
## Phase 2 — Real-World Optimizations

Phase 2 extends the core engine with optimizations that real production
LLM serving systems depend on.

### Chunked Prefill

Long prompts are split across multiple engine steps so shorter requests are not
starved.  Controlled by `Config.chunked_prefill_enabled` and
`Config.max_prefill_chunk_size`.  The scheduler integrates chunk budget into
the existing token-budget and decode-first logic — decode sequences keep
generating while long prompts are incrementally prefilled over several steps.

Relevant modules:
- `Config` — `chunked_prefill_enabled`, `max_prefill_chunk_size`
- `Scheduler.schedule()` — Phase 4 (chunked-prefill continue) & Phase 5 (admit with chunked budget)
- `FakeModelExecutor.prefill()` / `QwenExecutor.prefill()` — chunk-aware prefill from cursor position

### Prefix Cache

Requests with common prompt prefixes share KV cache blocks instead of
recomputing them.  Uses block-level hashing: a prompt's token IDs are divided
into logical blocks, each block is hashed, and the hash is looked up in a
global prefix cache.  Matching blocks are shared via reference counting in
`BlockAllocator`.

Key design:
- **Read-only probe** — `BlockManager.probe_prefix_cache()` is called by the
  scheduler **before** budget computation to determine how many prompt tokens
  are already cached.  No reference counts are modified.
- **Shared allocation** — `BlockManager.allocate_for_seq()` prepopulates the
  block table with shared physical blocks for matching prefix hashes, then
  non-matching blocks are allocated on-demand during execution.
- **Copy-on-Write ready** — `BlockTableEntry.is_shared` flags shared blocks
  so the executor can trigger COW before writing divergent tokens.

Relevant modules:
- `cache/prefix_cache.py` — `PrefixCache` + `PrefixCacheProbeResult`
- `cache/block_table.py` — `BlockTableEntry.is_shared`
- `cache/allocator.py` — `increment_ref()`, `get_ref_count()`, ref-count-aware `free()`
- `cache/manager.py` — `probe_prefix_cache()`, `allocate_for_seq()`, `ensure_block()`
- `scheduler/scheduler.py` — Phase 5 prefix probe integration

### Qwen2-0.5B Executor

A real model executor using HuggingFace Transformers' `Qwen/Qwen2-0.5B` for
end-to-end inference.  Includes proper tokenization, `past_key_values`
management for continuous batching, and per-sequence KV cache cleanup.

Controlled by `Config.executor_type = "qwen"`.

Relevant modules:
- `executor/qwen_executor.py` — `QwenExecutor` with real prefill/decode
- `worker/qwen_worker.py` — Qwen worker factory

> **Phase 2 说明：** 本阶段在核心引擎之上增加了生产级优化。
> - **Chunked Prefill（分块预填充）** — 长 prompt 被拆成多个 chunk 分步处理，避免阻塞短请求的 decode。
>   调度器将 chunk budget 集成到 token budget 体系，decode 序列持续生成的同时长 prompt 逐步完成 prefill。
> - **Prefix Cache（前缀缓存）** — 通过 block-level hashing 共享相同 prompt 前缀的 KV Cache 块。
>   采用"只读探测（read-only probe）→ 共享分配（shared allocation）→ 写时复制（Copy-on-Write ready）"三级设计。
>   BlockAllocator 使用引用计数（ref_count）管理共享块的声明周期。
> - **Qwen2-0.5B Executor** — 真实模型执行器，通过 HuggingFace Transformers 运行 Qwen2-0.5B 模型。
>   支持真正的 tokenizer、past_key_values 管理和 per-sequence 清理。

## Phase 3 / Serving Extension — Production HTTP Serving

The serving layer is an independent extension built on top of the core engine
via the `LLMEngine` public API.  It handles request lifecycle and service
governance rather than scheduling or model execution.

### Features

- **SSE streaming** — Server-Sent Events for real-time token-by-token output
- **Rate limiting** — RPM (requests per minute) and TPM (tokens per minute) admission control
- **Request cancellation** — cancel active requests and free their resources
- **Timeout enforcement** — auto-cancel requests exceeding `Config.request_timeout_s`
- **Client disconnect lifecycle** — detect disconnected HTTP clients and clean up orphaned engine resources

Relevant modules (in `mini_vllm/serving/`):
- `serving/` — HTTP server, SSE handler, rate limiter, lifecycle manager

> **Phase 3 / Serving Extension 说明：** 服务层是建立在核心引擎之上的独立扩展层，通过 LLMEngine 公共接口驱动底层引擎，
> 不涉及调度和模型执行本身。主要处理：HTTP SSE 流式输出（实时逐 token 推送）、RPM/TPM 速率限制、
> 请求取消与超时自动清理、HTTP 客户端断连检测与资源回收。位于 `mini_vllm/serving/` 目录下。

## Quickstart

```bash
cd mini-vllm-continuous-batching-lab

# Core engine demo (zero external dependencies — just Python 3.10+)
python examples/demo_fake_engine.py

# Benchmark with metrics report
python examples/benchmark.py --executor fake --requests 4

# Stage breakdown profiling
python examples/demo_stage_breakdown.py --executor fake --requests 16 --tokens 16

# Run tests
pip install pytest
pytest -q

# Qwen2-0.5B executor (requires torch + transformers)
# pip install torch transformers
# python examples/benchmark.py --executor qwen --requests 2 --tokens 16
```

> 快速开始说明：
> - `demo_fake_engine.py` 零外部依赖，只需 Python 3.10+，展示核心引擎循环
> - `benchmark.py` 运行多请求负载并输出 TTFT、TPOT、Throughput 等指标报告
> - `demo_stage_breakdown.py` 将端到端延迟分解为各个阶段的耗时分析
> - Qwen 执行器需要额外安装 `torch` 和 `transformers`

## Project Structure

```
mini-vllm-continuous-batching-lab/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── Phase1_Architecture.md          # Architecture deep-dive
│   ├── VLLM_Mapping.md                 # Module-by-module mapping to vLLM
│   ├── Learning_Notes.md               # Design decisions & lessons learned
│   ├── Runtime_Timeline.md             # Engine step lifecycle, sequence state machine
│   ├── Scheduler.md                    # 6-phase scheduling, chunked prefill, budget model
│   └── Memory_Manager.md               # 3-layer cache, on-demand allocation, eager comparison
├── mini_vllm/
│   ├── __init__.py                     # Public API exports
│   ├── config.py                       # Engine-wide configuration
│   ├── sequence/                       # Data model layer
│   │   ├── __init__.py
│   │   ├── status.py                   # Status enum (WAITING/PREFILL/RUNNING/...)
│   │   ├── sampling_params.py          # SamplingParams dataclass
│   │   ├── sequence.py                 # Sequence (per-generation state)
│   │   ├── sequence_group.py           # SequenceGroup (user request → sequences)
│   │   └── request_queue.py            # 4-pool queue (all store SequenceGroup)
│   ├── cache/                          # KV cache layer
│   │   ├── __init__.py
│   │   ├── block.py                    # Block dataclass
│   │   ├── block_table.py              # Logical → physical mapping (PagedAttention)
│   │   ├── allocator.py                # Free-list allocator with ref_count
│   │   ├── manager.py                  # Per-sequence allocation + prefix cache integration
│   │   └── prefix_cache.py             # Block-hash based prefix cache
│   ├── scheduler/                      # Scheduling layer
│   │   ├── __init__.py
│   │   ├── scheduler.py                # 6-phase continuous-batching scheduler
│   │   └── schedule_result.py          # ScheduleResult dataclass
│   ├── executor/                       # Model execution layer
│   │   ├── __init__.py
│   │   ├── base.py                     # Executor protocol (interface)
│   │   ├── executor.py                 # FakeModelExecutor (pure Python)
│   │   └── qwen_executor.py            # Qwen2-0.5B executor (Phase 2)
│   ├── model/                          # Model layer
│   │   ├── __init__.py
│   │   └── fake_model.py               # Fake model (mathematical simulation)
│   ├── engine/                         # Engine layer
│   │   ├── __init__.py
│   │   ├── engine_core.py              # Inner step loop
│   │   ├── engine.py                   # LLMEngine public API
│   │   ├── metrics.py                  # MetricsCollector (TTFT, TPOT, throughput)
│   │   └── stage_profiler.py           # Stage-level latency profiler
│   ├── worker/                         # Worker factory layer
│   │   ├── __init__.py
│   │   ├── fake_worker.py              # Fake worker
│   │   └── qwen_worker.py              # Qwen worker (Phase 2)
│   └── serving/                        # Serving extension (Phase 3)
│       ├── __init__.py
│       ├── server.py                   # HTTP server with SSE streaming
│       ├── rate_limiter.py             # RPM/TPM rate limiter
│       └── lifecycle.py                # Client disconnect + request lifecycle
├── examples/
│   ├── demo_fake_engine.py
│   ├── benchmark.py                    # Benchmark with metrics report
│   └── demo_stage_breakdown.py         # Stage profiling demo
└── tests/
    ├── test_request.py
    ├── test_kv_cache_manager.py
    ├── test_scheduler.py
    ├── test_engine.py
    ├── test_metrics.py
    ├── test_prefix_cache.py
    ├── test_stage_profiler.py
    ├── test_serving_layer.py
    └── test_fault_injection.py
```

> 项目结构概览：
> - `mini_vllm/` 按层组织：sequence（数据模型）→ cache（KV Cache）→ scheduler（调度器）→ executor（执行器）
>   → engine（引擎循环+指标+剖析器）→ worker（工厂模式）
> - `model/` 存放 FakeModel（纯数学模拟），与 executor 分离
> - `serving/` 是独立的 HTTP 服务层，通过 `LLMEngine` 公共 API 调用引擎
> - `tests/` 包含针对各层级的单元测试，以及 serving 层集成测试和故障注入测试
> - `examples/` 提供多种运行入口：demo、benchmark、stage profiling

## Stage Breakdown Profiling

This project includes a lightweight **Stage Profiler** for decomposing the
end-to-end serving request latency into individual stages:

- `request_queue_waiting` — time from arrival to first scheduling
- `scheduler_step` — scheduler overhead (token budget, admission, prefix probe)
- `kv_cache_allocation` — physical block allocation time
- `prefix_cache_lookup` — prefix cache hash computation and probe time
- `executor_forward` — combined model execution (prefill + decode)
- `prefill` — prompt processing / KV cache write
- `decode` — auto-regressive token generation
- `kv_cache_release` — block free overhead
- `metrics_update` — metrics collection overhead
- `engine_step_total` — total wall-clock per engine step

The profiler uses Python `time.time()` and context managers — zero external
dependencies. It is designed for **teaching and interview demonstrations**,
not production GPU profiling.

```bash
# Fake executor (pure CPU, no GPU needed)
python examples/demo_stage_breakdown.py --executor fake --requests 16 --tokens 16

# Qwen executor (real model, requires torch + transformers)
python examples/demo_stage_breakdown.py --executor qwen --requests 4 --tokens 16
```

For details, see [`docs/stage_breakdown_profiling.md`](docs/stage_breakdown_profiling.md).

**Important caveats:**
- This is **not** a replacement for Nsight Systems or PyTorch profiler.
- With the fake executor, all numbers reflect Python-level CPU overhead,
  not real inference performance.
- GPU kernel-level breakdown (attention, FFN, sampling) is not provided.

> Stage Profiler 说明：这是一个轻量级的阶段耗时分析工具，使用 Python `time.time()` 和 context manager，
> 零外部依赖。它把端到端的 LLM 服务延迟分解为：请求排队等待 → 调度器开销 → KV Cache 分配 →
> 前缀缓存查询 → prefill → decode → KV Cache 释放 → 指标更新等各个阶段，方便识别性能瓶颈。
> **注意：** 这不是 Nsight Systems 或 PyTorch Profiler 的替代品。FakeExecutor 的数据反映的是
> Python 级 CPU 开销，不是真实推理性能。

---

## PagedAttention Benchmarks

Two benchmark scripts measure the PagedAttention decode kernel against a PyTorch SDPA baseline:

| Script | Scope | Command |
|--------|-------|---------|
| `benchmark_paged_attention.py` | Isolated attention layer | `python -m benchmarks.benchmark_paged_attention` |
| `benchmark_decode_e2e.py` | Full Qwen2.5-0.5B decode step | `python -m benchmarks.benchmark_decode_e2e` |

Results are saved to `benchmark_results/` as JSON and CSV. See [`docs/benchmark_paged_attention.md`](docs/benchmark_paged_attention.md) for full details.

```bash
# Kernel benchmark (5 batch sizes × 7 context lengths)
python -m benchmarks.benchmark_paged_attention

# E2E decode benchmark (Qwen2.5-0.5B model)
python -m benchmarks.benchmark_decode_e2e
```

---

## License / 许可证

MIT

本项目采用 MIT 开源许可证。
