---
labels: ready-for-agent
Status: ready-for-agent
---

# mini-vLLM Continuous Batching Lab — PRD
# mini-vLLM 连续批处理实验项目 — 产品需求文档

## Problem Statement / 问题陈述

**English:**
LLM serving systems like vLLM achieve high GPU utilisation through Continuous Batching — a technique that allows new requests to preempt the decode batch at every token-generation step. However, the production codebases that implement this are vast (vLLM alone is 100k+ lines), making it difficult for engineers and students to understand the core architecture.

There is no concise, educational reference that:
- Breaks down Continuous Batching into its fundamental modules (Sequence, Scheduler, KV Cache, Executor, Engine)
- Shows how those modules interact across a single engine step
- Provides a real (Qwen2-0.5B) executor alongside a pure-Python fake for comparative study
- Isolates serving-layer concerns from core engine concerns so each can be studied independently

This project fills that gap with a deliberately minimal implementation that mirrors vLLM's module boundaries, naming, and architecture.

**中文：**
像 vLLM 这样的 LLM 服务系统通过连续批处理（Continuous Batching）实现高 GPU 利用率——这种技术允许新请求在每个 token 生成步骤抢占 decode batch。然而，实现这一技术的生产代码库非常庞大（仅 vLLM 就超过 10 万行），导致工程师和学生难以理解其核心架构。

目前缺少一个简洁的教学参考，能够：
- 将连续批处理分解为各个基础模块（Sequence、Scheduler、KV Cache、Executor、Engine）
- 展示这些模块在单个引擎步骤中的交互方式
- 同时提供真实（Qwen2-0.5B）执行器和纯 Python 假执行器，方便对比学习
- 将服务层关注点与核心引擎关注点分离，使两者可以独立学习

本项目的目标是提供一个刻意保持极简的实现，同时镜像 vLLM 的模块边界、命名和架构。

## Solution / 解决方案

**English:**
A three-phase, layered implementation of vLLM's Continuous Batching engine:

- **Phase 1 (Core Engine):** Pure-Python core with Sequence/SequenceGroup data model, three-layer PagedAttention KV Cache (BlockAllocator → BlockManager → BlockTable), a six-phase continuous-batching Scheduler (decode-first, token-budget), FakeModelExecutor, LLMEngine/EngineCore split, and MetricsCollector.
- **Phase 2 (Real-World Optimizations):** Chunked Prefill, Prefix Cache (block-level hashing, read-only probe, shared allocation, Copy-on-Write ready), and a real Qwen2-0.5B Executor via HuggingFace Transformers.
- **Phase 3 (Serving Extension):** Independent HTTP/SSE serving layer with rate limiting (RPM/TPM), request cancellation, timeout enforcement, and client disconnect lifecycle management.

All three phases share the same core engine API — serving is an extension, not a fork.

**中文：**
一个三层架构的 vLLM 连续批处理引擎实现：

- **Phase 1（核心引擎）：** 纯 Python 核心，包括 Sequence/SequenceGroup 数据模型、三层 PagedAttention KV Cache（BlockAllocator → BlockManager → BlockTable）、六阶段连续批处理调度器（decode-first、token-budget）、FakeModelExecutor、LLMEngine/EngineCore 分层和 MetricsCollector。
- **Phase 2（生产级优化）：** 分块 Prefill、前缀缓存（block-level hashing、只读探测、共享分配、写时复制就绪）和通过 HuggingFace Transformers 运行的 Qwen2-0.5B 真实执行器。
- **Phase 3（服务扩展层）：** 独立的 HTTP/SSE 服务层，提供速率限制（RPM/TPM）、请求取消、超时强制终止和客户端断连生命周期管理。

三个阶段共享同一核心引擎 API —— 服务层是扩展，不是分支。

## User Stories / 用户故事

1. **As a student learning LLM serving systems**, I want to run a minimal end-to-end engine demo with zero dependencies, so that I can see Continuous Batching in action without setting up GPU infrastructure.

2. **作为一名学习 LLM 服务系统的学生**，我希望运行一个零依赖的端到端引擎演示，从而无需配置 GPU 基础设施就能看到连续批处理的实际运行。

3. **As an engineer studying vLLM internals**, I want a module-by-module mapping to vLLM's source, so that I can transfer my understanding directly to the production codebase.

4. **作为一名研究 vLLM 内部机制的工程师**，我希望获得一个模块级到 vLLM 源码的映射，从而将我的理解直接迁移到生产代码库。

5. **As a student**, I want to trace the full lifecycle of a request through the engine (add_request → schedule → prefill → decode → finish), so that I understand how each layer contributes to the overall process.

6. **作为一名学生**，我希望追踪一个请求在引擎中的完整生命周期（add_request → schedule → prefill → decode → finish），从而理解每层在整个过程中的作用。

7. **As an engineer debugging a scheduler issue**, I want the scheduler to report rich debug information (token budget, ignored reasons, rejected reasons) at each step, so that I can diagnose scheduling anomalies without stepping through code.

8. **作为一名调试调度器问题的工程师**，我希望调度器在每一步报告丰富的调试信息（token budget、忽略原因、拒绝原因），从而无需单步调试就能诊断调度异常。

9. **As a performance engineer**, I want metrics on TTFT (Time to First Token), TPOT (Time Per Output Token), and throughput (req/s, tok/s), so that I can quantify the impact of scheduling and cache decisions.

10. **作为一名性能工程师**，我希望获得 TTFT（首 token 延迟）、TPOT（每输出 token 时间）和吞吐量（req/s、tok/s）指标，从而量化调度和缓存决策的影响。

11. **As a researcher studying Continuous Batching**, I want to swap between a fake executor (pure CPU arithmetic) and a real model executor (Qwen2-0.5B), so that I can separate algorithmic correctness from model execution performance.

12. **作为一名研究连续批处理的研究人员**，我希望在假执行器（纯 CPU 运算）和真实模型执行器（Qwen2-0.5B）之间切换，从而将算法正确性与模型执行性能分离。

13. **As a student**, I want chunked prefill to split long prompts across multiple steps, so that I can observe how short requests avoid starvation behind long prompts.

14. **作为一名学生**，我希望分块 prefill 能将长 prompt 拆到多个步骤处理，从而观察短请求如何避免被长 prompt 阻塞。

15. **As a performance engineer**, I want a prefix cache that shares KV blocks across requests with common prefixes, so that I can measure the reduction in prefill computation and memory usage.

16. **作为一名性能工程师**，我希望前缀缓存能跨请求共享相同前缀的 KV 块，从而衡量 prefill 计算量和内存用量的减少。

17. **As a system designer**, I want to see the stage-level breakdown of end-to-end latency (request queue waiting, scheduler, KV allocation, prefill, decode, KV release, metrics), so that I can identify the bottleneck in any given workload.

18. **作为一名系统设计师**，我希望看到端到端延迟的阶段级分解（请求排队等待、调度器、KV 分配、prefill、decode、KV 释放、指标更新），从而识别任意工作负载下的瓶颈。

19. **As a DevOps engineer deploying the system**, I want rate limiting (RPM/TPM) and request timeout enforcement, so that the engine is protected from misbehaving clients.

20. **作为一名部署系统的 DevOps 工程师**，我希望有速率限制（RPM/TPM）和请求超时强制终止功能，从而保护引擎免受异常客户端的影响。

21. **As a platform developer**, I want SSE streaming so that clients receive tokens incrementally, improving perceived latency for interactive applications.

22. **作为一名平台开发人员**，我希望支持 SSE 流式输出，使客户端逐步接收 token，从而改善交互式应用的感知延迟。

23. **As a platform developer**, I want the serving layer to detect client disconnects and automatically cancel orphaned engine requests, so that compute resources are not wasted on clients that have already left.

24. **作为一名平台开发人员**，我希望服务层能检测客户端断连并自动取消孤儿引擎请求，从而避免将计算资源浪费在已离开的客户端上。

25. **As an engineer debugging resource leaks**, I want a fault injection test suite that exercises OOM, queue overflow, block exhaustion, and client disconnect scenarios, so that I am confident the engine recovers cleanly from every failure mode.

26. **作为一名调试资源泄漏的工程师**，我希望有一个故障注入测试套件，能够验证 OOM、队列溢出、块耗尽和客户端断连等场景，从而确信引擎能从所有故障模式中干净地恢复。

27. **As a student comparing allocation strategies**, I want on-demand block allocation (allocating blocks only when needed) alongside eager allocation commentary, so that I understand the memory-efficiency trade-offs.

28. **作为一名对比分配策略的学生**，我希望在按需分配（仅在需要时分配块）的同时附带预分配方案的对比说明，从而理解内存效率的权衡。

29. **As an engineer debugging the KV cache**, I want memory tracing (allocated/freed blocks per step, free list dump, block table dump), so that I can verify that block reference counts and on-demand allocation behave correctly.

30. **作为一名调试 KV Cache 的工程师**，我希望有内存追踪功能（每步分配/释放的块、空闲列表转储、块表转储），从而验证块引用计数和按需分配的正确行为。

## Implementation Decisions / 实现决策

### Architecture / 架构

- **Three-layer KV Cache (BlockAllocator → BlockManager → BlockTable):** Mirrors vLLM's PagedAttention memory management. BlockAllocator is the low-level free-list with reference-counted shared blocks; BlockManager coordinates per-sequence allocation and prefix cache integration; BlockTable provides logical-to-physical mapping with shared-block tracking. This separation allows each layer to be tested and understood independently.

- **三层 KV Cache（BlockAllocator → BlockManager → BlockTable）：** 镜像 vLLM 的 PagedAttention 内存管理。BlockAllocator 是底层空闲列表，带引用计数的共享块；BlockManager 协调 per-sequence 分配和前缀缓存集成；BlockTable 提供逻辑到物理块的映射并跟踪共享块。这种分离使每一层可以独立测试和理解。

- **LLMEngine / EngineCore split:** EngineCore owns the scheduler and executor and runs the inner step loop. LLMEngine provides the public API (`add_request`, `step`, `run_until_done`, `get_outputs`) and handles output capture and logging. This mirrors vLLM's public API vs. inner-loop separation and allows EngineCore to be tested without API-layer concerns.

- **LLMEngine / EngineCore 分层：** EngineCore 持有调度器和执行器，运行内部 step 循环。LLMEngine 提供公共 API（`add_request`、`step`、`run_until_done`、`get_outputs`），处理输出捕获和日志。这镜像了 vLLM 公共 API 与内部循环的分离，使 EngineCore 可以在没有 API 层干扰的情况下被测试。

- **Six-phase Scheduler:** The scheduler runs six phases per step: (1) finish completed sequences, (2) categorize running groups into decode vs. prefill-continue, (3) deduct decode budget first (decode-first policy), (4) continue chunked prefill for mid-prefill groups, (5) admit new waiting groups with prefix cache awareness, (6) compute token counts and debug reason. The decode-first policy ensures decode latency is not starved by prefill.

- **六阶段调度器：** 调度器每步运行六个阶段：（1）终止已完成序列，（2）将运行中的组分类为 decode 或 prefill-continue，（3）优先扣除 decode 预算（decode-first 策略），（4）对中间状态的 PREFILL 组继续分块 prefill，（5）以前缀缓存感知的方式准入新等待组，（6）计算 token 计数和调试原因。decode-first 策略确保 decode 延迟不被 prefill 阻塞。

- **SequenceGroup / Sequence split:** The SequenceGroup owns user-level request metadata (prompt, sampling params, arrival time). Each Sequence owns per-generation state (token buffers, KV block table, prefill cursor, generation timing). Queue pools hold SequenceGroups; Scheduler and ScheduleResult operate on groups. This mirrors vLLM's data model and enables future parallel generation (multiple sequences per request).

- **SequenceGroup / Sequence 分离：** SequenceGroup 拥有用户级请求元数据（prompt、采样参数、到达时间）。每个 Sequence 拥有自己的生成状态（token 缓冲区、KV 块表、prefill 游标、生成时间戳）。队列池保存 SequenceGroup；调度器和 ScheduleResult 以组为单位操作。这镜像了 vLLM 的数据模型，并为未来的并行生成（每个请求多个 sequence）提供了支持。

- **On-demand block allocation:** Sequences start with zero blocks. Blocks are allocated one at a time through `BlockManager.ensure_block()`, called by the executor during prefill and decode. This differs from vLLM's eager allocation and is a deliberate simplification for educational clarity — the trade-off is lower memory waste at the cost of more frequent allocator calls.

- **按需块分配：** Sequence 从零个块开始。块通过 `BlockManager.ensure_block()` 逐个分配，由执行器在 prefill 和 decode 期间调用。这与 vLLM 的预分配不同，是为了教学清晰性而刻意简化——代价是更频繁的分配器调用，换来的好处是更低的内存浪费。

- **Prefix Cache with read-only probe:** `BlockManager.probe_prefix_cache()` is called by the scheduler before budget computation to determine how many prompt tokens are already cached — without modifying reference counts. This allows the scheduler to make admission decisions without side effects. Only when `allocate_for_seq()` confirms admission does the cache share the matching blocks via `increment_ref()`.

- **带只读探测的前缀缓存：** `BlockManager.probe_prefix_cache()` 在预算计算前被调度器调用，用于确定有多少 prompt token 已缓存——且不修改引用计数。这使得调度器可以在无副作用的情况下做准入决策。只有 `allocate_for_seq()` 确认准入后，缓存才通过 `increment_ref()` 共享匹配的块。

- **Executor Protocol:** The `Executor` abstract protocol defines a uniform interface (tokenize, prefill, decode, cleanup_sequence, KV callbacks) that both `FakeModelExecutor` and `QwenExecutor` implement. This enables phase-by-phase learning — understand the algorithm with the fake executor, then swap in the real model without changing any engine code.

- **Executor 协议：** `Executor` 抽象协议定义了一个统一接口（tokenize、prefill、decode、cleanup_sequence、KV 回调），`FakeModelExecutor` 和 `QwenExecutor` 都实现了该接口。这使得学习者可以分阶段学习——先用假执行器理解算法，然后切换为真实模型而不改动任何引擎代码。

- **MetricsCollector:** Centralized metrics collection — not log lines scattered across modules. Tracks TTFT (first_token_time − arrival_time), TPOT ((finish_time − first_token_time) / max(output_tokens−1, 1)), throughput (req/s, tok/s for both wall-clock and active-time), KV utilization (peak and average), block utilization (tokens per block), and scheduler latency.

- **MetricsCollector（指标采集器）：** 集中式的指标采集，而非日志散落在各个模块中。跟踪 TTFT（first_token_time − arrival_time）、TPOT（(finish_time − first_token_time) / max(output_tokens−1, 1)）、吞吐量（按挂钟时间和活跃时间计算的 req/s 和 tok/s）、KV 利用率（峰值和平均）、块利用率（每块 token 数）和调度器延迟。

- **Serving as independent extension:** The serving layer (`mini_vllm/serving/`) is a separate package that drives the engine through the `LLMEngine` public API only. It handles network concerns (HTTP, SSE, WebSocket lifecycle) and service governance (rate limiting, admission control). This separation ensures core engine tests do not require HTTP infrastructure, and serving tests can mock the engine.

- **服务层作为独立扩展：** 服务层（`mini_vllm/serving/`）是一个独立的包，仅通过 `LLMEngine` 公共 API 驱动引擎。它处理网络关注点（HTTP、SSE、WebSocket 生命周期）和服务治理（速率限制、准入控制）。这种分离确保核心引擎测试不需要 HTTP 基础设施，服务测试也可以 mock 引擎。

- **StageProfiler for educational profiling:** A lightweight (pure Python `time.time()` + context managers) profiler that decomposes end-to-end latency into stages. Uses `record_raw()` for direct timing injection (key for `request_queue_waiting` and `kv_cache_allocation` which are measured outside the step function) and context managers for `scheduler_step`, `prefill`, `decode`, `executor_forward`, `metrics_update`. Outputs sorted-by-total-time report with bottleneck detection hints.

- **StageProfiler 教学性能分析器：** 一个轻量级（纯 Python `time.time()` + context manager）分析器，将端到端延迟分解为各个阶段。使用 `record_raw()` 直接注入计时（对于 `request_queue_waiting` 和 `kv_cache_allocation` 这些在 step 函数外部测量的阶段关键），使用 context manager 包裹 `scheduler_step`、`prefill`、`decode`、`executor_forward`、`metrics_update`。输出按总时间排序的报告，附带瓶颈检测提示。

### Module interfaces / 模块接口

- `Scheduler.schedule() → ScheduleResult` — single entry point, returns structured result with prefill/decode/finished/ignored/rejected groups, token counts, budget remaining, and debug reason string.
- `Executor.prefill(sequences)` — handles chunk-aware prefill from cursor position; `Executor.decode(sequences)` — one token per sequence per call.
- `BlockManager.ensure_block(seq, position) → int` — returns physical block ID, allocating on demand; `BlockManager.free(seq_id)` — decrements ref counts, releases when zero.
- `BlockAllocator.allocate(n) → List[int] | None` — returns block IDs or None on OOM; `BlockAllocator.increment_ref(pid)` and ref-count-aware `free()` support shared blocks.
- `MetricsCollector.record_step()` + `MetricsCollector.report() → dict` — per-step recording, final aggregation.
- `StageProfiler.record(stage_name)` context manager — wraps any block with timing; `report() → dict` for programmatic access.

- `Scheduler.schedule() → ScheduleResult` — 单一入口，返回结构化结果，包含 prefill/decode/finished/ignored/rejected 各组、token 计数、剩余预算和调试原因字符串。
- `Executor.prefill(sequences)` — 处理分块感知的 prefill，从游标位置开始；`Executor.decode(sequences)` — 每条序列每步一个 token。
- `BlockManager.ensure_block(seq, position) → int` — 返回物理块 ID，按需分配；`BlockManager.free(seq_id)` — 递减引用计数，归零时释放。
- `BlockAllocator.allocate(n) → List[int] | None` — 返回块 ID 列表，OOM 时返回 None；`BlockAllocator.increment_ref(pid)` 和引用计数感知的 `free()` 支持共享块。
- `MetricsCollector.record_step()` + `MetricsCollector.report() → dict` — 每步记录，最终聚合。
- `StageProfiler.record(stage_name)` context manager — 为任意代码块添加计时；`report() → dict` 支持程序化访问。

## Testing Decisions / 测试决策

English:
The project tests at **three seams**, each testing at the highest practical level of integration. The guiding principle: **test external behavior, not implementation details** — verify what the module produces (outputs, states, metrics), not how it internally sequences loops.

| Seam | What it tests | Prior art |
|------|--------------|-----------|
| **Module unit seam** (`test_kv_cache_manager.py`, `test_prefix_cache.py`, `test_metrics.py`, `test_stage_profiler.py`) | BlockAllocator allocate/free/OOM/ref-count, BlockTable mapping, PrefixCache insert/lookup/probe, MetricsCollector formula correctness, StageProfiler timing aggregation | Test helper `_make_seq()` creates bare Sequences; tests call allocator/manager methods directly and assert on states |
| **Scheduler integration seam** (`test_scheduler.py`) | Scheduler admission, decode-first priority, chunked prefill, token budget enforcement, ignored/rejected reasons, sequence lifecycle | Test helpers `_make_config()`, `_make_group()`, `_make_scheduler()` construct a real Scheduler with real BlockManager + BlockAllocator + RequestQueue; tests call `schedule()` and assert on `ScheduleResult` |
| **Engine integration seam** (`test_engine.py`) | End-to-end correctness: `add_request` → step loop → output collection, continuous batching with staggered arrivals, OOM recovery, KV cache writes and fake logit dependence on cache content | Test helper `_engine()` creates a real `LLMEngine` with real FakeModelExecutor; tests add requests, drive steps, and assert on outputs, token counts, and engine state |
| **Serving integration seam** (`test_serving_layer.py`, `test_fault_injection.py`) | HTTP SSE streaming, RPM/TPM rate limiting, request cancellation, timeout, client disconnect lifecycle, OOM/queue exhaust/block exhaust recovery | Tests use a real HTTP server with `httpx` client; fault injection tests exercise every failure mode and verify resource cleanup |

These seams mirror the learning path: start with a single module, combine into the scheduler, integrate with the engine, and finally verify the HTTP layer.

中文：
本项目在**三个测试接缝**上进行测试，每个接缝都在实际可行的最高集成级别进行测试。指导原则：**只测试外部行为，不测试实现细节**——验证模块产生了什么（输出、状态、指标），而不是它如何在内部编排循环。

| 测试接缝 | 测试内容 | 现有参考 |
|---------|---------|---------|
| **模块单元接缝**（`test_kv_cache_manager.py`、`test_prefix_cache.py`、`test_metrics.py`、`test_stage_profiler.py`） | BlockAllocator allocate/free/OOM/引用计数、BlockTable 映射、PrefixCache insert/lookup/probe、MetricsCollector 公式正确性、StageProfiler 计时聚合 | 测试辅助函数 `_make_seq()` 创建裸 Sequence；测试直接调用 allocator/manager 方法并断言状态 |
| **调度器集成接缝**（`test_scheduler.py`） | 调度器准入、decode-first 优先级、分块 prefill、token budget 强制执行、ignored/rejected 原因、序列生命周期 | 测试辅助函数 `_make_config()`、`_make_group()`、`_make_scheduler()` 构建一个真实的 Scheduler 加真实的 BlockManager + BlockAllocator + RequestQueue；测试调用 `schedule()` 并断言 `ScheduleResult` |
| **引擎集成接缝**（`test_engine.py`） | 端到端正确性：`add_request` → step 循环 → 输出收集、有交错到达的连续批处理、OOM 恢复、KV cache 写入和假 logits 对缓存内容的依赖 | 测试辅助函数 `_engine()` 创建一个真实的 `LLMEngine` 加真实的 FakeModelExecutor；测试添加请求、驱动步骤、断言输出、token 计数和引擎状态 |
| **服务集成接缝**（`test_serving_layer.py`、`test_fault_injection.py`） | HTTP SSE 流式输出、RPM/TPM 速率限制、请求取消、超时、客户端断连生命周期、OOM/队列耗尽/块耗尽恢复 | 测试使用真实的 HTTP 服务器加 `httpx` 客户端；故障注入测试覆盖每种故障模式并验证资源清理 |

这些接缝反映了学习路径：从单一模块开始，组合为调度器，集成到引擎中，最后验证 HTTP 层。

## Out of Scope / 非目标

- **Real GPU kernels (CUDA / PagedAttention CUDA kernel):** The project simulates PagedAttention's memory management but uses fake KV cache values for educational purposes. CUDA kernel development is outside the project's scope.
- **真正的 GPU kernels（CUDA / PagedAttention CUDA kernel）：** 项目模拟了 PagedAttention 的内存管理，但使用假的 KV cache 值做教学用途。CUDA kernel 开发不在项目范围内。
- **Preemption / swapping:** GPU memory pressure handling (swapping KV blocks to CPU) is not implemented. The project uses a fixed number of GPU blocks with OOM as the failure mode.
- **抢占/交换（swap）：** GPU 内存压力处理（将 KV 块交换到 CPU）未实现。项目使用固定数量的 GPU 块，以 OOM 作为故障模式。
- **Speculative decoding:** Multi-token speculative generation is not implemented.
- **投机解码：** 多 token 投机生成未实现。
- **Multi-node / distributed serving:** The project is single-process.
- **多节点/分布式服务：** 项目为单进程。
- **Production-grade prefix cache eviction:** The prefix cache only grows — blocks are removed only when their last reference is freed. No LRU/TTL eviction is implemented.
- **生产级前缀缓存淘汰策略：** 前缀缓存只增不减——块仅在最后一个引用被释放时移除。没有实现 LRU/TTL 淘汰。
- **Production GPU profiling:** The StageProfiler is for teaching and interview demonstrations, not a replacement for Nsight Systems or PyTorch profiler.
- **生产级 GPU 性能分析：** StageProfiler 用于教学和面试演示，不是 Nsight Systems 或 PyTorch Profiler 的替代品。

## Further Notes / 补充说明

1. **Learning path:** The project is designed to be studied in phase order. Phase 1 (Core Engine) can be understood without GPU knowledge. Phase 2 adds real-world optimizations one at a time. Phase 3 can be studied independently or skipped entirely.

2. **学习路径：** 项目按阶段顺序设计学习。Phase 1（核心引擎）无需 GPU 知识即可理解。Phase 2 逐一增加生产级优化。Phase 3 可独立学习或完全跳过。

3. **Zero-friction start:** `examples/demo_fake_engine.py` requires only Python 3.10+ with zero pip dependencies. This is intentional — a student should see "Hello, world!" output from the Continuous Batching engine within 30 seconds of cloning.

4. **零摩擦启动：** `examples/demo_fake_engine.py` 只需要 Python 3.10+，零 pip 依赖。这是刻意为之——学生在 clone 后 30 秒内就能看到连续批处理引擎输出 "Hello, world!"。

5. **vLLM mapping:** The `docs/VLLM_Mapping.md` document provides a module-by-module mapping to vLLM's actual source structure, enabling direct transfer of understanding.

6. **vLLM 映射：** `docs/VLLM_Mapping.md` 文档提供了模块级到 vLLM 实际源码结构的映射，使理解可以直接迁移。

7. **Serving independence:** The serving layer is a Phase 3 concern and is deliberately separable. Changes to the serving layer should never require changes to the core engine, and vice versa. This is enforced architecturally — serving only accesses the engine through `LLMEngine.add_request()`, `step()`, and `cancel_request()`.

8. **服务层独立性：** 服务层是 Phase 3 的关注点，刻意设计为可分离的。服务层的变更不应要求核心引擎改变，反之亦然。这在架构上得到保证——服务层仅通过 `LLMEngine.add_request()`、`step()` 和 `cancel_request()` 访问引擎。

9. **Fault injection testing:** The fault injection test suite (`test_fault_injection.py`) is a distinguishing feature of this project. It systematically exercises every resource-exhaustion and lifecycle scenario to verify that the engine recovers cleanly — crucial for interview discussions about production reliability.

10. **故障注入测试：** 故障注入测试套件（`test_fault_injection.py`）是本项目的一个特色功能。它系统地测试每种资源耗尽和生命周期场景，以验证引擎能够干净地恢复——这对面试中讨论生产可靠性至关重要。
