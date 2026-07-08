# mini-vLLM — Resume Project Description

> **AI Infra · LLM Serving · Continuous Batching · PagedAttention · Prefix Cache**

---

## 背景

从零复现 vLLM 核心架构（Continuous Batching + PagedAttention + Prefix Cache），完整实现 Scheduler、BlockManager、Executor 三层架构。支持 fake executor 测试和 Qwen2-0.5B 真实模型推理双模式。79 个单元测试验证正确性。

## 简历版本（4~6 Bullets）

- **Designed and implemented a continuous-batching LLM inference engine from scratch**, replicating vLLM's core architecture including a 6-phase scheduler with decode-first priority, token-budget admission control, and chunked prefill for long prompts.
- **Built a PagedAttention-based KV cache manager with on-demand block allocation**, eliminating memory waste by allocating physical blocks at write time rather than admission time, with BlockTable mapping logical→physical addresses.
- **Implemented a Scheduler-aware prefix cache with two-phase probe-and-allocate mechanism**: probe phase is read-only (no ref_count side effects) for accurate token budget calculation, allocate phase increments ref_count and attaches shared blocks, achieving zero-copy KV sharing across requests with identical prompt prefixes.
- **Developed a reference-counted BlockAllocator** that safely manages shared block lifecycles — blocks are freed only when ref_count reaches zero, preventing use-after-free when multiple sequences share cached KV blocks.
- **Integrated HuggingFace Qwen2-0.5B as a real model backend** via a Protocol-based Executor abstraction, demonstrating model-agnostic scheduler design where swapping executor backends requires zero changes to scheduling or memory management code.
- **Achieved 79 passing unit tests covering** scheduler phases, token budget accounting, chunked prefill, prefix cache probe/allocate semantics, ref_count lifecycle, stale entry detection, and end-to-end engine integration with metrics (TTFT, TPOT, throughput, prefix cache hit rate).

---

## 项目结构

```
mini_vllm/
├── engine/           # LLMEngine, EngineCore, MetricsCollector
├── scheduler/        # Scheduler (6-phase), ScheduleResult
├── cache/            # BlockAllocator, BlockTable, BlockManager, PrefixCache
├── executor/         # FakeModelExecutor, QwenExecutor (Protocol-based)
├── worker/           # FakeWorker, QwenWorker
├── sequence/         # Sequence, SequenceGroup, RequestQueue, Status
├── model/            # FakeModel
└── config.py         # Global configuration

tests/                # 79 tests across scheduler, cache, prefix cache, engine
```

## 关键指标

| 指标 | 值 |
|------|-----|
| 测试覆盖 | 79 tests, all passing |
| Executor 支持 | Fake (educational) + Qwen2-0.5B (real) |
| Block level sharing | Zero-copy KV via ref_count |
| Prefix cache | Hash-based, block-level, stale-safe |
| Config knobs | max_num_seqs, max_batched_tokens, block_size, chunk_size |
| Stage Profiler | 10-stage breakdown for TTFT/TPOT bottleneck analysis |

## 面试亮点：Stage Breakdown Profiling

新增轻量级 stage profiler，可在一次 serving 请求中将端到端耗时拆解为：

- **request_queue_waiting** — 队列等待
- **scheduler_step** — 调度开销
- **kv_cache_allocation** — KV block 分配
- **prefix_cache_lookup** — 前缀缓存查询
- **executor_forward** — 模型推理（prefill + decode）
- **prefill** / **decode** — 分别计时
- **kv_cache_release** — block 释放
- **metrics_update** — 指标采集
- **engine_step_total** — 引擎 step 总耗时

支持 fake executor（纯 CPU、无依赖）和 Qwen executor（真实模型）双模式。
输出瓶颈提示，帮助面试中系统性地解释 TTFT/TPOT/P99 的瓶颈来源。

## 技术栈

Python, HuggingFace Transformers, PyTorch, PagedAttention, Continuous Batching

---

*项目地址: https://github.com/lmjlmj1234/mini-vllm-continuous-batching-lab*
