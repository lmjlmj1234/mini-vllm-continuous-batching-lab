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

## 技术栈

Python, HuggingFace Transformers, PyTorch, PagedAttention, Continuous Batching

---

*项目地址: https://github.com/lmjlmj1234/mini-vllm-continuous-batching-lab*
