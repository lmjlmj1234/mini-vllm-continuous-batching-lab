---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 07: Chunked Prefill
# Issue 07：分块 Prefill

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

Chunked prefill support: long prompts are split across multiple engine steps so that decode sequences are not starved. The scheduler integrates chunk budget into the existing token-budget and decode-first logic. Both executors (fake and Qwen) support cursor-based chunk-aware prefill.

分块 prefill 支持：长 prompt 被拆成多个引擎步骤处理，避免 decode 序列被阻塞。调度器将 chunk budget 集成到现有的 token-budget 和 decode-first 逻辑中。两个执行器（fake 和 Qwen）都支持基于游标的分块感知 prefill。

## Vertical slice / 垂直切片描述

This slice extends the scheduler and executors so that a long prompt is incrementally prefilled over several steps. Mid-prefill sequences carry a `prefill_cursor` tracking how many prompt tokens have been written to KV cache so far. Each step processes at most `max_prefill_chunk_size` new tokens.

本切片扩展了调度器和执行器，使长 prompt 可以分多步增量完成 prefill。中间状态的 prefill 序列带有 `prefill_cursor`，追踪已写入 KV cache 的 prompt token 数量。每步最多处理 `max_prefill_chunk_size` 个新 token。

## Acceptance criteria / 验收标准

- [x] `Config.chunked_prefill_enabled` boolean flag controls whether chunking is active
- [x] `Config.max_prefill_chunk_size` sets the max prompt tokens processed per prefill step
- [x] Scheduler Phase 4: mid-prefill groups continue from their cursor position, consuming chunk budget
- [x] Scheduler Phase 5: new waiting groups are admitted with chunk budget applied
- [x] `Sequence.prefill_cursor` tracks how many prompt tokens have been written to KV cache
- [x] `Sequence.is_prefill_finished` property returns `True` when `prefill_cursor ≥ prompt_length`
- [x] `FakeModelExecutor.prefill()` processes prompt tokens from cursor to `min(prompt_length, cursor + chunk_size)`
- [x] `QwenExecutor.prefill()` processes prompt tokens from cursor with proper attention mask for incremental prefill
- [x] Sequences remain in `Status.PREFILL` until their full prompt is processed (not prematurely set to RUNNING)
- [x] Decode continues normally during ongoing chunked prefill (decode-first budget policy)

## Key code / 核心代码

- `mini_vllm/config.py` — `chunked_prefill_enabled`, `max_prefill_chunk_size`
- `mini_vllm/scheduler/scheduler.py` — Phase 4 (continue) & Phase 5 (admit with chunk)
- `mini_vllm/executor/executor.py` — `FakeModelExecutor.prefill()` cursor logic
- `mini_vllm/executor/qwen_executor.py` — `QwenExecutor.prefill()` cursor + attention mask logic
- `mini_vllm/sequence/sequence.py` — `Sequence.prefill_cursor`, `is_prefill_finished`

## Key tests / 核心测试

- `tests/test_scheduler.py` — `test_chunked_prefill()` validates multi-step prefill with chunk boundaries
- `tests/test_engine.py` — engine integration validates chunk-aware prefill via engine step loop
- `tests/test_prefix_cache.py` — `test_cache_hit_reduces_prefill_tokens_in_scheduler()` verifies prefix cache integration with chunk budget

## Blocked by / 前置依赖

- [03: Core Scheduler](./03-core-scheduler.md) — scheduler must already handle prefill/decode lifecycle
- [04: Fake Model & Executor](./04-fake-model-executor.md) — executor must support cursor-based prefill
