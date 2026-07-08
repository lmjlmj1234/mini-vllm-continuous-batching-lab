---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 05: Engine Step Loop & Public API
# Issue 05：引擎步骤循环与公共 API

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

The inner engine loop (`EngineCore`) and the public-facing API (`LLMEngine`). `EngineCore` owns the scheduler and executor and runs one step at a time: check timeouts → schedule → prefill → decode → cleanup → record metrics. `LLMEngine` provides `add_request()`, `step()`, `run_until_done()`, `get_outputs()`, `cancel_request()`, and step-level debug printing.

内部引擎循环（`EngineCore`）和面向外部的 API（`LLMEngine`）。`EngineCore` 持有调度器和执行器，每次运行一个步骤：检查超时 → 调度 → prefill → decode → 清理 → 记录指标。`LLMEngine` 提供 `add_request()`、`step()`、`run_until_done()`、`get_outputs()`、`cancel_request()` 和步骤级调试打印。

## Vertical slice / 垂直切片描述

This slice wires all previous layers (data model → KV cache → scheduler → executor) into a running engine. A user can add prompts, drive the step loop, and collect generated text. Engine config (`Config` dataclass) centralizes all tuning knobs.

本切片将所有之前的层（数据模型 → KV Cache → 调度器 → 执行器）连接成一个可运行的引擎。用户可以添加 prompt、驱动步骤循环、收集生成的文本。引擎配置（`Config` dataclass）集中管理所有调优参数。

## Acceptance criteria / 验收标准

- [x] `Config` dataclass provides all engine tuning knobs: `max_num_seqs`, `max_num_batched_tokens`, `max_num_prefill_tokens`, `chunked_prefill_enabled`, `max_prefill_chunk_size`, `decode_first`, `block_size`, `num_gpu_blocks`, `max_model_len`, `vocab_size`, `executor_type`, `print_step_events`, `memory_trace`, serving params (`max_queue_len`, `max_num_streams`, `rate_limit_rpm/tpm`, `request_timeout_s`)
- [x] `Config.__post_init__()` validates all parameters are positive and `executor_type` is `"fake"` or `"qwen"`
- [x] `LLMEngine.__init__()` builds the full stack: worker → executor (wired with BlockManager callbacks) → BlockAllocator → BlockManager → Scheduler → EngineCore
- [x] `LLMEngine.add_request()` creates a SequenceGroup, tokenizes the prompt, enqueues to RequestQueue, returns request ID
- [x] `LLMEngine.step()` runs one engine iteration, captures finished outputs via `Executor.detokenize()`, prints step events
- [x] `LLMEngine.run_until_done()` loops `step()` until all requests finished, returns `{request_id: output_text}`
- [x] `LLMEngine.cancel_request()` cancels a request by ID, freeing all engine resources
- [x] `EngineCore.step()` runs: timeout check → scheduler.schedule() → executor.prefill() → executor.decode() → cleanup → metrics recording
- [x] `EngineCore._check_timeouts()` cancels requests exceeding `request_timeout_s`
- [x] `EngineCore.cancel_request()` marks sequences CANCELLED, frees blocks, cleans up executor, updates metrics
- [x] Step event logging: prints waiting/running/scheduled-prefill/scheduled-decode/ignored/rejected/finished groups with detailed state

## Key code / 核心代码

- `mini_vllm/config.py` — Config dataclass
- `mini_vllm/engine/engine_core.py` — EngineCore
- `mini_vllm/engine/engine.py` — LLMEngine

## Key tests / 核心测试

- `tests/test_engine.py` — end-to-end: add requests, step loop, output correctness, continuous batching with staggered arrivals, OOM recovery, KV write tracking, KV output dependence, ScheduleResult fields
- `tests/test_metrics.py` — engine integration tests that verify TTFT, TPOT, throughput via the metrics pipeline

## Blocked by / 前置依赖

- [01: Sequence & Request Data Model](./01-sequence-data-model.md) — engine uses SequenceGroup/Sequence/RequestQueue
- [02: Paged KV Cache](./02-paged-kv-cache.md) — engine creates BlockManager/BlockAllocator
- [03: Core Scheduler](./03-core-scheduler.md) — EngineCore owns Scheduler
- [04: Fake Model & Executor](./04-fake-model-executor.md) — EngineCore owns Executor
