---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 10: Stage-Level Profiler
# Issue 10：阶段级分析器

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

A lightweight, zero-dependency stage-level profiler (`StageProfiler`) that decomposes end-to-end serving request latency into individual stages: request queue waiting, scheduler step, KV cache allocation, prefix cache lookup, prefill, decode, executor forward, KV cache release, metrics update, and engine step total.

一个轻量级、零依赖的阶段级分析器（`StageProfiler`），将端到端服务请求延迟分解为各个阶段：请求排队等待、调度器步骤、KV Cache 分配、前缀缓存查询、prefill、decode、执行器前向、KV Cache 释放、指标更新和引擎步骤总计时。

## Vertical slice / 垂直切片描述

This slice provides a complete profiling tool with context-manager API, start/end API, direct timing injection, and a formatted report with bottleneck detection hints. Wired into EngineCore for automatic profiling of every engine step.

本切片提供一个完整的性能分析工具，包含 context-manager API、start/end API、直接计时注入和带瓶颈检测提示的格式化报告。集成到 EngineCore 中，可自动分析每个引擎步骤。

## Acceptance criteria / 验收标准

- [x] Context-manager API: `with profiler.record("stage_name"):` — times the block and records duration in seconds
- [x] Start/end API: `profiler.start("stage_name")` → `profiler.end("stage_name")` for manual timing
- [x] `record_raw(stage, duration_s)` for direct timing injection (used for queue waiting and KV allocation)
- [x] `report()` returns structured dict with per-stage: count, total_ms, avg_ms, max_ms, percent_of_total
- [x] `print_report()` formats as a sorted table with bottleneck detection hints
- [x] Stages sorted by total_ms descending in output
- [x] Percentage calculated relative to `engine_step_total` (falls back to sum of all stages)
- [x] EngineCore integration: 9 profiler calls per `step()` covering all stages
- [x] `request_queue_waiting` recorded per newly admitted sequence (arrival → first_scheduled_time)
- [x] `prefix_cache_lookup` recorded per probe_prefix_cache() call
- [x] `kv_cache_allocation` recorded per ensure_block() allocation
- [x] `kv_cache_release` recorded per free() call
- [x] Zero external dependencies — pure Python `time.time()`
- [x] `reset()` clears all recorded data
- [x] Exception-safe: context manager records timing even if the wrapped block raises

## Key code / 核心代码

- `mini_vllm/engine/stage_profiler.py` — StageProfiler
- `mini_vllm/engine/engine_core.py` — profiler integration in `step()`
- `mini_vllm/cache/manager.py` — profiler calls in `ensure_block()`, `free()`, `probe_prefix_cache()`
- `examples/demo_stage_breakdown.py` — demonstration script

## Key tests / 核心测试

- `tests/test_stage_profiler.py` — 13 tests covering: single stage recording, stats aggregation (count/total/avg/max), context manager exception handling, empty report, reset, start/end API, missing end ignored, percent-of-total, increment counters, multiple stage independence, engine core integration, cancel does not crash profiler

## Blocked by / 前置依赖

- [05: Engine Loop & Public API](./05-engine-loop-public-api.md) — profiler is wired into EngineCore
