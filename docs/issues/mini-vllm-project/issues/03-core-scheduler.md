---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 03: Core Continuous Batching Scheduler
# Issue 03：核心连续批处理调度器

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

The six-phase continuous-batching scheduler: admits waiting requests, manages prefill/decode/finished lifecycle, enforces token budget, implements decode-first priority, and produces a rich `ScheduleResult` with debug information.

六阶段连续批处理调度器：准入等待请求、管理 prefill/decode/finished 生命周期、强制执行 token budget、实现 decode-first 优先级策略、生成包含调试信息的丰富 `ScheduleResult`。

## Vertical slice / 垂直切片描述

This slice provides the complete scheduling logic as a pure function (`scheduler.schedule()` → `ScheduleResult`). It integrates with BlockManager and RequestQueue. At each engine step, it decides which groups prefill, which decode, which finish, which are ignored (and why), and which are rejected.

本切片提供作为纯函数的完整调度逻辑（`scheduler.schedule()` → `ScheduleResult`）。它与 BlockManager 和 RequestQueue 集成。在每个引擎步骤中，它决定哪些组应该 prefill、哪些 decode、哪些完成、哪些被忽略（以及原因）、哪些被拒绝。

## Acceptance criteria / 验收标准

- [x] `ScheduleResult` dataclass reports: `scheduled_prefill_groups`, `scheduled_decode_groups`, `finished_groups`, `ignored_groups`, `rejected_groups`, `ignored_reasons` (per-request-id), `num_prefill_tokens`, `num_decode_tokens`, `num_batched_tokens`, `token_budget_remaining`, `cached_token_count`, `num_uncached_prefill_tokens`, `matched_block_count`, `debug_reason`
- [x] Phase 1 (Finish): running groups whose sequences hit `max_tokens` are marked FINISHED and moved to finished pool
- [x] Phase 2 (Categorize): remaining groups split into decode (status=RUNNING) and prefill-continue (status=PREFILL)
- [x] Phase 3 (Decode-first budget): decode sequences consume budget first — 1 token per decode seq
- [x] Phase 4 (Chunked-prefill continue): mid-prefill groups advance their prefill cursor
- [x] Phase 5 (Admit new): waiting groups create a Sequence, allocate blocks, and enter the running pool with chunked budget; prefix cache probe reduces prefill cursor start
- [x] Phase 6 (Token counts & debug_reason): aggregates all counts and builds debug string
- [x] `max_num_seqs` limit: when sequence budget exhausted, remaining groups marked IGNORED with reason `MAX_NUM_SEQS_LIMIT`
- [x] Token budget: total `num_batched_tokens` capped by `max_num_batched_tokens`; prefill tokens additionally capped by `max_num_prefill_tokens`
- [x] Rejection: requests whose uncached prompt tokens exceed `max_num_batched_tokens` are rejected (moved to rejected pool)
- [x] Ignore mechanism: requests that fit but lack budget this step are ignored (remain in waiting queue for next step)
- [x] `Scheduler.block_manager_stats()` exposes BlockManager stats for metrics reporting

## Key code / 核心代码

- `mini_vllm/scheduler/scheduler.py` — Scheduler class with 6-phase `schedule()`
- `mini_vllm/scheduler/schedule_result.py` — ScheduleResult dataclass

## Key tests / 核心测试

- `tests/test_scheduler.py` — admission, prefill→decode transitions, finished removal, max_num_seqs limits, on-demand block allocation, token counts, decode-first priority, chunked prefill, ignored/rejected reasons

## Blocked by / 前置依赖

- [01: Sequence & Request Data Model](./01-sequence-data-model.md) — scheduler operates on SequenceGroup/Sequence/RequestQueue
- [02: Paged KV Cache](./02-paged-kv-cache.md) — scheduler calls `BlockManager.allocate_for_seq()`, `free()`, `probe_prefix_cache()`
