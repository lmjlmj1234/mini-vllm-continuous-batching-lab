---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 08: Prefix Cache
# Issue 08：前缀缓存

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

The block-hash based prefix cache: requests with common prompt prefixes share KV cache blocks instead of recomputing them. Uses block-level hashing of prompt token IDs, a global prefix cache dictionary, read-only probe for the scheduler, shared block allocation, and Copy-on-Write ready metadata.

基于块哈希的前缀缓存：具有相同 prompt 前缀的请求共享 KV cache 块，而不是重新计算。使用 prompt token ID 的块级哈希、全局前缀缓存字典、调度器的只读探测、共享块分配和写时复制就绪元数据。

## Vertical slice / 垂直切片描述

This slice extends BlockManager and BlockAllocator with prefix cache semantics. The scheduler probes the cache before budget computation (read-only, no side effects). On admission, matching blocks are shared via reference counting. Non-matching blocks are allocated on-demand during execution. The `is_shared` flag on BlockTableEntry enables future Copy-on-Write.

本切片为 BlockManager 和 BlockAllocator 增加了前缀缓存语义。调度器在预算计算前探测缓存（只读，无副作用）。准入时，匹配的块通过引用计数共享。不匹配的块在执行期间按需分配。BlockTableEntry 上的 `is_shared` 标志支持未来的写时复制。

## Acceptance criteria / 验收标准

- [x] `PrefixCache` maintains a `Dict[int, int]` mapping block hash → physical block ID
- [x] `PrefixCache.insert(hash, pid)` and `lookup(hash) → Optional[int]` with deterministic hash computation
- [x] Block hashes computed via `compute_block_hashes(prompt_token_ids, block_size)` — deterministic, partial-block aware
- [x] `PrefixCacheProbeResult` reports `matched_block_count`, `cached_token_count`, `matched_physical_block_ids`
- [x] `BlockManager.probe_prefix_cache()` is a read-only query — no reference counts modified
- [x] `BlockManager.allocate_for_seq()` prepopulates block table with shared blocks for matching prefix hashes
- [x] Only consecutive matches from block index 0 count — first miss breaks the chain
- [x] `BlockAllocator.increment_ref(pid)` bumps ref_count for shared blocks
- [x] `BlockAllocator.free()` decrements ref_count, only releases when ref_count reaches zero
- [x] `Shared blocks are skipped during KV write (`is_block_shared()` → `is_shared` flag in executor prefill)
- [x] `BlockTableEntry.is_shared` flag enables Copy-on-Write detection
- [x] Stale cache entries (block freed) are detected via `get_ref_count() > 0` check — probed as misses
- [x] Late-arriving requests with common prefix share blocks with already-running requests

## Key code / 核心代码

- `mini_vllm/cache/prefix_cache.py` — PrefixCache, PrefixCacheProbeResult, compute_block_hashes
- `mini_vllm/cache/block_table.py` — BlockTableEntry.is_shared
- `mini_vllm/cache/allocator.py` — increment_ref(), get_ref_count(), ref_count-aware free()
- `mini_vllm/cache/manager.py` — probe_prefix_cache(), allocate_for_seq(), ensure_block(), is_block_shared()
- `mini_vllm/scheduler/scheduler.py` — Phase 5 prefix probe integration

## Key tests / 核心测试

- `tests/test_prefix_cache.py` — 30 tests covering: empty cache, insert/lookup, span hits/misses, hash determinism, partial blocks, ref_count lifecycle (allocate, increment, free, double-free), shared prefix blocks skip KV writes, ref_count increases with sharers, partial prefix match, is_block_shared, different prompts don't match, late arrival sharing, cache persists after partial free, probe correctness, probe does not change ref_count, stale cache entries, partial prefix probe, scheduler integration with cache hits

## Blocked by / 前置依赖

- [02: Paged KV Cache](./02-paged-kv-cache.md) — prefix cache is built on top of BlockAllocator/BlockManager
- [03: Core Scheduler](./03-core-scheduler.md) — scheduler must support prefix probe in admission phase
