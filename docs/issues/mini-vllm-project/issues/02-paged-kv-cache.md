---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 02: Paged KV Cache (BlockAllocator → BlockManager → BlockTable)
# Issue 02：分页 KV Cache（BlockAllocator → BlockManager → BlockTable）

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

The three-layer PagedAttention KV cache: `BlockAllocator` (low-level free-list with reference counting), `BlockManager` (per-sequence allocation coordination with prefix cache integration), and `BlockTable` (logical-to-physical block mapping with shared-block tracking).

三层 PagedAttention KV Cache：`BlockAllocator`（底层空闲列表，带引用计数）、`BlockManager`（per-sequence 分配协调，集成前缀缓存）、`BlockTable`（逻辑块到物理块的映射，跟踪共享块）。

## Vertical slice / 垂直切片描述

This slice provides the complete memory management layer. Physical blocks are tracked in a free-list allocator. Sequences start with zero blocks and allocate on-demand. Block tables map logical positions to physical block IDs. Reference counting supports block sharing (used by prefix cache).

本切片提供了完整的内存管理层。物理块在空闲列表分配器中追踪。序列从零个块开始按需分配。块表将逻辑位置映射到物理块 ID。引用计数支持块共享（供前缀缓存使用）。

## Acceptance criteria / 验收标准

- [x] `BlockAllocator` maintains a free-list (`List[bool]`) and reference counts (`List[int]`) for all physical blocks
- [x] `BlockAllocator.allocate(n)` returns block IDs with `ref_count=1`, returns `None` on OOM
- [x] `BlockAllocator.free(pids)` decrements ref_count per block, returns to free pool only when `ref_count` reaches zero
- [x] `BlockAllocator.increment_ref(pid)` bumps reference count for shared blocks
- [x] `BlockAllocator.set_callbacks()` wires on_allocate/on_free hooks for executor KV storage sync
- [x] `BlockTable` holds `List[BlockTableEntry]` — each entry maps `logical → physical` block with `is_shared` flag
- [x] `BlockTable` supports `add_block()` (exclusive), `add_shared_block()` (shared via prefix cache), `get_block_ids()`, `get_physical_block(position)`, `is_shared_at(position)`, `dump_mapping()`
- [x] `BlockManager` coordinates per-sequence allocation: `allocate_for_seq()`, `ensure_block()`, `free()`
- [x] `BlockManager.ensure_block()` allocates on-demand when a token position crosses a block boundary, checks prefix cache before allocating
- [x] `BlockManager.free()` decrements ref counts — blocks only released when all references are removed
- [x] `BlockManager` provides prefix cache integration: `probe_prefix_cache()`, `compute_block_hashes()`, `is_block_shared()`
- [x] Memory tracing: `dump_tables()`, `get_trace_allocated()`, `get_trace_freed()`, `clear_trace_events()`

## Key code / 核心代码

- `mini_vllm/cache/block.py` — Block dataclass
- `mini_vllm/cache/block_table.py` — BlockTable, BlockTableEntry
- `mini_vllm/cache/allocator.py` — BlockAllocator
- `mini_vllm/cache/manager.py` — BlockManager

## Key tests / 核心测试

- `tests/test_kv_cache_manager.py` — BlockAllocator allocate/free/OOM/callbacks, BlockTable mapping, BlockManager on-demand allocation/OOM/stats
- `tests/test_prefix_cache.py` — BlockAllocator ref_count lifecycle, shared block semantics, BlockManager prefix cache integration (partially — core ref_count tests live here)

## Blocked by / 前置依赖

- [01: Sequence & Request Data Model](./01-sequence-data-model.md) — uses `Sequence` for block table storage
