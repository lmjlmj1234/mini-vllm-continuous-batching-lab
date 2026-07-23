from __future__ import annotations
import time
from typing import Dict, List, Optional, Tuple

from ..sequence.sequence import Sequence
from .allocator import BlockAllocator
from .block_table import BlockTable
from .prefix_cache import (
    PrefixCache,
    PrefixCacheProbeResult,
    compute_block_hashes,
)


class BlockManager:
    """On-demand BlockAllocator with Prefix Cache support.
    它不自己管物理块，而是封装了 Allocator + PrefixCache + BlockTable 三者的协作
    ``Sequence`` starts with **zero blocks**.  Blocks are allocated one at a
    time through ``ensure_block()``.  However, when a sequence is admitted,
    its prompt prefix is hashed and looked up in the Prefix Cache —
    matching blocks are **shared** (ref_count incremented) and prepopulated
    in the block table, avoiding both allocation and recomputation.

    Allocation flow::

        Scheduler → allocate_for_seq(seq)   [prepopulate shared blocks]
        Executor  → ensure_block(seq, pos)  [allocate or use cached]
                     ↓
        PrefixCache.lookup(hash)  → hit → share (increment_ref)
                                  → miss → allocate → insert(hash, pid)
    """

    def __init__(self, block_size: int, allocator: BlockAllocator,
                 profiler: Optional[object] = None,
                 enable_prefix_caching: bool = True) -> None:
        self._block_size = block_size #块大小
        self._allocator = allocator #持有底层 BlockAllocator 的引用（不是自己创建）
        self._tables: Dict[str, BlockTable] = {} # 请求 ID → BlockTable 的字典，一个 Sequence 对应一张表
        self._prefix_cache = PrefixCache() if enable_prefix_caching else None
        self._enable_prefix_caching = enable_prefix_caching
        self._profiler = profiler
        """Optional profiler for stage timing (kv_cache_allocation, etc.)."""

        # Per-seq metadata
        self._shared_prefix_blocks: Dict[str, int] = {} # 每条序列的共享块数量。记录它的 prompt 前几个逻辑块是通过缓存共享的
        """Number of logical blocks shared via prefix cache."""
        self._block_hashes: Dict[str, List[int]] = {} # 每条序列预计算的 prompt 块 hash 列表
        """Pre-computed block hashes for a sequence's prompt."""

        # Memory trace events #分配/释放追踪，用于调试和测试
        self._trace_allocated: Dict[str, List[int]] = {}
        self._trace_freed: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # Prefix Cache info
    # ------------------------------------------------------------------

    @property
    def prefix_cache(self) -> Optional[PrefixCache]:
        # 暴露前缀缓存实例（供外部只读访问）
        return self._prefix_cache

    def compute_block_hashes(self, seq: Sequence) -> List[int]:
        # 把 Sequence 的 prompt token IDs 按 block_size 切成 N 个块，每个块算一个 hash。比如 block_size=4，prompt 长度 10，就会算 3 个 hash（4+4+2）。结果用于前缀缓存匹配
        """Pre-compute block-level hashes for a sequence's prompt."""
        return compute_block_hashes(seq.prompt_token_ids, self._block_size)

    def probe_prefix_cache(
        self,
        prompt_token_ids: List[int],
    ) -> PrefixCacheProbeResult:
        """Read-only probe of the prefix cache for a prompt.

        The Scheduler calls this **before** budget computation to know
        how many prompt tokens are already cached.  No reference counts
        are modified — this is a pure query.

        Only consecutive matches from block index 0 count.  The first
        miss or stale entry ends the matched prefix.

        When ``enable_prefix_caching`` is False, always returns zero
        cached tokens — every prompt starts from scratch.
        """
        if not self._enable_prefix_caching:
            return PrefixCacheProbeResult(
                matched_block_count=0,
                cached_token_count=0,
                matched_physical_block_ids=[],
            )

        t0 = time.time()
        hashes = compute_block_hashes(prompt_token_ids, self._block_size)
        matched_pids: List[int] = []

        for h in hashes:
            cached_pid = self._prefix_cache.lookup(h)
            if cached_pid is not None and self._allocator.get_ref_count(cached_pid) > 0:
                matched_pids.append(cached_pid)
            else:
                break

        matched_count = len(matched_pids)
        # Cap at prompt length: the last matched block may be partial,
        # so block_size * matched_count could exceed the actual prompt.
        max_cached = matched_count * self._block_size
        result = PrefixCacheProbeResult(
            matched_block_count=matched_count,
            cached_token_count=min(max_cached, len(prompt_token_ids)),
            matched_physical_block_ids=matched_pids,
        )
        if self._profiler:
            self._profiler.record_raw("prefix_cache_lookup", time.time() - t0)
        return result

    # ------------------------------------------------------------------
    # On-demand allocation with Prefix Cache
    # ------------------------------------------------------------------

    def allocate_for_seq(self, seq: Sequence) -> None:
        """Register a sequence with an empty block table.

        Before admitting, checks the Prefix Cache for matching prompt
        blocks.  Matching blocks are shared (ref_count incremented) and
        prepopulated in the block table.  Non-matching blocks will be
        allocated on-demand during ``ensure_block()``.

        When prefix caching is disabled, all blocks are allocated
        on-demand.
        """
        table = BlockTable(seq.seq_id, self._block_size)
        self._tables[seq.seq_id] = table

        if not self._enable_prefix_caching:
            self._shared_prefix_blocks[seq.seq_id] = 0
            return

        # Compute block hashes and check cache
        hashes = self.compute_block_hashes(seq)
        self._block_hashes[seq.seq_id] = hashes

        shared_count = 0
        for h in hashes:
            cached_pid = self._prefix_cache.lookup(h)
            if cached_pid is not None and self._allocator.get_ref_count(cached_pid) > 0:
                # Valid cache entry: share this block
                self._allocator.increment_ref(cached_pid)
                table.add_shared_block(cached_pid)
                shared_count += 1
            else:
                # Stale cache entry or cache miss.  The prefix chain is
                # broken — remaining blocks will be allocated on-demand.
                break

        self._shared_prefix_blocks[seq.seq_id] = shared_count

        # No mirror sync to seq.block_table — BlockManager is the single
        # truth source.  Consumers use get_block_table().
        # First non-matching block will be allocated by ensure_block().
        # NOTE: we intentionally don't register new blocks here — they
        # will be registered when ensure_block() allocates them during
        # the executor's prefill write.

    def ensure_block(self, seq: Sequence, position: int) -> int:
        """Ensure a physical block exists for the given token position.
         "确保某个 token 位置已经有物理块了"。执行器在写 KV cache 时调用这个。返回物理块 ID
        Returns the physical block ID.  For positions within an already-
        shared or already-allocated block, returns immediately.

        For positions that cross a block boundary *beyond* the shared
        prefix: checks the Prefix Cache first (the block may have been
        registered by another sequence between admission and now).
        If no cache hit, allocates a new block and registers it.
        """
        logical_idx = position // self._block_size #位置 → 逻辑块编号。比如 block_size=4，position=6 → logical_idx=1
        table = self._tables.get(seq.seq_id)

        if table is None:
            table = BlockTable(seq.seq_id, self._block_size)
            self._tables[seq.seq_id] = table

        shared_count = self._shared_prefix_blocks.get(seq.seq_id, 0)

        while logical_idx >= table.num_blocks():#如果需要的逻辑块编号大于当前表的长度，说明需要加新块
            # We need to add a new block.  Check prefix cache first.
            is_prompt_position = position < len(seq.prompt_token_ids)
            cached_pid: Optional[int] = None

            if is_prompt_position and self._enable_prefix_caching:
                hashes = self._block_hashes.get(seq.seq_id)
                if hashes is not None and logical_idx < len(hashes):
                    h = hashes[logical_idx]
                    cached_pid = self._prefix_cache.lookup(h)
            # 先查缓存：如果这个位置是 prompt 的一部分（不是 decode 生成的 token），就拿着预计算的 hash 去前缀缓存里查查看。可能有其他序列在分配后又缓存了新块
            shared = False
            if cached_pid is not None:
                if self._allocator.get_ref_count(cached_pid) > 0:
                    # Valid cache hit: share the block
                    self._allocator.increment_ref(cached_pid)
                    table.add_shared_block(cached_pid)
                    shared = True
                # else: stale cache entry — treat as miss, allocate below
            # 缓存命中、引用有效 → 共享，加引用计数
            if not shared:
                # Allocate new block
                alloc_t0 = time.time()
                pids = self._allocator.allocate(1)
                if pids is None:
                    raise RuntimeError(
                        f"OOM: no free block for seq={seq.seq_id} "
                        f"position={position} "
                        f"(shared_prefix={self._shared_prefix_blocks.get(seq.seq_id, 0)})"
                    )
                pid = pids[0]
                table.add_block(pid)
                #没有命中 → 真正从 Allocator 分配一个物理块。如果 Allocator 返回 None（没空闲块了），就抛 OOM 异常
                if self._profiler:
                    self._profiler.record_raw("kv_cache_allocation",
                                              time.time() - alloc_t0)

                # Register the new block in prefix cache (only for prompt
                # positions — decode tokens are unpredictable)
                if is_prompt_position and self._enable_prefix_caching:
                    hashes = self._block_hashes.get(seq.seq_id)
                    if hashes is not None and logical_idx < len(hashes):
                        self._prefix_cache.insert(hashes[logical_idx], pid)
                #立即注册到前缀缓存 — prompt 的块在写之前就知道内容了，可以提前注册。这样后续的请求就能命中这个块。
                # Trace
                self._trace_allocated.setdefault(seq.seq_id, []).append(pid)

            # NOTE: mirror sync removed — see Phase 1.
        return table.get_physical_block(position)

    def is_block_shared(self, seq: Sequence, position: int) -> bool:
        """Check whether the block at *position* is shared (Prefix Cache).

        The executor can use this to skip KV writes for shared blocks
        (the data already exists).
        """
        #  "这个位置的块是共享的吗？" 执行器在 decode 阶段写 KV cache 之前调用。如果是共享块，需要走 Copy-on-Write（分配新块、拷贝数据），不能直接覆盖。
        return self._tables.get(seq.seq_id, BlockTable("_", self._block_size)).is_shared_at(position)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Free & cleanup
    # ------------------------------------------------------------------

    def free(self, seq_id: str) -> None:
        """Free all blocks owned by a sequence.

        Because blocks may be shared, ``free()`` decrements the ref_count
        in BlockAllocator rather than directly returning blocks to the
        free pool.  A block is only truly returned when its ref_count
        reaches zero.
        """
        free_t0 = time.time()
        table = self._tables.pop(seq_id, None)
        if table is None:
            return
        pids = table.get_block_ids()
        self._allocator.free(pids)
        if pids:
            self._trace_freed[seq_id] = pids
        table.clear()
        if self._profiler:
            self._profiler.record_raw("kv_cache_release", time.time() - free_t0)
        self._shared_prefix_blocks.pop(seq_id, None)
        self._block_hashes.pop(seq_id, None)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_table(self, seq_id: str) -> BlockTable | None:
        return self._tables.get(seq_id)

    def get_block_table(self, seq_id: str) -> List[int]:
        """Read-only view of block IDs for a sequence.

        Single truth source for logical-to-physical mapping.
        ``Sequence.block_table`` has been removed — all consumers
        must read from this method.
        """
        table = self._tables.get(seq_id)
        if table is None:
            return []
        return table.get_block_ids()

    def ensure_block_by_ids(
        self, seq_id: str, position: int, prompt_len: int
    ) -> int:
        """Lightweight ``ensure_block`` using ids instead of a Sequence object.

        Does NOT support prefix-cache matching (the full ``ensure_block``
        with a Sequence object should be used for that).  With prefix cache
        disabled, this is equivalent for test/simulation executors that
        cannot hold Sequence objects.
        """
        logical_idx = position // self._block_size
        table = self._tables.get(seq_id)

        if table is None:
            table = BlockTable(seq_id, self._block_size)
            self._tables[seq_id] = table

        while logical_idx >= table.num_blocks():
            alloc_t0 = time.time()
            pids = self._allocator.allocate(1)
            if pids is None:
                raise RuntimeError(
                    f"OOM: no free block for seq={seq_id} position={position}"
                )
            pid = pids[0]
            table.add_block(pid)
            if self._profiler:
                self._profiler.record_raw("kv_cache_allocation",
                                          time.time() - alloc_t0)
            self._trace_allocated.setdefault(seq_id, []).append(pid)

        return table.get_physical_block(position)

    def get_shared_prefix_length(self, seq_id: str) -> int:
        """Return how many logical blocks are shared for this sequence."""
        return self._shared_prefix_blocks.get(seq_id, 0)

    # ------------------------------------------------------------------
    # Memory trace helpers
    # ------------------------------------------------------------------

    def clear_trace_events(self) -> None:
        self._trace_allocated.clear()
        self._trace_freed.clear()

    def get_trace_allocated(self) -> Dict[str, List[int]]:
        return dict(self._trace_allocated)

    def get_trace_freed(self) -> Dict[str, List[int]]:
        return dict(self._trace_freed)

    def dump_tables(self) -> Dict[str, List[dict]]:
        result: Dict[str, List[dict]] = {}
        for seq_id, table in self._tables.items():
            result[seq_id] = table.dump_mapping()
        return result

    def stats(self) -> dict:
        s = self._allocator.stats()
        s["prefix_cache_entries"] = self._prefix_cache.size() if self._prefix_cache else 0
        return s
