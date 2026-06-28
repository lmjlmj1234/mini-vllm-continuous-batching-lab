from __future__ import annotations

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

    def __init__(self, block_size: int, allocator: BlockAllocator) -> None:
        self._block_size = block_size
        self._allocator = allocator
        self._tables: Dict[str, BlockTable] = {}
        self._prefix_cache = PrefixCache()

        # Per-seq metadata
        self._shared_prefix_blocks: Dict[str, int] = {}
        """Number of logical blocks shared via prefix cache."""
        self._block_hashes: Dict[str, List[int]] = {}
        """Pre-computed block hashes for a sequence's prompt."""

        # Memory trace events
        self._trace_allocated: Dict[str, List[int]] = {}
        self._trace_freed: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # Prefix Cache info
    # ------------------------------------------------------------------

    @property
    def prefix_cache(self) -> PrefixCache:
        return self._prefix_cache

    def compute_block_hashes(self, seq: Sequence) -> List[int]:
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
        """
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
        return PrefixCacheProbeResult(
            matched_block_count=matched_count,
            cached_token_count=min(max_cached, len(prompt_token_ids)),
            matched_physical_block_ids=matched_pids,
        )

    # ------------------------------------------------------------------
    # On-demand allocation with Prefix Cache
    # ------------------------------------------------------------------

    def allocate_for_seq(self, seq: Sequence) -> None:
        """Register a sequence with an empty block table.

        Before admitting, checks the Prefix Cache for matching prompt
        blocks.  Matching blocks are shared (ref_count incremented) and
        prepopulated in the block table.  Non-matching blocks will be
        allocated on-demand during ``ensure_block()``.
        """
        table = BlockTable(seq.seq_id, self._block_size)
        self._tables[seq.seq_id] = table

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
        seq.block_table = table.get_block_ids()

        # First non-matching block will be allocated by ensure_block().
        # NOTE: we intentionally don't register new blocks here — they
        # will be registered when ensure_block() allocates them during
        # the executor's prefill write.

    def ensure_block(self, seq: Sequence, position: int) -> int:
        """Ensure a physical block exists for the given token position.

        Returns the physical block ID.  For positions within an already-
        shared or already-allocated block, returns immediately.

        For positions that cross a block boundary *beyond* the shared
        prefix: checks the Prefix Cache first (the block may have been
        registered by another sequence between admission and now).
        If no cache hit, allocates a new block and registers it.
        """
        logical_idx = position // self._block_size
        table = self._tables.get(seq.seq_id)

        if table is None:
            table = BlockTable(seq.seq_id, self._block_size)
            self._tables[seq.seq_id] = table

        shared_count = self._shared_prefix_blocks.get(seq.seq_id, 0)

        while logical_idx >= table.num_blocks():
            # We need to add a new block.  Check prefix cache first.
            is_prompt_position = position < len(seq.prompt_token_ids)
            cached_pid: Optional[int] = None

            if is_prompt_position:
                hashes = self._block_hashes.get(seq.seq_id)
                if hashes is not None and logical_idx < len(hashes):
                    h = hashes[logical_idx]
                    cached_pid = self._prefix_cache.lookup(h)

            shared = False
            if cached_pid is not None:
                if self._allocator.get_ref_count(cached_pid) > 0:
                    # Valid cache hit: share the block
                    self._allocator.increment_ref(cached_pid)
                    table.add_shared_block(cached_pid)
                    shared = True
                # else: stale cache entry — treat as miss, allocate below

            if not shared:
                # Allocate new block
                pids = self._allocator.allocate(1)
                if pids is None:
                    raise RuntimeError(
                        f"OOM: no free block for seq={seq.seq_id} "
                        f"position={position} "
                        f"(shared_prefix={self._shared_prefix_blocks.get(seq.seq_id, 0)})"
                    )
                pid = pids[0]
                table.add_block(pid)

                # Register the new block in prefix cache (only for prompt
                # positions — decode tokens are unpredictable)
                if is_prompt_position:
                    hashes = self._block_hashes.get(seq.seq_id)
                    if hashes is not None and logical_idx < len(hashes):
                        self._prefix_cache.insert(hashes[logical_idx], pid)

                # Trace
                self._trace_allocated.setdefault(seq.seq_id, []).append(pid)

            seq.block_table = table.get_block_ids()

        return table.get_physical_block(position)

    def is_block_shared(self, seq: Sequence, position: int) -> bool:
        """Check whether the block at *position* is shared (Prefix Cache).

        The executor can use this to skip KV writes for shared blocks
        (the data already exists).
        """
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
        table = self._tables.pop(seq_id, None)
        if table is None:
            return
        pids = table.get_block_ids()
        self._allocator.free(pids)
        if pids:
            self._trace_freed[seq_id] = pids
        table.clear()
        self._shared_prefix_blocks.pop(seq_id, None)
        self._block_hashes.pop(seq_id, None)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_table(self, seq_id: str) -> BlockTable | None:
        return self._tables.get(seq_id)

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
        s["prefix_cache_entries"] = self._prefix_cache.size()
        return s
