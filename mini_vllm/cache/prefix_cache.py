from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PrefixCacheProbeResult:
    """Result of probing the prefix cache for a request's prompt.

    Returned by ``BlockManager.probe_prefix_cache()``.  This is a
    **read-only** query: no reference counts are modified, no blocks
    are allocated or shared.

    The Scheduler uses this information to compute uncached token
    counts for budget management and chunked prefill.
    """
    matched_block_count: int = 0
    """How many consecutive logical blocks (from index 0) match the cache."""

    cached_token_count: int = 0
    """Total prompt tokens that are already in cache (= matched_block_count * block_size)."""

    matched_physical_block_ids: List[int] = field(default_factory=list)
    """Physical block IDs of the matched blocks (for reference)."""


def _block_hash(tokens: List[int]) -> int:
    """Deterministic hash for a list of token IDs.

    Uses Python's built-in ``hash()`` over a tuple for simplicity.
    In real vLLM, a more robust hash (e.g., xxhash) may be used to
    avoid collisions and ensure portability.
    """
    return hash(tuple(tokens))


def compute_block_hashes(
    prompt_token_ids: List[int],
    block_size: int,
) -> List[int]:
    """Compute one hash per logical block of *block_size* tokens.

    Returns a list where ``hashes[i]`` is the hash of tokens in
    logical block *i*.  The last block may be partial.
    """
    hashes: List[int] = []
    for i in range(0, len(prompt_token_ids), block_size):
        chunk = prompt_token_ids[i : i + block_size]
        hashes.append(_block_hash(chunk))
    return hashes


class PrefixCache:
    """Hash-based prefix cache for prompt KV blocks.

    Stores a mapping::

        block_hash → physical_block_id

    When a new sequence begins prefill, its prompt token blocks are
    hashed and looked up in this cache.  Matching blocks are *shared*
    (ref_count incremented in BlockAllocator) rather than recomputed.

    Design notes:

    - Blocks are registered in the cache **immediately upon allocation
      during prefill**, not after they are fully written.  This is
      correct because for prompt tokens we know exactly which tokens
      go into each block before the forward pass writes them.

    - The cache has **no eviction** in this version.
      A production system would add LRU or capacity-based eviction.

    - This cache is the foundation for Copy-on-Write: when a shared
      block's data diverges (e.g., during decode), the writer creates
      a new block and the old hash entry remains valid for others.
    """

    def __init__(self) -> None:
        self._cache: Dict[int, int] = {}
        """hash(int) → physical_block_id."""

    def lookup(self, block_hash: int) -> Optional[int]:
        """Return the cached physical block ID, or ``None``."""
        return self._cache.get(block_hash)

    def insert(self, block_hash: int, physical_block_id: int) -> None:
        """Register a block in the cache."""
        self._cache[block_hash] = physical_block_id

    def size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def lookup_span(
        self,
        block_hashes: List[int],
    ) -> List[Optional[int]]:
        """Look up a sequence of block hashes.

        Returns a list of the same length as *block_hashes*, where
        each element is the cached physical block ID or ``None`` if
        the hash is not found.
        """
        return [self._cache.get(h) for h in block_hashes]

    def insert_span(
        self,
        block_hashes: List[int],
        physical_block_ids: List[int],
    ) -> None:
        """Register a sequence of hash → block mappings."""
        assert len(block_hashes) == len(physical_block_ids)
        for h, pid in zip(block_hashes, physical_block_ids):
            self._cache[h] = pid
