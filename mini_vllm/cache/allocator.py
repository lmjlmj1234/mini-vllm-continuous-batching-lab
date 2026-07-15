from __future__ import annotations
from typing import Callable, List, Optional


class BlockAllocator:
    """Low-level physical block pool with reference-count tracking.

    Each physical block carries a ``ref_count``.  When a block is shared
    via Prefix Cache, ``increment_ref()`` bumps the count.  ``free()``
    decrements the count and only truly releases the block when the
    reference count reaches zero.

    This design naturally supports Copy-on-Write: when a shared block
    needs to diverge (two sequences write different tokens to the same
    logical position), the writer allocates a *new* physical block,
    copies data, and decrements the original's ref_count.
    """

    def __init__(
        self,
        num_blocks: int,
        on_allocate: Optional[Callable[[int], None]] = None,
        on_free: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._num_blocks = num_blocks

        # Free-list: True = free, False = in use
        self._free: List[bool] = [True] * num_blocks

        # Reference count per block.  0 = free; >0 = number of references.
        self._ref_counts: List[int] = [0] * num_blocks

        self._on_allocate = on_allocate
        self._on_free = on_free

    def set_callbacks(
        self,
        on_allocate: Optional[Callable[[int], None]] = None,
        on_free: Optional[Callable[[int], None]] = None,
    ) -> None:
        if on_allocate is not None:
            self._on_allocate = on_allocate
        if on_free is not None:
            self._on_free = on_free

    # ------------------------------------------------------------------
    # Allocation / Free with ref-count semantics
    # ------------------------------------------------------------------

    def allocate(self, num_blocks: int) -> Optional[List[int]]:
        """Allocate *num_blocks* physical blocks.

        Sets ``ref_count = 1`` for each newly allocated block.
        Returns ``None`` if insufficient free blocks.
        """
        if num_blocks > self.num_free_blocks:
            return None

        indices: List[int] = []
        for i, free in enumerate(self._free):
            if free:
                indices.append(i)
                if len(indices) == num_blocks:
                    break

        for pid in indices:
            self._free[pid] = False
            self._ref_counts[pid] = 1  # first reference
            if self._on_allocate:
                self._on_allocate(pid)

        return indices

    def free(self, physical_block_ids: List[int]) -> None:
        """Release one reference per block.

        The block is only returned to the free pool when its ref_count
        reaches zero.  Calling ``free`` on a block that this caller does
        *not* own is safe — it merely decrements the shared count.
        """
        for pid in physical_block_ids:
            if self._ref_counts[pid] == 0:
                continue  # already free; guard against double-free
            self._ref_counts[pid] -= 1
            if self._ref_counts[pid] == 0:
                self._free[pid] = True
                if self._on_free:
                    self._on_free(pid)

    def increment_ref(self, pid: int) -> None:
        """Bump reference count for a shared block (Prefix Cache)."""
        assert not self._free[pid], f"Cannot increment ref on free block {pid}"
        self._ref_counts[pid] += 1

    def get_ref_count(self, pid: int) -> int:
        return self._ref_counts[pid]

    # ------------------------------------------------------------------
    # Invariant checks
    # ------------------------------------------------------------------

    def check_invariants(self) -> List[str]:
        """Verify allocator integrity.

        Returns a list of violation messages (empty = all good).
        """
        violations: List[str] = []
        used_count = 0
        for i in range(self._num_blocks):
            if not self._free[i] and self._ref_counts[i] == 0:
                violations.append(
                    f"Block {i}: in-use but ref_count = 0"
                )
            if self._free[i] and self._ref_counts[i] != 0:
                violations.append(
                    f"Block {i}: free but ref_count = {self._ref_counts[i]}"
                )
            if not self._free[i]:
                used_count += 1
        free_count = self._num_blocks - used_count
        if free_count != self.num_free_blocks:
            violations.append(
                f"Free count mismatch: computed={free_count}, "
                f"reported={self.num_free_blocks}"
            )
        return violations

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def max_blocks(self) -> int:
        """Alias for ``num_total_blocks`` — used for GPU pool sizing."""
        return self._num_blocks

    @property
    def num_free_blocks(self) -> int:
        return sum(self._free)

    @property
    def num_total_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_used_blocks(self) -> int:
        return self._num_blocks - self.num_free_blocks

    def dump_free_list(self) -> List[int]:
        return [i for i, free in enumerate(self._free) if free]

    def dump_used_list(self) -> List[int]:
        return [i for i, free in enumerate(self._free) if not free]

    def dump_ref_counts(self) -> List[int]:
        """Expose ref counts (for debugging / tests)."""
        return list(self._ref_counts)

    def stats(self) -> dict:
        return {
            "total_blocks": self._num_blocks,
            "free_blocks": self.num_free_blocks,
            "used_blocks": self.num_used_blocks,
        }
