from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BlockTableEntry:
    """A single entry in the logical-to-physical block mapping.

    ``is_shared`` is ``True`` when this block was obtained via Prefix
    Cache (i.e., the block belongs to another sequence and we are
    sharing it).  In the future, this flag will trigger Copy-on-Write:
    when a sequence needs to write to a shared block, the executor
    allocates a new block, copies data, and replaces this entry.
    """
    physical_block_id: int
    is_shared: bool = False

    def to_tuple(self) -> Tuple[int, bool]:
        return (self.physical_block_id, self.is_shared)


class BlockTable:
    """Logical-to-physical block mapping (PagedAttention).

    Each entry maps a logical block → physical block.  Entries may be
    shared (multiple BlockTables pointing to the same physical block
    via Prefix Cache).  The ``is_shared`` flag on each entry enables
    future Copy-on-Write: when a shared block needs to diverge, the
    writer allocates a new physical block and replaces the entry.
    """

    def __init__(self, request_id: str, block_size: int) -> None:
        self._request_id = request_id
        self._block_size = block_size
        self._entries: List[BlockTableEntry] = []

    def add_block(self, physical_block_id: int) -> None:
        """Add a non-shared block (owned exclusively by this sequence)."""
        self._entries.append(BlockTableEntry(physical_block_id, is_shared=False))

    def add_shared_block(self, physical_block_id: int) -> None:
        """Add a block shared via Prefix Cache.

        The physical block is owned by another sequence; this entry
        shares it via reference counting in the BlockAllocator.
        """
        self._entries.append(BlockTableEntry(physical_block_id, is_shared=True))

    def clear(self) -> None:
        self._entries.clear()

    def num_blocks(self) -> int:
        return len(self._entries)

    def get_block_ids(self) -> List[int]:
        return [e.physical_block_id for e in self._entries]

    def get_shared_flags(self) -> List[bool]:
        """Return whether each block is shared (for COW detection)."""
        return [e.is_shared for e in self._entries]

    def get_entries(self) -> List[BlockTableEntry]:
        """Return the full entries list (for COW mutation)."""
        return list(self._entries)

    def get_physical_block(self, token_position: int) -> int | None:
        logical_idx = token_position // self._block_size
        if logical_idx < len(self._entries):
            return self._entries[logical_idx].physical_block_id
        return None

    def is_shared_at(self, token_position: int) -> bool:
        """Check whether the block at this token position is shared.

        Used by the executor during decode to detect shared blocks
        that need Copy-on-Write before writing.
        """
        logical_idx = token_position // self._block_size
        if logical_idx < len(self._entries):
            return self._entries[logical_idx].is_shared
        return False

    def dump_mapping(self) -> List[dict]:
        return [
            {"logical": i, "physical": e.physical_block_id, "shared": e.is_shared}
            for i, e in enumerate(self._entries)
        ]

    def __repr__(self) -> str:
        ids = [e.physical_block_id for e in self._entries]
        return f"BlockTable(request={self._request_id}, blocks={ids})"
