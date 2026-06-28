from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Block:
    """A physical block in the KV cache pool."""
    block_id: int
