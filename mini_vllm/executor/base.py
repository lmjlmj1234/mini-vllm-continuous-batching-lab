from __future__ import annotations

from typing import Dict, List, Protocol, TYPE_CHECKING, runtime_checkable

from ..config import Config
from ..sequence.sequence import Sequence

if TYPE_CHECKING:
    from ..cache.manager import BlockManager


@runtime_checkable
class Executor(Protocol):
    """Abstract protocol for model executors.

    Defines the contract between the engine core and any model runner,
    whether fake (educational) or real (Qwen/HuggingFace).

    Responsibilities:
    - Tokenize / detokenize text using the model's tokenizer
    - Run prefill: process prompt tokens, write KV cache, produce first token
    - Run decode: read KV cache, produce one new token, write new KV
    - Maintain KV cache storage in sync with BlockAllocator lifecycle
    - Provide KV cache statistics for monitoring
    """

    def __init__(self, config: Config, block_manager: BlockManager | None = None) -> None: ...

    # ------------------------------------------------------------------
    # Block allocator callbacks
    # ------------------------------------------------------------------

    def prepare_block(self, block_id: int) -> None:
        """Called when BlockAllocator allocates a new physical block.

        Subclasses should create KV data storage for this block.
        """
        ...

    def release_block(self, block_id: int) -> None:
        """Called when BlockAllocator frees a physical block.

        Subclasses should release KV data storage for this block.
        """
        ...

    def make_block_allocator_callbacks(self) -> dict:
        """Return callbacks dict for BlockAllocator.set_callbacks()."""
        ...

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> List[int]: ...

    def detokenize(self, token_ids: List[int]) -> str: ...

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def prefill(self, sequences: List[Sequence]) -> None:
        """Process prompt tokens and write KV cache.

        Post-condition: sequences with finished prefill have their first
        output token set and status changed to RUNNING.
        """
        ...

    def decode(self, sequences: List[Sequence]) -> None:
        """Produce one new output token per sequence.

        Reads existing KV cache, produces logits, samples a token, and
        writes the new token's KV data back to the cache.
        """
        ...

    def cleanup_sequence(self, seq_id: str) -> None:
        """Remove per-sequence state (KV cache, etc.) after finish.

        Called by EngineCore when a group finishes, so the executor
        can release any sequence-specific resources.
        """
        ...

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_kv_stats(self) -> Dict[str, int]:
        """Return KV cache usage statistics."""
        ...

    @property
    def total_tokens_processed(self) -> int:
        """Total KV write operations performed so far."""
        ...
