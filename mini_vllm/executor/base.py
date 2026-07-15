from __future__ import annotations

from typing import Dict, List, Protocol, TYPE_CHECKING, runtime_checkable

from ..config import Config
from ..model_runner.base import ModelInput, ModelRunnerOutput
from ..sequence.sequence import Sequence

if TYPE_CHECKING:
    from ..cache.manager import BlockManager


@runtime_checkable
class Executor(Protocol):
    """Abstract protocol for model executors.

    Defines the contract between the engine core and any model runner,
    whether fake (educational) or real (Qwen/HuggingFace).

    Phase 1 additions:
    - ``execute()`` method for unified prefill+decode execution
    - ``prefill()`` and ``decode()`` are kept as optional wrappers
      for backward compatibility with ``FakeModelExecutor`` and
      ``QwenExecutor``.
    """

    def __init__(self, config: Config, block_manager: BlockManager | None = None) -> None: ...

    # ------------------------------------------------------------------
    # Block allocator callbacks
    # ------------------------------------------------------------------

    def prepare_block(self, block_id: int) -> None:
        """Called when BlockAllocator allocates a new physical block."""
        ...

    def release_block(self, block_id: int) -> None:
        """Called when BlockAllocator frees a physical block."""
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

    def execute(self, model_input: ModelInput) -> ModelRunnerOutput:
        """Unified prefill+decode execution (Phase 1.5+).

        Takes a ``ModelInput`` containing both prefill and decode
        sequences, runs the full Transformer layer loop once, and
        returns sampled token IDs mapped to source sequence IDs.

        All executors (Fake, Paged) share this single signature.
        """
        ...

    def prefill(self, sequences: List[Sequence]) -> None:
        """Legacy: per-sequence prefill.

        Kept for backward compatibility with ``FakeModelExecutor``.
        New executors should use ``execute()`` instead.
        """
        ...

    def decode(self, sequences: List[Sequence]) -> None:
        """Legacy: per-sequence decode.

        Kept for backward compatibility.  New executors should use
        ``execute()`` instead.
        """
        ...

    def cleanup_sequence(self, seq_id: str) -> None:
        """Remove per-sequence state after finish."""
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
