from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

from ..config import Config
from ..model.fake_model import FakeModel
from ..sequence.sequence import Sequence
from ..sequence.status import Status
from .base import Executor

if TYPE_CHECKING:
    from ..cache.manager import BlockManager


class FakeModelExecutor:
    """Simulated executor: fake KV cache + fake model.

    Responsibilities:
    - Maintains an in-memory KV cache (Dict[block_id, List[int]])
    - Listens to BlockAllocator callbacks to create/release KV storage
    - Runs prefill: writes prompt tokens to KV, computes first output token
    - Runs decode: reads KV, produces next token, writes generated token KV
    - Fake tokenizer / detokenizer for engine integration

    Implements the ``Executor`` protocol.
    """

    def __init__(self, config: Config, block_manager: BlockManager | None = None) -> None:
        self._config = config
        self._block_manager = block_manager
        self._model = FakeModel(vocab_size=config.vocab_size)
        self._kv_cache: Dict[int, List[int]] = {}
        self._total_tokens_processed: int = 0

    # ------------------------------------------------------------------
    # Block allocator callbacks
    # ------------------------------------------------------------------

    def prepare_block(self, block_id: int) -> None:
        self._kv_cache[block_id] = []

    def release_block(self, block_id: int) -> None:
        self._kv_cache.pop(block_id, None)

    def make_block_allocator_callbacks(self) -> dict:
        return {
            "on_allocate": self.prepare_block,
            "on_free": self.release_block,
        }

    # ------------------------------------------------------------------
    # KV cache simulation
    # ------------------------------------------------------------------

    def _write_to_kv(self, seq: Sequence, token_position: int, token_id: int) -> None:
        """Write a token's KV data, allocating blocks on-demand.

        For blocks shared via Prefix Cache (is_block_shared), the KV
        data already exists — we skip the write and the token count.
        """
        if self._block_manager is not None:
            if self._block_manager.is_block_shared(seq, token_position):
                # Block is shared via Prefix Cache: data already written
                # by the original sequence.  Still ensure block exists in
                # our table (it was prepopulated by allocate_for_seq).
                return
            pid = self._block_manager.ensure_block(seq, token_position)
        else:
            pid = seq.block_table[token_position // self._config.block_size]
        key = self._model._fake_key(token_id)
        value = self._model._fake_value(token_id)
        self._kv_cache[pid].extend([key, value])
        self._total_tokens_processed += 1

    def _read_from_kv(self, seq: Sequence, token_position: int) -> int:
        """Read KV bias for a token position."""
        if self._block_manager is not None:
            pid = self._block_manager.ensure_block(seq, token_position)
        else:
            pid = seq.block_table[token_position // self._config.block_size]
        kv_data = self._kv_cache.get(pid, [])
        return sum(kv_data) % self._model.vocab_size if kv_data else 0

    # ------------------------------------------------------------------
    # Prefill / Decode
    # ------------------------------------------------------------------

    def prefill(self, sequences: List[Sequence]) -> None:
        """Chunk-aware prefill: write from cursor, only complete when done."""
        chunk_size = self._config.max_prefill_chunk_size
        for seq in sequences:
            start = seq.prefill_cursor
            end = min(len(seq.prompt_token_ids), start + chunk_size)
            for pos in range(start, end):
                self._write_to_kv(seq, pos, seq.prompt_token_ids[pos])
            seq.prefill_cursor = end

            if seq.is_prefill_finished:
                first_token = self._model.prefill_token(seq.prompt_token_ids[-1])
                seq.output_token_ids = [first_token]
                seq.num_generated_tokens = 1
                seq.status = Status.RUNNING

    def decode(self, sequences: List[Sequence]) -> None:
        """Read KV, produce next token, write generated token KV."""
        for seq in sequences:
            position = len(seq.prompt_token_ids) + len(seq.output_token_ids) - 1
            kv_bias = self._read_from_kv(seq, position)
            prev = seq.output_token_ids[-1]
            next_token = self._model.decode_token(prev, kv_bias)

            # Write generated token back to KV (real LLMs do this too)
            new_pos = len(seq.prompt_token_ids) + len(seq.output_token_ids)
            self._write_to_kv(seq, new_pos, next_token)

            seq.output_token_ids.append(next_token)
            seq.num_generated_tokens += 1

    # ------------------------------------------------------------------
    # Fake tokenizer / detokenizer
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> List[int]:
        """Convert a string to a list of fake token IDs."""
        vocab = self._model.vocab_size
        return [ord(c) % vocab for c in prompt]

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to a fake string."""
        chars = []
        for t in token_ids:
            c = chr(t % 95 + 32)  # printable ASCII range
            chars.append(c)
        return "".join(chars)

    # ------------------------------------------------------------------
    # Sequence cleanup
    # ------------------------------------------------------------------

    def cleanup_sequence(self, seq_id: str) -> None:
        """No-op: FakeModelExecutor doesn't maintain per-sequence state."""
        pass

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_kv_stats(self) -> dict:
        return {
            "kv_tokens_written": self._total_tokens_processed,
            "kv_slot_capacity": len(self._kv_cache) * self._config.block_size,
            "allocated_blocks": len(self._kv_cache),
        }

    @property
    def total_tokens_processed(self) -> int:
        return self._total_tokens_processed
