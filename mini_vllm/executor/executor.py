from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

from ..config import Config
from ..model.fake_model import FakeModel
from ..model_runner.base import ModelInput, ModelRunnerOutput, SequenceExecutionInfo
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
        # Simulation state for unified execute() path (Phase 1.5)
        # Maps seq_id → {"last_token": int}
        self._sim_state: Dict[str, Dict] = {}

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

    def _get_block_ids(self, seq: Sequence) -> List[int]:
        """Read block IDs from BlockManager (single truth source).

        Falls back to an empty list if no BlockManager is available.
        """
        if self._block_manager is not None:
            return self._block_manager.get_block_table(seq.seq_id)
        return []

    def _write_to_kv(self, seq: Sequence, token_position: int, token_id: int) -> None:
        """Write a token's KV data, allocating blocks on-demand."""
        if self._block_manager is not None:
            if self._block_manager.is_block_shared(seq, token_position):
                return
            pid = self._block_manager.ensure_block(seq, token_position)
        else:
            block_ids = self._get_block_ids(seq)
            logical_idx = token_position // self._config.block_size
            if logical_idx >= len(block_ids):
                # Auto-create a dummy block ID for standalone tests
                dummy_pid = logical_idx
                self._kv_cache.setdefault(dummy_pid, [])
                pid = dummy_pid
            else:
                pid = block_ids[logical_idx]

        key = self._model._fake_key(token_id)
        value = self._model._fake_value(token_id)
        self._kv_cache[pid].extend([key, value])
        self._total_tokens_processed += 1

    def _read_from_kv(self, seq: Sequence, token_position: int) -> int:
        """Read KV bias for a token position."""
        if self._block_manager is not None:
            pid = self._block_manager.ensure_block(seq, token_position)
        else:
            block_ids = self._get_block_ids(seq)
            logical_idx = token_position // self._config.block_size
            if logical_idx >= len(block_ids):
                return 0
            pid = block_ids[logical_idx]

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
        """Read KV, produce next token, write generated token KV.

        NOTE: this legacy API is kept for QwenExecutor backward compat.
        The unified path uses ``execute()`` instead.  The position
        semantics are fixed to match the unified path:
        - READ at ``prompt_len + num_generated - 2`` (cached_len_before - 1)
        - WRITE at ``prompt_len + num_generated - 1`` (cached_len_before)
        """
        for seq in sequences:
            # READ at the last cached position (cached_len_before - 1)
            read_pos = len(seq.prompt_token_ids) + seq.num_generated_tokens - 2
            kv_bias = self._read_from_kv(seq, read_pos)
            prev = seq.output_token_ids[-1]
            next_token = self._model.decode_token(prev, kv_bias)

            # WRITE new KV at cached_len_before
            write_pos = len(seq.prompt_token_ids) + seq.num_generated_tokens - 1
            self._write_to_kv(seq, write_pos, next_token)

            seq.output_token_ids.append(next_token)
            seq.num_generated_tokens += 1

    # ------------------------------------------------------------------
    # Fake tokenizer / detokenizer
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> List[int]:
        vocab = self._model.vocab_size
        return [ord(c) % vocab for c in prompt]

    def detokenize(self, token_ids: List[int]) -> str:
        chars = []
        for t in token_ids:
            c = chr(t % 95 + 32)
            chars.append(c)
        return "".join(chars)

    # ------------------------------------------------------------------
    # Unified execution (Phase 1.5)
    # ------------------------------------------------------------------

    def execute(self, model_input: ModelInput) -> ModelRunnerOutput:
        """Unified prefill+decode using SequenceExecutionInfo snapshots.

        Does NOT receive Sequence objects.  Reads per-sequence metadata
        from ``model_input.sequence_info`` and maintains its own
        simulation state (``_sim_state``).
        """
        sampled_token_ids: List[int] = []
        sampled_sequence_ids: List[str] = []
        block_size = self._config.block_size
        vocab = self._model.vocab_size

        for info in model_input.sequence_info:
            if info.phase == "prefill":
                # Simulate KV writes for the prefill chunk
                for i in range(info.query_len):
                    pos = info.query_start + i
                    # Use position as pseudo-token-id for KV simulation
                    pseudo_tid = pos % vocab
                    self._simulate_kv_write(info.sequence_id, pos, pseudo_tid)

                if info.sample_output_index is not None:
                    # Prefill completes — produce first output token
                    first_token = (pos * 3 + 1) % vocab
                    self._sim_state.setdefault(info.sequence_id, {})
                    self._sim_state[info.sequence_id]["last_token"] = first_token
                    sampled_token_ids.append(first_token)
                    sampled_sequence_ids.append(info.sequence_id)

            elif info.phase == "decode":
                # Decode: read KV bias from the previous token position
                prev_position = info.cached_len_before - 1
                kv_bias = self._simulate_read_kv_bias(info.sequence_id, prev_position)

                state = self._sim_state.setdefault(info.sequence_id, {})
                prev_token = state.get("last_token", 42)
                next_token = (prev_token + 7 + kv_bias) % vocab

                # Write KV for the NEW token at cached_len_before
                self._simulate_kv_write(info.sequence_id, info.cached_len_before, next_token)

                state["last_token"] = next_token
                sampled_token_ids.append(next_token)
                sampled_sequence_ids.append(info.sequence_id)

        return ModelRunnerOutput(
            sampled_token_ids=tuple(sampled_token_ids),
            sampled_sequence_ids=tuple(sampled_sequence_ids),
        )

    def _simulate_kv_write(self, seq_id: str, position: int, token_id: int) -> None:
        """Simulate KV write using seq_id instead of Sequence object."""
        if self._block_manager is not None:
            pid = self._block_manager.ensure_block_by_ids(
                seq_id, position, position
            )
            key = self._model._fake_key(token_id)
            value = self._model._fake_value(token_id)
            self._kv_cache.setdefault(pid, []).extend([key, value])
            self._total_tokens_processed += 1
        else:
            logical_idx = position // self._config.block_size
            self._kv_cache.setdefault(logical_idx, [])
            key = self._model._fake_key(token_id)
            value = self._model._fake_value(token_id)
            self._kv_cache[logical_idx].extend([key, value])
            self._total_tokens_processed += 1

    def _simulate_read_kv_bias(self, seq_id: str, position: int) -> int:
        """Read simulated KV bias using seq_id instead of Sequence object."""
        if self._block_manager is not None:
            block_ids = self._block_manager.get_block_table(seq_id)
            logical_idx = position // self._config.block_size
            if logical_idx < len(block_ids):
                pid = block_ids[logical_idx]
                kv_data = self._kv_cache.get(pid, [])
                return sum(kv_data) % self._model.vocab_size if kv_data else 0
        else:
            logical_idx = position // self._config.block_size
            kv_data = self._kv_cache.get(logical_idx, [])
            return sum(kv_data) % self._model.vocab_size if kv_data else 0
        return 0

    # ------------------------------------------------------------------
    # Sequence cleanup
    # ------------------------------------------------------------------

    def cleanup_sequence(self, seq_id: str) -> None:
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
