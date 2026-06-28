from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

import torch

from ..config import Config
from ..sequence.sequence import Sequence
from ..sequence.status import Status

if TYPE_CHECKING:
    from ..cache.manager import BlockManager


_HF_CACHE: Optional[str] = None
"""Optional HF_HOME override (can be set before import)."""


def _get_model_and_tokenizer() -> Tuple[Any, Any]:
    """Lazy-load Qwen2-0.5B from HuggingFace.

    Import is inside the function so ``mini_vllm`` can be imported without
    torch/transformers installed (only the QwenExecutor path needs them).
    """
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _HF_CACHE is not None:
        os.environ["HF_HOME"] = _HF_CACHE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=dtype,
        device_map=device,
        use_cache=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


class QwenExecutor:
    """Real model executor using Qwen2-0.5B via HuggingFace Transformers.

    Responsibilities:
    - Loads Qwen2-0.5B model + tokenizer (lazy, on first instantiation)
    - Runs real prefill (prompt → KV cache → first token)
    - Runs real decode (KV cache → next token)
    - Manages per-sequence ``past_key_values`` for continuous batching
    - Tracks per-block KV storage for BlockAllocator callbacks
    - Cleans up per-sequence state when ``cleanup_sequence()`` is called

    Implements the ``Executor`` protocol.
    """

    def __init__(self, config: Config, block_manager: BlockManager | None = None) -> None:
        self._config = config
        self._block_manager = block_manager
        self._model, self._tokenizer = _get_model_and_tokenizer()
        self._model_config = self._model.config

        # Per-block KV tracking: {block_id: list_of_token_positions_in_block}
        # Actual tensors live in _seq_kv, keyed by seq_id.
        self._kv_cache: Dict[int, List[int]] = {}

        # Per-sequence past_key_values (transformers KV cache format)
        self._seq_kv: Dict[str, Any] = {}

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
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> List[int]:
        return self._tokenizer.encode(prompt, add_special_tokens=True)

    def detokenize(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prefill(self, sequences: List[Sequence]) -> None:
        chunk_size = self._config.max_prefill_chunk_size
        device = self._model.device

        for seq in sequences:
            start = seq.prefill_cursor
            end = min(len(seq.prompt_token_ids), start + chunk_size)
            token_ids = seq.prompt_token_ids[start:end]

            input_ids = torch.tensor([token_ids], device=device)
            past_kv = self._seq_kv.get(seq.seq_id)

            outputs = self._model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
                attention_mask=self._build_attention_mask(seq, past_kv, len(token_ids), device),
            )
            self._seq_kv[seq.seq_id] = outputs.past_key_values

            # Update block tracking for each written token
            for pos in range(start, end):
                if self._block_manager is not None:
                    block_id = self._block_manager.ensure_block(seq, pos)
                    self._kv_cache.setdefault(block_id, []).append(pos)

            self._total_tokens_processed += end - start
            seq.prefill_cursor = end

            if seq.is_prefill_finished:
                logits = outputs.logits[0, -1, :]
                next_token = self._sample_token(logits)
                seq.output_token_ids = [next_token]
                seq.num_generated_tokens = 1
                seq.status = Status.RUNNING
                seq.first_token_time = time.time()

    @torch.no_grad()
    def decode(self, sequences: List[Sequence]) -> None:
        device = self._model.device

        for seq in sequences:
            prev_token = seq.output_token_ids[-1]
            input_ids = torch.tensor([[prev_token]], device=device)
            past_kv = self._seq_kv.get(seq.seq_id)

            outputs = self._model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            self._seq_kv[seq.seq_id] = outputs.past_key_values

            # Sample next token from logits
            logits = outputs.logits[0, -1, :]
            next_token = self._sample_token(logits)

            # Track the new token's KV write in BlockManager
            new_pos = len(seq.prompt_token_ids) + len(seq.output_token_ids)
            if self._block_manager is not None:
                block_id = self._block_manager.ensure_block(seq, new_pos)
                self._kv_cache.setdefault(block_id, []).append(new_pos)

            self._total_tokens_processed += 1
            seq.output_token_ids.append(next_token)
            seq.num_generated_tokens += 1

    # ------------------------------------------------------------------
    # Sequence cleanup
    # ------------------------------------------------------------------

    def cleanup_sequence(self, seq_id: str) -> None:
        """Remove per-sequence KV cache for a finished sequence.

        Block-level KV tracking (``_kv_cache``) is cleaned up by
        ``release_block()`` callbacks when the BlockAllocator frees
        blocks.  Here we only clean the per-sequence past_key_values.
        """
        self._seq_kv.pop(seq_id, None)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_kv_stats(self) -> Dict[str, int]:
        return {
            "kv_tokens_written": self._total_tokens_processed,
            "kv_slot_capacity": len(self._kv_cache) * self._config.block_size,
            "allocated_blocks": len(self._kv_cache),
        }

    @property
    def total_tokens_processed(self) -> int:
        return self._total_tokens_processed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_token(logits: torch.Tensor) -> int:
        """Greedy sampling (argmax)."""
        return torch.argmax(logits).item()

    @staticmethod
    def _build_attention_mask(
        seq: Sequence,
        past_kv: Any,
        new_tokens: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Build attention mask for prefill with existing past_key_values.

        For chunked prefill, the sequence already has some tokens in KV.
        The attention mask must cover both existing and new tokens.
        """
        if past_kv is None:
            return None  # transformers will create a causal mask

        total_len = seq.prefill_cursor + new_tokens
        mask = torch.ones(1, total_len, dtype=torch.long, device=device)
        return mask
