from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from ..attention.backend import AttentionBackend
from ..config import Config
from ..model_runner.base import ModelInput, ModelRunnerOutput
from ..model_runner.config_adapter import ConfigAdapter
from ..model_runner.qwen_runner import QwenModelRunner
from .base import Executor

if TYPE_CHECKING:
    from ..cache.manager import BlockManager


class PagedExecutor:
    """Executor that uses QwenModelRunner with paged KV cache.

    Implements the ``Executor`` protocol via a single ``execute()`` method.
    No legacy ``prefill()``/``decode()`` methods.  No HF ``past_key_values``.

    Architecture::

        PagedExecutor
          → QwenModelRunner.execute_model(ModelInput) → logits
          → greedy argmax → sampled_token_ids
          → ModelRunnerOutput(sampled_token_ids, sampled_sequence_ids)
    """

    def __init__(
        self,
        config: Config,
        block_manager: Optional[BlockManager] = None,
    ) -> None:
        self._config = config
        self._block_manager = block_manager
        self._device = torch.device(config.device)
        self._total_tokens_processed: int = 0

        # Read model config
        model_config = ConfigAdapter.from_pretrained(config.model_path)
        ConfigAdapter.validate_for_attention(model_config)

        # Create attention backend (from config, not hardcoded)
        self._attention_backend = AttentionBackend.create(
            model_config, backend=config.attention_backend,
        )

        # Create model runner
        self._model_runner = QwenModelRunner(
            model_path=config.model_path,
            attention_backend=self._attention_backend,
            config=model_config,
            device=self._device,
            block_size=config.block_size,
            peak_runtime_estimate=config.peak_runtime_estimate,
        )

        # HF tokenizer
        self._tokenizer = self._load_tokenizer(config.model_path)

    # ------------------------------------------------------------------
    # Block allocator callbacks
    # ------------------------------------------------------------------

    def prepare_block(self, block_id: int) -> None:
        pass

    def release_block(self, block_id: int) -> None:
        pass

    def make_block_allocator_callbacks(self) -> dict:
        return {"on_allocate": self.prepare_block, "on_free": self.release_block}

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> List[int]:
        return self._tokenizer.encode(prompt)

    def detokenize(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    @staticmethod
    def _load_tokenizer(model_path: str):
        """Load HF tokenizer from local model path."""
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_path)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer from {model_path}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, model_input: ModelInput) -> ModelRunnerOutput:
        """Run prefill + decode in one unified forward pass.

        Args:
            model_input: Packed input for one step.

        Returns:
            ModelRunnerOutput with sampled tokens and sequence IDs.
        """
        logits = self._model_runner.execute_model(model_input)
        # logits: [num_samples, vocab_size]

        # Greedy argmax sampling
        sampled_ids = torch.argmax(logits, dim=-1).tolist()  # List[int]
        if not isinstance(sampled_ids, list):
            sampled_ids = [sampled_ids]

        # Sequence IDs from ModelInput.sequence_info
        seq_ids: List[str] = []
        sample_idx = 0
        for info in model_input.sequence_info:
            if info.sample_output_index is not None:
                # Map sample output index back to sequence
                assert info.sample_output_index == sample_idx, (
                    f"sample_output_index mismatch: expected {sample_idx}, "
                    f"got {info.sample_output_index}"
                )
                seq_ids.append(info.sequence_id)
                sample_idx += 1

        self._total_tokens_processed += model_input.input_ids.shape[0]

        return ModelRunnerOutput(
            sampled_token_ids=tuple(sampled_ids),
            sampled_sequence_ids=tuple(seq_ids),
        )

    # ------------------------------------------------------------------
    # Legacy stubs
    # ------------------------------------------------------------------

    def prefill(self, sequences) -> None:
        raise NotImplementedError("PagedExecutor: use execute() instead")

    def decode(self, sequences) -> None:
        raise NotImplementedError("PagedExecutor: use execute() instead")

    def cleanup_sequence(self, seq_id: str) -> None:
        pass

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_kv_stats(self) -> Dict[str, int]:
        pool = self._model_runner.pool
        return {
            "num_blocks": pool.num_blocks,
            "block_size": pool.block_size,
            "num_layers": pool.num_layers,
            "num_kv_heads": pool.num_kv_heads,
            "head_dim": pool.head_dim,
            "total_slots": pool.total_slots,
            "total_bytes": pool.total_bytes,
            "tokens_processed": self._total_tokens_processed,
        }

    @property
    def total_tokens_processed(self) -> int:
        return self._total_tokens_processed
