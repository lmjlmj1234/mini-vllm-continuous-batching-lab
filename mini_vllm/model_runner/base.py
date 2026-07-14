from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# ModelConfig — read from HF config.json, never hardcoded to a specific model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """All model dimensions read from the HuggingFace config.json.

    No Qwen2.5-0.5B-specific defaults — every value comes from the config
    file at runtime.
    """
    model_type: str = ""
    num_layers: int = 0
    hidden_size: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 32768
    hidden_act: str = "silu"
    intermediate_size: int = 0
    tie_word_embeddings: bool = True
    rope_scaling: dict | None = None
    """Optional RoPE scaling configuration (e.g. Yarn for Qwen2.5)."""
    dtype: torch.dtype = torch.float16
    """Native model weight dtype (bf16 for Qwen2.5)."""

    activation_dtype: torch.dtype = torch.float16
    """Computation dtype for KV cache and attention."""


# ---------------------------------------------------------------------------
# Per-step metadata (single definitions, no duplicates)
# ---------------------------------------------------------------------------

@dataclass
class AttentionGroup:
    """A group of sequences that share the same attention type.

    ``seq_indices`` are indices into the flattened per-step sequence list.
    The ModelRunner iterates groups and dispatches to the correct
    AttentionBackend method based on ``attention_type``.
    """
    seq_indices: List[int]           # indices into flattened seq list
    attention_type: str              # "prefill_ref" | "prefill_gpu" | "decode_gpu"
    cached_len_before: torch.Tensor  # [num_seqs_in_group]
    query_len: torch.Tensor          # [num_seqs_in_group]
    kv_len_after: torch.Tensor       # [num_seqs_in_group]


@dataclass
class AttentionMetadata:
    """Per-step attention metadata for both prefill and decode sequences.

    Every tensor dimension is determined at build time — no batch-wide
    KV padding, no hardcoded max lengths.
    """
    groups: List[AttentionGroup] = field(default_factory=list)

    # --- Prefill fields ---
    prefill_slot_mapping: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    prefill_block_tables: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    prefill_positions: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))

    # --- Decode fields ---
    decode_block_tables: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    decode_slot_mapping: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    decode_positions: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))

    # --- Shared config ---
    block_size: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0


@dataclass
class ModelInput:
    """All data needed for one ModelRunner.execute_model() call.

    ``input_ids``, ``positions``, and ``slot_mapping`` are concatenated
    tensors covering BOTH prefill and decode tokens.  The per-group
    metadata in ``attn_metadata`` disambiguates which tokens belong to
    which attention computation.

    ``sample_token_indices`` specifies which positions in ``input_ids``
    need LM head computation (last token of completed prefill + every
    decode token — not all prefill intermediate tokens).
    """
    input_ids: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    positions: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    slot_mapping: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    attn_metadata: AttentionMetadata = field(default_factory=AttentionMetadata)
    sample_token_indices: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    sequence_info: Tuple["SequenceExecutionInfo", ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Phase 1.5: Execution snapshot + output mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceExecutionInfo:
    """Read-only execution snapshot for one sequence in a single step.

    Carried inside ``ModelInput.sequence_info``.  The executor uses this
    to determine what each sequence expects without touching ``Sequence``
    objects.  Phase 2 (GPU) executors can ignore this entirely.

    ``sample_output_index`` is the position in ``ModelInput.sample_token_indices``
    where this sequence's sample lives, or ``None`` if this sequence does
    not produce a sample (unfinished prefill chunk).
    """
    sequence_id: str
    phase: str                           # "prefill" | "decode"
    query_start: int                     # absolute start position (cached_len_before)
    query_len: int                       # tokens processed this step
    cached_len_before: int               # same as query_start, explicit
    kv_len_after: int                    # cached_len_before + query_len
    sample_output_index: Optional[int]   # index into sample_token_indices, or None


@dataclass(frozen=True)
class ModelRunnerOutput:
    """Explicit mapping from sampled tokens back to source sequences.

    ``sampled_token_ids[i]`` was sampled for sequence
    ``sampled_sequence_ids[i]``.
    """
    sampled_token_ids: Tuple[int, ...] = field(default_factory=tuple)
    sampled_sequence_ids: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# ModelRunner interface
# ---------------------------------------------------------------------------

class ModelRunner(ABC):
    """Abstract interface for the Transformer layer loop.

    A ModelRunner loads model weights from a HuggingFace checkpoint and
    runs the full forward pass (embedding → N decoder layers → LM head).

    Key design properties:
    - Single ``execute_model()`` call per EngineCore.step()
    - No HF ``past_key_values`` — K/V is managed by the AttentionBackend
    - No per-sequence Python loop in the hot path
    """

    @abstractmethod
    def __init__(
        self,
        model_path: str,
        attention_backend: Any,
        config: ModelConfig,
        device: torch.device,
    ) -> None:
        """Load model weights. Do NOT enable HF cache."""
        ...

    @abstractmethod
    def execute_model(self, model_input: ModelInput) -> torch.Tensor:
        """Run the full model forward pass.

        Returns logits only at ``sample_token_indices`` positions, shape
        ``[num_sample_positions, vocab_size]``.
        """
        ...

    @property
    @abstractmethod
    def config(self) -> ModelConfig:
        ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        ...
