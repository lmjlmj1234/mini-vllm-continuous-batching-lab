from __future__ import annotations

from typing import Any, Optional

import torch

from ..attention.backend import AttentionBackend
from ..cache.pool import KVCachePool
from ..model.qwen_model import QwenModel
from ..model.rotary import RotaryEmbedding
from ..model.weight_loader import load_qwen_weights
from .base import (
    AttentionGroup,
    AttentionMetadata,
    ModelConfig,
    ModelInput,
    ModelRunner,
)


class QwenModelRunner(ModelRunner):
    """ModelRunner for Qwen2.5 models with paged KV cache.

    Runs the full Transformer forward pass (embedding → N decoder layers →
    final RMSNorm → LM head) with a SINGLE unified layer loop for both
    prefill and decode sequences.  K/V is written to the paged cache before
    attention (方案B).

    Architecture per layer::

        RMSNorm → QKV projection → reshape Q/K/V → RoPE
        → cache write → paged attention → output projection
        → residual → RMSNorm → SwiGLU MLP → residual
    """

    def __init__(
        self,
        model_path: str,
        attention_backend: AttentionBackend,
        config: ModelConfig,
        device: torch.device,
        block_size: int = 4,
        num_gpu_blocks_override: Optional[int] = None,
        peak_runtime_estimate: int = 0,
    ) -> None:
        self._model_config = config
        self._device = device
        self._dtype = config.dtype
        self._block_size = block_size
        self._num_gpu_blocks_override = num_gpu_blocks_override
        self._peak_runtime_estimate = peak_runtime_estimate

        # Build the Qwen model
        self.model = QwenModel(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            intermediate_size=config.intermediate_size,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
        )
        self.model.to(device=device, dtype=config.dtype)

        # Load weights
        load_qwen_weights(
            self.model, model_path, device=device, dtype=config.dtype,
        )
        self.model.eval()

        # RoPE
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=device,
            dtype=config.dtype,
        )

        # Attention backend
        self._attention_backend = attention_backend

        # Allocate KV cache pool
        self._pool = attention_backend.allocate_pool(
            num_layers=config.num_layers,
            num_blocks=self._resolve_num_blocks(),
            block_size=block_size,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dtype=config.activation_dtype,
            device=device,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ModelConfig:
        return self._model_config

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def pool(self) -> KVCachePool:
        return self._pool

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_model(self, model_input: ModelInput) -> torch.Tensor:
        """Run the full model forward pass.

        Args:
            model_input: Packed input for one step.

        Returns:
            Logits at ``sample_token_indices`` positions,
            shape ``[num_samples, vocab_size]``.
        """
        input_ids = model_input.input_ids       # [total_tokens]
        positions = model_input.positions       # [total_tokens]
        slot_mapping = model_input.slot_mapping # [total_tokens]
        attn_meta = model_input.attn_metadata

        total_tokens = input_ids.shape[0]
        num_prefill_tokens = (
            attn_meta.prefill_slot_mapping.shape[0]
            if attn_meta.prefill_slot_mapping.numel() > 0 else 0
        )
        num_decode_tokens = (
            attn_meta.decode_slot_mapping.shape[0]
            if attn_meta.decode_slot_mapping.numel() > 0 else 0
        )

        # Build corrected decode metadata (0-based seq_indices for decode
        # block tables — the original input_builder uses global seq indices
        # that don't match decode_block_tables shape)
        decode_meta = self._build_decode_meta(attn_meta)

        # ---- 1. Token embedding ----
        hidden = self.model.embed_tokens(input_ids)  # [total_tokens, hidden_size]

        # ---- 2. Per-layer loop ----
        for layer_idx in range(self.model.num_layers):
            layer = self.model.layers[layer_idx]

            # Pre-norm residual
            residual = hidden
            normed = layer.input_layernorm(hidden)

            # QKV projection (fused weight)
            q, k, v = layer.attention.qkv_proj(normed)
            # q: [total_tokens, num_heads, head_dim]
            # k: [total_tokens, num_kv_heads, head_dim]
            # v: [total_tokens, num_kv_heads, head_dim]

            # RoPE
            q = self.rope(q, positions)
            k = self.rope(k, positions)

            # Write K/V to paged cache (方案B: write-first)
            self._attention_backend.write_kv_cache(
                layer_idx, k, v, slot_mapping,
            )

            # Attention — read from cache (方案B)
            attn_out = torch.zeros_like(q)  # [total_tokens, num_heads, hd]

            if num_prefill_tokens > 0:
                pref_q = q[:num_prefill_tokens]
                pref_k = k[:num_prefill_tokens]
                pref_v = v[:num_prefill_tokens]
                pref_result = self._attention_backend.prefill_attention(
                    layer_idx, pref_q, pref_k, pref_v,
                    attn_meta, self._pool,
                )
                attn_out[:num_prefill_tokens] = pref_result

            if num_decode_tokens > 0 and decode_meta is not None:
                dec_q = q[num_prefill_tokens:]
                dec_result = self._attention_backend.decode_attention(
                    layer_idx, dec_q, decode_meta, self._pool,
                )
                attn_out[num_prefill_tokens:] = dec_result

            # Output projection + residual
            attn_flat = attn_out.reshape(total_tokens, -1)  # [total_tokens, num_heads*hd]
            attn_proj = layer.attention.o_proj(attn_flat)   # [total_tokens, hidden]

            # Post-attention: RMSNorm → SwiGLU MLP → residual
            hidden = layer.post_attention(attn_proj, residual)

        # ---- 3. Final RMSNorm + LM head ----
        hidden = self.model.norm(hidden)  # [total_tokens, hidden]
        logits = self.model.lm_head(hidden)  # [total_tokens, vocab_size]

        # ---- 4. Gather sample positions ----
        if model_input.sample_token_indices.numel() > 0:
            logits = logits[model_input.sample_token_indices]

        return logits

    # ------------------------------------------------------------------
    # Decode metadata helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_decode_meta(
        attn_meta: AttentionMetadata,
    ) -> Optional[AttentionMetadata]:
        """Build corrected AttentionMetadata for decode dispatch.

        The original ``input_builder`` sets decode group ``seq_indices`` to
        global offsets (e.g. ``[num_prefill, num_prefill+1, ...]``), but
        ``decode_block_tables`` only has rows for decode sequences.  This
        helper rebuilds the group with 0-based indices so the backend can
        correctly index ``decode_block_tables``.
        """
        # Find the original decode group
        decode_group = None
        for g in attn_meta.groups:
            if g.attention_type == "decode_gpu":
                decode_group = g
                break
        if decode_group is None:
            return None

        num_decode = len(decode_group.seq_indices)
        new_group = AttentionGroup(
            seq_indices=list(range(num_decode)),
            attention_type="decode_gpu",
            cached_len_before=decode_group.cached_len_before,
            query_len=decode_group.query_len,
            kv_len_after=decode_group.kv_len_after,
        )

        return AttentionMetadata(
            groups=[new_group],
            decode_block_tables=attn_meta.decode_block_tables,
            decode_slot_mapping=attn_meta.decode_slot_mapping,
            decode_positions=attn_meta.decode_positions,
            block_size=attn_meta.block_size,
            num_kv_heads=attn_meta.num_kv_heads,
            head_dim=attn_meta.head_dim,
        )

    def _resolve_num_blocks(self) -> int:
        """Compute number of GPU KV cache blocks from available memory.

        Uses ``compute_num_gpu_blocks()`` from the cache pool module.
        The ``peak_runtime_estimate`` is set to 0 initially; the StageProfiler
        will refine it in a future milestone.
        """
        from mini_vllm.cache.pool import compute_num_gpu_blocks
        return compute_num_gpu_blocks(
            num_layers=self._model_config.num_layers,
            num_kv_heads=self._model_config.num_kv_heads,
            head_dim=self._model_config.head_dim,
            block_size=self._block_size,
            dtype=self._dtype,
            device=self._device,
            gpu_memory_utilization=0.90,
            peak_runtime_estimate=self._peak_runtime_estimate,
            num_gpu_blocks_override=self._num_gpu_blocks_override,
        )
