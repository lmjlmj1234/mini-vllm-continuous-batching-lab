from __future__ import annotations

from typing import Any

import torch

from .base import ModelConfig


class ConfigAdapter:
    """Read model configuration from HuggingFace ``config.json``.

    No hardcoded Qwen2.5-0.5B values — every dimension is read
    dynamically from the checkpoint's configuration.
    """

    @staticmethod
    def from_pretrained(model_path: str) -> ModelConfig:
        """Load and validate config from a local model directory."""
        import json
        import os

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Model config not found at {config_path}"
            )

        with open(config_path) as f:
            hf_config = json.load(f)

        return ConfigAdapter.from_hf_config(hf_config)

    @staticmethod
    def from_hf_config(hf_config: dict[str, Any]) -> ModelConfig:
        """Convert HuggingFace config dict to ModelConfig."""
        hidden_size = hf_config.get("hidden_size", 0)
        num_heads = hf_config.get("num_attention_heads", 0)
        head_dim = hf_config.get("head_dim", 0)
        if head_dim == 0 and num_heads > 0:
            head_dim = hidden_size // num_heads

        return ModelConfig(
            model_type=hf_config.get("model_type", ""),
            num_layers=hf_config.get("num_hidden_layers", 0),
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=hf_config.get("num_key_value_heads", num_heads),
            head_dim=head_dim,
            vocab_size=hf_config.get("vocab_size", 0),
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            rope_theta=hf_config.get("rope_theta", 10000.0),
            max_position_embeddings=hf_config.get("max_position_embeddings", 32768),
            hidden_act=hf_config.get("hidden_act", "silu"),
            intermediate_size=hf_config.get("intermediate_size", 0),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", True),
            rope_scaling=hf_config.get("rope_scaling", None),
            dtype=torch.float16,
            activation_dtype=torch.float16,
        )

    @staticmethod
    def validate_for_attention(config: ModelConfig) -> None:
        """Check that the config is compatible with PagedAttention.

        Raises ``ValueError`` with a clear message if any check fails.
        """
        if config.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {config.num_heads}")
        if config.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be > 0, got {config.num_kv_heads}")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({config.num_heads}) must be divisible by "
                f"num_kv_heads ({config.num_kv_heads}) for GQA"
            )
        if config.head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {config.head_dim}")
        if config.head_dim > 128:
            raise ValueError(
                f"head_dim ({config.head_dim}) > 128 not supported "
                f"by first-generation PagedAttention kernel"
            )
