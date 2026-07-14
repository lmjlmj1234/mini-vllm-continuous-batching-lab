"""HF checkpoint weight loader for Qwen2.5 models.

Loads weights from a HuggingFace ``state_dict`` into our custom ``nn.Module``
subclasses.  Handles QKV fusion (q_proj + k_proj + v_proj → qkv_weight) and
gate+up fusion (gate_proj + up_proj → gate_up_weight).
"""

from __future__ import annotations

import os
from typing import Any

import torch

from .qwen_model import QwenModel


def load_qwen_weights(
    model: QwenModel,
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Load HF checkpoint weights into a QwenModel.

    Args:
        model: Target QwenModel (custom nn.Module, not HF).
        model_path: Path to HF model directory containing ``pytorch_model*.bin``
            or ``model-*.safetensors``.
        device: Target device for weights.
        dtype: Target dtype for weights.
    """
    # Try safetensors first, then bin
    state_dict: dict[str, torch.Tensor] = _load_state_dict(model_path, device, dtype)

    # Verify required keys are present
    _check_keys(state_dict, model.num_layers)

    # ---- Embedding ----
    model.embed_tokens.weight.data.copy_(state_dict["model.embed_tokens.weight"])

    # ---- Final RMSNorm ----
    model.norm.weight.data.copy_(state_dict["model.norm.weight"])

    # ---- LM head ----
    if model.tie_word_embeddings:
        model.lm_head.weight.data.copy_(state_dict["model.embed_tokens.weight"])
    else:
        model.lm_head.weight.data.copy_(state_dict["lm_head.weight"])

    # ---- Per-layer ----
    for i in range(model.num_layers):
        prefix = f"model.layers.{i}"

        # Input layernorm (RMSNorm)
        model.layers[i].input_layernorm.weight.data.copy_(
            state_dict[f"{prefix}.input_layernorm.weight"]
        )

        # Post-attention layernorm (RMSNorm)
        model.layers[i].post_attn_layernorm.weight.data.copy_(
            state_dict[f"{prefix}.post_attention_layernorm.weight"]
        )

        # Fused QKV (weight + bias)
        q_w = state_dict[f"{prefix}.self_attn.q_proj.weight"]
        k_w = state_dict[f"{prefix}.self_attn.k_proj.weight"]
        v_w = state_dict[f"{prefix}.self_attn.v_proj.weight"]
        qkv = torch.cat([q_w, k_w, v_w], dim=0)  # [Q+K+V, hidden]
        model.layers[i].attention.qkv_proj.qkv_weight.data.copy_(qkv)
        q_b = state_dict[f"{prefix}.self_attn.q_proj.bias"]
        k_b = state_dict[f"{prefix}.self_attn.k_proj.bias"]
        v_b = state_dict[f"{prefix}.self_attn.v_proj.bias"]
        qkv_b = torch.cat([q_b, k_b, v_b], dim=0)  # [Q+K+V]
        model.layers[i].attention.qkv_proj.qkv_bias.data.copy_(qkv_b)

        # Output projection
        model.layers[i].attention.o_proj.weight.data.copy_(
            state_dict[f"{prefix}.self_attn.o_proj.weight"]
        )

        # Fused gate+up
        gate_w = state_dict[f"{prefix}.mlp.gate_proj.weight"]
        up_w = state_dict[f"{prefix}.mlp.up_proj.weight"]
        gate_up = torch.cat([gate_w, up_w], dim=0)  # [2*intermediate, hidden]
        model.layers[i].mlp.gate_up_weight.data.copy_(gate_up)

        # Down projection
        model.layers[i].mlp.down_proj.weight.data.copy_(
            state_dict[f"{prefix}.mlp.down_proj.weight"]
        )


def _load_state_dict(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Load HF state dict from safetensors or bin."""
    # Try safetensors
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file
            sd = load_file(safetensors_path, device=str(device))
            return {k: v.to(dtype=dtype) for k, v in sd.items()}
        except ImportError:
            pass

    # Try sharded safetensors (index file)
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        try:
            from safetensors.torch import load_file
            import json
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            shard_files = set(weight_map.values())
            sd = {}
            for fname in shard_files:
                shard_path = os.path.join(model_path, fname)
                shard = load_file(shard_path, device=str(device))
                sd.update(shard)
            return {k: v.to(dtype=dtype) for k, v in sd.items()}
        except ImportError:
            pass

    # Fallback: pytorch_model.bin
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location=device, weights_only=True)
        return {k: v.to(dtype=dtype) for k, v in sd.items()}

    # Try sharded bin
    import glob
    bin_files = sorted(glob.glob(os.path.join(model_path, "pytorch_model-*.bin")))
    if bin_files:
        sd = {}
        for fpath in bin_files:
            shard = torch.load(fpath, map_location=device, weights_only=True)
            sd.update(shard)
        return {k: v.to(dtype=dtype) for k, v in sd.items()}

    raise FileNotFoundError(
        f"No model weights found in {model_path}. "
        f"Expected model.safetensors, model.safetensors.index.json, "
        f"pytorch_model.bin, or pytorch_model-*.bin"
    )


def _check_keys(state_dict: dict[str, Any], num_layers: int) -> None:
    """Verify essential keys exist in the loaded state dict."""
    required_global = [
        "model.embed_tokens.weight",
        "model.norm.weight",
    ]
    for key in required_global:
        if key not in state_dict:
            raise KeyError(f"Missing required key: {key}")

    # lm_head.weight is optional when tie_word_embeddings=True (HF skips it)

    for i in range(num_layers):
        required_per_layer = [
            f"model.layers.{i}.input_layernorm.weight",
            f"model.layers.{i}.post_attention_layernorm.weight",
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"model.layers.{i}.self_attn.q_proj.bias",
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"model.layers.{i}.self_attn.k_proj.bias",
            f"model.layers.{i}.self_attn.v_proj.weight",
            f"model.layers.{i}.self_attn.v_proj.bias",
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"model.layers.{i}.mlp.gate_proj.weight",
            f"model.layers.{i}.mlp.up_proj.weight",
            f"model.layers.{i}.mlp.down_proj.weight",
        ]
        for key in required_per_layer:
            if key not in state_dict:
                raise KeyError(f"Missing required key: {key} (layer {i})")
