"""Unit tests for SwiGLUMLP module."""

import torch
import torch.nn.functional as F

from mini_vllm.model.mlp import SwiGLUMLP


def test_mlp_shapes():
    """SwiGLUMLP preserves hidden_size."""
    mlp = SwiGLUMLP(hidden_size=128, intermediate_size=512)
    x = torch.randn(4, 128)
    out = mlp(x)
    assert out.shape == (4, 128)


def test_mlp_fused_gate_up():
    """Fused gate+up weight has correct shape."""
    h = 64
    inter = 256
    mlp = SwiGLUMLP(h, inter)
    assert mlp.gate_up_weight.shape == (2 * inter, h), (
        f"gate_up_weight shape: {mlp.gate_up_weight.shape}"
    )


def test_mlp_swiglu_formula():
    """SwiGLU formula: SiLU(x @ gate) * (x @ up)."""
    h = 8
    inter = 16
    mlp = SwiGLUMLP(h, inter)
    x = torch.randn(1, h)

    # Manual computation
    gate = x @ mlp.gate_up_weight[:inter].T
    up = x @ mlp.gate_up_weight[inter:].T
    expected = F.silu(gate) * up
    expected = expected @ mlp.down_proj.weight.T

    out = mlp(x)
    assert torch.allclose(out, expected, atol=1e-6), (
        f"SwiGLU formula mismatch: max={(out - expected).abs().max().item()}"
    )


def test_mlp_batched():
    """SwiGLUMLP works with batched input."""
    mlp = SwiGLUMLP(64, 256)
    x = torch.randn(2, 8, 64)
    out = mlp(x)
    assert out.shape == (2, 8, 64)


def test_mlp_vs_hf():
    """Numerical correctness vs HuggingFace Qwen2MLP."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    except ImportError:
        import pytest
        pytest.skip("transformers not installed")

    torch.manual_seed(42)
    h = 64
    inter = 256

    # HF Qwen2MLP takes a config object
    from transformers import Qwen2Config
    hf_config = Qwen2Config(
        hidden_size=h,
        intermediate_size=inter,
        hidden_act="silu",
    )
    hf_mlp = Qwen2MLP(hf_config)
    our_mlp = SwiGLUMLP(h, inter)

    # Copy weights
    # HF has gate_proj, up_proj, down_proj
    gate_w = hf_mlp.gate_proj.weight.data
    up_w = hf_mlp.up_proj.weight.data
    down_w = hf_mlp.down_proj.weight.data

    our_mlp.gate_up_weight.data[:inter] = gate_w
    our_mlp.gate_up_weight.data[inter:] = up_w
    our_mlp.down_proj.weight.data.copy_(down_w)

    x = torch.randn(3, h)
    hf_out = hf_mlp(x)
    our_out = our_mlp(x)

    assert torch.allclose(hf_out, our_out, atol=1e-5), (
        f"Max diff: {(hf_out - our_out).abs().max().item()}"
    )


def test_mlp_fp16():
    """SwiGLUMLP works with fp16."""
    mlp = SwiGLUMLP(64, 256).half()
    x = torch.randn(2, 64).half()
    out = mlp(x)
    assert out.dtype == torch.float16
    assert out.shape == (2, 64)
