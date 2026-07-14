"""Unit tests for RMSNorm module."""

import torch

from mini_vllm.model.rms_norm import RMSNorm


def test_rmsnorm_shape():
    """RMSNorm preserves input shape."""
    norm = RMSNorm(128)
    x = torch.randn(16, 128)
    out = norm(x)
    assert out.shape == (16, 128)


def test_rmsnorm_3d():
    """RMSNorm works with 3D input."""
    norm = RMSNorm(64)
    x = torch.randn(4, 32, 64)
    out = norm(x)
    assert out.shape == (4, 32, 64)


def test_rmsnorm_output_not_nan():
    """RMSNorm output is finite (no NaN/Inf)."""
    norm = RMSNorm(32)
    x = torch.randn(8, 32)
    out = norm(x)
    assert torch.isfinite(out).all()


def test_rmsnorm_nonzero():
    """RMSNorm output is not all zeros (weight is 1, eps small)."""
    norm = RMSNorm(16)
    x = torch.randn(4, 16) * 10.0
    out = norm(x)
    assert (out.abs() > 1e-8).any()


def test_rmsnorm_vs_hf():
    """Numerical correctness vs HuggingFace Qwen2RMSNorm."""
    try:
        from transformers.modeling_utils import (
            is_deepspeed_zero3_enabled as _,
        )
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    except ImportError:
        import pytest
        pytest.skip("transformers not installed")

    torch.manual_seed(42)
    hidden_size = 64
    eps = 1e-6

    hf_norm = Qwen2RMSNorm(hidden_size, eps=eps)
    our_norm = RMSNorm(hidden_size, eps=eps)
    our_norm.weight.data.copy_(hf_norm.weight.data)

    x = torch.randn(4, 32, hidden_size)
    hf_out = hf_norm(x)
    our_out = our_norm(x)

    assert torch.allclose(hf_out, our_out, atol=1e-6), (
        f"Max diff: {(hf_out - our_out).abs().max().item()}"
    )


def test_rmsnorm_vs_hf_small_tensor():
    """RMSNorm correctness with small tensors."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    except ImportError:
        import pytest
        pytest.skip("transformers not installed")

    rms = RMSNorm(8, eps=1e-6)
    hf_rms = Qwen2RMSNorm(8, eps=1e-6)
    rms.weight.data.copy_(hf_rms.weight.data)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    out = rms(x)
    ref = hf_rms(x)
    assert torch.allclose(out, ref, atol=1e-7), (
        f"RMSNorm mismatch: max={(out - ref).abs().max().item()}"
    )


def test_rmsnorm_eps_effect():
    """Changing eps affects numerical output."""
    norm1 = RMSNorm(32, eps=1e-1)
    norm2 = RMSNorm(32, eps=1e-6)
    x = torch.randn(2, 32)
    out1 = norm1(x)
    out2 = norm2(x)
    diff = (out1 - out2).abs().max().item()
    assert diff > 1e-8, "Different eps should produce different outputs"
