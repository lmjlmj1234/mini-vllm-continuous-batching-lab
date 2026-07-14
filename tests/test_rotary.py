"""Unit tests for RotaryEmbedding module."""

import torch

from mini_vllm.model.rotary import RotaryEmbedding


def test_rope_shape():
    """RoPE preserves input shape."""
    rope = RotaryEmbedding(head_dim=64, theta=10000.0)
    x = torch.randn(10, 8, 64)
    positions = torch.arange(10)
    out = rope(x, positions)
    assert out.shape == (10, 8, 64)


def test_rope_zero_position():
    """RoPE at position 0 is identity (cos=1, sin=0)."""
    rope = RotaryEmbedding(head_dim=64, theta=10000.0)
    x = torch.randn(3, 4, 64)
    positions = torch.zeros(3, dtype=torch.long)
    out = rope(x, positions)
    assert torch.allclose(out, x, atol=1e-6), (
        "RoPE at pos=0 should be identity"
    )


def test_rope_vs_hf():
    """Numerical correctness vs HuggingFace Qwen2RotaryEmbedding."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2RotaryEmbedding,
            apply_rotary_pos_emb,
        )
    except ImportError:
        import pytest
        pytest.skip("transformers not installed")

    torch.manual_seed(42)
    head_dim = 64
    theta = 10000.0
    max_seq = 128
    dtype = torch.float32

    # HF expects a config object
    from transformers import Qwen2Config
    hf_config = Qwen2Config(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=head_dim,
        rope_theta=theta,
        max_position_embeddings=max_seq,
    )
    hf_rope = Qwen2RotaryEmbedding(hf_config)
    our_rope = RotaryEmbedding(head_dim, theta, max_seq, dtype=dtype)

    x = torch.randn(1, 8, 5, head_dim)  # [B, H, L, D]
    positions = torch.tensor([[0, 5, 10, 15, 20]], dtype=torch.long)

    # Our RoPE: per-token, [L, H, D] layout
    x_our = x.squeeze(0).permute(1, 0, 2)  # [L, H, D] = [5, 8, 64]
    our_out = our_rope(x_our, positions[0])  # [5, 8, 64]
    our_out = our_out.permute(1, 0, 2).unsqueeze(0)  # [1, 8, 5, 64]

    # HF RoPE: expects [B, H, L, D]
    cos, sin = hf_rope(x, positions)
    hf_out, _ = apply_rotary_pos_emb(x, x, cos, sin)

    assert torch.allclose(our_out, hf_out, atol=1e-5), (
        f"Max diff: {(our_out - hf_out).abs().max().item()}"
    )


def test_rope_positions_affect_output():
    """Different positions produce different rotations."""
    rope = RotaryEmbedding(head_dim=32, theta=1000.0)
    x = torch.randn(1, 2, 32).expand(2, -1, -1).clone()
    pos1 = torch.tensor([0])
    pos2 = torch.tensor([1])
    out1 = rope(x[:1], pos1)
    out2 = rope(x[:1], pos2)
    diff = (out1 - out2).abs().max().item()
    assert diff > 1e-6, "Different positions should produce different rotations"


def test_rope_kv_heads():
    """RoPE works with num_heads != num_kv_heads (both dim 1)."""
    rope = RotaryEmbedding(head_dim=64, theta=10000.0)
    q = torch.randn(5, 8, 64)   # num_heads=8
    k = torch.randn(5, 2, 64)   # num_kv_heads=2
    positions = torch.arange(5)
    q_out = rope(q, positions)
    k_out = rope(k, positions)
    assert q_out.shape == (5, 8, 64)
    assert k_out.shape == (5, 2, 64)


def test_rope_dim_semantics():
    """RoPE rotation: first half uses cos, second half adds -x2*sin."""
    rope = RotaryEmbedding(head_dim=4, theta=1000.0, max_seq_len=10)
    x = torch.ones(1, 1, 4)
    positions = torch.tensor([1])

    # theta=1000, inv_freq = [1.0, 1000.0**(-2/4)] = [1.0, 0.0316228]
    # cos(1): [cos(1*1), cos(1*0.0316), cos(1*1), cos(1*0.0316)]
    #        ≈ [0.5403, 0.9995, 0.5403, 0.9995]
    # sin(1): [sin(1*1), sin(1*0.0316), sin(1*1), sin(1*0.0316)]
    #        ≈ [0.8415, 0.0316, 0.8415, 0.0316]
    #
    # x * cos = [0.5403, 0.9995, 0.5403, 0.9995]
    # rotate_half(x) = [-1, -1, 1, 1]
    # rotate_half(x) * sin = [-0.8415, -0.0316, 0.8415, 0.0316]
    # result = x*cos + rotate_half(x)*sin
    #        = [-0.3012, 0.9679, 1.3818, 1.0311]
    out = rope(x, positions)
    assert out[0, 0, 0] != 1.0, "First element should be rotated"
    # First two elements: cos - sin
    assert abs(out[0, 0, 0].item() - (-0.3012)) < 0.1
    assert abs(out[0, 0, 1].item() - 0.9679) < 0.1
    # Last two elements: cos + sin
    assert abs(out[0, 0, 2].item() - 1.3818) < 0.1
    assert abs(out[0, 0, 3].item() - 1.0311) < 0.1
