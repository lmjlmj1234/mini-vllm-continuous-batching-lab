"""GPU component tests for QwenDecoderLayer."""

import pytest
import torch

from mini_vllm.model.transformer_layer import QwenDecoderLayer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_decoder_layer_forward():
    """Single decoder layer forward produces correct shapes."""
    layer = QwenDecoderLayer(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-6,
    ).cuda()

    hidden_states = torch.randn(3, 64).cuda()
    q, k, v, residual = layer(hidden_states)

    assert q.shape == (3, 4, 16), f"Q shape: {q.shape}"
    assert k.shape == (3, 2, 16), f"K shape: {k.shape}"
    assert v.shape == (3, 2, 16), f"V shape: {v.shape}"
    assert residual.shape == (3, 64), f"Residual shape: {residual.shape}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_decoder_layer_post_attention():
    """Post-attention path (o_proj + MLP + residual) produces correct shape."""
    layer = QwenDecoderLayer(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-6,
    ).cuda()

    hidden_states = torch.randn(3, 64).cuda()
    _, _, _, residual = layer(hidden_states)

    # Simulate what the ModelRunner does: attention output → o_proj → residual
    attn_out = torch.randn(3, 4, 16).cuda()
    attn_flat = attn_out.reshape(3, -1)
    attn_proj = layer.attention.o_proj(attn_flat)
    assert attn_proj.shape == (3, 64)

    final = layer.post_attention(attn_proj, residual)
    assert final.shape == (3, 64), f"Post-attention output shape: {final.shape}"
    assert torch.isfinite(final).all(), "Output contains NaN/Inf"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_decoder_layer_fp16():
    """Decoder layer works with fp16 on GPU."""
    layer = QwenDecoderLayer(
        hidden_size=128,
        num_heads=8,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=256,
        rms_norm_eps=1e-6,
    ).half().cuda()

    x = torch.randn(5, 128).half().cuda()
    q, k, v, residual = layer(x)
    assert q.dtype == torch.float16
    assert q.shape == (5, 8, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_decoder_layer_batched():
    """Decoder layer works with 2D (batched) input."""
    layer = QwenDecoderLayer(
        hidden_size=32,
        num_heads=2,
        num_kv_heads=1,
        head_dim=16,
        intermediate_size=64,
        rms_norm_eps=1e-6,
    ).cuda()

    x = torch.randn(2, 8, 32).cuda()  # [B, S, H]
    q, k, v, residual = layer(x)
    assert q.shape == (2, 8, 2, 16)
    assert residual.shape == (2, 8, 32)
