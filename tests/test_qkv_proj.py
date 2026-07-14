"""Unit tests for QKVProjection module."""

import torch

from mini_vllm.model.qkv_proj import QKVProjection


def test_qkv_shapes():
    """QKVProjection produces correct Q/K/V shapes with GQA."""
    hd = 64
    n_heads = 8
    n_kv_heads = 2
    proj = QKVProjection(hidden_size=512, num_heads=n_heads,
                         num_kv_heads=n_kv_heads, head_dim=hd)

    x = torch.randn(4, 512)
    q, k, v = proj(x)
    assert q.shape == (4, n_heads, hd), f"Q shape: {q.shape}"
    assert k.shape == (4, n_kv_heads, hd), f"K shape: {k.shape}"
    assert v.shape == (4, n_kv_heads, hd), f"V shape: {v.shape}"


def test_qkv_no_gqa():
    """When num_heads == num_kv_heads, all shapes are equal."""
    hd = 32
    n_heads = 4
    proj = QKVProjection(hidden_size=128, num_heads=n_heads,
                         num_kv_heads=n_heads, head_dim=hd)
    x = torch.randn(2, 128)
    q, k, v = proj(x)
    assert q.shape == k.shape and k.shape == v.shape, (
        f"QKV shapes should match when no GQA: Q={q.shape} K={k.shape} V={v.shape}"
    )


def test_qkv_weight_fusion():
    """Fused QKV weight has correct total size."""
    hd = 64
    n_heads = 8
    n_kv = 2
    proj = QKVProjection(512, n_heads, n_kv, hd)
    q_sz = n_heads * hd
    kv_sz = n_kv * hd
    expected = q_sz + 2 * kv_sz
    assert proj.qkv_weight.shape[0] == expected, (
        f"Fused weight rows {proj.qkv_weight.shape[0]} != expected {expected}"
    )
    assert proj.qkv_weight.shape[1] == 512


def test_qkv_batched():
    """QKVProjection works with batched input."""
    proj = QKVProjection(64, 4, 1, 16)
    x = torch.randn(3, 8, 64)
    q, k, v = proj(x)
    assert q.shape == (3, 8, 4, 16)
    assert k.shape == (3, 8, 1, 16)


def test_qkv_deterministic():
    """Same input produces same output."""
    proj = QKVProjection(32, 2, 1, 16)
    x = torch.randn(5, 32)
    q1, k1, v1 = proj(x)
    q2, k2, v2 = proj(x)
    assert torch.equal(q1, q2)
    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)


def test_qkv_fp16():
    """QKVProjection works with fp16."""
    hd = 32
    proj = QKVProjection(128, 4, 2, hd)
    proj = proj.half()
    x = torch.randn(3, 128).half()
    q, k, v = proj(x)
    assert q.dtype == torch.float16
    assert q.shape == (3, 4, 32)
    assert k.shape == (3, 2, 32)
