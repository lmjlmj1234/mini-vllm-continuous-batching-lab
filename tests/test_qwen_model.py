"""GPU component tests for QwenModel."""

import pytest
import torch

from mini_vllm.model.qwen_model import QwenModel


def _small_config():
    return dict(
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_qwen_model_forward():
    """Full model forward produces correct logit shape."""
    config = _small_config()
    model = QwenModel(**config).cuda()
    input_ids = torch.randint(0, config["vocab_size"], (2, 8)).cuda()
    logits = model(input_ids)
    assert logits.shape == (2, 8, config["vocab_size"]), (
        f"Logits shape: {logits.shape}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_qwen_model_fp16():
    """Full model works with fp16 on GPU."""
    config = _small_config()
    model = QwenModel(**config).half().cuda()
    input_ids = torch.randint(0, config["vocab_size"], (1, 4)).cuda()
    logits = model(input_ids)
    assert logits.shape == (1, 4, config["vocab_size"])
    assert logits.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_qwen_model_tied_embeddings():
    """When tied, lm_head and embed_tokens share weights."""
    config = _small_config()
    config["tie_word_embeddings"] = True
    model = QwenModel(**config).cuda()
    input_ids = torch.randint(0, config["vocab_size"], (1, 3)).cuda()
    logits = model(input_ids)
    assert logits.shape == (1, 3, config["vocab_size"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_qwen_model_not_tied():
    """When not tied, lm_head has separate weights."""
    config = _small_config()
    config["tie_word_embeddings"] = False
    model = QwenModel(**config).cuda()
    input_ids = torch.randint(0, config["vocab_size"], (1, 3)).cuda()
    logits = model(input_ids)
    assert logits.shape == (1, 3, config["vocab_size"])
    # Verify lm_head weight is NOT the same object as embed_tokens weight
    assert model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_qwen_model_layer_count():
    """Model has correct number of layers."""
    config = _small_config()
    config["num_layers"] = 4
    model = QwenModel(**config).cuda()
    assert len(model.layers) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_qwen_model_lm_head_gather():
    """LM head with sample token indices produces correct logits."""
    config = _small_config()
    model = QwenModel(**config).cuda()
    with torch.no_grad():
        input_ids = torch.randint(0, config["vocab_size"], (1, 5)).cuda()
        hidden = model.embed_tokens(input_ids)
        for layer in model.layers:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            q, k, v = layer.attention.qkv_proj(hidden)
            attn_out = model._simple_attention(q, k, v)
            attn_flat = attn_out.reshape(*hidden.shape)
            attn_proj = layer.attention.o_proj(attn_flat)
            hidden = layer.post_attention(attn_proj, residual)
        hidden = model.norm(hidden)
        logits = model.lm_head(hidden)

    assert logits.shape == (1, 5, config["vocab_size"])
    # Gather test: last position logits
    last_logits = logits[:, -1, :]
    assert last_logits.shape == (1, config["vocab_size"])
