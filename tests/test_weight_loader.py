"""Integration tests for weight loading.

Requires a real Qwen2.5 model in HF cache or ``QWEN_MODEL_PATH`` env var.
Skipped when no model weights are available.
"""

import os
import pytest

MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
)


def _has_model():
    """Check if model weight files exist."""
    if not os.path.exists(MODEL_PATH):
        return False
    # Check for any weight file
    for fname in os.listdir(MODEL_PATH):
        if fname.endswith((".safetensors", ".bin")):
            return True
    return False


def test_config_loading():
    """ConfigAdapter can read Qwen2.5 config without model weights."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model path not found: {MODEL_PATH}")
    from mini_vllm.model_runner.config_adapter import ConfigAdapter

    model_config = ConfigAdapter.from_pretrained(MODEL_PATH)
    assert model_config.num_layers > 0, "num_layers should be > 0"
    assert model_config.hidden_size > 0
    assert model_config.num_heads > 0
    assert model_config.num_kv_heads > 0
    assert model_config.head_dim > 0
    assert model_config.intermediate_size > 0
    assert model_config.vocab_size > 0
    assert model_config.rope_theta > 0
    assert model_config.rms_norm_eps > 0


@pytest.mark.skipif(not _has_model(), reason="Qwen2.5 model weights not available")
def test_weight_loader_loading():
    """Load Qwen2.5 weights into QwenModel."""
    import torch
    from mini_vllm.model.qwen_model import QwenModel
    from mini_vllm.model_runner.config_adapter import ConfigAdapter
    from mini_vllm.model.weight_loader import load_qwen_weights

    model_config = ConfigAdapter.from_pretrained(MODEL_PATH)
    model = QwenModel(
        num_layers=model_config.num_layers,
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_heads,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
        intermediate_size=model_config.intermediate_size,
        vocab_size=model_config.vocab_size,
        rms_norm_eps=model_config.rms_norm_eps,
        tie_word_embeddings=model_config.tie_word_embeddings,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.float16)

    load_qwen_weights(model, MODEL_PATH, device=device, dtype=torch.float16)
    model.eval()

    # Verify weights are not zero
    for name, param in model.named_parameters():
        assert param.abs().sum().item() > 0, f"Parameter {name} is all zeros"


@pytest.mark.skipif(not _has_model(), reason="Qwen2.5 model weights not available")
def test_weight_loader_qkv_fusion():
    """Fused QKV weight has correct shape after loading."""
    import torch
    from mini_vllm.model.qwen_model import QwenModel
    from mini_vllm.model_runner.config_adapter import ConfigAdapter
    from mini_vllm.model.weight_loader import load_qwen_weights

    model_config = ConfigAdapter.from_pretrained(MODEL_PATH)
    model = QwenModel(
        num_layers=1,
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_heads,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
        intermediate_size=model_config.intermediate_size,
        vocab_size=model_config.vocab_size,
        rms_norm_eps=model_config.rms_norm_eps,
        tie_word_embeddings=model_config.tie_word_embeddings,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.float16)

    load_qwen_weights(model, MODEL_PATH, device=device, dtype=torch.float16)

    # Check fused QKV weight shape
    q_sz = model_config.num_heads * model_config.head_dim
    kv_sz = model_config.num_kv_heads * model_config.head_dim
    expected = q_sz + 2 * kv_sz
    actual = model.layers[0].attention.qkv_proj.qkv_weight.shape[0]
    assert actual == expected, (
        f"Fused QKV weight rows {actual} != expected {expected}"
    )


@pytest.mark.skipif(not _has_model(), reason="Qwen2.5 model weights not available")
def test_weight_loader_forward():
    """Loaded model can run forward pass (no KV cache)."""
    import torch
    from mini_vllm.model.qwen_model import QwenModel
    from mini_vllm.model_runner.config_adapter import ConfigAdapter
    from mini_vllm.model.weight_loader import load_qwen_weights

    model_config = ConfigAdapter.from_pretrained(MODEL_PATH)
    model = QwenModel(
        num_layers=model_config.num_layers,
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_heads,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
        intermediate_size=model_config.intermediate_size,
        vocab_size=model_config.vocab_size,
        rms_norm_eps=model_config.rms_norm_eps,
        tie_word_embeddings=model_config.tie_word_embeddings,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.float16)

    load_qwen_weights(model, MODEL_PATH, device=device, dtype=torch.float16)
    model.eval()

    with torch.no_grad():
        input_ids = torch.randint(0, model_config.vocab_size, (1, 4), device=device)
        logits = model(input_ids)

    assert logits.shape == (1, 4, model_config.vocab_size)
    assert torch.isfinite(logits).all()
