"""Integration tests for QwenModelRunner.

Requires a real Qwen2.5 model path (``QWEN_MODEL_PATH`` env var or HF cache).
Skipped when no model is available.
"""

import os

import pytest
import torch

MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
)


def _has_model():
    if not os.path.exists(MODEL_PATH):
        return False
    for fname in os.listdir(MODEL_PATH):
        if fname.endswith((".safetensors", ".bin")):
            return True
    return False


@pytest.mark.skipif(not _has_model(), reason="Qwen2.5 model weights not available")
def test_qwen_runner_execute_model():
    """QwenModelRunner.execute_model produces correct logit shape."""
    from mini_vllm.attention.backend import AttentionBackend
    from mini_vllm.model_runner.config_adapter import ConfigAdapter
    from mini_vllm.model_runner.qwen_runner import QwenModelRunner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ConfigAdapter.from_pretrained(MODEL_PATH)
    attention_backend = AttentionBackend.create(model_config, backend="reference")

    runner = QwenModelRunner(
        model_path=MODEL_PATH,
        attention_backend=attention_backend,
        config=model_config,
        device=device,
    )

    # Build a simple prefill-only ModelInput
    from mini_vllm.model_runner.base import (
        AttentionGroup, AttentionMetadata, ModelInput,
    )

    token_ids = torch.tensor([101, 102, 103, 104], device=device)
    positions = torch.tensor([0, 1, 2, 3], device=device)
    slot_mapping = torch.tensor([0, 1, 2, 3], device=device)
    sample_indices = torch.tensor([3], device=device)

    attn_meta = AttentionMetadata(
        groups=[
            AttentionGroup(
                seq_indices=[0],
                attention_type="prefill_gpu",
                cached_len_before=torch.tensor([0], device=device),
                query_len=torch.tensor([4], device=device),
                kv_len_after=torch.tensor([4], device=device),
            ),
        ],
        prefill_slot_mapping=slot_mapping,
        prefill_block_tables=torch.tensor([[0]], device=device),
        prefill_positions=positions,
        decode_block_tables=torch.zeros((0, 1), dtype=torch.long, device=device),
        decode_slot_mapping=torch.tensor([], dtype=torch.long, device=device),
        decode_positions=torch.tensor([], dtype=torch.long, device=device),
        block_size=4,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    )

    model_input = ModelInput(
        input_ids=token_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        attn_metadata=attn_meta,
        sample_token_indices=sample_indices,
    )

    with torch.no_grad():
        logits = runner.execute_model(model_input)

    assert logits.shape == (1, model_config.vocab_size), (
        f"Logits shape: {logits.shape}, expected (1, {model_config.vocab_size})"
    )
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not _has_model(), reason="Qwen2.5 model weights not available")
def test_qwen_runner_pool_allocated():
    """QwenModelRunner allocates KV cache pool at init."""
    from mini_vllm.attention.backend import AttentionBackend
    from mini_vllm.model_runner.config_adapter import ConfigAdapter
    from mini_vllm.model_runner.qwen_runner import QwenModelRunner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ConfigAdapter.from_pretrained(MODEL_PATH)
    attention_backend = AttentionBackend.create(model_config, backend="reference")

    runner = QwenModelRunner(
        model_path=MODEL_PATH,
        attention_backend=attention_backend,
        config=model_config,
        device=device,
    )

    pool = runner.pool
    assert pool.num_layers == model_config.num_layers
    assert pool.num_kv_heads == model_config.num_kv_heads
    assert pool.head_dim == model_config.head_dim
    assert pool.num_blocks > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(not _has_model(), reason="Qwen2.5 model weights not available")
def test_qwen_runner_decode_single_token():
    """ModelRunner can run decode after prefill."""
    from mini_vllm.attention.backend import AttentionBackend
    from mini_vllm.model_runner.config_adapter import ConfigAdapter
    from mini_vllm.model_runner.qwen_runner import QwenModelRunner
    from mini_vllm.model_runner.base import (
        AttentionGroup, AttentionMetadata, ModelInput,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ConfigAdapter.from_pretrained(MODEL_PATH)
    attention_backend = AttentionBackend.create(model_config, backend="reference")
    pool = attention_backend.allocate_pool(
        num_layers=model_config.num_layers,
        num_blocks=32,
        block_size=4,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
        dtype=torch.float16,
        device=device,
    )

    runner = QwenModelRunner(
        model_path=MODEL_PATH,
        attention_backend=attention_backend,
        config=model_config,
        device=device,
    )

    # Prefill 4 tokens, slot 0-3
    pref_ids = torch.tensor([101, 102, 103, 104], device=device)
    pref_pos = torch.tensor([0, 1, 2, 3], device=device)
    pref_slots = torch.tensor([0, 1, 2, 3], device=device)

    pref_meta = AttentionMetadata(
        groups=[
            AttentionGroup(
                seq_indices=[0],
                attention_type="prefill_gpu",
                cached_len_before=torch.tensor([0], device=device),
                query_len=torch.tensor([4], device=device),
                kv_len_after=torch.tensor([4], device=device),
            ),
        ],
        prefill_slot_mapping=pref_slots,
        prefill_block_tables=torch.tensor([[0]], device=device),
        prefill_positions=pref_pos,
        decode_block_tables=torch.zeros((0, 1), dtype=torch.long, device=device),
        decode_slot_mapping=torch.tensor([], dtype=torch.long, device=device),
        decode_positions=torch.tensor([], dtype=torch.long, device=device),
        block_size=4,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    )

    pref_input = ModelInput(
        input_ids=pref_ids,
        positions=pref_pos,
        slot_mapping=pref_slots,
        attn_metadata=pref_meta,
        sample_token_indices=torch.tensor([3], device=device),
    )

    with torch.no_grad():
        pref_logits = runner.execute_model(pref_input)
    assert pref_logits.shape == (1, model_config.vocab_size)
