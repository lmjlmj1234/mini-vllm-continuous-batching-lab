"""End-to-end tests for PagedExecutor with EngineCore.

Requires a real Qwen2.5 model path. Skipped when unavailable.
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
def test_paged_executor_execute():
    """PagedExecutor.execute returns ModelRunnerOutput."""
    from mini_vllm.executor.paged_executor import PagedExecutor
    from mini_vllm.config import Config
    from mini_vllm.model_runner.base import (
        AttentionGroup, AttentionMetadata, ModelInput,
        SequenceExecutionInfo,
    )
    from mini_vllm.cache.allocator import BlockAllocator
    from mini_vllm.cache.manager import BlockManager

    config = Config(
        model_path=MODEL_PATH,
        executor_type="paged",
        block_size=4,
        num_gpu_blocks=32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allocator = BlockAllocator(num_blocks=32)
    block_manager = BlockManager(block_size=4, allocator=allocator)

    executor = PagedExecutor(config, block_manager=block_manager)
    assert executor._block_manager is not None

    # Allocate blocks for the sequence
    pid = allocator.allocate(1)[0]

    # Build ModelInput for single prefill
    dummy_seq_id = "test-seq-0"
    token_ids = torch.tensor([101, 102, 103, 104], device=device)
    positions = torch.tensor([0, 1, 2, 3], device=device)
    slots = torch.tensor([0, 1, 2, 3], device=device)

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
        prefill_slot_mapping=slots,
        prefill_block_tables=torch.tensor([[pid]], device=device),
        prefill_positions=positions,
        decode_block_tables=torch.zeros((0, 1), dtype=torch.long, device=device),
        decode_slot_mapping=torch.tensor([], dtype=torch.long, device=device),
        decode_positions=torch.tensor([], dtype=torch.long, device=device),
        block_size=4,
        num_kv_heads=0,
        head_dim=0,
    )

    model_input = ModelInput(
        input_ids=token_ids,
        positions=positions,
        slot_mapping=slots,
        attn_metadata=attn_meta,
        sample_token_indices=torch.tensor([3], device=device),
        sequence_info=(
            SequenceExecutionInfo(
                sequence_id=dummy_seq_id,
                phase="prefill",
                query_start=0,
                query_len=4,
                cached_len_before=0,
                kv_len_after=4,
                sample_output_index=0,
            ),
        ),
    )

    output = executor.execute(model_input)

    assert len(output.sampled_token_ids) == 1
    assert len(output.sampled_sequence_ids) == 1
    assert output.sampled_sequence_ids[0] == dummy_seq_id

    # Cleanup
    allocator.free([pid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(not _has_model(), reason="Qwen2.5 model weights not available")
def test_paged_executor_kv_stats():
    """PagedExecutor.get_kv_stats() returns correct structure."""
    from mini_vllm.executor.paged_executor import PagedExecutor
    from mini_vllm.config import Config

    config = Config(
        model_path=MODEL_PATH,
        executor_type="paged",
    )

    executor = PagedExecutor(config)
    stats = executor.get_kv_stats()

    assert "num_blocks" in stats
    assert "num_layers" in stats
    assert "num_kv_heads" in stats
    assert "head_dim" in stats
    assert stats["num_layers"] > 0
    assert stats["num_kv_heads"] > 0
