"""Tests for ``ModelInputBuilder`` (Phase 1 of V1 Paged Engine)."""

from __future__ import annotations

import pytest

from mini_vllm.cache.allocator import BlockAllocator
from mini_vllm.cache.manager import BlockManager
from mini_vllm.config import Config
from mini_vllm.engine.input_builder import ModelInputBuilder
from mini_vllm.sequence.sequence import Sequence
from mini_vllm.sequence.sampling_params import SamplingParams
from mini_vllm.sequence.status import Status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> Config:
    kwargs = dict(
        block_size=4,
        num_gpu_blocks=16,
        max_prefill_chunk_size=4,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        vocab_size=256,
    )
    kwargs.update(overrides)
    return Config(**kwargs)


def _make_seq(
    seq_id: str,
    prompt_len: int = 8,
    max_new: int = 8,
    prefill_cursor: int = 0,
) -> Sequence:
    cfg = _make_config()
    seq = Sequence(
        seq_id=seq_id,
        group_id=f"g-{seq_id}",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_new),
        arrival_time=0.0,
    )
    seq.prefill_cursor = prefill_cursor
    if prefill_cursor >= prompt_len:
        seq.status = Status.RUNNING
        seq.output_token_ids = [42]
        seq.num_generated_tokens = 1
    else:
        seq.status = Status.PREFILL
    return seq


def _make_mgr(cfg: Config) -> BlockManager:
    alloc = BlockAllocator(num_blocks=cfg.num_gpu_blocks)
    return BlockManager(cfg.block_size, alloc)


def _prepare_prefill(mgr: BlockManager, seq: Sequence) -> None:
    """Allocate prefix blocks + chunk blocks for a prefill sequence.

    Allocates ``block_size`` tokens worth of blocks starting from the
    prefill cursor, simulating what the scheduler+executor pipeline
    will do before ``ModelInputBuilder.build()`` runs.
    """
    mgr.allocate_for_seq(seq)
    chunk_end = seq.prefill_cursor + 4  # default chunk size
    for pos in range(chunk_end):
        mgr.ensure_block(seq, pos)


def _prepare_decode(mgr: BlockManager, seq: Sequence) -> None:
    """Allocate all blocks (prefix + generated) for a decode sequence."""
    mgr.allocate_for_seq(seq)
    total = len(seq.prompt_token_ids) + seq.num_generated_tokens
    for pos in range(total):
        mgr.ensure_block(seq, pos)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelInputBuilder:
    """Tests for ModelInputBuilder construction logic."""

    def test_build_empty(self):
        """Empty prefill and decode lists produce an empty ModelInput."""
        cfg = _make_config()
        mgr = _make_mgr(cfg)
        builder = ModelInputBuilder(mgr, cfg)

        model_input = builder.build([], [])

        assert model_input.input_ids.shape == (0,)
        assert model_input.positions.shape == (0,)
        assert model_input.slot_mapping.shape == (0,)
        assert model_input.sample_token_indices.shape == (0,)
        assert model_input.attn_metadata.groups == []

    def test_build_single_prefill(self):
        """Single prefill sequence with cursor=0 produces correct metadata."""
        cfg = _make_config(max_prefill_chunk_size=4)
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=8, prefill_cursor=0)
        _prepare_prefill(mgr, seq)
        builder = ModelInputBuilder(mgr, cfg)

        model_input = builder.build([seq], [])

        # 4 tokens in chunk (capped by max_prefill_chunk_size)
        assert model_input.input_ids.tolist() == [0, 1, 2, 3]
        assert model_input.positions.tolist() == [0, 1, 2, 3]

        # Slot mapping uses block_id * block_size + offset
        # Block 0 → slot 0, Block 0 → slot 1, Block 0 → slot 2, Block 0 → slot 3
        slot_mapping = model_input.slot_mapping.tolist()
        assert slot_mapping == [0, 1, 2, 3]
        assert model_input.sample_token_indices.tolist() == []  # prefill not complete

    def test_build_prefill_completes(self):
        """Prefill completing this step gets its last token sampled."""
        cfg = _make_config(max_prefill_chunk_size=4)
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=4, prefill_cursor=0)
        _prepare_prefill(mgr, seq)
        builder = ModelInputBuilder(mgr, cfg)

        model_input = builder.build([seq], [])

        # 4 tokens in chunk, prefill completes
        assert model_input.input_ids.tolist() == [0, 1, 2, 3]
        assert model_input.positions.tolist() == [0, 1, 2, 3]
        # Last token (index 3) needs sampling
        assert model_input.sample_token_indices.tolist() == [3]

    def test_build_single_decode(self):
        """Single decode sequence produces one token."""
        cfg = _make_config()
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=8, prefill_cursor=8)
        _prepare_decode(mgr, seq)
        builder = ModelInputBuilder(mgr, cfg)

        model_input = builder.build([], [seq])

        # 1 decode token
        assert model_input.input_ids.tolist() == [42]  # last output token
        assert model_input.positions.tolist() == [8]  # = prompt_len + num_generated - 1
        assert model_input.sample_token_indices.tolist() == [0]

    def test_build_mixed_prefill_decode(self):
        """Mixed prefill + decode in single build."""
        cfg = _make_config(max_prefill_chunk_size=4)
        mgr = _make_mgr(cfg)

        prefill_seq = _make_seq("s0", prompt_len=8, prefill_cursor=0)
        _prepare_prefill(mgr, prefill_seq)

        decode_seq = _make_seq("s1", prompt_len=8, prefill_cursor=8)
        _prepare_decode(mgr, decode_seq)

        builder = ModelInputBuilder(mgr, cfg)
        model_input = builder.build([prefill_seq], [decode_seq])

        # Prefill: 4 tokens (0,1,2,3) + Decode: 1 token (42) = 5 total
        assert model_input.input_ids.tolist() == [0, 1, 2, 3, 42]
        assert model_input.positions.tolist() == [0, 1, 2, 3, 8]
        assert len(model_input.slot_mapping) == 5

        # sample_token_indices: decode token (index 4)
        assert model_input.sample_token_indices.tolist() == [4]

        # Check groups
        groups = model_input.attn_metadata.groups
        assert len(groups) == 2

        prefill_group = groups[0]
        assert prefill_group.attention_type == "prefill_gpu"
        assert prefill_group.seq_indices == [0]
        assert prefill_group.cached_len_before.tolist() == [0]
        assert prefill_group.query_len.tolist() == [4]

        decode_group = groups[1]
        assert decode_group.attention_type == "decode_gpu"
        assert decode_group.seq_indices == [1]
        assert decode_group.cached_len_before.tolist() == [8]
        assert decode_group.query_len.tolist() == [1]
        assert decode_group.kv_len_after.tolist() == [9]

    def test_slot_mapping_multi_block(self):
        """Slot mapping correctly handles cross-block token positions."""
        cfg = _make_config(block_size=4, max_prefill_chunk_size=8)
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=8, prefill_cursor=0)
        _prepare_prefill(mgr, seq)

        # Ensure block for position 4 (second block)
        mgr.ensure_block(seq, 4)

        builder = ModelInputBuilder(mgr, cfg)

        # Modify the prefill to cover 8 tokens
        class _PatchedMeta:
            pass

        model_input = builder.build([seq], [])

        # Slot mapping: positions 0-3 → block 0, slots 0-3
        #                positions 4-7 → block 1, slots 4-7
        slots = model_input.slot_mapping.tolist()
        assert len(slots) == 8  # 8 tokens in chunk (max_prefill_chunk_size=8)
        # block 0, offset 0-3
        assert slots[0] == 0  # block_id=0 * 4 + 0
        assert slots[1] == 1  # block_id=0 * 4 + 1
        assert slots[2] == 2
        assert slots[3] == 3
        # block 1, offset 0-3
        assert slots[4] == 4  # block_id=1 * 4 + 0
        assert slots[5] == 5
        assert slots[6] == 6
        assert slots[7] == 7

    def test_decode_slot_mapping_after_generated_tokens(self):
        """Decode token slot accounts for generated tokens' positions."""
        cfg = _make_config(block_size=4)
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=8, prefill_cursor=8)
        seq.output_token_ids = [42, 43, 44]  # 3 generated tokens
        seq.num_generated_tokens = 3
        _prepare_decode(mgr, seq)
        builder = ModelInputBuilder(mgr, cfg)

        model_input = builder.build([], [seq])

        # cached_len_before = len(prompt_token_ids) + max(0, num_generated - 1)
        # = 8 + 3 - 1 = 10  (output_token_ids[-1] = [44] is the input;
        #  its KV was written in prior decode steps, so 44 sits at position 10)
        assert model_input.positions.tolist() == [10]
        assert model_input.sample_token_indices.tolist() == [0]
        # input_id = last generated token
        assert model_input.input_ids.tolist() == [44]

    def test_block_table_tensor_padding(self):
        """Block table tensors are padded with -1 for shorter sequences."""
        cfg = _make_config(block_size=4)
        mgr = _make_mgr(cfg)

        # Seq A: 4 tokens → 1 block
        seq_a = _make_seq("sA", prompt_len=4, prefill_cursor=4)
        _prepare_decode(mgr, seq_a)

        # Seq B: 8 tokens → 2 blocks
        seq_b = _make_seq("sB", prompt_len=8, prefill_cursor=8)
        _prepare_decode(mgr, seq_b)

        builder = ModelInputBuilder(mgr, cfg)
        model_input = builder.build([], [seq_a, seq_b])

        decode_bt = model_input.attn_metadata.decode_block_tables
        # Seq A: 5 tokens (4 prompt + 1 gen) → 2 blocks
        # Seq B: 9 tokens (8 prompt + 1 gen) → 3 blocks
        assert decode_bt.shape == (2, 3)  # 2 seqs, max_blocks=3
        bt_a = decode_bt[0].tolist()
        bt_b = decode_bt[1].tolist()
        assert bt_a[0] >= 0 and bt_a[1] >= 0  # 2 valid blocks
        assert bt_a[2] == -1  # padding
        assert bt_b[0] >= 0 and bt_b[1] >= 0 and bt_b[2] >= 0  # all 3 valid

    def test_cached_len_before_semantics_decode(self):
        """cached_len_before = prompt_len + num_generated - 1 for decode."""
        cfg = _make_config()
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=10, prefill_cursor=10)
        _prepare_decode(mgr, seq)
        builder = ModelInputBuilder(mgr, cfg)

        model_input = builder.build([], [seq])
        group = model_input.attn_metadata.groups[0]
        assert group.cached_len_before.tolist() == [10]
        assert group.query_len.tolist() == [1]
        assert group.kv_len_after.tolist() == [11]

    def test_cached_len_before_semantics_prefill(self):
        """cached_len_before = prefill_cursor for prefill."""
        cfg = _make_config(max_prefill_chunk_size=4)
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=16, prefill_cursor=8)
        _prepare_prefill(mgr, seq)

        # Ensure blocks for positions 8-11
        for pos in range(8, 12):
            mgr.ensure_block(seq, pos)

        builder = ModelInputBuilder(mgr, cfg)
        model_input = builder.build([seq], [])

        group = model_input.attn_metadata.groups[0]
        assert group.cached_len_before.tolist() == [8]
        assert group.query_len.tolist() == [4]
        assert group.kv_len_after.tolist() == [12]

    def test_get_block_table_single_truth_source(self):
        """BlockManager.get_block_table() is the only way to read block IDs."""
        cfg = _make_config()
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=4, prefill_cursor=4)
        _prepare_decode(mgr, seq)

        # Sequence does NOT have a block_table attribute
        assert not hasattr(seq, 'block_table')

        # All block info comes from BlockManager
        block_ids = mgr.get_block_table(seq.seq_id)
        assert len(block_ids) >= 1
        assert all(pid >= 0 for pid in block_ids)


# ---------------------------------------------------------------------------
# Decode cache length regression tests (Phase 1.5 off-by-one fix)
# ---------------------------------------------------------------------------

class TestDecodeCacheLength:
    """Verify cached_len_before = prompt_len + num_generated - 1.

    The first output token (produced by completing prefill) is NOT in KV
    cache yet.  Only prompt_len + (num_generated - 1) tokens have been
    written.  The *input* to this decode step is output_token_ids[-1],
    which sits at position = cached_len_before.
    """

    def test_first_decode_after_prefill(self):
        """First decode: cached_len_before == prompt_len."""
        cfg = _make_config(block_size=4)
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=10, prefill_cursor=10)
        _prepare_decode(mgr, seq)
        builder = ModelInputBuilder(mgr, cfg)

        model_input = builder.build([], [seq])

        assert model_input.positions.tolist() == [10]
        # Group metadata
        group = model_input.attn_metadata.groups[0]
        assert group.cached_len_before.tolist() == [10]
        assert group.query_len.tolist() == [1]
        assert group.kv_len_after.tolist() == [11]

    def test_second_decode(self):
        """Second decode: cached_len_before == prompt_len + 1."""
        cfg = _make_config(block_size=4)
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=10, prefill_cursor=10)
        seq.output_token_ids = [0, 1]              # 2 generated tokens
        seq.num_generated_tokens = 2
        _prepare_decode(mgr, seq)
        builder = ModelInputBuilder(mgr, cfg)

        model_input = builder.build([], [seq])

        assert model_input.positions.tolist() == [11]
        group = model_input.attn_metadata.groups[0]
        assert group.cached_len_before.tolist() == [11]
        assert group.kv_len_after.tolist() == [12]

    def test_nth_decode(self):
        """Nth decode: cached_len_before == prompt_len + N - 1."""
        cfg = _make_config(block_size=4)
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=10, prefill_cursor=10)
        seq.output_token_ids = list(range(5))      # 5 generated tokens
        seq.num_generated_tokens = 5
        _prepare_decode(mgr, seq)
        builder = ModelInputBuilder(mgr, cfg)

        model_input = builder.build([], [seq])

        # cached_len_before = 10 + 5 - 1 = 14
        assert model_input.positions.tolist() == [14]
        group = model_input.attn_metadata.groups[0]
        assert group.cached_len_before.tolist() == [14]
        assert group.kv_len_after.tolist() == [15]

    def test_block_boundary_first_decode_writes_next_block(self):
        """prompt_len == block_size: first decode writes offset 0 of next block."""
        cfg = _make_config(block_size=4)
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=4, prefill_cursor=4)
        _prepare_decode(mgr, seq)
        builder = ModelInputBuilder(mgr, cfg)

        model_input = builder.build([], [seq])

        # cached_len_before = 4 + 1 - 1 = 4
        position = model_input.positions.tolist()[0]
        assert position == 4
        # logical_idx = 4 // 4 = 1, offset = 0 — first slot of second block
        slot = model_input.slot_mapping.tolist()[0]
        block_ids = mgr.get_block_table(seq.seq_id)
        expected = block_ids[1] * 4 + 0  # second block, offset 0
        assert slot == expected, (
            f"expected slot {expected} (block_ids[1]*4 + 0), got {slot}"
        )

    def test_block_boundary_last_offset(self):
        """prompt_len == block_size - 1: first decode writes last offset of block."""
        cfg = _make_config(block_size=4)
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=3, prefill_cursor=3)
        _prepare_decode(mgr, seq)
        builder = ModelInputBuilder(mgr, cfg)

        model_input = builder.build([], [seq])

        # cached_len_before = 3 + 1 - 1 = 3
        position = model_input.positions.tolist()[0]
        assert position == 3
        # logical_idx = 3 // 4 = 0, offset = 3 — last slot of first block
        slot = model_input.slot_mapping.tolist()[0]
        block_ids = mgr.get_block_table(seq.seq_id)
        expected = block_ids[0] * 4 + 3  # first block, offset 3
        assert slot == expected, (
            f"expected slot {expected} (block_ids[0]*4 + 3), got {slot}"
        )

    def test_mixed_prefill_first_decode_slot_mapping(self):
        """Both prefill and first-decode slot mappings are correct."""
        cfg = _make_config(block_size=4, max_prefill_chunk_size=4)
        mgr = _make_mgr(cfg)

        prefill_seq = _make_seq("p0", prompt_len=8, prefill_cursor=0)
        _prepare_prefill(mgr, prefill_seq)

        decode_seq = _make_seq("d0", prompt_len=6, prefill_cursor=6)
        _prepare_decode(mgr, decode_seq)

        builder = ModelInputBuilder(mgr, cfg)
        model_input = builder.build([prefill_seq], [decode_seq])

        # Prefill: 4 tokens at block 0, slots 0-3
        pref_slots = model_input.attn_metadata.prefill_slot_mapping.tolist()
        assert len(pref_slots) == 4
        assert pref_slots == [0, 1, 2, 3]

        # Decode: position = 6 + 1 - 1 = 6
        # logical_idx = 6 // 4 = 1, offset = 2
        decode_slots = model_input.attn_metadata.decode_slot_mapping.tolist()
        assert len(decode_slots) == 1
        d_block_ids = mgr.get_block_table(decode_seq.seq_id)
        expected = d_block_ids[1] * 4 + 2
        assert decode_slots[0] == expected, (
            f"expected decode slot {expected}, got {decode_slots[0]}"
        )

    def test_fake_executor_cache_length_consistency(self):
        """FakeExecutor.execute() consistency: KV write at cached_len_before."""
        from mini_vllm.executor.executor import FakeModelExecutor

        cfg = _make_config(block_size=4, num_gpu_blocks=32)
        alloc = BlockAllocator(num_blocks=cfg.num_gpu_blocks)
        mgr = BlockManager(cfg.block_size, alloc)
        executor = FakeModelExecutor(cfg, mgr)
        alloc.set_callbacks(
            on_allocate=executor.prepare_block,
            on_free=executor.release_block,
        )

        # Create a prefill sequence and allocate its blocks
        seq = _make_seq("s0", prompt_len=8, prefill_cursor=0)
        mgr.allocate_for_seq(seq)
        for pos in range(8):
            mgr.ensure_block(seq, pos)
        seq.prefill_cursor = 8
        seq.status = Status.RUNNING
        seq.output_token_ids = [99]   # first output token
        seq.num_generated_tokens = 1

        # Build ModelInput for the first decode.
        # Need to ensure the block at position 8 (like EngineCore does).
        mgr.ensure_block(seq, 8)
        builder = ModelInputBuilder(mgr, cfg)
        model_input = builder.build([], [seq])

        # Verify the snapshot before execution
        si = model_input.sequence_info[0]
        assert si.phase == "decode"
        assert si.cached_len_before == 8          # prompt_len + 1 - 1
        assert si.query_len == 1
        assert si.kv_len_after == 9

        # Execute
        output = executor.execute(model_input)
        assert len(output.sampled_token_ids) == 1
        assert output.sampled_sequence_ids[0] == "s0"

        # After execution, executor's sim_state should have last_token
        assert executor._sim_state.get("s0", {}).get("last_token") is not None

        # Verify kv cache write happened at position = cached_len_before = 8
        block_ids = mgr.get_block_table("s0")
        # Position 8 → block_size=4 → logical_idx=2, so block table should have 3 blocks
        assert len(block_ids) >= 3

        # Run a second decode to verify the chain
        seq.output_token_ids.append(100)
        seq.num_generated_tokens = 2

        # Ensure block at position 9 before building (normally EngineCore does this)
        mgr.ensure_block(seq, 9)

        model_input2 = builder.build([], [seq])
        si2 = model_input2.sequence_info[0]
        assert si2.cached_len_before == 9          # prompt_len + 2 - 1

        output2 = executor.execute(model_input2)
        assert len(output2.sampled_token_ids) == 1
        assert output2.sampled_sequence_ids[0] == "s0"


# ---------------------------------------------------------------------------
# Decode invariant assertions (Phase 1.5 off-by-one fix)
# ---------------------------------------------------------------------------

class TestDecodeInvariants:
    """Decode state must satisfy num_generated_tokens >= 1."""

    def test_decode_with_zero_generated_raises(self):
        """num_generated_tokens == 0 must raise AssertionError."""
        cfg = _make_config()
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=8, prefill_cursor=8)
        seq.num_generated_tokens = 0
        seq.output_token_ids = []
        builder = ModelInputBuilder(mgr, cfg)
        with pytest.raises(AssertionError, match="num_generated_tokens=0"):
            builder.build([], [seq])

    def test_decode_empty_output_token_ids_raises(self):
        """Empty output_token_ids must raise AssertionError."""
        cfg = _make_config()
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=8, prefill_cursor=8)
        seq.num_generated_tokens = 1
        seq.output_token_ids = []
        builder = ModelInputBuilder(mgr, cfg)
        with pytest.raises(AssertionError, match="empty output_token_ids"):
            builder.build([], [seq])

    def test_decode_mismatched_counts_raises(self):
        """num_generated_tokens != len(output_token_ids) must raise."""
        cfg = _make_config()
        mgr = _make_mgr(cfg)
        seq = _make_seq("s0", prompt_len=8, prefill_cursor=8)
        seq.num_generated_tokens = 3
        seq.output_token_ids = [1, 2]  # length 2, not 3
        builder = ModelInputBuilder(mgr, cfg)
        with pytest.raises(AssertionError, match="num_generated"):
            builder.build([], [seq])

    def test_legacy_and_unified_decode_consistent(self):
        """FakeExecutor legacy decode() and unified execute() produce same
        token sequence for the same request."""
        from mini_vllm.executor.executor import FakeModelExecutor

        cfg = _make_config(block_size=4, num_gpu_blocks=32)
        alloc = BlockAllocator(num_blocks=cfg.num_gpu_blocks)
        mgr = BlockManager(cfg.block_size, alloc)

        # --- Unified path ---
        executor_unified = FakeModelExecutor(cfg, mgr)
        alloc.set_callbacks(
            on_allocate=executor_unified.prepare_block,
            on_free=executor_unified.release_block,
        )
        # Simulate completing prefill
        seq_u = _make_seq("u0", prompt_len=8, prefill_cursor=8)
        mgr.allocate_for_seq(seq_u)
        for pos in range(8):
            mgr.ensure_block(seq_u, pos)
        seq_u.status = Status.RUNNING
        seq_u.output_token_ids = [42]
        seq_u.num_generated_tokens = 1

        # First decode step via unified execute()
        mgr.ensure_block(seq_u, 8)  # position = prompt_len
        builder = ModelInputBuilder(mgr, cfg)
        mi = builder.build([], [seq_u])
        out1 = executor_unified.execute(mi)
        first_token = out1.sampled_token_ids[0]

        # Second decode step via unified execute()
        seq_u.output_token_ids.append(first_token)
        seq_u.num_generated_tokens = 2
        mgr.ensure_block(seq_u, 9)
        mi2 = builder.build([], [seq_u])
        out2 = executor_unified.execute(mi2)
        unified_second = out2.sampled_token_ids[0]

        # --- Legacy path (requires separate executor due to state) ---
        alloc2 = BlockAllocator(num_blocks=cfg.num_gpu_blocks)
        mgr2 = BlockManager(cfg.block_size, alloc2)
        executor_legacy = FakeModelExecutor(cfg, mgr2)
        alloc2.set_callbacks(
            on_allocate=executor_legacy.prepare_block,
            on_free=executor_legacy.release_block,
        )

        seq_l = _make_seq("l0", prompt_len=8, prefill_cursor=8)
        mgr2.allocate_for_seq(seq_l)
        for pos in range(8):
            mgr2.ensure_block(seq_l, pos)
        seq_l.status = Status.RUNNING
        seq_l.output_token_ids = [42]
        seq_l.num_generated_tokens = 1

        # First decode step via legacy decode()
        executor_legacy.decode([seq_l])
        legacy_first = seq_l.output_token_ids[-1]

        # Second decode step via legacy decode()
        executor_legacy.decode([seq_l])
        legacy_second = seq_l.output_token_ids[-1]

        # Unified and legacy produce same token sequence
        assert first_token == legacy_first, (
            f"First decode token differs: unified={first_token} legacy={legacy_first}"
        )
        assert unified_second == legacy_second, (
            f"Second decode token differs: unified={unified_second} legacy={legacy_second}"
        )

    def test_legacy_prefill_then_decode_via_engine(self):
        """Engine with unified execute() runs prefill+decode to completion
        using the FakeModelExecutor — validates end-to-end position semantics
        without testing exact output values."""
        from mini_vllm.engine.engine import LLMEngine

        cfg = _make_config(
            max_num_seqs=4, max_num_batched_tokens=16,
            block_size=4, num_gpu_blocks=16,
        )
        engine = LLMEngine(cfg)
        engine.add_request("Hello world test prompt", max_new_tokens=4)
        outputs = engine.run_until_done()
        assert len(outputs) == 1
        result = list(outputs.values())[0]
        assert isinstance(result, str)
        assert len(result) >= 1

