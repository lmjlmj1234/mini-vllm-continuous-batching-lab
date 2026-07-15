"""Tests for BlockTable, BlockAllocator, and BlockManager."""

import pytest

from mini_vllm import BlockAllocator, BlockManager, BlockTable, Config, SamplingParams
from mini_vllm.sequence.sequence import Sequence


def _make_seq(seq_id="s0", group_id="g0", prompt_len=4, max_new=4) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        group_id=group_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_new),
        arrival_time=0.0,
    )


class TestBlockTable:
    def test_add_and_count(self) -> None:
        table = BlockTable("r0", block_size=4)
        assert table.num_blocks() == 0
        table.add_block(12)
        table.add_block(7)
        assert table.num_blocks() == 2
        assert table.get_block_ids() == [12, 7]

    def test_get_physical_block(self) -> None:
        table = BlockTable("r0", block_size=4)
        table.add_block(10)
        table.add_block(20)
        assert table.get_physical_block(0) == 10
        assert table.get_physical_block(3) == 10
        assert table.get_physical_block(4) == 20
        assert table.get_physical_block(7) == 20
        assert table.get_physical_block(8) is None

    def test_clear(self) -> None:
        table = BlockTable("r0", block_size=4)
        table.add_block(1)
        table.add_block(2)
        table.clear()
        assert table.num_blocks() == 0


class TestBlockAllocator:
    def test_allocate_and_free(self) -> None:
        alloc = BlockAllocator(num_blocks=8)
        assert alloc.num_free_blocks == 8

        pids = alloc.allocate(3)
        assert pids == [0, 1, 2]
        assert alloc.num_free_blocks == 5

        alloc.free(pids)
        assert alloc.num_free_blocks == 8

    def test_oom(self) -> None:
        alloc = BlockAllocator(num_blocks=2)
        t1 = alloc.allocate(2)
        assert t1 is not None
        t2 = alloc.allocate(1)
        assert t2 is None

    def test_callback_on_allocate(self) -> None:
        events = []
        alloc = BlockAllocator(
            num_blocks=4,
            on_allocate=lambda pid: events.append(("alloc", pid)),
            on_free=lambda pid: events.append(("free", pid)),
        )
        pids = alloc.allocate(2)
        assert events == [("alloc", 0), ("alloc", 1)]
        alloc.free(pids)
        assert events == [("alloc", 0), ("alloc", 1), ("free", 0), ("free", 1)]

    def test_stats(self) -> None:
        alloc = BlockAllocator(num_blocks=8)
        alloc.allocate(2)
        s = alloc.stats()
        assert s["total_blocks"] == 8
        assert s["free_blocks"] == 6
        assert s["used_blocks"] == 2


class TestBlockManager:
    def test_allocate_for_seq_starts_empty(self) -> None:
        """On-demand: allocate_for_seq creates empty block table."""
        config = Config(num_gpu_blocks=8, block_size=4)
        alloc = BlockAllocator(num_blocks=config.num_gpu_blocks)
        mgr = BlockManager(config.block_size, alloc)

        seq = _make_seq()
        mgr.allocate_for_seq(seq)
        # No blocks allocated at admission
        assert mgr.get_block_table(seq.seq_id) == []
        assert alloc.num_free_blocks == 8
        # Block table is registered internally
        assert mgr.get_table(seq.seq_id) is not None

    def test_ensure_block_allocates_on_demand(self) -> None:
        """ensure_block() allocates one block at a time."""
        config = Config(num_gpu_blocks=8, block_size=4)
        alloc = BlockAllocator(num_blocks=config.num_gpu_blocks)
        mgr = BlockManager(config.block_size, alloc)

        seq = _make_seq()
        mgr.allocate_for_seq(seq)
        assert len(mgr.get_block_table(seq.seq_id)) == 0
        assert alloc.num_free_blocks == 8

        # First token: position 0 → allocate block 0
        pid = mgr.ensure_block(seq, 0)
        assert pid == 0
        assert len(mgr.get_block_table(seq.seq_id)) == 1
        assert alloc.num_free_blocks == 7

        # Same block position 1-3: no new alloc
        pid = mgr.ensure_block(seq, 3)
        assert pid == 0
        assert len(mgr.get_block_table(seq.seq_id)) == 1

        # Token position 4 → new logical block → allocate block 1
        pid = mgr.ensure_block(seq, 4)
        assert pid == 1
        assert len(mgr.get_block_table(seq.seq_id)) == 2
        assert alloc.num_free_blocks == 6

    def test_free(self) -> None:
        config = Config(num_gpu_blocks=8, block_size=4)
        alloc = BlockAllocator(num_blocks=config.num_gpu_blocks)
        mgr = BlockManager(config.block_size, alloc)

        seq = _make_seq()
        mgr.allocate_for_seq(seq)
        mgr.ensure_block(seq, 0)
        mgr.ensure_block(seq, 4)
        assert alloc.num_free_blocks == 6

        mgr.free(seq.seq_id)
        assert alloc.num_free_blocks == 8
        assert mgr.get_table(seq.seq_id) is None

    def test_oom_during_ensure_block(self) -> None:
        """OOM occurs at execution time, not admission."""
        config = Config(num_gpu_blocks=2, block_size=4)
        alloc = BlockAllocator(num_blocks=config.num_gpu_blocks)
        mgr = BlockManager(config.block_size, alloc)

        seq1 = _make_seq("s1")
        mgr.allocate_for_seq(seq1)
        mgr.ensure_block(seq1, 0)   # alloc block 0
        mgr.ensure_block(seq1, 4)   # alloc block 1  → pool exhausted

        seq2 = _make_seq("s2")
        # Different prompt so prefix cache can't share
        seq2.prompt_token_ids = [9, 10, 11, 12]
        mgr.allocate_for_seq(seq2)
        import pytest
        with pytest.raises(RuntimeError, match="OOM"):
            mgr.ensure_block(seq2, 0)  # no free blocks left

    def test_stats(self) -> None:
        config = Config(num_gpu_blocks=8, block_size=4)
        alloc = BlockAllocator(num_blocks=config.num_gpu_blocks)
        mgr = BlockManager(config.block_size, alloc)

        seq = _make_seq()
        mgr.allocate_for_seq(seq)
        mgr.ensure_block(seq, 0)
        mgr.ensure_block(seq, 4)
        s = mgr.stats()
        assert s["total_blocks"] == 8
        assert s["free_blocks"] == 6
        assert s["used_blocks"] == 2
