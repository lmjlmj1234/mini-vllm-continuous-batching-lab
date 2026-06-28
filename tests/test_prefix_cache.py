"""Tests for Prefix Cache: caching, sharing, ref counts, and integration."""

import pytest

from mini_vllm import (
    BlockAllocator,
    BlockManager,
    BlockTable,
    Config,
    LLMEngine,
    PrefixCache,
    PrefixCacheProbeResult,
    SamplingParams,
    SequenceGroup,
    Status,
)
from mini_vllm.cache.prefix_cache import compute_block_hashes
from mini_vllm.sequence.sequence import Sequence


def _make_seq(seq_id="s0", group_id="g0", prompt_len=4, max_new=4) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        group_id=group_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_new),
        arrival_time=0.0,
    )


# ======================================================================
# PrefixCache unit tests
# ======================================================================


class TestPrefixCache:
    def test_empty_cache_returns_none(self) -> None:
        cache = PrefixCache()
        assert cache.lookup(42) is None
        assert cache.size() == 0

    def test_insert_and_lookup(self) -> None:
        cache = PrefixCache()
        cache.insert(123, 5)
        assert cache.lookup(123) == 5
        assert cache.size() == 1

    def test_lookup_span_all_hits(self) -> None:
        cache = PrefixCache()
        for h, pid in [(10, 1), (20, 2), (30, 3)]:
            cache.insert(h, pid)
        hashes = [10, 20, 30]
        assert cache.lookup_span(hashes) == [1, 2, 3]

    def test_lookup_span_with_misses(self) -> None:
        cache = PrefixCache()
        cache.insert(10, 1)
        cache.insert(30, 3)
        hashes = [10, 20, 30]
        assert cache.lookup_span(hashes) == [1, None, 3]

    def test_insert_span(self) -> None:
        cache = PrefixCache()
        hashes = [10, 20, 30]
        pids = [1, 2, 3]
        cache.insert_span(hashes, pids)
        assert cache.lookup(10) == 1
        assert cache.lookup(20) == 2
        assert cache.lookup(30) == 3

    def test_hash_determinism(self) -> None:
        tokens = [101, 102, 103, 104]
        h1 = compute_block_hashes(tokens, block_size=2)
        h2 = compute_block_hashes(tokens, block_size=2)
        assert h1 == h2

    def test_hash_partial_block(self) -> None:
        """Last block may be smaller than block_size."""
        tokens = [101, 102, 103]
        hashes = compute_block_hashes(tokens, block_size=4)
        assert len(hashes) == 1  # one partial block


# ======================================================================
# BlockAllocator ref-count unit tests
# ======================================================================


class TestBlockAllocatorRefCount:
    def test_allocate_sets_ref_count_one(self) -> None:
        alloc = BlockAllocator(num_blocks=8)
        pids = alloc.allocate(2)
        assert alloc.get_ref_count(0) == 1
        assert alloc.get_ref_count(1) == 1

    def test_increment_ref(self) -> None:
        alloc = BlockAllocator(num_blocks=8)
        pids = alloc.allocate(1)
        assert alloc.get_ref_count(0) == 1
        alloc.increment_ref(0)
        assert alloc.get_ref_count(0) == 2

    def test_free_decrements_ref_and_releases_at_zero(self) -> None:
        alloc = BlockAllocator(num_blocks=8)
        pid = alloc.allocate(1)[0]
        alloc.increment_ref(pid)  # ref = 2
        assert alloc.num_free_blocks == 7

        alloc.free([pid])  # ref = 1
        assert alloc.num_free_blocks == 7  # not freed yet

        alloc.free([pid])  # ref = 0 → freed
        assert alloc.num_free_blocks == 8

    def test_on_free_called_only_when_ref_reaches_zero(self) -> None:
        events = []
        alloc = BlockAllocator(
            num_blocks=8,
            on_free=lambda pid: events.append(("free", pid)),
        )
        pid = alloc.allocate(1)[0]
        alloc.increment_ref(pid)  # ref = 2
        alloc.free([pid])         # ref = 1 → no on_free
        assert events == []       # still alive (other reference)
        alloc.free([pid])         # ref = 0 → on_free
        assert events == [("free", pid)]

    def test_double_free_is_safe(self) -> None:
        alloc = BlockAllocator(num_blocks=8)
        pid = alloc.allocate(1)[0]
        alloc.free([pid])   # ref 0 → freed
        alloc.free([pid])   # already free → no-op
        assert alloc.num_free_blocks == 8


# ======================================================================
# BlockManager Prefix Cache integration tests
# ======================================================================


def _make_mgr(num_blocks=16, block_size=4) -> tuple:
    alloc = BlockAllocator(num_blocks=num_blocks)
    mgr = BlockManager(block_size, alloc)
    return mgr, alloc


class TestBlockManagerPrefixCache:
    def test_allocate_for_seq_without_cache(self) -> None:
        """First request: no cache hit, shared_prefix_blocks remains 0."""
        mgr, _ = _make_mgr()
        seq = _make_seq("s0", prompt_len=8)
        mgr.allocate_for_seq(seq)

        assert seq.block_table == []  # no blocks allocated
        assert mgr.get_shared_prefix_length("s0") == 0

    def test_ensure_block_registers_in_cache(self) -> None:
        """New blocks for prompt tokens are registered in prefix cache."""
        mgr, alloc = _make_mgr()
        seq = _make_seq("s0", prompt_len=8)

        mgr.allocate_for_seq(seq)
        # First block boundary
        mgr.ensure_block(seq, 0)
        # Second block boundary
        mgr.ensure_block(seq, 4)

        # Two blocks should now be in the cache
        assert mgr.prefix_cache.size() == 2

    def test_shared_prefix_blocks_not_written(self) -> None:
        """Second sequence with same prefix shares blocks."""
        mgr, alloc = _make_mgr()

        # First sequence: populate cache
        seq_a = _make_seq("sA", prompt_len=8)
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)  # alloc block 0, register in cache
        mgr.ensure_block(seq_a, 4)  # alloc block 1, register in cache

        # Second sequence: same prompt → should share
        seq_b = _make_seq("sB", prompt_len=8)
        mgr.allocate_for_seq(seq_b)

        assert mgr.get_shared_prefix_length("sB") == 2  # both blocks shared
        assert seq_b.block_table == seq_a.block_table  # same PIDs

        # Both blocks should be shared (check BlockTable entries)
        tbl_b = mgr.get_table("sB")
        assert tbl_b is not None
        entries = tbl_b.get_entries()
        assert entries[0].is_shared
        assert entries[1].is_shared

    def test_ref_count_increases_with_each_sharer(self) -> None:
        """Sharing a block increments its ref_count."""
        mgr, alloc = _make_mgr()

        seq_a = _make_seq("sA", prompt_len=4)
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)  # alloc block 0, ref=1

        assert alloc.get_ref_count(0) == 1

        seq_b = _make_seq("sB", prompt_len=4)
        mgr.allocate_for_seq(seq_b)
        assert alloc.get_ref_count(0) == 2  # shared

        seq_c = _make_seq("sC", prompt_len=4)
        mgr.allocate_for_seq(seq_c)
        assert alloc.get_ref_count(0) == 3  # shared again

    def test_free_sharer_does_not_release_block(self) -> None:
        """Freeing one sharer keeps the block alive for others."""
        mgr, alloc = _make_mgr()

        seq_a = _make_seq("sA", prompt_len=4)
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)

        seq_b = _make_seq("sB", prompt_len=4)
        mgr.allocate_for_seq(seq_b)

        assert alloc.get_ref_count(0) == 2
        assert alloc.num_free_blocks == 15  # 15 of 16 free (block 0 used)

        # Free first sequence
        mgr.free("sA")
        assert alloc.get_ref_count(0) == 1  # still referenced by sB
        assert alloc.num_free_blocks == 15  # block 0 still in use

        # Free second sequence
        mgr.free("sB")
        assert alloc.get_ref_count(0) == 0  # truly freed
        assert alloc.num_free_blocks == 16  # back to full pool

    def test_partial_prefix_match(self) -> None:
        """Only matching prefix blocks are shared; rest allocated on-demand."""
        mgr, alloc = _make_mgr(block_size=4, num_blocks=16)

        # Sequence A: tokens [0, 1, 2, 3, 10, 11, 12, 13]
        seq_a = _make_seq("sA", prompt_len=8)
        seq_a.prompt_token_ids = [0, 1, 2, 3, 10, 11, 12, 13]
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)  # alloc block 0 → hash for [0,1,2,3]
        mgr.ensure_block(seq_a, 4)  # alloc block 1 → hash for [10,11,12,13]

        # Sequence B: tokens [0, 1, 2, 3, 20, 21, 22, 23]
        seq_b = _make_seq("sB", prompt_len=8)
        seq_b.prompt_token_ids = [0, 1, 2, 3, 20, 21, 22, 23]
        mgr.allocate_for_seq(seq_b)

        # Block 0 should be shared (same hash), block 1 NOT shared (different)
        assert mgr.get_shared_prefix_length("sB") == 1
        tbl_b = mgr.get_table("sB")
        assert tbl_b is not None
        assert tbl_b.num_blocks() == 1  # only block 0 prepopulated
        assert tbl_b.get_entries()[0].is_shared

        # Block 1 should be allocated on-demand
        mgr.ensure_block(seq_b, 4)
        assert seq_b.block_table is not None
        assert len(seq_b.block_table) == 2  # 2 blocks total

    def test_is_block_shared(self) -> None:
        """is_block_shared correctly identifies shared vs owned blocks."""
        mgr, alloc = _make_mgr()

        seq_a = _make_seq("sA", prompt_len=4)
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)

        seq_b = _make_seq("sB", prompt_len=4)
        mgr.allocate_for_seq(seq_b)

        # Position 0 → in shared block
        assert mgr.is_block_shared(seq_b, 0) is True
        assert mgr.is_block_shared(seq_b, 1) is True  # same block
        assert mgr.is_block_shared(seq_b, 3) is True  # same block

    def test_no_prefix_match_for_different_prompts(self) -> None:
        """Entirely different prompt → no sharing."""
        mgr, alloc = _make_mgr()

        seq_a = _make_seq("sA", prompt_len=4)
        seq_a.prompt_token_ids = [0, 1, 2, 3]
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)

        seq_b = _make_seq("sB", prompt_len=4)
        seq_b.prompt_token_ids = [100, 101, 102, 103]  # different!
        mgr.allocate_for_seq(seq_b)

        assert mgr.get_shared_prefix_length("sB") == 0
        assert mgr.is_block_shared(seq_b, 0) is False

    def test_late_arrival_can_share(self) -> None:
        """A sequence arriving after another finished can still share."""
        mgr, alloc = _make_mgr()

        # First request: run to completion
        seq_a = _make_seq("sA", prompt_len=4)
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)  # register in cache
        mgr.free("sA")

        # Block is freed, but cache entry exists
        assert mgr.prefix_cache.size() == 1

        # Second request: should find stale cache entry and re-create
        # (ref_count was 0, so allocate_for_seq skips the stale entry)
        seq_b = _make_seq("sB", prompt_len=4)
        mgr.allocate_for_seq(seq_b)
        assert mgr.get_shared_prefix_length("sB") == 0  # stale → no share

        # ensure_block will re-allocate and re-register
        mgr.ensure_block(seq_b, 0)
        assert mgr.prefix_cache.size() == 1  # re-registered
        assert alloc.get_ref_count(0) == 1  # now sB owns it

    def test_cache_persists_after_partial_free(self) -> None:
        """Blocks shared by multiple sequences persist until last ref."""
        mgr, alloc = _make_mgr()

        seq_a = _make_seq("sA", prompt_len=4)
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)

        seq_b = _make_seq("sB", prompt_len=4)
        mgr.allocate_for_seq(seq_b)

        mgr.free("sA")  # ref: 2→1, block still alive
        assert alloc.get_ref_count(0) == 1
        assert alloc.num_free_blocks == 15  # block 0 still used by sB

        # seq_b can still read from block 0
        pid = mgr.ensure_block(seq_b, 0)
        assert pid == seq_a.block_table[0]  # same physical block


# ======================================================================
# PrefixCache Probe tests
# ======================================================================


class TestPrefixCacheProbe:
    """Tests for read-only probe_prefix_cache and Scheduler integration."""

    def test_probe_returns_correct_cached_count(self) -> None:
        """Probe returns correct matched_block_count and cached_token_count."""
        mgr, alloc = _make_mgr(block_size=4, num_blocks=16)

        # Populate cache with first sequence
        seq_a = _make_seq("sA", prompt_len=8)
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)  # alloc block 0 → register in cache
        mgr.ensure_block(seq_a, 4)  # alloc block 1 → register in cache

        # Probe second sequence with same prompt
        tokens = list(range(8))
        probe = mgr.probe_prefix_cache(tokens)

        assert probe.matched_block_count == 2
        assert probe.cached_token_count == 8  # 2 blocks * 4 block_size
        assert len(probe.matched_physical_block_ids) == 2

    def test_probe_does_not_change_ref_count(self) -> None:
        """Probe is read-only: ref_count must not change."""
        mgr, alloc = _make_mgr(block_size=4, num_blocks=16)

        # Populate cache
        seq_a = _make_seq("sA", prompt_len=4)
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)  # alloc block 0, ref=1

        assert alloc.get_ref_count(0) == 1

        # Probe — should NOT change ref_count
        tokens = list(range(4))
        probe = mgr.probe_prefix_cache(tokens)
        assert probe.matched_block_count == 1
        assert alloc.get_ref_count(0) == 1  # unchanged

        # Another probe — still unchanged
        probe2 = mgr.probe_prefix_cache(tokens)
        assert probe2.matched_block_count == 1
        assert alloc.get_ref_count(0) == 1  # still 1

    def test_allocate_for_seq_changes_ref_count_after_probe(self) -> None:
        """After probe returns correct info, allocate_for_seq actually increments ref.

        This validates the two-phase flow: Scheduler probes (read-only),
        then BlockManager attaches (ref_count++).
        """
        mgr, alloc = _make_mgr(block_size=4, num_blocks=16)

        # Populate cache
        seq_a = _make_seq("sA", prompt_len=4)
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)
        assert alloc.get_ref_count(0) == 1

        # Probe
        tokens = list(range(4))
        probe = mgr.probe_prefix_cache(tokens)
        assert probe.matched_block_count == 1
        assert alloc.get_ref_count(0) == 1  # probe didn't change

        # Now allocate — ref_count should increase
        seq_b = _make_seq("sB", prompt_len=4)
        mgr.allocate_for_seq(seq_b)
        assert alloc.get_ref_count(0) == 2  # allocate_for_seq incremented

    def test_probe_stale_cache_entry_returns_zero(self) -> None:
        """A stale cache entry (ref_count=0) is excluded from probe result."""
        mgr, alloc = _make_mgr(block_size=4, num_blocks=16)

        # Populate cache
        seq_a = _make_seq("sA", prompt_len=4)
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)
        assert mgr.prefix_cache.size() == 1

        # Free all references to the block
        mgr.free("sA")
        assert alloc.get_ref_count(0) == 0  # fully freed
        assert mgr.prefix_cache.size() == 1  # cache entry still exists (stale)

        # Probe — stale entry, should return 0 matched blocks
        tokens = list(range(4))
        probe = mgr.probe_prefix_cache(tokens)
        assert probe.matched_block_count == 0
        assert probe.cached_token_count == 0
        assert probe.matched_physical_block_ids == []

    def test_probe_partial_prefix(self) -> None:
        """Only consecutive blocks from index 0 count."""
        mgr, alloc = _make_mgr(block_size=4, num_blocks=16)

        # Populate cache with tokens [0,1,2,3, 10,11,12,13]
        seq_a = _make_seq("sA", prompt_len=8)
        seq_a.prompt_token_ids = [0, 1, 2, 3, 10, 11, 12, 13]
        mgr.allocate_for_seq(seq_a)
        mgr.ensure_block(seq_a, 0)  # hash for [0,1,2,3] → cache
        mgr.ensure_block(seq_a, 4)  # hash for [10,11,12,13] → cache

        # Probe with tokens [0,1,2,3, 20,21,22,23]
        tokens = [0, 1, 2, 3, 20, 21, 22, 23]
        probe = mgr.probe_prefix_cache(tokens)

        # Only block 0 matches (same hash for [0,1,2,3])
        assert probe.matched_block_count == 1
        assert probe.cached_token_count == 4  # 1 block * 4

    def test_probe_empty_cache(self) -> None:
        """Empty cache should return zero matches."""
        mgr, alloc = _make_mgr(block_size=4, num_blocks=16)

        tokens = [0, 1, 2, 3]
        probe = mgr.probe_prefix_cache(tokens)
        assert probe.matched_block_count == 0
        assert probe.cached_token_count == 0


# ======================================================================
# Scheduler-aware Prefix Cache integration tests
# ======================================================================


def _make_engine_config(**kwargs) -> Config:
    """Create a config suitable for prefix cache engine tests."""
    defaults = dict(
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_num_prefill_tokens=32,
        max_prefill_chunk_size=16,
        block_size=4,
        num_gpu_blocks=32,
        chunked_prefill_enabled=True,
        print_step_events=False,
    )
    defaults.update(kwargs)
    return Config(**defaults)


class TestSchedulerPrefixCache:
    """Scheduler-aware prefix cache: budget and ScheduleResult changes."""

    def test_cache_not_populated_yet(self) -> None:
        """ScheduleResult has zero cache stats when there's no prefix hit."""
        config = _make_engine_config()
        engine = LLMEngine(config)
        engine.add_request("Hello world", max_new_tokens=4)
        result = engine.step()

        # No cache hit for the first request
        assert result.cached_token_count == 0
        assert result.num_uncached_prefill_tokens > 0  # some prefill happened
        assert result.matched_block_count == 0
        # num_prefill_tokens == num_uncached_prefill_tokens (no cache)
        assert result.num_prefill_tokens == result.num_uncached_prefill_tokens

    def test_cache_hit_reduces_prefill_tokens_in_scheduler(self) -> None:
        """Second request with same prompt has cached_token_count > 0.

        The first request's blocks must still be alive for the cache
        to be valid.  We keep A running while admitting B.
        """
        config = _make_engine_config(
            max_num_seqs=4,
            num_gpu_blocks=64,
        )
        engine = LLMEngine(config)

        # Request A: populate cache (keep A running so blocks stay alive)
        engine.add_request("Hello world", max_new_tokens=16)
        engine.step()  # admit A, prefill → cache populated, A → RUNNING

        # Request B: same prompt → should see cache hit
        engine.add_request("Hello world", max_new_tokens=4)
        result = engine.step()

        assert result.cached_token_count > 0, (
            f"Expected cached_token_count > 0, got {result.cached_token_count}"
        )
        # At least some of A's blocks should be shared with B
        assert result.matched_block_count > 0

    def test_cache_hit_compare_with_no_cache(self) -> None:
        """Same request consumes less prefill budget with prefix cache."""
        config_with_cache = _make_engine_config(
            max_num_seqs=4,
            num_gpu_blocks=64,
        )
        engine1 = LLMEngine(config_with_cache)

        # First request populates cache (keep alive)
        engine1.add_request("The capital of France is", max_new_tokens=16)
        engine1.step()  # admit A, prefill fills cache, A → RUNNING

        # Second request with same prompt → cache hit
        engine1.add_request("The capital of France is", max_new_tokens=4)
        result_cache = engine1.step()

        # Now run without any cache (fresh engine, first request)
        config_no_cache = _make_engine_config()
        engine2 = LLMEngine(config_no_cache)
        engine2.add_request("The capital of France is", max_new_tokens=4)
        result_no_cache = engine2.step()

        # With cache: should have cached tokens > 0
        assert result_cache.cached_token_count > 0
        # Cached version computes fewer tokens
        # (the full prompt includes cached tokens that need no compute)
        assert result_cache.num_uncached_prefill_tokens <= result_no_cache.num_prefill_tokens

    def test_identical_requests_metrics_show_cache_hits(self) -> None:
        """MetricsCollector reports cached token counts."""
        config = _make_engine_config(
            max_num_seqs=1,
            max_num_batched_tokens=16,
            max_prefill_chunk_size=4,
            block_size=4,
            num_gpu_blocks=32,
        )
        engine = LLMEngine(config)

        # First request: long prompt, populates cache
        engine.add_request("A" * 10, max_new_tokens=2)
        engine.run_until_done()

        # Second request: same prompt → cache hit
        engine.add_request("A" * 10, max_new_tokens=2)
        engine.run_until_done()

        # Verify via metrics that cached tokens were tracked
        metrics = engine.engine_core.metrics_collector
        report = metrics.report()
        assert "total_cached_tokens" in report
        assert "prefix_cache_hit_rate" in report
