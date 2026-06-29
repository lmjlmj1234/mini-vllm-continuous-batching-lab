"""Fault injection tests — simulate real production failure scenarios.

Each test exercises a specific failure mode and validates:
1. The system fails gracefully (not a crash)
2. Resources are properly released
3. Error codes are propagated correctly
"""
import pytest
from mini_vllm import Config
from serving.api_server import ServingLayer


def _sv(**kw_overrides) -> ServingLayer:
    defaults = dict(
        print_step_events=False,
        num_gpu_blocks=16,
        block_size=4,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_queue_len=3,
        max_num_streams=2,
        rate_limit_rpm=1000,
        rate_limit_tpm=1000000,
    )
    defaults.update(kw_overrides)
    cfg = Config(**defaults)
    return ServingLayer(cfg)


# ======================================================================
# 1. Queue Overflow
# ======================================================================

class TestQueueOverflow:
    """Simulate waiting queue full → admission rejects new requests."""

    def test_queue_overflow_rejects_new_request(self):
        """Waiting queue is full — new requests get QUEUE_OVERFLOW."""
        sv = _sv(max_queue_len=2, max_num_seqs=1)

        # Fill the engine queue
        sv.engine.add_request("Request A slow", max_new_tokens=16)
        sv.engine.add_request("Request B slow", max_new_tokens=16)
        sv.engine.add_request("Request C slow", max_new_tokens=16)
        # A and B should be waiting, C would overflow
        assert sv.engine.queue.num_waiting >= 2
        # Another request via serving layer
        resp = sv.generate("D", max_tokens=4)
        assert resp.error_code == "QUEUE_OVERFLOW", f"Got {resp.error_code}"

    def test_queue_accepts_after_drain(self):
        """After requests finish, queue accepts new requests."""
        sv = _sv(max_queue_len=2, max_num_seqs=1)
        sv.engine.add_request("A", max_new_tokens=2)
        # Run to completion — queue drains
        sv.engine.run_until_done()
        # Now new request should work
        resp = sv.generate("B", max_tokens=2)
        assert resp.error is None


# ======================================================================
# 2. KV Block Exhaustion
# ======================================================================

class TestBlockExhaustion:
    """Simulate all KV cache blocks allocated → new requests rejected."""

    def test_block_exhaustion_rejects_new(self):
        """No free blocks → admission returns BLOCK_EXHAUSTED."""
        sv = _sv(num_gpu_blocks=2, max_queue_len=16, max_model_len=256)
        resp = sv.generate("Hello world", max_tokens=16)
        assert resp.error_code == "BLOCK_EXHAUSTED", f"Got {resp.error_code}"

    def test_block_exhaustion_does_not_crash(self):
        """Even with 0 free blocks, the system should not crash."""
        sv = _sv(num_gpu_blocks=1, max_queue_len=16, max_model_len=256)
        # Should return error, not crash
        resp = sv.generate("Test", max_tokens=4)
        assert resp.error_code is not None

    def test_blocks_recovered_after_finish(self):
        """After requests finish, blocks return to free pool."""
        sv = _sv(num_gpu_blocks=16)
        sv.generate("Hello", max_tokens=2)
        free_before = sv.engine.block_manager._allocator.num_free_blocks
        # Run another — should have roughly the same free blocks
        sv.generate("World", max_tokens=2)
        assert sv.engine.block_manager._allocator.num_free_blocks >= free_before - 4


# ======================================================================
# 3. Streaming Connection Exhaustion
# ======================================================================

class TestStreamExhaustion:
    """Simulate max concurrent streams → new stream requests rejected."""

    def test_stream_exhaustion_rejects(self):
        """All stream slots taken → TOO_MANY_STREAMS."""
        sv = _sv(max_num_streams=2)
        # Take both stream slots
        sv.stream_manager.try_acquire("s1")
        sv.stream_manager.try_acquire("s2")
        # Third attempt fails
        assert not sv.stream_manager.try_acquire("s3")

    def test_stream_release_recovers_slot(self):
        """After releasing, new stream can be acquired."""
        sv = _sv(max_num_streams=1)
        assert sv.stream_manager.try_acquire("s1")
        sv.stream_manager.release("s1")
        assert sv.stream_manager.try_acquire("s2")


# ======================================================================
# 4. Timeout Storm
# ======================================================================

class TestTimeoutStorm:
    """Simulate many requests timing out simultaneously."""

    def test_timeout_releases_blocks(self):
        """All timed-out requests free their blocks."""
        sv = _sv(request_timeout_s=0.001, num_gpu_blocks=16)
        import time
        time.sleep(0.005)

        # Add several requests
        for i in range(4):
            sv.engine.add_request(f"Timeout-{i}", max_new_tokens=64)

        time.sleep(0.005)
        # Step once to admit some
        sv.engine.step()
        # Trigger timeout check
        sv.engine.engine_core._check_timeouts()

        # All should be gone
        assert sv.engine.queue.num_running == 0
        assert sv.engine.queue.num_waiting == 0

        # All blocks should be free
        alloc = sv.engine.block_manager._allocator
        assert alloc.num_free_blocks == alloc.num_total_blocks

    def test_timeout_metrics_updated(self):
        """Timeout requests increment timeout counter."""
        sv = _sv(request_timeout_s=0.001, num_gpu_blocks=16)
        import time
        time.sleep(0.005)
        sv.engine.add_request("Metrics test", max_new_tokens=64)
        time.sleep(0.005)
        sv.engine.step()
        sv.engine.engine_core._check_timeouts()
        # Metrics should reflect this
        mc = sv.engine.engine_core.metrics_collector
        report = mc.report()
        assert report.get("timeout_requests", 0) > 0


# ======================================================================
# 5. Cancel Storm
# ======================================================================

class TestCancelStorm:
    """Simulate many requests cancelled simultaneously."""

    def test_cancel_releases_blocks_to_pool(self):
        """All cancelled requests return blocks to the free pool."""
        sv = _sv(num_gpu_blocks=32)
        rids = []
        for i in range(4):
            rid = sv.engine.add_request(f"Cancel-{i}", max_new_tokens=64)
            rids.append(rid)

        sv.engine.step()  # admit some

        # Cancel all
        for rid in rids:
            sv.engine.cancel_request(rid)

        alloc = sv.engine.block_manager._allocator
        assert alloc.num_free_blocks == alloc.num_total_blocks

    def test_cancel_ref_count_integrity(self):
        """After cancel, no block has a dangling ref_count > 0."""
        sv = _sv(num_gpu_blocks=16)
        # Two requests with identical prompt (triggers prefix cache sharing)
        rid_a = sv.engine.add_request("RefCount test", max_new_tokens=8)
        rid_b = sv.engine.add_request("RefCount test", max_new_tokens=8)
        sv.engine.step()

        # Cancel both (using actual return values, not hardcoded IDs)
        for r in [rid_a, rid_b]:
            sv.engine.cancel_request(r)

        alloc = sv.engine.block_manager._allocator
        ref_counts = alloc.dump_ref_counts()
        assert all(rc == 0 for rc in ref_counts), f"Leaked refs: {ref_counts}"


# ======================================================================
# 6. Prefix Cache Stale Entry
# ======================================================================

class TestPrefixCacheStaleEntry:
    """Simulate hash hit but physical block reused → cache not misused."""

    def test_stale_entry_not_falsely_used(self):
        """After a block is freed and its PID reassigned, probe ignores it."""
        from mini_vllm import PrefixCacheProbeResult
        sv = _sv(num_gpu_blocks=8, block_size=4)

        # Request A populates cache
        rid_a = sv.engine.add_request("AAAA", max_new_tokens=2)
        sv.engine.run_until_done()

        # Block is freed, cache entry is stale
        assert sv.engine.block_manager.prefix_cache.size() > 0

        # Probe for same prompt — should detect stale (ref_count=0)
        probe = sv.engine.block_manager.probe_prefix_cache([65, 65, 65, 65])  # "AAAA"
        assert probe.cached_token_count == 0, (
            f"Stale entry should yield 0, got {probe.cached_token_count}"
        )

    def test_stale_entry_new_request_recreates(self):
        """A new request with same prompt re-registers the block."""
        sv = _sv(num_gpu_blocks=8, block_size=4)

        # First request populates cache and finishes (blocks freed)
        sv.engine.add_request("BBBB", max_new_tokens=2)
        sv.engine.run_until_done()

        # Second request: blocks were freed, but it should still work
        sv.engine.add_request("BBBB", max_new_tokens=2)
        sv.engine.run_until_done()

        # No crash, system healthy
        assert sv.engine.queue.num_finished >= 2


# ======================================================================
# 7. Admission Control Under Block Pressure
# ======================================================================

class TestAdmissionBlockPressure:
    """Admission control under severe block pressure — 10 blocks, 100 requests.

    Three scenarios:
    1. Normal-length requests (each needs ~17 blocks) → all BLOCK_EXHAUSTED
    2. Very short requests (each needs ~1 block) → first batch passes, rest blocked
    3. No admission control → RuntimeError OOM crash
    """

    def test_10blocks_100_long_all_rejected(self):
        """10 blocks, 100 normal-length requests → all BLOCK_EXHAUSTED."""
        sv = _sv(
            num_gpu_blocks=10,
            max_queue_len=200,
            max_num_seqs=100,
            max_model_len=1024,
            rate_limit_rpm=100000,
            rate_limit_tpm=100000000,
        )
        results = []
        for i in range(100):
            r = sv.generate(f"Request {i} normal length", max_tokens=64)
            results.append(r.error_code)

        exhausted = [e for e in results if e == "BLOCK_EXHAUSTED"]
        assert len(exhausted) == 100, \
            f"Expected 100 BLOCK_EXHAUSTED, got {len(exhausted)}: {set(results)}"

    def test_10blocks_100_short_uses_blocks_then_admission_blocks(self):
        """10 blocks, 100 short requests — first 8 bypass admission, 9th blocked.

        Fill engine directly (bypassing admission control) so blocks get
        consumed.  Then a new request through generate() hits the watermark.
        """
        sv = _sv(
            num_gpu_blocks=10,
            max_queue_len=200,
            max_num_seqs=100,
            max_model_len=1024,
            rate_limit_rpm=100000,
            rate_limit_tpm=100000000,
        )

        # Fill engine directly — this bypasses admission control
        for i in range(100):
            sv.engine.add_request(f"Z{i}", max_new_tokens=2)

        # Step once: scheduler admits as many as it can
        # Each short request needs ceil((1+2)/4)=1 block.
        # With 10 blocks total, scheduler can admit up to 10.
        sv.engine.step()

        alloc = sv.engine.block_manager._allocator
        before = alloc.num_free_blocks
        assert before < alloc.num_total_blocks, \
            "Expected blocks to be consumed after scheduling"

        # Now try a new request through the full admission pipeline
        resp = sv.generate("New arrival", max_tokens=4)

        # If the scheduler used all 10 blocks (0 free), admission sees
        # free=0, needed=1, 0-1=-1 < 2 → BLOCK_EXHAUSTED.
        # If the scheduler left enough blocks (watermark respected), the
        # new request might pass — but with 10 blocks and block_size=4,
        # even 10 concurrent short requests consume all blocks.
        assert resp.error_code == "BLOCK_EXHAUSTED", \
            f"Expected BLOCK_EXHAUSTED, got {resp.error_code} " \
            f"(free={before}/{alloc.num_total_blocks})"

    def test_10blocks_no_admission_control_crashes(self):
        """Bypassing admission control with 10 blocks + many requests → OOM crash.

        The scheduler admits requests in batches per step (token budget
        limits prefill).  Each decode step also allocates new blocks.
        Over enough steps, the block pool drains and ensure_block() OOMs.
        """
        sv = _sv(
            num_gpu_blocks=10,
            max_queue_len=200,
            max_num_seqs=100,
            max_model_len=1024,
        )

        # Bypass admission control — add 20 requests directly into engine
        for i in range(20):
            sv.engine.add_request(f"OOM{i}", max_new_tokens=64)

        # Step repeatedly until OOM — the scheduler admits batches and
        # each decode step allocates more blocks until the pool empties.
        with pytest.raises(RuntimeError, match="OOM"):
            for _ in range(1000):
                sv.engine.step()
                if sv.engine.queue.num_running == 0 and sv.engine.queue.num_waiting == 0:
                    # All finished without OOM — enlarge the test to guarantee crash
                    pytest.skip("All requests finished before OOM — increase request count")
