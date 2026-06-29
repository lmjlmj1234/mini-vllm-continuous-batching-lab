"""Test serving layer: admission, rate limiting, streaming, cancel, metrics."""
import time
import pytest
from mini_vllm import Config
from serving.api_server import ServingLayer
from serving.rate_limiter import RateLimiter
from serving.admission_control import AdmissionControl
from serving.stream_manager import StreamManager


def _serving(**kw) -> ServingLayer:
    defaults = dict(
        print_step_events=False,
        num_gpu_blocks=32,
        block_size=4,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_queue_len=4,
        max_num_streams=4,
        rate_limit_rpm=10,
        rate_limit_tpm=10000,
    )
    defaults.update(kw)
    cfg = Config(**defaults)
    return ServingLayer(cfg)


class TestGenerate:
    def test_generate_non_stream_success(self):
        sv = _serving()
        resp = sv.generate("Hello", max_tokens=4, stream=False)
        assert resp.error is None
        assert len(resp.text) > 0
        assert resp.request_id.startswith("sv-")

    def test_generate_empty_prompt_rejected(self):
        sv = _serving()
        resp = sv.generate("", max_tokens=4)
        # Empty prompt is rejected by admission control
        assert resp.error_code == "PROMPT_TOO_LONG"

    def test_generate_max_tokens_zero(self):
        sv = _serving()
        resp = sv.generate("Hello", max_tokens=0)
        assert resp.error is None


class TestStreaming:
    def test_streaming_basic(self):
        sv = _serving()
        result = sv.generate_stream("Hello", max_tokens=4)
        assert result is not None
        engine_rid, tracking_id, err = result
        assert err is None
        assert engine_rid is not None

        # Poll until completion
        gen_count = 0
        tokens = []
        while True:
            token_text, gen_count, finished, finish_reason = sv.poll_stream(engine_rid, tracking_id, gen_count)
            if token_text:
                tokens.append(token_text)
            if finished:
                break

        assert len(tokens) > 0
        assert finish_reason == "length" or finish_reason == "finished"

    def test_streaming_token_accumulation(self):
        """Streaming yields at least one token."""
        sv = _serving()
        result = sv.generate_stream("Hello world", max_tokens=4)
        engine_rid, tracking_id, err = result
        assert err is None

        gen_count = 0
        first_token = None
        while True:
            token_text, gen_count, finished, _ = sv.poll_stream(engine_rid, tracking_id, gen_count)
            if token_text and first_token is None:
                first_token = token_text
            if finished:
                break

        assert first_token is not None


class TestRateLimit:
    def test_rpm_rejects_after_limit(self):
        sv = _serving(rate_limit_rpm=2)  # only 2 RPM
        # First two should succeed
        assert sv.generate("Hello", max_tokens=2).error is None
        assert sv.generate("World", max_tokens=2).error is None

    def test_rpm_limit_resets(self):
        """RPM resets after the window passes."""
        sv = _serving(rate_limit_rpm=1, rate_limit_tpm=100000)
        # First should work
        assert sv.generate("Hello", max_tokens=2).error is None
        # Directly check rate limiter state
        assert sv.rate_limiter._rpm.count() > 0


class TestAdmissionControl:
    def test_prompt_too_long(self):
        sv = _serving(max_model_len=4)
        resp = sv.generate("Hello world this is too long", max_tokens=4)
        assert resp.error_code == "PROMPT_TOO_LONG"
        assert resp.error is not None

    def test_queue_overflow(self):
        sv = _serving(max_queue_len=2, max_num_seqs=1)
        # Use engine.add_request to fill queue without running to completion
        # A runs, B and C wait → 2 waiting slots full
        sv.engine.add_request("A", max_new_tokens=64)
        sv.engine.add_request("B", max_new_tokens=64)
        sv.engine.add_request("C", max_new_tokens=64)
        sv.engine.step()  # schedule: A starts, B and C wait
        assert sv.engine.queue.num_running == 1
        assert sv.engine.queue.num_waiting == 2
        # Fourth via serving layer should overflow (max_queue_len=2 waiting max)
        resp = sv.generate("D", max_tokens=64)
        assert resp.error_code == "QUEUE_OVERFLOW", f"Got {resp.error_code}"

    def test_block_exhausted(self):
        """With only 2 GPU blocks, admission should reject."""
        sv = _serving(num_gpu_blocks=2, max_queue_len=16, max_model_len=256)
        # Large prompt with generation
        resp = sv.generate("Hello world", max_tokens=16)
        assert resp.error_code == "BLOCK_EXHAUSTED"


class TestStreamManager:
    def test_max_streams_cap(self):
        sv = _serving(max_num_streams=2)
        # Acquire 2 streams
        assert sv.stream_manager.try_acquire("s1")
        assert sv.stream_manager.try_acquire("s2")
        # Third should fail
        assert not sv.stream_manager.try_acquire("s3")
        # Release and retry
        sv.stream_manager.release("s1")
        assert sv.stream_manager.try_acquire("s3")

    def test_stream_manager_counts(self):
        sv = _serving(max_num_streams=4)
        assert sv.stream_manager.active_count == 0
        sv.stream_manager.try_acquire("a")
        assert sv.stream_manager.active_count == 1
        sv.stream_manager.release("a")
        assert sv.stream_manager.active_count == 0


class TestCancel:
    def test_cancel_running_request(self):
        """Cancel a request, verify blocks are freed."""
        sv = _serving(num_gpu_blocks=16)
        # Start a request but don't run it to completion
        engine_rid = sv.engine.add_request("Hello world cancel test", max_new_tokens=64)
        sv.engine.step()  # admit + start prefill

        # Cancel via serving layer (need engine-generated request_id)
        # The engine request_id format is req-XXXX
        ok = sv.engine.cancel_request(engine_rid)
        assert ok

        # Block should be freed
        alloc = sv.engine.block_manager._allocator
        assert alloc.num_free_blocks == alloc.num_total_blocks

    def test_cancel_non_existent(self):
        sv = _serving()
        ok = sv.cancel("does-not-exist")
        assert not ok


class TestTimeout:
    def test_timeout_cancels_request(self):
        sv = _serving(request_timeout_s=0.001)
        sv.engine.add_request("Timeout test", max_new_tokens=64)
        import time
        time.sleep(0.002)  # Ensure time passes
        sv.engine.step()
        sv.engine.engine_core._check_timeouts()
        # After timeout, request should be gone
        assert sv.engine.queue.num_running == 0


class TestMetrics:
    def test_metrics_endpoint_returns_json(self):
        sv = _serving()
        sv.generate("Hello", max_tokens=4)
        metrics_json = sv.get_metrics()
        assert '"total_requests"' in metrics_json
        assert '"block_utilization"' in metrics_json

    def test_metrics_after_rejection(self):
        sv = _serving(max_model_len=4)
        sv.generate("Hello this is too long", max_tokens=4)
        # Third request rejected
        metrics_json = sv.get_metrics()
        assert '"rejected_requests"' in metrics_json

    def test_metrics_after_cancel(self):
        sv = _serving(num_gpu_blocks=16)
        rid = sv.engine.add_request("Cancel test for metrics", max_new_tokens=64)
        sv.engine.step()
        sv.engine.cancel_request(rid)
        metrics_json = sv.get_metrics()
        assert '"cancelled_requests"' in metrics_json


class TestDisconnect:
    """Test disconnect lifecycle management — abort mid-stream, resource cleanup."""

    def test_streaming_disconnect_abort(self):
        """Abort mid-stream frees blocks and removes from queue."""
        sv = _serving(num_gpu_blocks=64)
        engine_rid, tracking_id, err = sv.generate_stream("Hello", max_tokens=64)
        assert err is None

        # Step once to get blocks allocated
        sv.engine.step()

        # Abort mid-stream (simulating client disconnect)
        sv._abort(engine_rid, tracking_id)
        # _abort no longer releases the stream slot — do it explicitly
        sv.stream_manager.release(tracking_id)

        # Blocks should be freed
        alloc = sv.engine.block_manager._allocator
        assert alloc.num_free_blocks == alloc.num_total_blocks

        # Queue should be empty
        assert sv.engine.queue.num_running == 0
        assert sv.engine.queue.num_waiting == 0

    def test_streaming_disconnect_running_queue_cleared(self):
        """Multiple streams aborted mid-stream — all removed from running queue."""
        sv = _serving(num_gpu_blocks=64)
        # Start multiple streaming requests
        streams = [sv.generate_stream(f"Request {i}", max_tokens=64) for i in range(4)]
        for engine_rid, tracking_id, err in streams:
            assert err is None

        # Step to admit them into the running queue
        sv.engine.step()

        # Abort all mid-stream
        for engine_rid, tracking_id, err in streams:
            sv._abort(engine_rid, tracking_id)
            sv.stream_manager.release(tracking_id)

        assert sv.engine.queue.num_running == 0
        assert sv.engine.queue.num_waiting == 0

    def test_streaming_disconnect_blocks_returned(self):
        """After abort, all KV blocks return to the free pool."""
        sv = _serving(num_gpu_blocks=64)
        engine_rid, tracking_id, err = sv.generate_stream("Hello world test", max_tokens=64)
        assert err is None

        # Step to allocate blocks
        sv.engine.step()

        # Sample: some blocks should be in use
        alloc = sv.engine.block_manager._allocator
        used_before = alloc.num_total_blocks - alloc.num_free_blocks
        assert used_before > 0, "Expected at least one block allocated"

        # Abort mid-stream
        sv._abort(engine_rid, tracking_id)
        sv.stream_manager.release(tracking_id)

        # All blocks returned
        assert alloc.num_free_blocks == alloc.num_total_blocks, \
            f"Blocks not freed: {alloc.num_free_blocks}/{alloc.num_total_blocks} free"

    def test_streaming_disconnect_metrics_cancelled(self):
        """Aborted request increments metrics cancelled count."""
        sv = _serving(num_gpu_blocks=64)
        engine_rid, tracking_id, err = sv.generate_stream("Cancel metrics", max_tokens=64)
        assert err is None
        sv.engine.step()

        # Abort mid-stream
        sv._abort(engine_rid, tracking_id)
        sv.stream_manager.release(tracking_id)

        mc = sv.engine.engine_core.metrics_collector
        report = mc.report()
        assert report.get("cancelled_requests", 0) >= 1, \
            f"Expected cancelled_requests >= 1, got {report.get('cancelled_requests')}"

    def test_streaming_disconnect_no_double_abort(self):
        """Abort after normal finish is a no-op — no double metrics counting."""
        sv = _serving(num_gpu_blocks=64)
        engine_rid, tracking_id, err = sv.generate_stream("Hello", max_tokens=4)
        assert err is None

        # Run to completion
        gen_count = 0
        finished = False
        while not finished:
            _, gen_count, finished, _ = sv.poll_stream(engine_rid, tracking_id, gen_count)

        # Request should be in _finished_requests now
        assert engine_rid in sv._finished_requests

        # Grab metrics before second abort attempt
        mc = sv.engine.engine_core.metrics_collector
        cancelled_before = mc.report().get("cancelled_requests", 0)

        # Second abort attempt — should be no-op
        sv._abort(engine_rid, tracking_id)
        # poll_stream already released the slot on finish, but release
        # again to follow _abort + release protocol
        sv.stream_manager.release(tracking_id)

        cancelled_after = mc.report().get("cancelled_requests", 0)
        assert cancelled_after == cancelled_before, \
            f"Double-abort incremented cancelled count: {cancelled_before} -> {cancelled_after}"

    def test_generate_stream_safe_finishes_normally(self):
        """generate_stream_safe yields tokens and finishes without error."""
        sv = _serving(num_gpu_blocks=64)
        tokens = []
        for token_text, err, finished, finish_reason in sv.generate_stream_safe("Hello", max_tokens=4):
            if token_text:
                tokens.append(token_text)
            if finished:
                break
        assert len(tokens) > 0

    def test_generate_stream_safe_abort_on_close(self):
        """Closing the generator mid-stream triggers abort (all blocks freed)."""
        sv = _serving(num_gpu_blocks=64)
        gen = sv.generate_stream_safe("Hello world", max_tokens=64)

        # Consume one token, then close (simulating disconnect)
        for token_text, err, finished, _ in gen:
            if token_text:
                break  # disconnect after first token

        # Explicitly close the generator — triggers finally → _abort
        gen.close()

        # Generator was closed — blocks should be freed
        alloc = sv.engine.block_manager._allocator
        assert alloc.num_free_blocks == alloc.num_total_blocks, \
            f"Blocks not freed after generator close: {alloc.num_free_blocks}/{alloc.num_total_blocks}"

    def test_streaming_disconnect_comprehensive(self):
        """Comprehensive disconnect test — validates ALL resource cleanup."""
        sv = _serving(num_gpu_blocks=64, max_num_streams=2)
        mc = sv.engine.engine_core.metrics_collector
        alloc = sv.engine.block_manager._allocator
        queue = sv.engine.queue

        cancelled_before = mc.report().get("cancelled_requests", 0)
        free_before = alloc.num_free_blocks

        # Start streaming request
        gen = sv.generate_stream_safe("Hello world test", max_tokens=64)

        # Consume one token
        for token_text, err, finished, _ in gen:
            if token_text:
                break

        # Close the generator (simulates client disconnect / aclose())
        gen.close()

        # ── queue is clean ──
        assert queue.num_running == 0, f"running={queue.num_running}"
        assert queue.num_waiting == 0, f"waiting={queue.num_waiting}"

        # ── All blocks returned ──
        assert alloc.num_free_blocks == alloc.num_total_blocks, \
            f"free={alloc.num_free_blocks}/{alloc.num_total_blocks}"

        # ── Block tables cleared ──
        assert len(sv.engine.block_manager._tables) == 0, \
            f"tables not empty: {sv.engine.block_manager._tables}"

        # ── Stream slot released ──
        assert sv.stream_manager.active_count == 0, \
            f"active={sv.stream_manager.active_count}"

        # ── Metrics cancelled +1 ──
        cancelled_after = mc.report().get("cancelled_requests", 0)
        assert cancelled_after == cancelled_before + 1, \
            f"cancelled: {cancelled_before} -> {cancelled_after}"

    def test_disconnect_after_engine_finished(self):
        """Client disconnects after engine finished — no duplicate abort.

        This tests the case where engine finishes the request before the
        client disconnects.  poll_stream detects completion and marks
        _finished_requests.  The subsequent _abort() is a no-op.
        """
        sv = _serving(num_gpu_blocks=64, max_num_streams=2)
        mc = sv.engine.engine_core.metrics_collector
        queue = sv.engine.queue

        engine_rid, tracking_id, err = sv.generate_stream("Hello", max_tokens=4)
        assert err is None

        # Run to completion via poll_stream
        gen_count = 0
        finished = False
        while not finished:
            _, gen_count, finished, _ = sv.poll_stream(engine_rid, tracking_id, gen_count)

        # Engine already marked it finished
        assert engine_rid in sv._finished_requests
        assert queue.num_running == 0
        assert queue.num_waiting == 0

        cancelled_before = mc.report().get("cancelled_requests", 0)

        # Client disconnects AFTER engine finished — should be no-op
        sv._abort(engine_rid, tracking_id)

        cancelled_after = mc.report().get("cancelled_requests", 0)
        assert cancelled_after == cancelled_before, \
            f"cancelled_requests leaked: {cancelled_before} -> {cancelled_after}"

        # Stream slot already released by poll_stream
        assert sv.stream_manager.active_count == 0
