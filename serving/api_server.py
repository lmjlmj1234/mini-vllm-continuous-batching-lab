"""API Server — production-oriented LLM serving gateway.

Architecture::

    POST /generate
        ↓
    AdmissionControl.check()
        ↓  (PROMPT_TOO_LONG / QUEUE_OVERFLOW / BLOCK_EXHAUSTED)
    RateLimiter.check()
        ↓  (RATE_LIMITED_RPM / RATE_LIMITED_TPM)
    StreamManager.try_acquire()   [if stream=True]
        ↓  (TOO_MANY_STREAMS)
    Engine.add_request()
        ↓
    loop Engine.step() / collect tokens [stream] or wait [non-stream]
        ↓
    StreamManager.release()
    Response

Protects: The entire pipeline — each guard catches a specific failure mode.
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Generator, Set
from mini_vllm import Config
from mini_vllm.engine.engine import LLMEngine
from .request import ServingRequest, ServingResponse, StreamEvent, ERROR_CODES
from .sse import format_sse_event
from .rate_limiter import RateLimiter
from .admission_control import AdmissionControl
from .request_cancel import CancelManager
from .stream_manager import StreamManager
from .metrics_endpoint import MetricsEndpoint


class ServingLayer:
    """Orchestrates all serving components around the LLM engine.

    This is the single entry point for ``/generate`` — it runs the
    full admission → rate-limit → stream-control → engine pipeline.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()
        self._engine = LLMEngine(self._config)

        # Serving sub-components
        self._rate_limiter = RateLimiter(
            rpm_limit=self._config.rate_limit_rpm,
            tpm_limit=self._config.rate_limit_tpm,
        )
        self._admission = AdmissionControl(
            self._config,
            self._engine.block_manager._allocator,
            current_waiting=lambda: self._engine.queue.num_waiting,
        )
        self._stream_manager = StreamManager(
            self._engine,
            max_streams=self._config.max_num_streams,
        )
        self._cancel_mgr = CancelManager(
            self._engine,
            timeout_s=self._config.request_timeout_s,
        )
        self._metrics_endpoint = MetricsEndpoint(
            self._engine,
            self._stream_manager,
            self._rate_limiter,
            self._admission,
        )

        # Track normally-finished requests to prevent double-abort
        self._finished_requests: Set[str] = set()

    # ------------------------------------------------------------------
    # Generate API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        stream: bool = False,
    ) -> ServingResponse:
        """Synchronous /generate handler.

        The serving layer generates a tracking request_id for lifecycle
        management.  The engine assigns its own internal request_id
        (req-0000, req-0001, ...).  We map between them.
        """
        tracking_id = f"sv-{uuid.uuid4().hex[:8]}"
        prompt_token_ids = self._engine.executor.tokenize(prompt)

        # --- Admission gate ---
        err = self._admission.check(prompt_token_ids, max_tokens)
        if err:
            self._engine.engine_core.metrics_collector.count_rejected()
            return ServingResponse(
                request_id=tracking_id, text="", error="Admission rejected", error_code=err,
            )

        # --- Rate limit gate ---
        total_tokens = len(prompt_token_ids) + max_tokens
        err2 = self._rate_limiter.check(total_tokens)
        if err2:
            self._engine.engine_core.metrics_collector.count_rejected()
            self._engine.engine_core.metrics_collector.count_rpm_rejected()
            return ServingResponse(
                request_id=tracking_id, text="", error="Rate limited", error_code="RATE_LIMITED",
            )

        # --- Stream gate ---
        if stream and not self._stream_manager.try_acquire(tracking_id):
            self._engine.engine_core.metrics_collector.count_rejected()
            return ServingResponse(
                request_id=tracking_id, text="", error="Too many streams", error_code="TOO_MANY_STREAMS",
            )

        # --- Engine ---
        self._engine.engine_core.metrics_collector.count_request()
        engine_rid = self._engine.add_request(prompt, max_new_tokens=max_tokens)
        self._rate_limiter.record(total_tokens)

        # Run to completion
        outputs = self._engine.run_until_done()

        if stream:
            self._stream_manager.release(tracking_id)

        output_text = outputs.get(engine_rid, "")
        return ServingResponse(
            request_id=tracking_id,
            text=output_text,
            generated_tokens=0,
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 64,
    ):
        """Streaming generate — yields tokens one at a time via a callback.

        Uses a callback-based approach: ``on_token(token_text)`` is called
        for each generated token.  Returns a ServingResponse on completion.

        This avoids the complexity of async generators while supporting
        the same token-by-token delivery pattern.
        """
        tracking_id = f"sv-{uuid.uuid4().hex[:8]}"
        prompt_token_ids = self._engine.executor.tokenize(prompt)

        # --- Admission gate ---
        err = self._admission.check(prompt_token_ids, max_tokens)
        if err:
            self._engine.engine_core.metrics_collector.count_rejected()
            return None, "Admission rejected", err

        # --- Rate limit gate ---
        total_tokens = len(prompt_token_ids) + max_tokens
        err2 = self._rate_limiter.check(total_tokens)
        if err2:
            self._engine.engine_core.metrics_collector.count_rejected()
            return None, "Rate limited", "RATE_LIMITED"

        # --- Stream gate ---
        if not self._stream_manager.try_acquire(tracking_id):
            self._engine.engine_core.metrics_collector.count_rejected()
            return None, "Too many streams", "TOO_MANY_STREAMS"

        try:
            engine_rid = self._engine.add_request(prompt, max_new_tokens=max_tokens)
            self._rate_limiter.record(total_tokens)
        except Exception:
            self._stream_manager.release(tracking_id)
            raise

        return engine_rid, tracking_id, None

    def poll_stream(
        self,
        engine_rid: str,
        tracking_id: str,
        gen_count: int = 0,
    ) -> tuple:
        """Poll one step of a streaming request.

        Returns (token_text, gen_count, finished).
        """
        result = self._engine.step()
        self._engine.engine_core._check_timeouts()

        # Check finished groups
        for sg in result.finished_groups:
            for seq in sg.seqs:
                if seq.group_id == engine_rid and seq.finished:
                    if gen_count < len(seq.output_token_ids):
                        gen_count = len(seq.output_token_ids)
                    self._finished_requests.add(engine_rid)
                    self._stream_manager.release(tracking_id)
                    return ("", gen_count, True, seq.status.name.lower())

        # Check decode groups for new tokens
        for sg in result.scheduled_decode_groups:
            for seq in sg.get_unfinished_seqs():
                if seq.group_id == engine_rid:
                    while gen_count < seq.num_generated_tokens:
                        tok = seq.output_token_ids[gen_count]
                        token_text = self._engine.executor.detokenize([tok])
                        gen_count += 1
                        return (token_text, gen_count, False, None)

        return ("", gen_count, False, None)

    def generate_stream_safe(
        self,
        prompt: str,
        max_tokens: int = 64,
    ):
        """Disconnect-safe streaming wrapper.

        Yields ``(token_text, error, finished, finish_reason)`` tuples.

        Guarantees cleanup via ``_abort()`` in a ``finally`` block:
        - If the caller stops iterating (e.g. generator ``close()`` on
          disconnect), the ``finally`` runs and releases all resources.
        - If the request finishes normally, ``_abort()`` is a no-op
          because ``poll_stream()`` already marked the request as
          finished via ``_finished_requests``.
        - If an exception occurs mid-stream, the ``finally`` ensures
          cleanup before the exception propagates.
        """
        engine_rid, tracking_id, err = self.generate_stream(prompt, max_tokens)
        if err:
            yield (None, err, True, None)
            return
        gen_count = 0
        finished = False
        finish_reason = None
        try:
            while not finished:
                token_text, gen_count, finished, finish_reason = (
                    self.poll_stream(engine_rid, tracking_id, gen_count)
                )
                yield (token_text, None, finished, finish_reason)
        finally:
            # *Always* release the stream slot — even on exception,
            # generator close, or abort skip.  discard() is safe for
            # double-release (poll_stream may have already released).
            try:
                if not finished:
                    self._abort(engine_rid, tracking_id)
            finally:
                self._stream_manager.release(tracking_id)

    # ------------------------------------------------------------------
    # Abort (cleanup on disconnect / failure)
    # ------------------------------------------------------------------

    def _abort(self, engine_rid: str, tracking_id: str) -> None:
        """Cancel a running request in the engine and free all engine resources.

        Handles:
        - Engine cancel_request (→ blocks freed, queue cleaned, metrics +1)

        Does NOT release the stream slot — the caller's ``finally`` is
        responsible for that (see ``generate_stream_safe``).

        Safe to call multiple times — if the request already finished
        normally (``_finished_requests``), this is a no-op.
        """
        if engine_rid in self._finished_requests:
            return
        self._finished_requests.add(engine_rid)
        self._cancel_mgr.cancel(engine_rid)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(self, request_id: str) -> bool:
        """Cancel a running request."""
        return self._cancel_mgr.cancel(request_id)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> str:
        """Return /metrics as JSON."""
        return self._metrics_endpoint.render()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def engine(self) -> LLMEngine:
        return self._engine

    @property
    def admission(self) -> AdmissionControl:
        return self._admission

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    @property
    def stream_manager(self) -> StreamManager:
        return self._stream_manager
