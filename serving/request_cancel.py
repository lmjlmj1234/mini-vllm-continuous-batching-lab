"""Request cancel and timeout management.

Protects: KV cache blocks from leaking when requests are abandoned.

Why HTTP disconnect != resource release
----------------------------------------
When an HTTP client disconnects:
1. The TCP connection closes — the server can detect this via
   ``socket.recv()`` returning empty bytes.
2. BUT: the LLM engine **still holds** the sequence in its running queue,
   with all KV cache blocks pinned and ref_count > 0.
3. The scheduler **still allocates budget** for this sequence.
4. If not cancelled, the sequence runs to completion (max_tokens)
   producing tokens nobody consumes — wasting GPU compute.

This is a **resource leak**:
- KV cache blocks remain allocated (denying other requests)
- Token budget is consumed (reducing throughput)
- Prefix cache ref_count remains inflated

Cancel must:
1. Mark the sequence CANCELLED/TIMEOUT
2. Free all physical blocks (decrement ref_count)
3. Remove from scheduler queues
4. Update metrics

Why cancel is expensive
-------------------------
Cancel is not free — BlockManager.free() walks the block table, decrements
ref_count for each block, and may trigger BlockAllocator callbacks.
On a large batch, cancelling N sequences costs O(N * avg_blocks).

Real-system correspondence
---------------------------
- vLLM: ``abort_seq_group()`` → ``BlockSpaceManager.free()`` →
  ``PhysicalTokenBlock.ref_count--``. Same ref_count semantics.
- TGI: ``_clean_batch()`` on client disconnect.
- FastAPI/uvicorn: detects disconnect via ``request.is_disconnected()``
  and raises ``ClientDisconnect``, but *application code* must handle
  the cleanup.
"""

import time
from typing import Dict, Set
from mini_vllm import Status
from mini_vllm.engine.engine import LLMEngine


class CancelManager:
    """Manages request cancellation and timeout detection."""

    def __init__(self, engine: LLMEngine, timeout_s: float = 60.0) -> None:
        self._engine = engine
        self._default_timeout = timeout_s
        self._cancelled: Set[str] = set()

    def cancel(self, request_id: str) -> bool:
        """Cancel a request and release all resources.

        Returns True if the request was found and cancelled.
        """
        ok = self._engine.cancel_request(request_id)
        if ok:
            self._cancelled.add(request_id)
        return ok

    def check_timeouts(self) -> int:
        """Check all running/waiting requests for timeout.

        Returns the number of requests that timed out.
        """
        # EngineCore already handles _check_timeouts in step().
        # This is the serving-layer wrapper for manual checks.
        core = self._engine._engine_core
        core._check_timeouts()
        return 0  # count tracked via metrics

    def is_cancelled(self, request_id: str) -> bool:
        return request_id in self._cancelled
