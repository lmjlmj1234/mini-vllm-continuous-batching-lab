"""Streaming connection manager — tracks active SSE streams.

Protects: Worker threads, socket file descriptors, and server memory.

Why streaming is NOT free
--------------------------
Each active SSE stream holds:
1. A socket connection (file descriptor) — limited by OS ulimit
2. A worker/goroutine context — limited thread pool
3. Memory buffers — output tokens accumulate, backpressure needed
4. A slot in the engine's running queue — KV cache blocks are pinned

If streams are unbounded, the server will:
- Exhaust file descriptors (``EMFILE``)
- Run out of worker threads (starve other requests)
- Pin all KV cache blocks (deny new requests)

The ``max_streams`` cap prevents this.

Real-system correspondence
---------------------------
- vLLM: ``--max-num-seqs`` limits active sequences, but streams can exceed
  this if the client reads slowly (backpressure).
- TGI (HuggingFace): per-model max_concurrent_requests.
- OpenAPI gateways: nginx ``worker_connections`` + upstream timeouts.

What goes wrong without it
---------------------------
A slow client that reads 1 byte/minute holds a stream for hours,
pinning KV cache blocks that could serve hundreds of other requests.
Stream manager is the lid on this pressure cooker.
"""

from typing import Dict, Set
from mini_vllm.engine.engine import LLMEngine


class StreamManager:
    """Manages active streaming connections with a hard cap.

    ``active_streams`` counts the number of currently streaming requests.
    When a new request arrives with ``stream=True``, the manager checks
    whether the cap would be exceeded **before** admitting.

    The cap protects against connection exhaustion and KV cache pinning.
    """

    def __init__(self, engine: LLMEngine, max_streams: int = 16) -> None:
        self._engine = engine
        self._max_streams = max_streams
        self._active: Set[str] = set()

    def try_acquire(self, request_id: str) -> bool:
        """Try to register a new stream. Returns False if at capacity."""
        if len(self._active) >= self._max_streams:
            return False
        self._active.add(request_id)
        return True

    def release(self, request_id: str) -> None:
        """Release a stream (on finish, cancel, or disconnect)."""
        self._active.discard(request_id)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def max_streams(self) -> int:
        return self._max_streams

    def is_full(self) -> bool:
        return len(self._active) >= self._max_streams
