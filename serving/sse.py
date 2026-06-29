"""Server-Sent Events (SSE) formatter for token streaming.

SSE format::

    data: {"token": "Hello", "finished": false}\n\n
    data: {"token": " world", "finished": false}\n\n
    data: {"token": "", "finished": true, "generated_tokens": 2}\n\n

Each event is a JSON line prefixed with ``data: `` and terminated by
a double newline.

Why SSE for LLM?
-----------------
- SSE is a *single long-lived HTTP connection* with incremental writes.
  The client opens one connection and reads tokens as they arrive.
- Alternative (WebSocket) adds bidirectional complexity we don't need.
- Alternative (polling) wastes server resources on requests with no new data.
- SSE uses standard HTTP — no upgrade needed, works behind every load balancer
  and reverse proxy.
"""

import json
from typing import Optional

from .request import StreamEvent


def format_sse_event(event: StreamEvent) -> str:
    """Format a StreamEvent into an SSE message string."""
    payload = {
        "token": event.token,
        "finished": event.finished,
    }
    if event.request_id:
        payload["request_id"] = event.request_id
    if event.generated_tokens > 0:
        payload["generated_tokens"] = event.generated_tokens
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
