"""Shared request/response models for the serving layer."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ServingRequest:
    """Normalised internal representation of an API /generate request."""
    request_id: str
    prompt: str
    max_tokens: int = 64
    stream: bool = False
    arrival_time: float = 0.0


@dataclass
class StreamEvent:
    """A single streaming event (token or final)."""
    token: str = ""
    finished: bool = False
    request_id: str = ""
    generated_tokens: int = 0


@dataclass
class ServingResponse:
    """Final non-streaming response."""
    request_id: str
    text: str
    generated_tokens: int = 0
    error: Optional[str] = None
    error_code: Optional[str] = None
    finish_reason: str = "length"


ERROR_CODES = {
    "PROMPT_TOO_LONG": "PROMPT_TOO_LONG",
    "QUEUE_OVERFLOW": "QUEUE_OVERFLOW",
    "BLOCK_EXHAUSTED": "BLOCK_EXHAUSTED",
    "RATE_LIMITED": "RATE_LIMITED",
    "TOO_MANY_STREAMS": "TOO_MANY_STREAMS",
    "REQUEST_CANCELLED": "REQUEST_CANCELLED",
    "REQUEST_TIMEOUT": "REQUEST_TIMEOUT",
}
