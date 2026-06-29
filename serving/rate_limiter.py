"""Token-aware rate limiter — RPM + TPM dual limits.

Protects: GPU compute budget — the scarcest resource in LLM serving.

Why LLM APIs cannot use QPS alone
-----------------------------------
A single request can have 16 prompt tokens (cheap) or 4096 (expensive).
QPS treats them equally — a burst of large requests can OOM the GPU.

TPM (tokens per minute) is the token-count metric:
- Prompt tokens: each is processed once during prefill
- Generated tokens: each is produced one-at-a-time during decode
- Total = prompt_tokens + max_new_tokens (the actual tokens the model touches)

Why both RPM and TPM
---------------------
- RPM catches request-flood attacks: 10k tiny requests/sec
- TPM catches compute-flood attacks: 1 large request eating 10k tokens
- Together they cover both dimensions of cost

Real-system correspondence
---------------------------
- OpenAI: tokens-per-minute (TPM) + requests-per-minute (RPM) on API keys
- Anthropic: tokens-per-minute per API key
- vLLM + multiple LoRA: per-adapter rate limits are common

What goes wrong without it
---------------------------
A single user sends prompts of increasing length until the GPU OOMs.
Rate limiting with TPM flattens the cost regardless of request shape.
"""

import time
from typing import List


class SlidingWindowCounter:
    """Sliding window counter for rate limiting.

    Uses a list of timestamps. Old entries (outside the window) are
    discarded on each check.  For production, a bucketed counter
    (e.g., token-bucket algorithm) would be more memory-efficient.
    """

    def __init__(self, window_s: float = 60.0) -> None:
        self._window = window_s
        self._events: List[float] = []

    def _purge(self, now: float) -> None:
        cutoff = now - self._window
        self._events = [t for t in self._events if t > cutoff]

    def count(self) -> int:
        now = time.time()
        self._purge(now)
        return len(self._events)

    def record(self) -> None:
        self._events.append(time.time())

    def would_exceed(self, limit: int) -> bool:
        """Check if recording another event would exceed *limit*."""
        now = time.time()
        self._purge(now)
        return len(self._events) >= limit


class RateLimiter:
    """Dual sliding-window rate limiter: RPM + TPM."""

    def __init__(self, rpm_limit: int = 60, tpm_limit: int = 100000) -> None:
        self._rpm = SlidingWindowCounter()
        self._tpm = SlidingWindowCounter()
        self._rpm_limit = rpm_limit
        self._tpm_limit = tpm_limit

    def check(self, total_tokens: int = 0) -> str | None:
        """Check if request passes rate limits.

        Returns an error code string if rate limited, or None if OK.

        ``total_tokens`` = prompt_tokens + max_new_tokens (the actual
        number of tokens the model will process for this request).
        """
        if self._rpm.would_exceed(self._rpm_limit):
            return "RATE_LIMITED_RPM"
        if self._tpm.would_exceed(self._tpm_limit):
            return "RATE_LIMITED_TPM"
        return None

    def record(self, total_tokens: int = 0) -> None:
        """Record a request against both counters."""
        self._rpm.record()
        for _ in range(total_tokens):
            self._tpm.record()

    @property
    def rpm_count(self) -> int:
        return self._rpm.count()

    @property
    def tpm_count(self) -> int:
        return self._tpm.count()
