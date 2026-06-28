from __future__ import annotations

from typing import List


class FakeModel:
    """Simulated model with fake logits and KV cache reads/writes.

    Uses a deterministic, input-dependent formula:
    - Prefill: writes prompt tokens to KV, returns generated first token
    - Decode: reads KV history + previous token to produce next token
    """

    def __init__(self, vocab_size: int = 500) -> None:
        self._vocab_size = vocab_size

    def _fake_key(self, token_id: int) -> int:
        return (token_id * 7 + 3) % self._vocab_size

    def _fake_value(self, token_id: int) -> int:
        return (token_id * 13 + 5) % self._vocab_size

    def prefill_token(self, token_id: int) -> int:
        """Return simulated first generated token from a prompt token."""
        return (token_id * 3 + 1) % self._vocab_size

    def decode_token(self, prev_token: int, kv_bias: int = 0) -> int:
        """Return simulated next token given previous token and KV bias."""
        return (prev_token + 7 + kv_bias) % self._vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
