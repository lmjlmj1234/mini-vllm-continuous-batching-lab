"""Pre-scheduler admission control — the first line of defence.

Protects: Scheduler stability, KV cache pool, and queue memory.

Why Admission Control must be BEFORE the Scheduler
-----------------------------------------------------
Without it, EVERY request reaches the Scheduler's Phase 5 (Admit).
Problems:
1. A prompt exceeding max_model_len wastes scheduler compute — it must
   traverse 5 phases before being rejected.
2. A queue overflow forces the Scheduler to iterate over hundreds of
   waiting requests every step, slowing scheduling latency.
3. KV block exhaustion only surfaces during ``ensure_block()`` — by then
   the request is already admitted and consuming executor time.

Admission Control catches these BEFORE they enter the scheduling loop.

Checks performed (in order)
-----------------------------
1. **Prompt length** ── len(prompt) > max_model_len -> PROMPT_TOO_LONG
2. **Queue depth**    ── num_waiting >= max_queue_len -> QUEUE_OVERFLOW
3. **Block pressure** ── free_blocks below watermark -> BLOCK_EXHAUSTED
4. **Token budget**   ── prompt tokens exceed remaining budget -> BUDGET_EXCEEDED

Real-system correspondence
---------------------------
- vLLM: no explicit admission control (relies on scheduler rejection).
  Production deployments add an API gateway layer for this.
- TGI: ``max_input_length`` and ``max_batch_prefill_tokens``.
- Triton Inference Server: ``--max-queue-delay-us`` shapes queue depth.
- Nvidia ``nv-device-plugin``: blocks GPU memory oversubscription.

What goes wrong without it
---------------------------
A 10k-token prompt that slips past tokenizer limits enters the Scheduler,
gets chunked across 2500 steps, blocks decode for all other requests,
and eventually exhausts KV cache blocks — all before being rejected.
A single large request can DOS the entire serving system.
"""

from typing import Optional
from mini_vllm import Config
from mini_vllm import BlockAllocator


class AdmissionControl:
    """Pre-scheduler admission gate."""

    def __init__(
        self,
        config: Config,
        allocator: BlockAllocator,
        current_waiting: callable,
    ) -> None:
        self._config = config
        self._allocator = allocator
        self._current_waiting = current_waiting
        # Block watermark: reject if free blocks drop below this %
        self._block_watermark_pct = 0.2

    def check(self, prompt_token_ids, max_tokens: int = 64) -> Optional[str]:
        """Run all admission checks. Returns error code string or None."""
        prompt_len = len(prompt_token_ids)

        # 0. Empty prompt
        if prompt_len == 0:
            return "PROMPT_TOO_LONG"

        # 1. Prompt length vs model capacity
        if prompt_len > self._config.max_model_len:
            return "PROMPT_TOO_LONG"

        # 2. Queue depth
        if self._current_waiting() >= self._config.max_queue_len:
            return "QUEUE_OVERFLOW"

        # 3. Block pressure — is the KV cache pool close to full?
        total = self._allocator.num_total_blocks
        free = self._allocator.num_free_blocks
        # Estimate blocks needed: prompt blocks + max generation blocks
        needed_blocks = (prompt_len + max_tokens + self._config.block_size - 1) // self._config.block_size
        watermark_blocks = int(total * self._block_watermark_pct)
        if free - needed_blocks < watermark_blocks:
            return "BLOCK_EXHAUSTED"

        return None
