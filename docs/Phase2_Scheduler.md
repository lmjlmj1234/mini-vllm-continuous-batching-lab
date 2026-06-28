# Phase 2: Scheduler Deepening

## Overview

Phase 2 transforms the scheduler from a simple admit-then-decode loop into a
production-grade continuous-batching scheduler with **chunked prefill** and
**decode-first priority**. These are the two mechanisms that allow real
serving systems (vLLM, TensorRT-LLM) to handle diverse workloads without
head-of-line blocking.

---

## Decode-First Scheduling

### The Problem

In a serving system, users waiting for *existing* requests to generate tokens
are actively watching output. Every millisecond of added decode latency is
directly visible. New requests arriving during decoding *also* need attention,
but their first token is less time-sensitive than the Nth token of an ongoing
generation.

### The Policy

The scheduler deducts token budget for running **decode** sequences
*first*. Only the remaining budget is available for **prefill** (both
completing partial prefill and admitting new requests)::

    max_num_batched_tokens = 16
    max_num_prefill_tokens = 8   ← soft cap on prefill
    
    Step 1: decode tokens = 3    (3 running sequences)
    Step 2: remaining = 16 - 3 = 13
    Step 3: prefill budget = min(13, 8) = 8    ← decode already guaranteed

This guarantees that decode latency is never starved by prefill. In practice
`max_num_prefill_tokens` is often set lower than `max_num_batched_tokens` to
leave headroom for decode even under heavy prefill load.

### Why Decode Is More Sensitive

| Phase | Characteristic | User Impact |
|-------|---------------|-------------|
| **Prefill** | Compute-bound (process many tokens) | User sees first token latency (TTFT) |
| **Decode** | Memory-bound (read KV cache) | User sees per-token latency (TPOT) |

If decode is delayed, every active user feels a stall in their stream. If
prefill is delayed by one step, one new user sees a slightly longer TTFT.
The trade-off favors decode.

---

## Chunked Prefill

### The Problem

A request with a 4096-token prompt needs 4096 KV cache entries written
before the first output token. Without chunking, the scheduler must fit all
4096 tokens into a single step or ignore the request entirely::

    # Without chunking — prompt_len=4096, max_num_batched_tokens=1024
    prompt_len 4096 > max_num_batched_tokens 1024 → REJECTED

    # Even if it fits, 4096 prefill tokens block decode for the whole step

### The Solution

Chunked prefill splits the prompt into smaller pieces. The sequence tracks
progress with a **prefill_cursor** and is only moved to decode status when
all prompt tokens have been written to KV::

    Sequence lifecycle (with chunked prefill):

    WAITING ──→ PREFILL (cursor=0)
                   │
                   │  step N:  write chunk_size tokens → cursor += chunk_size
                   │  step N+1: write next chunk       → cursor += chunk_size
                   │  ...
                   ▼
               cursor ≥ prompt_length
                   │
                   ▼
               RUNNING (first output token generated)
                   │
                   ▼  decode until max_tokens
               FINISHED

### Benefits

1. **No more head-of-line blocking** — A long prompt no longer blocks all
   decode sequences for an entire step. The scheduler processes one chunk,
   then lets decode happen.

2. **Fine-grained budget control** — The scheduler can admit a long prompt
   by allocating its first chunk, then continue on subsequent steps. Other
   sequences are never starved.

3. **Mirrors vLLM** — vLLM uses the same mechanism (``Scheduler._schedule_prefills``
   iterates over waiting and running-prefill sequences, advancing cursor
   by ``max_num_prefill_tokens / num_prefill_seqs``).

### Implementation Details

- **Sequence.prefill_cursor** — tracks how many prompt tokens have been
  written to KV. Starts at 0, incremented by the executor after each chunk.
- **Sequence.is_prefill_finished** — property: ``prefill_cursor >= prompt_length``.
- **Scheduler** — prefill-continue groups (already in PREFILL state with
  cursor > 0) are admitted during Phase 4, ahead of new waiting groups.
  Each step advances the cursor by ``max_prefill_chunk_size`` tokens.
- **Executor.prefill()** — reads from ``seq.prefill_cursor``, writes
  ``chunk_size`` tokens to KV. Only on the final chunk does it generate
  the first output token and transition status to RUNNING.

---

## Config Changes

| Field | Default | Description |
|-------|---------|-------------|
| `max_num_prefill_tokens` | 16 | Cap on prefill tokens per step |
| `chunked_prefill_enabled` | True | Enable/disable chunked prefill |
| `max_prefill_chunk_size` | 4 | Prompt tokens processed per prefill step |
| `decode_first` | True | Decode sequences consume budget first |

---

## ScheduleResult Changes

| Field | Type | Description |
|-------|------|-------------|
| `preempted_groups` | `List[SequenceGroup]` | Placeholder — not yet implemented |
| `token_budget_remaining` | `int` | Remaining token budget after scheduling |
| `debug_reason` | `str` | Human-readable scheduling summary (replaces `reason`) |
| `ignored_reasons` | `Dict[str, str]` | Maps ignored group request_id → reason string |

Reason strings for ignored groups:
- `NO_TOKEN_BUDGET` — Not enough prefill token budget this step.
- `MAX_NUM_SEQS_LIMIT` — Sequence count cap reached.
- `NO_KV_BLOCK` — No free KV cache blocks (currently leads to rejection).
- `WAITING_FOR_NEXT_STEP` — Mid-prefill sequence deferred (cursor cannot advance).

---

## vLLM Correspondence

| mini-vLLM | vLLM | Notes |
|-----------|------|-------|
| `Config.decode_first` | `Scheduler._schedule()` decode-first loop | Both prioritize running decode |
| `Config.chunked_prefill_enabled` | `SchedulerConfig.chunked_prefill_enabled` | Same name, same semantics |
| `Config.max_num_prefill_tokens` | `SchedulerConfig.max_num_batched_tokens` in prefill context | vLLM uses a single budget but computes separate limits |
| `Config.max_prefill_chunk_size` | `CacheConfig.max_num_batched_tokens` / scheduling heuristic | vLLM derives per-step tokens dynamically |
| `Sequence.prefill_cursor` | `Sequence.data.num_prefilled_tokens` in vLLM | Same tracking mechanism |
| `ScheduleResult.token_budget_remaining` | `SchedulerOutputs.remaining_budget` | Same diagnostics |
| `ignored_reasons` | Scheduler output logging | vLLM logs similar diagnostics |
