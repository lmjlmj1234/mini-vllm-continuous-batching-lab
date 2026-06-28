# Scheduler — Continuous Batching with Chunked Prefill

> **Design Doc** — How mini-vLLM schedules sequences, allocates token budgets,
> and handles priority between prefill and decode.

---

## 1. Scheduling Algorithm (6 Phases)

```
                        Scheduler.schedule()
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Phase 1: Finish    │
                    │  Check              │
                    │                     │
                    │  for each running   │
                    │  group: if all seqs │
                    │  at max_tokens →    │
                    │  free blocks,       │
                    │  mark_finished      │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Phase 2:           │
                    │  Categorize         │
                    │                     │
                    │  decode_groups ←    │
                    │    status=RUNNING   │
                    │  prefill_continue ← │
                    │    status=PREFILL   │
                    │    (cursor > 0)     │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Phase 3:           │
                    │  Decode-First       │
                    │  Budget             │
                    │                     │
                    │  remaining_budget = │
                    │    max_batched -    │
                    │    num_decode_seqs  │
                    │  prefill_budget =   │
                    │    min(remaining,   │
                    │      max_prefill)   │
                    │  remaining_seq =    │
                    │    max_seqs -       │
                    │    num_decode_seqs  │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Phase 4:           │
                    │  Chunked-Prefill    │
                    │  Continue           │
                    │                     │
                    │  for each mid-      │
                    │  prefill group:     │
                    │    this_chunk =     │
                    │    min(remaining,   │
                    │      chunk_size)    │
                    │    deduct from      │
                    │    prefill_budget   │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Phase 5:           │
                    │  Admit New          │
                    │                     │
                    │  for each waiting:  │
                    │    if seq/ token    │
                    │    budget OK:       │
                    │    → allocate_for_  │
                    │      seq(seq)       │
                    │      mark_running   │
                    │    else: ignore/    │
                    │    reject           │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Phase 6:           │
                    │  Token Counts       │
                    │                     │
                    │  num_prefill_tokens │
                    │  num_decode_tokens  │
                    │  num_batched_tokens │
                    │  remaining_budget   │
                    │  debug_reason       │
                    └─────────────────────┘
```

---

## 2. Token Budget Model

The scheduler tracks two constraints on every step:

```
max_num_batched_tokens = 16        max_num_seqs = 4
max_num_prefill_tokens = 16        chunk_size = 4

Step budget allocation:

total_budget = max_num_batched_tokens (16)
    │
    ├── decode_budget: num_decode_seqs × 1 token each
    │   (guaranteed by decode-first — never starved)
    │
    └── prefill_budget: min(remaining, max_num_prefill_tokens)
         │
         ├── continue_budget: mid-prefill chunks
         │   (existing sequences that haven't finished prefill)
         │
         └── admit_budget: new waiting groups
             (may be zero if continue consumed all budget)
```

### Example budget flow (step 3 from demo):

```
starting budget:                     max_batched=16, max_prefill=16

Phase 2: decode=[], prefill_continue=[A, B]

Phase 3: remaining=16, prefill_budget=16, remaining_seq=4

Phase 4:
  A (prompt 11, cursor 8):  remaining_prompt=3, chunk=min(3,4)=3
    prefill_budget: 16→13
  B (prompt 13, cursor 8):  remaining_prompt=5, chunk=min(5,4)=4
    prefill_budget: 13→9,  remaining_seq: 4→3

Phase 5:
  C (prompt 9, waiting):    chunk=min(9,4)=4
    prefill_budget: 9→5,    remaining_seq: 3→2

Result: prefill_tokens=11, decode_tokens=0, budget_remaining=5
```

---

## 3. Decode-First Priority

```
Why decode-first?
  1. Prefill is compute-bound (high throughput per token)
  2. Decode is memory-bound (low throughput, high latency sensitivity)
  3. Users perceive "responsiveness" = time between output tokens

Without decode-first:
  A long prefill chunk can delay all decode sequences by one entire step.
  Users see a pause in output — bad UX.

With decode-first:
  Decode sequences always run, regardless of prefill pressure.
  TTFT for new requests may increase, but TPOT for existing requests is stable.
```

**Implementation**: Phase 3 deducts decode tokens from budget **before** any
prefill computation. This means:

```
Case: max_batched=16, 2 decode seqs, 3 waiting with long prompts

Phase 3:  remaining = 16 - 2 = 14
Phase 4:  (no mid-prefill groups)
Phase 5:  admit chunk budget = min(14, max_prefill) with remaining_seq=2

If max_prefill=16: admit up to 14 tokens of new prefill
If max_prefill=4:  admit up to 4 tokens (even though 14 remaining!)

Decode tokens always go first, prefill gets what's left AND is capped.
```

---

## 4. Chunked Prefill

### Why chunked?

Without chunked prefill, a prompt longer than `max_num_batched_tokens` must be
rejected. With chunked prefill:

- Prompt is split into `ceil(prompt_len / chunk_size)` chunks
- Each step processes one chunk
- The sequence stays in `PREFILL` status until all chunks are consumed
- On the last chunk, the sequence transitions to `RUNNING`

### Chunk boundary handling

```
prompt_len = 12, chunk_size = 4

Step  Token Range  prefill_cursor  Status           Blocks
────  ───────────  ──────────────  ──────           ──────
1      0..3        4               PREFILL          1
2      4..7        8               PREFILL          2
3      8..11       12              RUNNING (done)   3
4      decode      -               RUNNING          3+

Note: blocks only grow as token positions are written.
No block is allocated for "future" chunks.
```

### Prefill cursor lifecycle

```
Created by scheduler          Advanced by executor    Read by scheduler
in Phase 5                    in prefill()            in Phase 4
     │                              │                      │
     │                              │                      │
     ▼                              ▼                      ▼
  cursor = 0               cursor += chunk_size     remaining = prompt - cursor
  (admission)              (each step)              (for next chunk)

Is prefill finished?
  cursor >= len(prompt_token_ids)
    → executor sets status = RUNNING, generates first output token
```

### Cursor position → KV block mapping

```
prompt tokens:  [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]
                  0   1   2   3   4   5   6   7   8   9   10
                  └──────L0──────┘  └──────L1──────┘  └──L2──
                  block_size=4

At cursor=4:   L0 has data (written), L1 empty (not yet allocated)
At cursor=8:   L0, L1 have data, L2 not yet allocated
At cursor=11:  L0, L1, L2 all have data (3 blocks allocated)
```

---

## 5. Admission Policy (Phase 5)

When a waiting group arrives, the scheduler checks:

```
                     New waiting group arrives
                              │
                              │
                    ┌─────────┴──────────┐
                    │                    │
                    ▼                    ▼
             remaining_seq       remaining_seq
             > 0                 <= 0
                    │                    │
                    │                    ▼
                    │             IGNORE: MAX_NUM_SEQS_LIMIT
                    │
                    ▼
             chunk_size <= prefill_budget?
                    │                    │
                    │                    │
              YES ──┘                    └── NO
                    │                       │
                    │                  ┌────┴────┐
                    │                  │         │
                    ▼                  ▼         ▼
             admit group          prompt >    prompt <=
             allocate_for_seq     max_batch  max_batch
             set cursor=0             │           │
             set PREFILL              │           │
             mark_running             ▼           ▼
                                  REJECT     IGNORE: NO_TOKEN_BUDGET
```

### Ignored vs Rejected

| Status | Meaning | Recovery |
|--------|---------|----------|
| IGNORE | "Skip this step, try again next step" | Automatically retried next `schedule()` call |
| REJECT | "This request can never be served" | Moved to rejected pool, never retried |

**Rejection reasons**: the only case is `prompt_len > max_num_batched_tokens`
AND chunked_prefill is disabled. With chunked prefill enabled, long prompts
are split and admitted.

**Ignore reasons**:

| Reason | When |
|--------|------|
| `MAX_NUM_SEQS_LIMIT` | Running sequences already at `max_num_seqs` capacity |
| `NO_TOKEN_BUDGET` | Not enough prefill tokens remaining after decode-first & continue |
| `WAITING_FOR_NEXT_STEP` | Mid-prefill sequence can't advance (temporary: budget consumed) |

---

## 6. ScheduleResult Structure

The scheduler returns a complete audit record of every decision:

```
ScheduleResult(
    scheduled_prefill_groups=[],   ← sequences to run prefill on
    scheduled_decode_groups=[],    ← sequences to run decode on
    ignored_groups=[],             ← groups skipped this step
    finished_groups=[],            ← groups that completed this step
    rejected_groups=[],            ← groups that can never be served

    num_batched_tokens=N,          ← total tokens in this batch
    num_prefill_tokens=N,          ← tokens from prefill groups
    num_decode_tokens=N,           ← tokens from decode groups
    token_budget_remaining=N,      ← unused budget this step

    debug_reason="...",            ← human-readable scheduler decision
    ignored_reasons={},            ← per-group ignore reason
    preempted_groups=[],           ← (future: preemption victims)
)
```

### Engine dispatches based on this result:

```python
# EngineCore.step():
result = scheduler.schedule()

# Dispatch prefill (only PREFILL-status sequences)
for sg in result.scheduled_prefill_groups:
    for seq in sg.get_unfinished_seqs():
        if seq.status == Status.PREFILL:
            only_prefill_seqs.append(seq)
if only_prefill_seqs:
    executor.prefill(only_prefill_seqs)

# Dispatch decode (all unfinished sequences in decode groups)
for sg in result.scheduled_decode_groups:
    decode_seqs.extend(sg.get_unfinished_seqs())
if decode_seqs:
    executor.decode(decode_seqs)
```

---

## 7. vLLM Scheduler Mapping

| mini-vLLM | vLLM | Notes |
|-----------|------|-------|
| `Scheduler.schedule()` | `Scheduler.schedule()` | Same entry point |
| `ScheduleResult` | `SchedulerOutputs` | Same structure |
| `scheduled_prefill_groups` | `seq_groups` (prefill) | vLLM also tracks `is_prefill=True` |
| `scheduled_decode_groups` | `seq_groups` (decode) | vLLM sets `is_prefill=False` |
| `ignored_groups` | `ignored_seq_groups` | Same concept |
| `max_num_batched_tokens` | `max_num_batched_tokens` | Same config |
| `max_num_seqs` | `max_num_seqs` | Same config |
| Chunked prefill | `enable_chunked_prefill` | Same mechanism |
| Decode-first priority | Implicit (decode always 1 token) | Same effect |
| On-demand alloc via manager | `BlockSpaceManager` | Same three-layer split |
| No preemption yet | Preemption via swap | Future work |
| No priority scheduling | Priority via `priority` field | Future work |

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 6 separate phases | Each phase has a single responsibility. Easy to trace, test, and extend (e.g., adding priority between phases). |
| Decode-first as Phase 3 | Guarantees decode never starves. Decode gets 1 token per seq, prefill gets the remaining budget. |
| `prefill_budget = min(remaining, max_prefill_tokens)` | Separate cap prevents prefill from dominating even when batched budget is large. |
| Scheduler doesn't allocate blocks | Clean separation: scheduler handles tokens, BlockManager handles memory. Avoids coupling scheduling policy to memory layout. |
| `ScheduleResult` carries ignored_reasons dict | Production debugging requires knowing WHY a group was skipped. `WAITING_FOR_NEXT_STEP` vs `NO_TOKEN_BUDGET` tells very different stories. |
| Cursor on Sequence, not ScheduleResult | Cursor is persistent state that must survive across steps. The scheduler reads it but doesn't own it. |
