# Runtime Timeline — Engine Step Lifecycle

> **Design Doc** — How one engine step flows through mini-vLLM's components, and
> how sequences transition through their lifecycle.

---

## 1. Engine Step: End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LLMEngine.step()                            │
│                                                                     │
│  1. EngineCore.step()                                               │
│     │                                                               │
│     ├─► 2. Scheduler.schedule()                                     │
│     │      │                                                        │
│     │      │  Phase 1 ── Finish check                               │
│     │      │  Phase 2 ── Categorize running (decode / prefill-cont) │
│     │      │  Phase 3 ── Decode-first budget deduction              │
│     │      │  Phase 4 ── Chunked-prefill continue                   │
│     │      │  Phase 5 ── Admit new waiting groups                   │
│     │      │  Phase 6 ── Token counts & debug_reason                │
│     │      │                                                        │
│     │      └─► Returns ScheduleResult                               │
│     │            ├─ scheduled_prefill_groups                        │
│     │            ├─ scheduled_decode_groups                         │
│     │            ├─ ignored_groups + ignored_reasons                │
│     │            ├─ finished_groups                                 │
│     │            └─ token counts & budget remaining                 │
│     │                                                               │
│     ├─► 3. Executor.prefill(prefill_seqs)                           │
│     │      │                                                        │
│     │      │  for each seq:                                         │
│     │      │    for pos in [cursor, cursor+chunk):                  │
│     │      │      _write_to_kv(seq, pos, token)                    │
│     │      │        └─► BlockManager.ensure_block(seq, pos)        │
│     │      │              └─► BlockAllocator.allocate(1)  ← on-demand│
│     │      │    cursor += chunk                                     │
│     │      │    if is_prefill_finished → status = RUNNING           │
│     │      │                                                        │
│     │      └─► first_token_time = now  (for TTFT)                  │
│     │                                                               │
│     └─► 4. Executor.decode(decode_seqs)                             │
│            │                                                        │
│            │  for each seq:                                         │
│            │    read KV → produce next_token                        │
│            │    _write_to_kv(seq, new_pos, token)  ← decode writes! │
│            │    append to output_token_ids                          │
│            │    num_generated_tokens++                              │
│            │                                                        │
│            └─► (no TTFT update — decode is not first token)        │
│                                                                     │
│  5. _print_step / _print_memory_trace (if enabled)                  │
│  6. Capture finished outputs                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Sequence State Machine

```
                    ┌──────────────┐
                    │   WAITING    │  Initial state, in RequestQueue.waiting
                    └──────┬───────┘
                           │
                    Scheduler Phase 5
                    (admit, allocate_for_seq, set cursor=0)
                           │
                           ▼
                    ┌──────────────┐
                    │   PREFILL    │  Prompt tokens being written to KV
                    └──────┬───────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             │    cursor < prompt_len    │ cursor >= prompt_len
             │    (more chunks remain)   │ (is_prefill_finished)
             │             │             │
             │    Next step: Phase 4     │
             │    scheduler continues    │
             │    prefill (same status)  │
             │             │             │
             │             └──────┬──────┘
             │                    │
             │                    ▼
             │          ┌──────────────┐
             │          │   RUNNING    │  Generating tokens one at a time
             │          └──────┬───────┘
             │                 │
             │        num_generated >= max_tokens
             │                 │
             │                 ▼
             │          ┌──────────────┐
             └──────────│  FINISHED    │  Blocks freed, output captured
                        └──────────────┘

    ── Additional states ──
    ┌────────────┐
    │  REJECTED  │  Prompt too long for max_num_batched_tokens
    └────────────┘
    (SequenceGroups also carry IGNORED per-step — not a sequence state,
     but a scheduling decision: "skip this step, try again next time")
```

---

## 3. Per-Step Component Interaction (Sequence-Level Trace)

### Step 1: Admit + First Prefill Chunk

```
  RequestQueue           Scheduler              Executor           BlockManager
  ────────────           ──────────             ────────           ────────────
  waiting: [A, B]
       │
       ├──── Phase 5 ──► allocate_for_seq(A) ────────────────► create empty table
       │                                                     ◄── seq.block_table = []
       │                allocate_for_seq(B) ────────────────► create empty table
       │                                                     ◄── seq.block_table = []
       │                status = PREFILL, cursor = 0
       │                mark_running(A), mark_running(B)
       │          ◄──── scheduled_prefill = [A, B]
       │
  running: [A, B]
       │
       └────────────► prefill([A, B])
                          │
                          ├─ A: _write_to_kv(pos=0..3)
                          │      └─ ensure_block(A, 0) ──► allocate(1) → P0
                          │      └─ ensure_block(A, 1) ──► no-op
                          │      └─ ensure_block(A, 2) ──► no-op
                          │      └─ ensure_block(A, 3) ──► no-op
                          │    cursor: 0 → 4
                          │
                          └─ B: _write_to_kv(pos=0..3)
                                 └─ ensure_block(B, 0) ──► allocate(1) → P1
                               cursor: 0 → 4

  Result: A(blocks=[P0]), B(blocks=[P1]), free=14/16
```

### Step 2: Second Prefill Chunk (No Admit)

```
  RequestQueue           Scheduler              Executor           BlockManager
  ────────────           ──────────             ────────           ────────────
  waiting: []
  running: [A, B]
       │
       ├──── Phase 2 ──► A=PREFILL(cursor=4), B=PREFILL(cursor=4)
       ├──── Phase 3 ──► decode_groups=[], token_budget=16
       ├──── Phase 4 ──► A: chunk=4, B: chunk=4
       │          ◄──── scheduled_prefill = [A, B]
       │
       └────────────► prefill([A, B])
                          │
                          ├─ A: _write_to_kv(pos=4..7)
                          │      └─ ensure_block(A, 4) ──► allocate(1) → P2
                          │      └─ ensure_block(A, 5..7) ──► no-op
                          │    cursor: 4 → 8
                          │
                          └─ B: _write_to_kv(pos=4..7)
                                 └─ ensure_block(B, 4) ──► allocate(1) → P3

  Result: A(blocks=[P0,P2]), B(blocks=[P1,P3]), free=12/16
```

### Step 3: Prefill Completes A, Admit C

```
  RequestQueue           Scheduler              Executor           BlockManager
  ────────────           ──────────             ────────           ────────────
  waiting: []          
  running: [A, B]
       │
       ├──── Phase 2 ──► A=PREFILL(cursor=8), B=PREFILL(cursor=8)
       ├──── Phase 4 ──► A: chunk=3(11-8), B: chunk=4(12-8)
       │                 prefill_budget: 16-3-4=9
       │
       │   ┌─ arrival ◄── C arrives (waiting=[C])
       │   │
       ├──── Phase 5 ──► C: chunk=4, admit
       │          ◄──── scheduled_prefill = [A, B, C]
       │
       └────────────► prefill([A, B, C])
                          │
                          ├─ A: _write_to_kv(pos=8..10)
                          │      └─ ensure_block(A, 8) ──► allocate(1) → P4
                          │    cursor: 8 → 11
                          │    is_prefill_finished? YES
                          │    → status = RUNNING, gen=1
                          │
                          ├─ B: _write_to_kv(pos=8..11)
                          │      └─ ensure_block(B, 8) ──► allocate(1) → P5
                          │    cursor: 8 → 12
                          │
                          └─ C: _write_to_kv(pos=0..3)
                                 └─ ensure_block(C, 0) ──► allocate(1) → P6

  Result: A(RUNNING, 3 blocks), B(PREFILL, 3 blocks), C(PREFILL, 1 block)
          free=9/16
```

### Step 4: First Decode + Prefill Continue

```
  RequestQueue           Scheduler              Executor           BlockManager
  ────────────           ──────────             ────────           ────────────
  running: [A, B, C]
       │
       ├──── Phase 2 ──► decode=[A], prefill_continue=[B, C]
       ├──── Phase 3 ──► deduct A: token_budget=16-1=15
       ├──── Phase 4 ──► B: chunk=1(13-12), C: chunk=4(8-4)
       │          ◄──── scheduled_decode = [A]
       │                scheduled_prefill = [B, C]
       │
       ├──── decode([A]) ─────────────────────────────────┐
       │    read KV at pos=11                             │
       │    compute next_token                             │
       │    _write_to_kv(A, pos=11+1=12, token)           │
       │      └─ ensure_block(A, 12) ──► allocate(1)→P9   │
       │    gen: 1 → 2                                    │
       │                                                  │
       └──── prefill([B, C]) ──────────────────────────┐  │
                  │                                     │  │
                  ├─ B: _write_to_kv(pos=12)            │  │
                  │      └─ ensure_block(B,12)→P7       │  │
                  │    cursor: 12 → 13                  │  │
                  │    is_prefill_finished? YES          │  │
                  │    → RUNNING, gen=1                 │  │
                  │                                     │  │
                  └─ C: _write_to_kv(pos=4..7)          │  │
                         └─ ensure_block(C,4)→P8        │  │
                       cursor: 4 → 8                    │  │
                                                        │  │
  Result: A(RUNNING, 4 blocks), B(RUNNING, 4 blocks), ◄──┘  │
          C(PREFILL, 2 blocks), free=6/16 ◄──────────────────┘

  Key observation: position 12 = logical block 3 for A → P9
                    position 12 = logical block 3 for B → P7
                    (same logical index, different sequences → different physical blocks)
```

---

## 4. Chunked Prefill Sequence (Single Request)

```
For a request with prompt_len=12, max_new=2, chunk_size=4:

Time    Scheduler                  Executor                     Sequence State
────    ──────────                  ────────                     ─────────────
T1      Phase 5: admit
        allocate_for_seq(seq)                                    block_table = []
        set cursor=0                                             Status = PREFILL
        mark_running
        │
        │ scheduled_prefill = [seq]
        │
        └──────────────►  prefill([seq])
                          _write_to_kv(pos=0) → ensure_block(0)  blocks = [P0]
                          _write_to_kv(pos=1) → no-op
                          _write_to_kv(pos=2) → no-op
                          _write_to_kv(pos=3) → no-op
                          cursor = 4                              Status = PREFILL

T2      Phase 4: continue
        cursor=4, remaining=8, chunk=4
        │
        └──────────────►  prefill([seq])
                          _write_to_kv(pos=4) → ensure_block(4)  blocks = [P0, P2]
                          _write_to_kv(pos=5..7) → no-op
                          cursor = 8                              Status = PREFILL

T3      Phase 4: continue
        cursor=8, remaining=4, chunk=4
        │
        └──────────────►  prefill([seq])
                          _write_to_kv(pos=8) → ensure_block(8)  blocks = [P0, P2, P4]
                          _write_to_kv(pos=9..11) → no-op
                          cursor = 12
                          is_prefill_finished? YES                Status = RUNNING
                          output_token_ids = [first_token]
                          num_generated_tokens = 1

T4      Phase 2: decode
        │
        └──────────────►  decode([seq])
                          read_kv(pos=11)
                          compute next_token
                          _write_to_kv(pos=12) → ensure_block(12)
                          append to output                       Status = RUNNING
                          num_generated_tokens = 2

T5      Finish check: num_generated(2) >= max_tokens(2)
        free(seq.seq_id)  ──►  allocator.free([P0, P2, P4])
                                blocks returned to free list
                                block_table cleared               Status = FINISHED
```

---

## 5. Where TTFT / TPOT Are Measured

```
         first_token_time (set once)
              │
              ▼
  ┌──────┐   ┌────────┐   ┌──────┐   ┌──────┐   ┌──────┐
  │Admit │   │Last    │   │Decode│   │Decode│   │Decode│
  │      │   │Prefill │   │  1   │   │  2   │   │  N   │
  └──────┘   └────────┘   └──────┘   └──────┘   └──────┘
       │          │            │          │          │
       │          ├────────────┤          ├──────────┤
       │          │   TTFT     │          │  TPOT    │
       │          │ (Time to   │          │(Time Per │
       │          │  First     │          │ Output   │
       │          │  Token)    │          │  Token)  │
       │          └────────────┘          └──────────┘
       │
  arrival_time
```

- **TTFT** = `first_token_time - arrival_time` (includes time waiting in queue)
- **TPOT** = measured per decode step from start to finish of `executor.decode()`

---

## 6. Component Ownership Map

```
┌────────────────────────────────────────────────────────┐
│                     LLMEngine                          │
│                                                        │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ Queue    │  │ BlockManager │  │  EngineCore      │ │
│  │ ──────── │  │ ──────────── │  │  ────────────    │ │
│  │ waiting  │  │ allocate/    │  │  Scheduler       │ │
│  │ running  │  │ free/        │  │  Executor        │ │
│  │ finished │  │ ensure_block │  │                  │ │
│  │ rejected │  │ trace events │  │                  │ │
│  └──────────┘  └──────┬───────┘  └──────────────────┘ │
│                        │                               │
│                        ▼                               │
│              ┌──────────────────┐                      │
│              │  BlockAllocator  │                      │
│              │  ──────────────  │                      │
│              │  allocate(N)     │  on_allocate/        │
│              │  free(pids)      │  on_free callbacks   │
│              │  free_list       │────► FakeModelExec   │
│              └──────────────────┘      (kv_cache)      │
└────────────────────────────────────────────────────────┘
```

---

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `EngineCore` as separate class | Mirrors vLLM's split: public `LLMEngine` + inner loop `EngineCore`. Enables async variant (`AsyncLLMEngine`) later. |
| `FakeModelExecutor` owns `_kv_cache` | BlockAllocator is a pure free-list — has no concept of "data". Executor reacts to alloc/free via callbacks, maintaining the actual KV storage. Clean separation. |
| Scheduler doesn't touch allocator | Scheduler operates on `SequenceGroup` level (tokens, budget, priority). Block allocation is `BlockManager`'s job. |
| `ensure_block` called per KV write | Simple, safe. The alternative (checking block boundaries in the executor) duplicates logic. `ensure_block` is idempotent for already-allocated blocks. |
| Prefill cursor stored on `Sequence` | Cursor must survive across engine steps. Scheduler doesn't touch it; executor advances it. `Sequence` is the natural owner. |
