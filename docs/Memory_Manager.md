# Memory Manager — KV Cache Allocation Architecture

> **Design Doc** — Three-layer cache hierarchy, on-demand vs eager allocation,
> and how PagedAttention maps logical blocks to physical blocks.

---

## 1. Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BlockAllocator                               │
│                                                                      │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐                  │
│  │ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7│ 8│ 9│10│11│12│13│14│15│  physical blocks │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘                  │
│   ▲     ▲        allocated (free=False)                              │
│   │     │        white: free (free=True)                             │
│   │     │                                                            │
│   │     └──►  on_allocate(pid) callback                              │
│   │              └─ executor._kv_cache[pid] = []                     │
│   │                                                                  │
│   └──►  on_free(pid) callback                                        │
│            └─ executor._kv_cache.pop(pid)                            │
│                                                                      │
│  allocate(N) → [pid, ...] or None                                    │
│  free([pids]) → marks free, fires callback                           │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
               holds reference (not ownership)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         BlockManager                                 │
│                                                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐                    │
│  │  BlockTable (seq A) │  │  BlockTable (seq B) │   per-sequence    │
│  │                     │  │                     │   mapping         │
│  │  logical  physical  │  │  logical  physical  │                    │
│  │  ────────┴────────  │  │  ────────┴────────  │                    │
│  │  L0    ──►  P0      │  │  L0    ──►  P1      │                    │
│  │  L1    ──►  P2      │  │  L1    ──►  P3      │                    │
│  │  L2    ──►  P4      │  │  L2    ──►  P5      │                    │
│  └─────────────────────┘  └─────────────────────┘                    │
│                                                                      │
│  allocate_for_seq(seq)  ──  register empty BlockTable                │
│  ensure_block(seq,pos)  ──  alloc on demand, returns physical ID    │
│  free(seq_id)           ──  release all blocks of sequence          │
│  dump_tables()          ──  trace support                            │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
            each BlockTable holds a list of physical block IDs
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         BlockTable (per Sequence)                     │
│                                                                      │
│  Sequence: req-0000-seq-0                                            │
│                                                                      │
│  ┌──────────┬────────────┬────────────────────────────────┐          │
│  │ Block ID │ Token Range│ Physical Block (P0)            │          │
│  │          │            │                                │          │
│  │ logical  │ position   │  ┌───┬───┬───┬───┬───┬───┬───┐│          │
│  │   0      │  0 .. 3    │  │ k │ v │ k │ v │ k │ v │k/v││  KV data │
│  │          │            │  └───┴───┴───┴───┴───┴───┴───┘│          │
│  ├──────────┼────────────┤  ┌───┬───┬───┬───┬───┬───┬───┐│          │
│  │   1      │  4 .. 7    │  │ k │ v │ k │ v │ k │ v │k/v││          │
│  │          │            │  └───┴───┴───┴───┴───┴───┴───┘│          │
│  └──────────┴────────────┴────────────────────────────────┘          │
│                                                                      │
│  get_physical_block(token_position) → physical_block_id              │
│  get_block_ids() → [P0, P2, P4, ...]                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Sequence-to-Physical-Block Resolution

When the executor writes to KV, it resolves the physical block through two
indirections:

```
Token position 5
       │
       │  position // block_size
       ▼
Logical block index:  5 // 4 = 1
       │
       │  BlockTable.get_physical_block(1)
       ▼
Physical block ID:  P2
       │
       │  _kv_cache[P2] += [key, value]
       ▼
KV data stored in physical block 2

BlockTable:
  L0 → P0
  L1 → P2    ←  position 5 falls here
  L2 → P4
```

**Why two indirections?** This is PagedAttention's key idea — logical→physical
indirection enables:
1. **Non-contiguous physical storage**: a sequence's KV data can be scattered
   across physical blocks (P0, P2, P4, ...), which wouldn't be possible with a
   flat array.
2. **On-demand allocation**: blocks are allocated one at a time only when a
   token position crosses a block boundary.
3. **Sharing**: in production vLLM, multiple sequences in beam search can share
   physical blocks via copy-on-write (ref counts in BlockTable).

---

## 3. On-Demand Allocation: How `ensure_block` Works

```
                        ensure_block(seq, token_position)
                               │
                               │
                               ▼
                    logical_idx = position // block_size
                               │
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    │                     │
                    ▼                     ▼
              logical_idx <            logical_idx >=
              table.num_blocks()       table.num_blocks()
                    │                     │
                    │                     │
                    ▼                     ▼
               return current        while logical_idx >= num_blocks():
               physical_block             │
                                    pids = allocator.allocate(1)
                                          │
                                    ┌─────┴────┐
                                    │          │
                                    ▼          ▼
                                 pids is    pids is None
                                 not None    (OOM)
                                    │          │
                                    │          ▼
                                    │    raise RuntimeError
                                    │    ("OOM: no free block")
                                    │
                                    ▼
                              table.add_block(pid)
                              seq.block_table = get_block_ids()
                              trace event: ALLOC
                                    │
                              ┌─────┘
                              │
                              ▼
                         return physical_block
```

### State machine of a single sequence's block table

```
Admission:
  block_table = []              num_blocks = 0

Write position 0 (first prefill token):
  ensure_block(seq, 0) → alloc   block_table = [P0]      num_blocks = 1

Write positions 1, 2, 3:
  ensure_block(seq, 1..3) → hit  block_table = [P0]       num_blocks = 1

Write position 4 (crosses block_size boundary):
  ensure_block(seq, 4) → alloc   block_table = [P0, P2]   num_blocks = 2

Write positions 5, 6, 7:
  ensure_block(seq, 5..7) → hit  block_table = [P0, P2]   num_blocks = 2

... (growth pattern repeats every block_size tokens until generation completes)

Finish:
  free(seq_id) → allocator.free([P0, P2, ...])
         block_table = [] (cleared)
```

### Decode also allocates

Real LLMs write the generated token's K and V at each decode step.
Our decode now mirrors this:

```
Decode step for seq A (prompt_len=11, gen=2 → writing position 12):

  new_pos = 11 + 2 = 13
  logical_idx = 13 // 4 = 3

  Current block_table = [P0, P2, P4] (num_blocks=3)
  logical_idx(3) >= num_blocks(3) → YES → allocate one more block

  allocate(1) → P9
  block_table = [P0, P2, P4, P9]
  write KV data to P9
```

**Without this, the KV cache would only hold prompt tokens**, which would make
the decode behavior unrealistic for memory analysis.

---

## 4. Eager vs On-Demand: Trace Comparison

### Same demo (3 req, 16 blocks, block_size=4, prompt A=11/B=13/C=9)

**Step 3 — peak pressure point**:

```
                   Eager Allocation                   On-Demand Allocation
               ┌──────────────────────┐           ┌──────────────────────┐
Physical Pool: │ 0 1 2 3 4 5 6 7 8 9 │          │ 0 1 2 3 4 5 6 7 8 9 │
               │ A A A A B B B B C C │          │ A A B B A B A C      │
               │ 0 1 2 3 4 5 6 7 8 9 │          │ 0 2 1 3 4 5 6       │
               │                      │          │                      │
               │ 10 11 12 13 14 15    │          │ 10 11 12 13 14 15   │
               │ B B C C C C          │          │   free               │
               │ 0 1 2 3             │          │                      │
               └──────────────────────┘          └──────────────────────┘
               ██ = contains data                  ██ = contains data
               ░░ = allocated but empty            ░░ = free

               Used: 16 / 16 (100%)                Used: 7 / 16 (44%)
               Free:  0 / 16 (0%)                  Free: 9 / 16 (56%)
               Data:  7 blocks                     Data: 7 blocks
               Waste: 9 blocks (56%)               Waste: 0 blocks
```

**Step-by-step table**:

| Step | Metric | Eager | On-Demand | Saved |
|------|--------|-------|-----------|-------|
| 1 | Used blocks | 12 (75%) | **2 (12%)** | 10 |
| 1 | Free blocks | 4 | **14** | |
| 3 | Used blocks | 16 (100%) | **7 (44%)** | 9 |
| 3 | Free blocks | **0** | **9** | |
| 3 | 4th request | REJECTED | **ADMITTABLE** | |
| 8 | Used blocks | 16 (stuck) | **14 (88%)** | 2 |
| 8 | Room to grow | 0 | **2** | |
| 11 | Used (B only) | 7 | **6** | 1 |
| 11 | Returned after A,C done | 9 | **10** | |

### The insight

**Eager allocation's pressure is proportional to sum(lifetime_max_tokens).**
**On-demand's pressure is proportional to sum(actual_tokens_written).**

In production serving:
- 50 concurrent requests with max_tokens=1024 and avg prompt_len=512
- Eager: `50 × ceil((512+1024)/16) = 50 × 97 = 4850 blocks`
- On-demand at steady state: `50 × ceil((512+50)/16) ≈ 50 × 36 = 1800 blocks`
- **Real saving: ~60% fewer blocks**

This over-commitment is what makes vLLM economically viable — you serve more
requests with less GPU memory.

---

## 5. OOM and Preemption

Current implementation: OOM during `ensure_block` raises `RuntimeError`.

```
ensure_block(seq, position)
  │
  └─► allocator.allocate(1)
        │
        ├─► success → continue
        │
        └─► None → RuntimeError("OOM: no free block for seq=...")
```

**vLLM's approach** (future work):
- OOM triggers **preemption**: select a victim sequence, move its blocks to
  CPU (swap out), reclaim physical blocks for the active sequence.
- Preemption policy determines the victim: earliest arrival, lowest priority,
  largest memory footprint, etc.
- After preemption, the victim is rescheduled from its checkpoint (either
  recompute or restore from CPU swap).

The current `RuntimeError` is a placeholder — the structure is in place to
replace it with a preemption handler.

---

## 6. Memory Trace Format

When `config.memory_trace=True`, each step prints:

```
+- BlockAllocator free list [step N]
|  free blocks:  [...]       ◄─ block IDs available for allocation
|  used blocks:  [...]       ◄─ block IDs currently allocated
|  [seq_id] BlockTable: ...  ◄─ logical→physical mapping per sequence
|  [seq_id] ALLOC: blocks [] ◄─ new allocations this step
|  [seq_id] FREE:  blocks [] ◄─ freed blocks this step
|  [seq_id] blocks=N         ◄─ current vs eager comparison
|           (would be M with eager), saved=N
+-
```

This format is designed to answer four questions at a glance:
1. **How full is the allocator?** — free/used list
2. **Which sequence owns which blocks?** — BlockTable per seq
3. **What changed this step?** — ALLOC/FREE events
4. **How on-demand compares to eager?** — saved blocks count

---

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `allocate_for_seq` allocates 0 blocks | Admission is a scheduling decision, not a memory decision. Blocks are only allocated when data is written. |
| `ensure_block` allocates 1 at a time | Minimizes waste; ties memory pressure to actual token production. |
| Decode writes to KV | Real LLMs produce new K,V at each decode step. Skipping it would make the memory trace unrealistically optimistic. |
| OOM = RuntimeError | Deliberate simplification. vLLM uses preemption; this placeholder keeps the architecture clean for future swap/restore logic. |
| BlockTable stored in `Sequence.block_table` (flat list) | Simplified from vLLM's ref-counted per-block entries. Fine for single-sequence-per-group (no beam search). |
| No copy-on-write | vLLM's COW for beam search requires ref counts per block. Our sequences are independent. |
