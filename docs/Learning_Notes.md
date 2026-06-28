# Learning Notes

Design decisions, observations, and lessons learned while building this.

**Design docs (recommended reading order):**
- [Runtime_Timeline.md](Runtime_Timeline.md) — Engine step lifecycle, sequence state machine, component interaction trace
- [Scheduler.md](Scheduler.md) — 6-phase scheduling algorithm, token budget model, decode-first priority, chunked prefill, admission policy
- [Memory_Manager.md](Memory_Manager.md) — Three-layer KV cache architecture, on-demand vs eager allocation, PagedAttention mapping, trace format

## Why Separate Prefill and Decode?

In transformer inference, **prefill** (processing the prompt) and **decode**
(generating one token at a time) have fundamentally different compute profiles:

- **Prefill** is *compute-bound* — large matrix multiplications over the full
  prompt length, high arithmetic intensity, high GPU utilisation.
- **Decode** is *memory-bound* — each step processes one token.  The dominant
  cost is loading model weights and KV cache from HBM.  Arithmetic intensity
  is low.

vLLM launches them as separate CUDA kernels.  The scheduler explicitly returns
two batches so the engine can dispatch them differently.

## Why Eager (Not On-Demand) KV Cache Allocation?

Phase 1 allocates all blocks a request will ever need at admission time:

```python
blocks_needed = ceil((prompt_len + max_new_tokens) / block_size)
```

This is **wasteful** — a request with `max_new_tokens=1024` reserves blocks
for tokens it may never generate (e.g. if the model hits EOS early).  vLLM
allocates blocks on-demand: one block during prefill, then one more whenever
the current block fills up.  This allows over-commitment and higher
utilisation.

We use eager allocation because it's simpler to reason about and guarantees
no request will ever OOM mid-decode — which is valuable when the goal is
learning the scheduling dynamics, not maximising cache utilisation.

## Block Size Trade-off

`block_size=4` is unrealistically small (vLLM typically uses 16).  But for
the fake demo, a small block size means the block table fills up fast and
allocation / free dynamics are visible within a few steps.  Real vLLM
systems use larger blocks (16-32 tokens) to amortise metadata overhead.

## Why Not Single-File?

Each module maps 1:1 to a vLLM counterpart.  Keeping them separate:

1. **Makes the vLLM mapping obvious** — open `mini_vllm/scheduler/scheduler.py`
   next to `vllm/core/scheduler.py` and the structural similarity is clear.
2. **Enforces clean interfaces** — no cross-module coupling beyond the
   `ScheduleResult` / `BlockTable` / `Sequence` contracts.
3. **Scales to later phases** — when we add CUDA kernels, preemption,
   swapping, etc., each module grows without creating a monster file.

## The Token Numbering Trap

In the scheduler, one detail that's easy to get wrong: a request's *total
tokens in the batch* changes between prefill and decode.  During prefill,
the request contributes `prompt_length` tokens to the batched-token budget.
During decode, it contributes exactly **1** token.  The scheduler must track
both budgets correctly or risk over-subscribing the GPU.

## Why Split SequenceGroup / Sequence?

The original design merged everything into a single `Request` class.  After
discussion, we split into three classes mirroring vLLM:

* `SequenceGroup` — user-facing metadata: prompt text, `SamplingParams`
  (temperature, top_p, max_tokens, stop strings), arrival time, and a list of
  child `Sequence` objects.  The scheduler moves groups between pools.

* `Sequence` — per-generation state: token buffers, status, block table,
  TTFT / finish timestamps.  The model runner operates on sequences.

* `SamplingParams` — generation knobs, extracted as its own class so future
  phases can add more fields (repetition penalty, min_p, etc.) without
  bloating `SequenceGroup`.

The key insight: the **scheduler reasons about groups** (budget, priority,
sampling config), but the **model runner reasons about sequences** (tokens,
KV blocks).  Queue pools store groups; `ScheduleResult` reports groups.
Individual sequence status is tracked *inside* each group — the scheduler
checks `sg.get_unfinished_seqs()` per step.

This split is load-bearing scaffolding for future phases: beam search
(multiple sequences per group), priority scheduling (group-level priority
drives admission order), and preemption (swap out entire groups).

## Why RequestQueue Holds Groups in All Pools

vLLM stores `SequenceGroup` objects in its scheduler-internal waiting,
running, and swapped queues — never raw `Sequence` objects.  Aligning with
this means:

- `RequestQueue.waiting` = `SequenceGroup`
- `RequestQueue.running` = `SequenceGroup`
- `RequestQueue.finished` = `SequenceGroup`
- `RequestQueue.rejected` = `SequenceGroup`

Pools keyed by group ID mean `get_by_id(rid)` always returns the group,
which owns its sequences.  The scheduler iterates running groups and asks
each for its unfinished sequences.

## ScheduleResult Mirrors SchedulerOutputs

`ScheduleResult` was refactored from a flat `(prefill_requests,
decode_requests, finished_requests)` to a structured object matching
vLLM's `SchedulerOutputs`::

    ScheduleResult(
        scheduled_prefill_groups=[],
        scheduled_decode_groups=[],
        ignored_groups=[],
        finished_groups=[],
        rejected_groups=[],
        num_batched_tokens=N,
        num_prefill_tokens=N,
        num_decode_tokens=N,
        reason="...",
    )

This richer structure turns the scheduler output from "here are some
sequences to process" into a full audit record of what was scheduled,
why some were skipped, and how many tokens each phase consumed — exactly
what a production system needs for logging, metrics, and debugging.

## Phase 2 — Chunked Prefill, Decode-First, Memory Trace

### Chunked Prefill Design

A prompt longer than `max_prefill_chunk_size` no longer gets rejected. Instead,
the scheduler admits it and splits the prefill work across multiple steps:

```
prompt_len=12, chunk_size=4:
  step 1:  write tokens 0..3   → cursor=4,  PREFILL
  step 2:  write tokens 4..7   → cursor=8,  PREFILL
  step 3:  write tokens 8..11  → cursor=12, PREFILL finished → RUNNING
```

The `Sequence.prefill_cursor` tracks how many prompt tokens have been written
to KV.  The `is_prefill_finished` property checks `cursor >= prompt_len`.
The executor reads from cursor and writes the chunk; the scheduler advances
cursor after execution.

### Decode-First Scheduling Priority

In each step, running decode sequences consume their token budget *before*
any prefill group is considered.  The scheduler categorises running groups
into `decode_groups` (Status.RUNNING) and `prefill_continue_groups`
(Status.PREFILL with cursor > 0), then:

1. Deduct `num_decode_seqs` from `max_num_batched_tokens` immediately
2. Compute `prefill_budget = min(remaining, max_num_prefill_tokens)`
3. Chunked-prefill continue groups consume from prefill_budget
4. Only then admit new waiting groups with what's left

This guarantees decode latency is never blocked by prefill — matching
vLLM's scheduler behaviour.

### Memory Trace — Eager Allocation Waste

The Memory Trace mode (`config.memory_trace=True`) revealed a critical
inefficiency.  Our BlockManager still uses **eager allocation**: at admission
time, a sequence is allocated `ceil((prompt_len + max_tokens) / block_size)`
blocks — its entire lifetime requirement up front.

The trace from the demo (3 requests, 16 blocks, block_size=4) shows the
consequences clearly:

| Step | Free blocks | Used | Actually needed | Slack (wasted) |
|------|------------|------|----------------|---------------|
| 1    | 4          | 12   | 2 (A:1, B:1)  | 10            |
| 3    | **0**      | 16   | 7 (A:3, B:3, C:1) | **9**    |
| 4-10 | **0**      | 16   | 7              | **9** (stuck) |
| 11   | 9          | 7    | 4 (B:4)        | 3             |

At step 3, all 16 blocks are allocated (100% utilisation reported), but only
7 blocks contain any KV data. 9 blocks (56% of the cache) are reserved for
future tokens that haven't been generated yet.  A 4th request arriving at
step 3 would be **rejected** despite having 9 blocks of effective free space.

Root cause: eager allocation computes `blocks_needed = (prompt_len + max_tokens
+ block_size - 1) // block_size` at admission and allocates that many physical
blocks.  During prefill, only `ceil(cursor / block_size)` blocks are actually
used.  During early decode, only `ceil((prompt_len + generated) / block_size)`
blocks are used.  The gap between "allocated" and "really needed" persists
for the entire lifetime of the request.

**The insight**: KV cache pressure in eager mode comes from the *sum of worst-
case lifetimes*, not from the *sum of actual data*.  This is the fundamental
motivation for **on-demand allocation** — allocate one block during prefill,
then one more whenever the current block fills up during decode.  With
on-demand, the cache can be over-committed: the system admits more requests
than physical blocks, relying on the fact that most requests won't need all
their blocks at the same time.

### Three KV Metrics Clarified

During development we confused three related but distinct metrics.  The final
definitions in `executor.py`:

- **kv_tokens_written** — actual token-level KV data written to cache
  (incremented per-token in `_write_to_kv`)
- **kv_slot_capacity** — total token capacity of all allocated blocks
  (= `allocated_blocks * block_size`)
- **allocated_blocks** — physical blocks currently allocated
  (= `allocator.num_used_blocks`)

The double-counting bug: `prefill()` incremented `_total_tokens_processed`
both inside `_write_to_kv()` (per token written) and after the loop
(`end - start`).  Fixed by removing the redundant post-loop increment.

### Phase 2 Test Coverage

4 new scheduler tests plus updated existing tests:
- `test_decode_first` — decode gets budget priority over new prefill
- `test_chunked_prefill` — 12-token prompt splits across 3 chunks of 4
- `test_prefill_not_finished_not_decode` — partial prefill stays PREFILL
- `test_ignored_reasons` — ignored groups carry non-empty reason string
- Updated `test_schedule_result_fields` for new `ScheduleResult` fields
- Updated executor tests for renamed KV stat keys

All 46 tests pass.

## Phase 3 — On-Demand Block Allocation

### The Insight

**Eager allocation is like `malloc(prompt_len + max_tokens)` upfront.**
**On-demand is like `std::vector::push_back()` — grow as you write.**

The core idea: `Sequence` starts with **zero blocks**.  Every token write
(`_write_to_kv`) calls `BlockManager.ensure_block(seq, position)`, which
allocates exactly one new physical block when `position // block_size`
exceeds the current number of blocks.  Otherwise it's a no-op.

This is the rewrite of `BlockManager`:

```
Before (eager):                     After (on-demand):

allocate_for_seq(seq, N):           allocate_for_seq(seq):
    pids = allocator.allocate(N)        # just register, no alloc
    table.add_blocks(pids)              seq.block_table = []
    seq.block_table = pids
                                    ensure_block(seq, pos):
                                        if pos // BS >= len(table):
Scheduler:                                allocator.allocate(1)
    blocks = ceil((plen+max)/BS)          table.add_block(pid)
    mgr.allocate_for_seq(seq,blocks)  # no block count in scheduler!

Executor:                            Executor:
    _write_to_kv(seq, pos):             _write_to_kv(seq, pos):
        pid = seq.block_table[pos//BS]      pid = mgr.ensure_block(seq, pos)
        kv_cache[pid] += [k, v]             kv_cache[pid] += [k, v]
```

### Eager vs On-Demand Trace Comparison

Same demo (3 requests, 16 blocks, block_size=4):

| Step | Metric | Eager | On-Demand |
|------|--------|-------|-----------|
| 1    | Used blocks | 12 (75%) | **2 (12%)** |
|      | Peak waste | 10 blocks | 0 |
| 3    | Used blocks | 16 (100%) | **7 (44%)** |
|      | Free blocks | **0** | **9** |
|      | Could admit 4th? | NO | YES |
| 8    | Used blocks | 16 (stuck) | **14 (88%)** |
|      | Room for growth | 0 | **2** |

At step 3 with eager: all 16 blocks locked, but only 7 have real data.
With on-demand: 9 blocks free, could easily admit 2 more requests.

### Decode Now Writes to KV

Previously, generated tokens were not written back to the KV cache.
`decode()` now calls `_write_to_kv()` for each generated token, mirroring
real LLM behaviour.  This means:

- During decode, blocks continue to grow as generated tokens fill blocks
- The KV cache trace reflects true data volume at every step
- `kv_tokens_written` correctly equals prompt_tokens + generated_tokens

### OOM Moves to Execution Time

In eager mode, OOM is a scheduler-time rejection (can't allocate lifetime
blocks → reject).  In on-demand mode, OOM is a runtime error thrown by
`ensure_block()` when no free physical block exists.  This is more realistic:
in production vLLM, OOM during decode triggers **preemption** — swap out a
victim sequence's blocks to CPU and reclaim them for the active sequence.

### Test Changes

- `test_reject_on_oom` → `test_ondemand_admits_without_allocating_blocks`
  (scheduler admits even with 2 blocks, seq.block_table starts empty)
- `test_rejected_when_oom` → `test_ondemand_oom_during_execution`
  (RuntimeError raised by ensure_block when no free blocks remain)
- 3 new BlockManager unit tests for `ensure_block` on-demand allocation

All 47 tests pass.

## KV Cache Three-Layer Split

vLLM separates KV cache management into three layers::

    BlockAllocator > BlockSpaceManager > BlockTable

We mirror this exactly:

- **BlockAllocator** — pure free-list of physical block IDs.  Takes optional
  `on_allocate` / `on_free` callbacks so the executor can react to block
  lifecycle events (allocate / release fake KV storage).  No knowledge of
  sequences or scheduling.

- **BlockManager** — coordinates the allocator.  `allocate_for_seq()`
  registers a sequence with an empty `BlockTable` (no blocks allocated).
  Actual blocks are allocated on-demand through `ensure_block()`, called
  by the executor before each KV write.  `free()` reclaims all blocks
  owned by a sequence.  The scheduler never touches the allocator
  directly — it always goes through the manager.

- **BlockTable** — per-sequence mapping.  Keeps a `List[int]` of physical
  block IDs.  `get_physical_block(token_position)` derives the logical
  block index from the token position and looks up the physical ID.

The old `KVCacheManager` merged all three into one class.  Splitting them
makes the system more vLLM-like and isolates each concern for future phases
(swapping needs a CPU allocator; preemption needs the manager to track
reference counts).

## Fake KV Cache in the Executor

The original `FakeModelRunner` was purely arithmetic — no state, no memory.
The new `FakeModelExecutor` maintains `_kv_cache: Dict[int, List[int]]`
as a simulated device-side KV cache::

    BlockAllocator.allocate()   >   executor._prepare_block(pid)   >   _kv_cache[pid] = []
    BlockAllocator.free()       >   executor._release_block(pid)   >   _kv_cache.pop(pid)

During inference:

- **Prefill**: for each prompt token, the executor writes a fake key/value
  into the corresponding physical block (looked up from `seq.block_table`).
  The output token is then sampled: `(sum(prompt_ids) + 1) % vocab_size`.

- **Decode**: the next token depends on *both* the previous output AND the
  current KV content::

      kv_bias = sum(kv_data_at_position) % vocab_size
      next_token = (prev_token + 7 + kv_bias) % vocab_size

  Each generated token is also written to KV (``_write_to_kv``),
  mirroring real LLM behaviour where every decode step produces new KV
  entries that affect subsequent attention reads.

  This is a crude simulation of attention — the KV cache actually influences
  what tokens come next.  Changing `block_size`, prompt content, or
  allocation strategy now produces different outputs (not just different
  scheduling traces).

The benefit: when we later swap in a real model (Qwen, Llama), the scheduler
and engine code won't need changes — only the executor (now an isolated
module) gets replaced.

## Engine Two-Layer Split

vLLM separates the public API from the inner loop::

    LLMEngine > EngineCore > ModelExecutor

We follow the same pattern:

- **LLMEngine** — public interface.  Owns the queue, block manager, output
  dict.  `add_request()` creates a `SequenceGroup`.  `step()` calls
  `EngineCore.step()` then captures outputs.  `run_until_done()` loops.

- **EngineCore** — the inner step loop.  Owns the scheduler and executor.
  `step()` calls `scheduler.schedule()` then dispatches prefill and
  decode to the executor.  Returns `ScheduleResult`.

This means `LLMEngine` is the only class users import.  `EngineCore` is
an internal component that could be swapped for an async version in future
phases.  vLLM's `AsyncLLMEngine` does exactly this — the core loop runs
in a background thread while the public API returns coroutines.

## Directory Restructure

The original flat `mini_vllm/` package grew to 12 files and was becoming
hard to navigate.  We reorganised into sub-packages mirroring vLLM's own
module structure::

    mini_vllm/
    ├── sequence/        # Status, SamplingParams, Sequence, SequenceGroup, RequestQueue
    ├── cache/           # BlockTable, BlockAllocator, BlockManager
    ├── scheduler/       # Scheduler, ScheduleResult
    ├── executor/        # FakeModelExecutor
    ├── engine/          # LLMEngine, EngineCore, MetricsCollector
    └── worker/          # (future GPUWorker)

Benefits:

1. **Navigation** — opening `mini_vllm/engine/` shows exactly the engine
   components; opening `mini_vllm/cache/` shows KV cache components.

2. **vLLM alignment** — paths map to vLLM source: `mini_vllm/scheduler/`
   vs `vllm/core/scheduler.py`, `mini_vllm/engine/` vs `vllm/engine/`.

3. **Scaling** — future modules get their own package (`model/` for model
   loading, `worker/` for GPU workers) without bloating any package.

4. **Testing** — `from mini_vllm.engine.llm_engine import LLMEngine` is
   explicit.  The root `__init__.py` re-exports all public symbols so
   existing code (`from mini_vllm import LLMEngine, Config`) still works.
