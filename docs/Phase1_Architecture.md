# Phase 1 Architecture

## Overview

```
                        ┌──────────────────┐
                        │   LLMEngine      │  public API
                        │ (add_request,    │
                        │  run_until_done) │
                        └──────┬───────────┘
                               │ delegates to
                        ┌──────▼───────────┐
                        │   EngineCore     │  inner step loop
                        │ (step → repeat)  │
                        └──────┬───────────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
             ┌──────────┐ ┌────────┐ ┌──────────────┐
             │ Scheduler │ │Config  │ │FakeModelExec │
             └─────┬─────┘ └────────┘ └──────┬───────┘
                   │                         │
          ┌────────┼────────┐                │
          ▼        ▼        ▼                │
   ┌──────────┐┌──────┐┌──────────┐          │
   │RequestQ  ││Block ││BlockTbl  │          │
   │(groups)  ││Mgr   ││          │          │
   └──────────┘└──┬───┘└──────────┘          │
                  ▼                          │
           ┌──────────┐                     │
           │BlockAlloc│←──(callbacks)──►    │
           │          │  _prepare_block     │
           └──────────┘  _release_block    │
                                           │
                  ┌─────────────────────────┘
                  ▼
           ┌──────────────┐
           │ Fake KV Cache│  Dict[block_id, List[int]]
           └──────────────┘
```

## Package Structure (rounded — modules with real logic)

```
mini_vllm/
├── config.py                 # Config dataclass
├── sequence/                 # Data model
│   ├── status.py             # Status enum (WAITING → PREFILL → RUNNING → FINISHED)
│   ├── sampling_params.py    # SamplingParams (max_tokens, temperature, top_p, …)
│   ├── sequence.py           # Sequence (token buffers, status, block table)
│   ├── sequence_group.py     # SequenceGroup (prompt, sampling params, child seqs)
│   └── request_queue.py      # 4-pool queue (all pools hold SequenceGroup)
├── cache/                    # KV cache
│   ├── block_table.py        # Logical → physical block mapping
│   ├── block_allocator.py    # Low-level: free-list allocate / free
│   └── block_manager.py      # High-level: allocate_for_seq, append_token, free
├── scheduler/
│   └── scheduler.py          # Scheduler + ScheduleResult
├── executor/
│   └── fake_executor.py      # Fake model with simulated KV & logits
├── engine/
│   ├── engine_core.py        # Inner step loop (scheduler → executor)
│   ├── llm_engine.py         # Public API (add_request, run_until_done, outputs)
│   └── metrics.py            # MetricsCollector (TTFT, TPOT, throughput)
└── worker/                   # Future: GPUWorker
```

## Lifecycle (SequenceGroup → Sequence)

```
  ┌──────────────────┐
  │ SequenceGroup    │  ← LLMEngine.add_request() creates a group
  │ (WAITING)        │     with SamplingParams + prompt tokens
  └───────┬──────────┘
          │ scheduler admits → creates Sequence, allocates KV blocks
          ▼
  ┌──────────────────┐
  │ Sequence         │
  │ (PREFILL)        │  ← executor.prefill(): writes all prompt tokens
  │                  │     to fake KV cache, generates first output token
  └───────┬──────────┘
          │
          ▼
  ┌──────────────────┐   ┌────────────┐
  │ Sequence         │ ←─│ each step: │  ← executor.decode(): reads KV
  │ (RUNNING)        │   │ decode()   │     cache to compute next token,
  └───────┬──────────┘   └────────────┘     writes new token to KV
          │  (loops until num_generated_tokens == max_tokens)
          ▼
  ┌──────────────────┐
  │ Sequence         │  ← KV blocks freed, output ready
  │ (FINISHED)       │
  └──────────────────┘

  (the SequenceGroup can be REJECTED if KV cache is full,
   before any Sequence is created)
```

## Scheduler Design

The scheduler runs once per engine step and produces a ``ScheduleResult``::

1. **Finish** — iterate running groups (SequenceGroup objects).  For each
   group, check its sequences: any that hit ``max_tokens`` are marked
   FINISHED and their KV blocks are freed.  If every sequence in a group
   is done, the group moves to the finished pool.
2. **Decode** — groups still running become the decode batch.
3. **Admit** — iterate waiting groups.  For each that fits within
   ``max_num_seqs`` and ``max_num_batched_tokens``, create a Sequence,
   allocate KV blocks, mark as PREFILL, move group to running.

**ScheduleResult** mirrors vLLM's ``SchedulerOutputs``::

    ScheduleResult(
        scheduled_prefill_groups=[],   # SequenceGroup objects admitted this step
        scheduled_decode_groups=[],    # SequenceGroup objects continuing decode
        ignored_groups=[],             # groups that couldn't fit this step
        finished_groups=[],            # groups that fully completed this step
        rejected_groups=[],            # groups dropped (OOM / never-fit)
        num_batched_tokens=N,
        num_prefill_tokens=N,          # prompt tokens processed
        num_decode_tokens=N,           # == |decode_groups|
        reason="prefill: r0 | decode: r1",
    )

## KV Cache Architecture (Three-Layer)

### Layer 1: BlockAllocator

The lowest layer manages a fixed-size pool of physical block IDs.

```python
class BlockAllocator:
    def allocate(num_blocks) -> List[int] | None  # first-fit free-list
    def free(physical_block_ids) -> None          # reclaim blocks
    stats() -> dict                               # used/free/total
```

Registering optional callbacks (``on_allocate`` / ``on_free``) lets higher
layers react to block events — e.g. the executor allocates fake KV storage
when a new block is handed out.

### Layer 2: BlockManager

The coordination layer sits between the scheduler and the allocator.

```python
class BlockManager:
    def allocate_for_seq(seq, num_blocks) -> BlockTable | None
    def free(seq_id) -> None
    def append_token(seq, token_position) -> bool  # on-demand alloc (future)
```

The scheduler calls ``block_manager.allocate_for_seq()`` — not the allocator
directly.  This mirrors vLLM where ``BlockSpaceManager`` coordinates between
the ``Scheduler`` and ``BlockAllocator``.

### Layer 3: BlockTable

Each sequence has one ``BlockTable`` mapping logical block indices (derived
from token position) to the physical block IDs allocated by the manager.

```python
class BlockTable:
    def get_physical_block(token_position) -> int | None
    def get_block_ids() -> List[int]
```

### Allocation Strategy

The current strategy is **eager** — all blocks a request will ever need are
reserved at admission time:

```
blocks_needed = ceil((prompt_length + max_new_tokens) / block_size)
```

This is wasteful (reserves blocks for tokens that may never be generated)
but guarantees no request ever OOMs mid-decode.  Future phases may switch
to **on-demand** allocation (one block during prefill, one more each time
the current block fills up).

## Fake KV Cache (Executor Layer)

The ``FakeModelExecutor`` maintains a simulated device-side KV cache::

    _kv_cache: Dict[physical_block_id, List[int]]

Each entry stores fake key/value pairs for every token slot in that physical
block.  The cache is populated by the ``BlockAllocator`` callbacks:

- ``_prepare_block(block_id)`` → creates an empty list for the new block
- ``_release_block(block_id)`` → removes the block's data

During inference:

- **Prefill**: for each prompt token, the executor looks up the physical
  block from ``seq.block_table`` and appends a fake key/value to that
  block's KV cache entry.  The first output token is then sampled
  deterministically.
- **Decode**: the next token depends on *both* the last output token AND
  the current KV cache content (simulating attention)::

      kv_bias = sum(kv_data_at_current_position) % vocab_size
      next_token = (prev_token + 7 + kv_bias) % vocab_size

This makes output genuinely depend on what was stored in the KV cache —
replacing the old arithmetic-only fake that had no memory state.

## Engine Architecture (Two-Layer)

### Layer 1: EngineCore

The inner step loop holds the scheduler and executor:

```python
class EngineCore:
    def step(self) -> ScheduleResult:
        result = self._scheduler.schedule()
        self._executor.prefill(prefill_seqs_from(result))
        self._executor.decode(decode_seqs_from(result))
        return result
```

### Layer 2: LLMEngine

The public-facing API holds all components and provides the user interface:

```python
class LLMEngine:
    add_request(prompt, max_new_tokens) -> request_id
    step() -> ScheduleResult
    run_until_done() -> Dict[request_id, output_text]
    get_outputs() -> Dict[request_id, output_text]
```

``LLMEngine`` is responsible for wiring components together (especially
the ``BlockAllocator`` callbacks to the executor), capturing finished
outputs, and printing step summaries.

## Engine Loop

```
while requests remain:
    1. result = engine_core.step()                # schedule + execute
       │
       ├── scheduler.schedule() → ScheduleResult
       ├── executor.prefill(prefill_seqs)          # writes prompt to KV
       └── executor.decode(decode_seqs)            # reads KV, writes new token
    2. llm_engine captures outputs from result.finished_groups
    3. print step summary (groups, cache stats, KV token count)
```
