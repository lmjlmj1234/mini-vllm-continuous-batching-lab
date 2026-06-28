# mini-vLLM Continuous Batching Lab

**A minimal, educational re-implementation of vLLM's Continuous Batching engine.**

## Why Continuous Batching?

Traditional LLM serving processes requests one at a time or in fixed-size
batches.  **Continuous Batching** (also called *in-flight* or *iterative-level*
batching) allows new requests to **preempt** the decode batch at every
iteration.  The result is dramatically higher GPU utilisation and lower latency
for interactive workloads.

vLLM popularised this technique with PagedAttention and a scheduler that
treats each *token generation step* as the scheduling granularity — rather
than waiting for an entire request to finish before scheduling the next one,
new requests merge into the batch on the very next step.

This project breaks down exactly how that works.

## Phase 1 — Project Skeleton & Fake Model Executor

Phase 1 implements the **core module boundaries** of vLLM's engine without any
GPU code.  Everything runs in pure Python.

### What's implemented

| Module | vLLM counterpart | Responsibility |
|--------|-----------------|----------------|
| `SequenceGroup` | `SequenceGroup` | User request metadata: prompt, sampling params, arrival time |
| `Sequence` | `Sequence` | Token buffers, status, KV block table, generation timing |
| `SamplingParams` | `SamplingParams` | Generation knobs (max_tokens, temperature, top_p, stop strings) |
| `RequestQueue` | (implicit in scheduler) | Four pools (waiting/running/finished/rejected), all store SequenceGroup |
| `BlockTable` | `BlockTable` | Logical → physical block mapping for PagedAttention |
| `BlockAllocator` | `BlockAllocator` | Low-level physical block free-list management |
| `BlockManager` | `BlockSpaceManager` | Coordinates BlockAllocator, manages per-sequence allocation & free |
| `Scheduler` / `ScheduleResult` | `Scheduler` / `SchedulerOutputs` | Admit waiting groups, build prefill+decode batches, rich result reporting |
| `FakeModelExecutor` | `ModelRunner` | Fake tokenisation, KV cache simulation, prefill & decode with fake logits |
| `LLMEngine` / `EngineCore` | `LLMEngine` / `EngineCore` | Public API (LLMEngine) and inner loop (EngineCore) |
| `MetricsCollector` | StatLogger / metrics | TTFT, TPOT, throughput stats |
| `Config` | `EngineConfig` | Single place for tuning knobs |

### Architecture layers

```
mini_vllm/
├── config.py                  # Engine-wide configuration
├── sequence/                  # Data model layer
│   ├── status.py              # Status enum
│   ├── sampling_params.py     # SamplingParams dataclass
│   ├── sequence.py            # Sequence (per-generation state)
│   ├── sequence_group.py      # SequenceGroup (user request → sequences)
│   └── request_queue.py       # 4-pool queue (all store SequenceGroup)
├── cache/                     # KV cache layer
│   ├── block_table.py         # Logical → physical mapping
│   ├── block_allocator.py     # Free-list allocator (low-level)
│   └── block_manager.py       # Alloc/free per seq, on-demand append (high-level)
├── scheduler/                 # Scheduling layer
│   └── scheduler.py           # Scheduler + ScheduleResult
├── executor/                  # Model execution layer
│   └── fake_executor.py       # Fake model with simulated KV cache & logits
├── engine/                    # Engine layer
│   ├── engine_core.py         # Inner step loop
│   ├── llm_engine.py          # Public API (add_request, run_until_done)
│   └── metrics.py             # Performance metrics
└── worker/                    # Future: GPUWorker package
    └── __init__.py
```

### Key design decisions

- **SequenceGroup / Sequence split** — mirrors vLLM.  The group owns sampling
  params and prompt metadata; each Sequence owns token buffers and KV block
  table.  Queue pools hold groups; `ScheduleResult` reports groups.

- **BlockAllocator → BlockManager → BlockTable** — three-layer KV cache.
  BlockAllocator manages the free-list; BlockManager coordinates per-sequence
  allocation; BlockTable provides logical-to-physical mapping.

- **Fake KV cache** — the executor maintains a simulated device-side KV cache
  (`Dict[int, List[int]]`).  Prefill writes prompt tokens to KV; decode reads
  from existing KV to influence the next token.  This makes the fake model
  output genuinely depend on KV cache content.

- **LLMEngine / EngineCore split** — EngineCore owns the scheduler+executor
  and runs the inner step loop.  LLMEngine provides the public API and
  handles output capture and logging.

### What's **not** in Phase 1

- Real GPU kernels (CUDA)
- PagedAttention CUDA kernel
- Preemption / swapping
- Prefix caching
- Chunked prefill
- Speculative decoding
- Multi-node support

## Quickstart

```bash
cd mini-vllm-continuous-batching-lab

# Phase 1 has zero external dependencies — just Python 3.10+
python examples/demo_fake_engine.py

# Run tests
pip install pytest
pytest -q
```

## Project Structure

```
mini-vllm-continuous-batching-lab/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── Phase1_Architecture.md      # Architecture deep-dive
│   ├── VLLM_Mapping.md             # Module-by-module mapping to vLLM
│   ├── Learning_Notes.md           # Design decisions & lessons learned
│   ├── Runtime_Timeline.md         # Engine step lifecycle, sequence state machine
│   ├── Scheduler.md                # 6-phase scheduling, chunked prefill, budget model
│   └── Memory_Manager.md           # 3-layer cache, on-demand allocation, eager comparison
├── mini_vllm/
│   ├── __init__.py
│   ├── config.py
│   ├── sequence/
│   │   ├── __init__.py
│   │   ├── status.py
│   │   ├── sampling_params.py
│   │   ├── sequence.py
│   │   ├── sequence_group.py
│   │   └── request_queue.py
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── block_table.py
│   │   ├── block_allocator.py
│   │   └── block_manager.py
│   ├── scheduler/
│   │   ├── __init__.py
│   │   └── scheduler.py
│   ├── executor/
│   │   ├── __init__.py
│   │   └── fake_executor.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── engine_core.py
│   │   ├── llm_engine.py
│   │   └── metrics.py
│   └── worker/
│       └── __init__.py
├── examples/
│   └── demo_fake_engine.py
└── tests/
    ├── test_request.py
    ├── test_kv_cache_manager.py
    ├── test_scheduler.py
    └── test_engine.py
```

## License

MIT
