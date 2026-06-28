# vLLM Module Mapping

This document maps every module in `mini-vllm-continuous-batching-lab` to its
counterpart in the real [vLLM](https://github.com/vllm-project/vllm) codebase.

## Core Modules

| mini-vLLM | vLLM | Notes |
|-----------|------|-------|
| `LLMEngine` | [`LLMEngine`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py) | Public API — add_request, run_until_done, get_outputs. |
| `EngineCore` | [`EngineCore`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/engine_core.py) | Inner step loop — owns scheduler + executor, runs schedule→execute each step. |
| `Scheduler` | [`Scheduler`](https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py) | The scheduling policy. vLLM's is much more complex: supports preemption, swapping, chunked prefill, and various scheduling strategies. |
| `SequenceGroup` | [`SequenceGroup`](https://github.com/vllm-project/vllm/blob/main/vllm/core/sequence.py) | Owns the request metadata: prompt text, sampling params, arrival timestamp. All queue pools hold SequenceGroup objects. |
| `Sequence` | [`Sequence`](https://github.com/vllm-project/vllm/blob/main/vllm/core/sequence.py) | A single generation candidate. Owns token buffers, status, block table, timing. Created inside its parent SequenceGroup at admission time. |
| `SamplingParams` | [`SamplingParams`](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py) | Generation parameters (max_tokens, temperature, top_p, top_k, stop strings). Belongs to SequenceGroup and inherited by each child Sequence. |
| `ScheduleResult` | [`SchedulerOutputs`](https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py) | Returned by every scheduler step. Contains prefill/decode group lists, token counts, ignored/rejected groups, and a human-readable reason string. |
| `RequestQueue` | (implicit in scheduler) | vLLM stores waiting/running/swapped sequence groups in scheduler-internal dicts. We extract it to a separate class for clarity. All four pools (waiting/running/finished/rejected) store SequenceGroup objects, matching vLLM's organisation. |
| `BlockAllocator` | [`BlockAllocator`](https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py) | Low-level: manages free-list of physical block IDs. Supports allocate/free with callbacks (used by executor to allocate fake KV storage). |
| `BlockManager` | [`BlockSpaceManager`](https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py) | High-level: coordinates BlockAllocator for per-sequence allocation and free. The scheduler calls `block_manager.allocate_for_seq()`, not the allocator directly. |
| `BlockTable` | [`BlockTable`](https://github.com/vllm-project/vllm/blob/main/vllm/core/block_table.py) | Nearly identical concept — maps logical token positions to physical block IDs for PagedAttention. |
| `FakeModelExecutor` | [`ModelRunner`](https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py) / [`ModelExecutor`](https://github.com/vllm-project/vllm/blob/main/vllm/executor/interfaces.py) | Fake model with simulated KV cache and logits. Maintains a device-side `Dict[block_id, List[int]]` that grows/shrinks with block allocation/free. Prefill writes prompt to KV; decode reads KV to influence output. |
| `Config` | [`EngineConfig` / `SchedulerConfig` / `CacheConfig`](https://github.com/vllm-project/vllm/blob/main/vllm/config.py) | vLLM splits configuration by subsystem. We keep everything in one `Config` for simplicity. |
| `MetricsCollector` | `StatLogger` / prometheus metrics | vLLM logs TTFT, TPOT, throughput through a structured logger and prometheus. Ours is a simple dict-based collector. |

## Request Status Lifecycle (vLLM)

| vLLM `SequenceStatus` | mini-vLLM `Status` | Description |
|----------------------|-------------------|-------------|
| `WAITING` | `WAITING` | In the waiting queue, not yet scheduled |
| `RUNNING` | `PREFILL` → `RUNNING` | vLLM merges prefill into RUNNING; we separate them |
| `SWAPPED` | — | Not implemented (GPU → CPU block offloading) |
| `FINISHED_STOPPED` | `FINISHED` | Normal completion |
| `FINISHED_LENGTH_CAPPED` | `FINISHED` | Hit max_tokens |
| `FINISHED_ABORTED` | `REJECTED` | Dropped before/during processing |

## Scheduler Differences

| Feature | vLLM | mini-vLLM (Phase 1) |
|---------|------|---------------------|
| Scheduling granularity | Token-level | Token-level |
| Prefill / decode separation | Yes | Yes |
| Chunked prefill | Yes | No |
| Preemption | Yes (swap in/out) | No |
| Prefix caching | Yes | No |
| Priority scheduling | Yes | No (FIFO) |
| Max num sequences | `max_num_seqs` | `max_num_seqs` |
| Max batched tokens | `max_num_batched_tokens` | `max_num_batched_tokens` |
| ScheduleResult | `SchedulerOutputs` with `scheduled_seq_groups`, `prefill_groups`, `decode_groups`, `ignored_seq_groups`, token counts | Mirrors vLLM: `scheduled_prefill_groups`, `scheduled_decode_groups`, `ignored_groups`, `finished_groups`, `rejected_groups`, `num_batched/prefill/decode_tokens`, `reason` |
| Queue pools | Scheduler-internal per-group dicts | `RequestQueue` class, all pools store `SequenceGroup` |
| KV cache layers | `BlockAllocator` → `BlockSpaceManager` → `BlockTable` | Same three-layer split: `BlockAllocator` → `BlockManager` → `BlockTable` |
| Executor / engine split | `EngineCore` → `ModelExecutor` → `Worker` | `EngineCore` → `FakeModelExecutor` (no Worker layer yet) |

## Terminology

| vLLM term | mini-vLLM term | Meaning |
|-----------|---------------|---------|
| `SequenceGroup` | `SequenceGroup` | A group of sequences sharing the same prompt |
| `Sequence` | `Sequence` | A single generation candidate with its own token buffers |
| `SamplingParams` | `SamplingParams` | Generation parameters (max_tokens, temperature, top_p, …) |
| Block | Physical block | Fixed-size contiguous KV cache unit |
| Block table | Block table | Logical → physical block ID mapping |
| `num_scheduled_tokens` | `max_num_batched_tokens` | Token budget per step |
| GPU blocks | Physical blocks | Device memory for KV cache |
