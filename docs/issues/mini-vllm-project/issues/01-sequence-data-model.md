---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 01: Sequence & Request Data Model
# Issue 01：序列与请求数据模型

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

The foundational data model layer: the request lifecycle state machine (`Status` enum), generation configuration (`SamplingParams`), per-generation token buffers and timestamps (`Sequence`), user-level request ownership (`SequenceGroup`), and the four-pool queue (`RequestQueue`).

基础数据模型层：请求生命周期状态机（`Status` 枚举）、生成配置（`SamplingParams`）、每条生成序列的 token 缓冲区和时间戳（`Sequence`）、用户级请求所有权（`SequenceGroup`），以及四池队列（`RequestQueue`）。

## Vertical slice / 垂直切片描述

This slice provides the complete type system that every other module depends on. A request enters as a `SequenceGroup`, spawns one or more `Sequence` objects, transitions through `WAITING → PREFILL → RUNNING → FINISHED/REJECTED/CANCELLED/TIMEOUT`, and is tracked in the appropriate queue pool.

本切片提供了所有其他模块依赖的完整类型系统。请求以 `SequenceGroup` 进入，创建一个或多个 `Sequence` 对象，经过 `WAITING → PREFILL → RUNNING → FINISHED/REJECTED/CANCELLED/TIMEOUT` 状态转换，并在相应的队列池中追踪。

## Acceptance criteria / 验收标准

- [x] `Status` enum defines `WAITING`, `PREFILL`, `RUNNING`, `FINISHED`, `REJECTED`, `CANCELLED`, `TIMEOUT` — each a distinct state
- [x] `SamplingParams` dataclass configures `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_token_ids`, `stop_strings` with safe default values (no mutable default sharing)
- [x] `Sequence` holds per-generation state: `seq_id`, `group_id`, `prompt_token_ids`, `output_token_ids`, `status`, `block_table`, timing fields (`arrival_time`, `first_token_time`, `first_scheduled_time`, `finish_time`), `prefill_cursor`, `num_generated_tokens`
- [x] `Sequence.finished` property returns `True` for any terminal status
- [x] `Sequence.is_prefill_finished` property checks `prefill_cursor ≥ prompt_length`
- [x] `Sequence._set_group()` establishes bidrectional link with `SequenceGroup`
- [x] `SequenceGroup` creates `Sequence` objects via `create_sequence()`, tracks `num_sequences`, `num_finished`, `is_finished`, provides `get_unfinished_seqs()`
- [x] `RequestQueue` manages four pools (`waiting`, `running`, `finished`, `rejected`), all storing `SequenceGroup` objects
- [x] `RequestQueue` provides `add()`, `mark_running()`, `mark_finished()`, `mark_rejected()`, and `get_by_id()` across all pools
- [x] `Sequence.to_dict()` serializes core state for logging/debugging

## Key code / 核心代码

- `mini_vllm/sequence/status.py` — Status enum
- `mini_vllm/sequence/sampling_params.py` — SamplingParams dataclass
- `mini_vllm/sequence/sequence.py` — Sequence class
- `mini_vllm/sequence/sequence_group.py` — SequenceGroup + RequestQueue

## Key tests / 核心测试

- `tests/test_request.py` — validates `_make_seq()` helper and foundational Sequence creation (tests focused on data model construction)

## Blocked by / 前置依赖

None — can start immediately. Foundational layer, all other slices depend on this.

无——可立即开始。基础层，所有其他切片依赖于此。
