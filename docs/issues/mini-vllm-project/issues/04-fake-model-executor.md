---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 04: Fake Model & Executor
# Issue 04：假模型与执行器

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

The pure-Python model simulation layer: `FakeModel` (mathematical functions for keys, values, and logits), `FakeModelExecutor` (implements the `Executor` protocol with in-memory KV cache), and the `Executor` abstract protocol definition.

纯 Python 模型模拟层：`FakeModel`（key、value 和 logits 的数学函数）、`FakeModelExecutor`（使用内存 KV cache 实现 `Executor` 协议）、以及 `Executor` 抽象协议定义。

## Vertical slice / 垂直切片描述

This slice provides a complete model execution pipeline that requires zero external dependencies. The executor tokenizes text via ASCII mapping, writes simulated KV data during prefill, reads KV during decode to influence the next token, and detokenizes output back to text. KV cache content genuinely affects output — not random.

本切片提供完整的模型执行管道，零外部依赖。执行器通过 ASCII 映射做 tokenize，在 prefill 期间写入模拟 KV 数据，在 decode 期间读取 KV 影响下一个 token，并将输出 detokenize 回文本。KV cache 内容真实影响输出——不是随机的。

## Acceptance criteria / 验收标准

- [x] `Executor` protocol defines: `tokenize()`, `detokenize()`, `prefill()`, `decode()`, `cleanup_sequence()`, `get_kv_stats()`, `prepare_block()`, `release_block()`, `make_block_allocator_callbacks()`
- [x] `Executor` is `@runtime_checkable` — supports `isinstance()` checks at runtime
- [x] `FakeModel` provides deterministic `_fake_key(token_id)`, `_fake_value(token_id)`, `prefill_token(last_token)`, `decode_token(prev_token, kv_bias)` using pure arithmetic
- [x] `FakeModelExecutor` maintains `_kv_cache: Dict[int, List[int]]` — simulated device-side KV storage keyed by physical block ID
- [x] Block allocator callbacks (`prepare_block`, `release_block`) keep KV storage in sync with physical block lifecycle
- [x] `prefill(sequences)`: chunk-aware, writes prompt tokens to KV from cursor position, produces first output token via `FakeModel.prefill_token()` when prefill completes
- [x] `decode(sequences)`: reads KV bias via `_read_from_kv()`, generates next token via `FakeModel.decode_token()`, writes new token's KV data
- [x] `tokenize(prompt)`: ASCII-to-token-ID mapping (`ord(c) % vocab_size`)
- [x] `detokenize(token_ids)`: token-ID-to-text mapping (`chr(t % 95 + 32)`)
- [x] KV stats: `get_kv_stats()` reports `kv_tokens_written`, `kv_slot_capacity`, `allocated_blocks`
- [x] On-demand block integration: calls `BlockManager.ensure_block()` during prefill/decode writes; skips KV write for shared prefix blocks

## Key code / 核心代码

- `mini_vllm/executor/base.py` — Executor protocol
- `mini_vllm/executor/executor.py` — FakeModelExecutor
- `mini_vllm/model/fake_model.py` — FakeModel
- `mini_vllm/worker/fake_worker.py` — FakeWorker factory

## Key tests / 核心测试

- `tests/test_engine.py` — indirectly tested through engine integration (KV writes tracked, KV output dependence verified)

## Blocked by / 前置依赖

- [01: Sequence & Request Data Model](./01-sequence-data-model.md) — executor operates on `Sequence` objects
- [02: Paged KV Cache](./02-paged-kv-cache.md) — executor calls `BlockManager.ensure_block()`, responds to allocator callbacks
