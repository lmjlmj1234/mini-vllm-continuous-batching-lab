---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 09: Qwen2-0.5B Executor
# Issue 09：Qwen2-0.5B 执行器

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

A real model executor using HuggingFace Transformers' `Qwen/Qwen2-0.5B` for end-to-end inference. Implements the `Executor` protocol with proper tokenization, `past_key_values` management for continuous batching, real prefill and decode, per-sequence KV cache cleanup, and block-level KV storage tracking.

使用 HuggingFace Transformers 的 `Qwen/Qwen2-0.5B` 实现的真实模型执行器，用于端到端推理。实现了 `Executor` 协议，包含正确的 tokenization、面向连续批处理的 `past_key_values` 管理、真实的 prefill 和 decode、per-sequence KV cache 清理和块级 KV 存储追踪。

## Vertical slice / 垂直切片描述

This slice provides a complete real-model pipeline as a drop-in replacement for `FakeModelExecutor`. Controlled by `Config.executor_type = "qwen"`. No engine changes are needed — the Executor protocol abstracts the difference.

本切片提供完整的真实模型管道，作为 `FakeModelExecutor` 的即插即用替代品。通过 `Config.executor_type = "qwen"` 控制。无需更改引擎代码——Executor 协议抽象了差异。

## Acceptance criteria / 验收标准

- [x] `QwenExecutor` implements the `Executor` protocol (tokenize, prefill, decode, cleanup_sequence, KV callbacks)
- [x] Model and tokenizer are lazy-loaded inside `_get_model_and_tokenizer()` — importing `mini_vllm` without torch/transformers does not fail
- [x] Device auto-detection: CUDA if available, CPU fallback
- [x] `tokenize()` uses real Qwen tokenizer with `add_special_tokens=True`
- [x] `detokenize()` uses real Qwen tokenizer with `skip_special_tokens=True`
- [x] `prefill()` runs real model forward with chunked prompt tokens, saves `past_key_values`, samples first output token via argmax
- [x] `decode()` runs real model forward with one new token + existing `past_key_values`, appends to sequence
- [x] `cleanup_sequence()` removes per-sequence `past_key_values` from `_seq_kv` dict
- [x] Block allocator callbacks (`prepare_block`, `release_block`) track per-block KV positions
- [x] `get_kv_stats()` reports real KV token counts and block capacity
- [x] Attention mask is correctly built for chunked prefill with existing `past_key_values`
- [x] `_build_attention_mask()` handles both first-chunk (returns None → transformers creates causal mask) and subsequent-chunks (custom mask covering existing + new tokens)

## Key code / 核心代码

- `mini_vllm/executor/qwen_executor.py` — QwenExecutor
- `mini_vllm/worker/qwen_worker.py` — QwenWorker factory

## Key tests / 核心测试

- (No dedicated test file for QwenExecutor — tested manually via `examples/benchmark.py --executor qwen` and `examples/demo_stage_breakdown.py --executor qwen`)

## Blocked by / 前置依赖

- [04: Fake Model & Executor](./04-fake-model-executor.md) — QwenExecutor implements the same Executor protocol
- [02: Paged KV Cache](./02-paged-kv-cache.md) — QwenExecutor calls BlockManager callbacks and ensure_block()
