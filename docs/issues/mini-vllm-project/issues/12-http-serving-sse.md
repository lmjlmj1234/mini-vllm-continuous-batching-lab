---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 12: HTTP Serving with SSE Streaming
# Issue 12：HTTP 服务与 SSE 流式输出

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

The HTTP serving layer: a standalone HTTP server that drives the core engine via the `LLMEngine` public API. Provides SSE (Server-Sent Events) streaming for real-time token-by-token output, non-streaming generation, and metrics endpoints.

HTTP 服务层：一个独立的 HTTP 服务器，通过 `LLMEngine` 公共 API 驱动核心引擎。提供 SSE（Server-Sent Events）流式输出（逐 token 实时推送）、非流式生成和指标端点。

## Vertical slice / 垂直切片描述

This slice provides a complete production-style HTTP interface to the engine. Clients can send requests via HTTP POST, receive tokens incrementally via SSE, query runtime metrics, and cancel in-flight requests. The serving layer is fully independent — no core engine modifications are needed.

本切片提供完整的生产风格 HTTP 接口给引擎。客户端可以通过 HTTP POST 发送请求、通过 SSE 增量接收 token、查询运行时指标和取消正在进行的请求。服务层完全独立——无需修改核心引擎。

## Acceptance criteria / 验收标准

- [x] Non-streaming generation endpoint: `POST /generate` — returns full output as JSON
- [x] SSE streaming endpoint: sends each token as a separate SSE event
- [x] Streaming token accumulation: tokens are buffered and sent incrementally to the client
- [x] `max_num_streams` cap — limits concurrent active SSE streams
- [x] Stream manager tracks active stream count per engine step
- [x] Empty prompt rejection
- [x] `max_tokens=0` handling
- [x] Metrics endpoint returns engine metrics as JSON
- [x] Metrics endpoint correctly reports rejected/cancelled/timeout counts
- [x] HTTP server integration with `LLMEngine` step loop — serving drives the loop, not the other way around

## Key code / 核心代码

- `mini_vllm/serving/server.py` — HTTP server, SSE handler

## Key tests / 核心测试

- `tests/test_serving_layer.py` — non-stream generation, empty prompt rejection, streaming basic, streaming token accumulation, metrics endpoint, metrics after rejection/cancel, streaming disconnect abort, running queue cleared after disconnect

## Blocked by / 前置依赖

- [05: Engine Loop & Public API](./05-engine-loop-public-api.md) — serving layer drives LLMEngine
