---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 13: Rate Limiting & Admission Control
# Issue 13：速率限制与准入控制

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

Rate limiting (RPM and TPM) and admission control for the serving layer. Protects the engine from overload by rejecting requests that exceed configurable per-minute limits on requests and tokens, or that overflow the engine's queue.

服务层的速率限制（RPM 和 TPM）和准入控制。通过拒绝超过可配置的每分钟请求数或 token 数限制的请求，以及防止引擎队列溢出，保护引擎免受过载。

## Vertical slice / 垂直切片描述

This slice extends the serving layer with two admission control mechanisms: rate-based (time-window RPM/TPM) and capacity-based (max queue depth). Rejected requests receive immediate HTTP error responses rather than consuming engine resources.

本切片扩展了服务层，增加了两种准入控制机制：基于速率（时间窗口 RPM/TPM）和基于容量（最大队列深度）。被拒绝的请求立即收到 HTTP 错误响应，而不是消耗引擎资源。

## Acceptance criteria / 验收标准

- [x] RPM limiting: requests exceeding `Config.rate_limit_rpm` within a 60-second sliding window are rejected
- [x] TPM limiting: requests whose prompt + generated tokens would exceed `Config.rate_limit_tpm` are rejected
- [x] RPM limit resets after the time window expires — subsequent requests are accepted
- [x] Queue overflow: requests exceeding `Config.max_queue_len` waiting requests are rejected
- [x] Prompt-too-long detection: requests exceeding engine capacity are rejected with appropriate error
- [x] Block exhaustion: when all KV cache blocks are in use, new requests are rejected (not crashed)
- [x] Blocks recovered after finish: completed requests release blocks, new requests can be admitted
- [x] Stream exhaustion: when `max_num_streams` is reached, new streaming requests are rejected
- [x] Stream release recovers slot: completed/cancelled streams free their slot

## Key code / 核心代码

- `mini_vllm/serving/rate_limiter.py` — RPM/TPM rate limiter
- `mini_vllm/serving/server.py` — admission control integration
- `mini_vllm/config.py` — rate limit configuration params

## Key tests / 核心测试

- `tests/test_serving_layer.py` — RPM rejection, RPM reset after window, prompt too long, queue overflow, block exhaustion, max streams cap, stream manager counts
- `tests/test_fault_injection.py` — queue overflow reject/accept after drain, block exhaustion reject/no crash/block recovery, stream exhaustion reject/slot recovery

## Blocked by / 前置依赖

- [12: HTTP Serving with SSE](./12-http-serving-sse.md) — rate limiting extends the serving layer
