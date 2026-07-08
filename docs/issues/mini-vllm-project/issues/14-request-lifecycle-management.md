---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 14: Request Lifecycle Management (Cancel, Timeout, Client Disconnect)
# Issue 14：请求生命周期管理（取消、超时、客户端断连）

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

Complete request lifecycle management: request cancellation (manual cancel via API), timeout enforcement (auto-cancel after `request_timeout_s`), and HTTP client disconnect detection (clean up orphaned engine resources when a client drops the connection).

完整的请求生命周期管理：请求取消（通过 API 手动取消）、超时强制终止（`request_timeout_s` 后自动取消）、HTTP 客户端断连检测（客户端断开连接时清理孤儿引擎资源）。

## Vertical slice / 垂直切片描述

This slice extends both the core engine and the serving layer with lifecycle management. The core engine handles cancel/timeout logic (sequence state management, block freeing, executor cleanup, metrics update). The serving layer adds client disconnect detection atop the engine's cancel mechanism.

本切片为核心引擎和服务层都增加了生命周期管理。核心引擎处理取消/超时逻辑（序列状态管理、块释放、执行器清理、指标更新）。服务层在引擎的取消机制之上增加了客户端断连检测。

## Acceptance criteria / 验收标准

- [x] `LLMEngine.cancel_request(request_id)` cancels a running or waiting request by ID
- [x] Cancel sets sequence status to `CANCELLED`, records `finish_time`, frees blocks, cleans up executor
- [x] Cancel removes request from waiting/running pool, moves to finished pool
- [x] Cancel returns `False` for non-existent requests without error
- [x] `MetricsCollector.count_cancelled()` tracks cancelled requests (excluded from throughput)
- [x] `EngineCore._check_timeouts()` scans running and waiting pools for expired requests
- [x] Timeout sets sequence status to `TIMEOUT`, frees blocks, cleans up executor
- [x] `MetricsCollector.count_timeout()` tracks timed-out requests (excluded from throughput)
- [x] Timeout works for both scheduled (has sequences) and unscheduled (waiting only) requests
- [x] Client disconnect detection: serving layer detects dropped HTTP connections and calls `cancel_request()`
- [x] Disconnect cleanup: cancelled requests have their resources freed, running queue slots released
- [x] Disconnect does not crash the engine or profiler
- [x] Timeout releases blocks back to the allocator pool — subsequent requests can use them
- [x] Timeout metrics correctly reflect the timeout counter

## Key code / 核心代码

- `mini_vllm/engine/engine_core.py` — `cancel_request()`, `_check_timeouts()`
- `mini_vllm/engine/engine.py` — `LLMEngine.cancel_request()`
- `mini_vllm/serving/lifecycle.py` — Client disconnect lifecycle handler
- `mini_vllm/serving/server.py` — HTTP disconnect detection integration

## Key tests / 核心测试

- `tests/test_serving_layer.py` — cancel running request, cancel non-existent, timeout cancels request, streaming disconnect abort, running queue cleared after disconnect
- `tests/test_fault_injection.py` — timeout releases blocks, timeout metrics, cancel releases blocks, cancel ref_count integrity
- `tests/test_engine.py` — (implicitly tested via engine lifecycle)
- `tests/test_stage_profiler.py` — `test_cancel_does_not_crash_profiler()`

## Blocked by / 前置依赖

- [05: Engine Loop & Public API](./05-engine-loop-public-api.md) — cancel/timeout are EngineCore features
- [12: HTTP Serving with SSE](./12-http-serving-sse.md) — disconnect detection extends the serving layer
