---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 15: Fault Injection & Recovery Test Suite
# Issue 15：故障注入与恢复测试套件

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

A systematic fault injection test suite (`test_fault_injection.py`) that exercises every resource-exhaustion and lifecycle scenario to verify that the engine recovers cleanly. Tests cover queue overflow, block exhaustion, stream exhaustion, timeout, cancel, stale prefix cache entries, and admission-control-vs-block-count edge cases.

一个系统性的故障注入测试套件（`test_fault_injection.py`），测试每种资源耗尽和生命周期场景，以验证引擎能干净地恢复。测试覆盖队列溢出、块耗尽、流耗尽、超时、取消、过期前缀缓存条目和准入控制与块数量的边界情况。

## Vertical slice / 垂直切片描述

This slice provides comprehensive fault injection coverage for the serving layer + engine. Each test provokes a specific failure mode, then verifies that (a) the engine does not crash, (b) resources are recovered cleanly, and (c) subsequent requests succeed.

本切片为服务层和引擎提供全面的故障注入覆盖。每个测试触发一个特定的故障模式，然后验证 (a) 引擎不会崩溃，(b) 资源被干净地恢复，(c) 后续请求可以成功。

## Acceptance criteria / 验收标准

- [x] Queue overflow: rejects new request when waiting queue is full; accepts again after queue drains
- [x] Block exhaustion: rejects new request when all GPU blocks are used; engine does not crash
- [x] Block recovery: completed requests release blocks, making room for new requests
- [x] Stream exhaustion: rejects new streaming request when max streams reached
- [x] Stream release: completed/cancelled stream recovers the slot for subsequent requests
- [x] Timeout releases blocks: timed-out requests free their blocks back to the allocator
- [x] Timeout metrics updated: timeout counter reflects the number of timed-out requests
- [x] Cancel releases blocks: cancelled requests free their blocks
- [x] Cancel ref_count integrity: block reference counts are correct after cancel (no leaks, no double-free)
- [x] Stale prefix cache entry: a freed block's hash entry is not falsely returned by probe
- [x] Stale entry new request recreates: after a stale cache miss, the new request correctly allocates a fresh block
- [x] Resource exhaustion under concurrency: 10-block pool with 100 long requests — all rejected (no crash)
- [x] Mixed workload with limited blocks: 10-block pool with short requests — first batch uses blocks, then admission blocks, then blocks freed → new admitted
- [x] Crash safety without admission control: no admission limits × limited blocks — verifies engine handles OOM gracefully without crashing

## Key code / 核心代码

- `tests/test_fault_injection.py` — 15 systematic fault injection tests
- `mini_vllm/serving/server.py` — serving layer resource management
- `mini_vllm/engine/engine_core.py` — cancel/timeout resource cleanup
- `mini_vllm/cache/allocator.py` — ref_count integrity
- `mini_vllm/cache/prefix_cache.py` — stale entry handling

## Key tests / 核心测试

- `tests/test_fault_injection.py` — the entire file is the deliverable

## Blocked by / 前置依赖

- [12: HTTP Serving with SSE](./12-http-serving-sse.md) — tests drive the serving layer
- [13: Rate Limiting & Admission Control](./13-rate-limiting-admission-control.md) — tests exercise admission limits
- [14: Request Lifecycle Management](./14-request-lifecycle-management.md) — tests exercise cancel/timeout/disconnect
