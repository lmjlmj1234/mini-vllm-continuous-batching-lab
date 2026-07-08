---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 06: Metrics Collection & Reporting
# Issue 06：指标采集与报告

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

The centralized `MetricsCollector` that tracks performance metrics across an engine run: TTFT (Time to First Token), TPOT (Time Per Output Token), throughput (req/s and tok/s for both wall-clock and active time), KV block utilization, block utilization (tokens per block), scheduler latency, prefix cache hit rate, and serving-layer counters (rejected/cancelled/timeout/rpm-rejected/tpm-rejected).

集中式 `MetricsCollector`，跟踪引擎运行的各项性能指标：TTFT（首 token 延迟）、TPOT（每输出 token 时间）、吞吐量（挂钟时间和活跃时间的 req/s 和 tok/s）、KV 块利用率、块利用率（每块 token 数）、调度器延迟、前缀缓存命中率和服务层计数器（rejected/cancelled/timeout/rpm-rejected/tpm-rejected）。

## Vertical slice / 垂直切片描述

This slice provides a complete metrics pipeline. Metrics are recorded per-step via `record_step()` and per-sequence via `register_sequence()`. The `report()` method computes all aggregates and returns a structured dict. `print_report()` formats the output for human reading.

本切片提供完整的指标管道。指标通过 `record_step()` 按步骤记录，通过 `register_sequence()` 按序列记录。`report()` 方法计算所有汇总值并返回结构化字典。`print_report()` 将输出格式化为人类可读格式。

## Acceptance criteria / 验收标准

- [x] TTFT calculation: `first_token_time − arrival_time`, reported as avg/min/max in ms
- [x] TPOT calculation: `(finish_time − first_token_time) / max(num_output_tokens − 1, 1)`, excludes single-token outputs
- [x] Throughput (wall-clock): `completed_requests / (max(finish_times) − min(arrival_times))` and `total_output_tokens / elapsed` — captures workload-level throughput including idle gaps
- [x] Throughput (active): `completed_requests / sum(step_times)` and `total_output_tokens / active_time` — system-level throughput excluding idle gaps
- [x] KV block utilization: peak and average percentage of total blocks in use
- [x] Block utilization: average tokens per allocated block across all finished sequences
- [x] Prefix cache hit rate: cached tokens as a percentage of total prompt tokens
- [x] Serving-layer counters: `total_requests`, `rejected_requests`, `cancelled_requests`, `timeout_requests`, `rpm_rejected`, `tpm_rejected`
- [x] Scheduler latency: avg and max per-step scheduler overhead in ms
- [x] Only FINISHED sequences counted in throughput (cancelled/timeout excluded)
- [x] `report()` returns a structured dict; `print_report()` formats for console

## Key code / 核心代码

- `mini_vllm/engine/metrics.py` — MetricsCollector

## Key tests / 核心测试

- `tests/test_metrics.py` — 29 tests covering: TTFT existence/ordering/formula, TPOT existence/denominator/single-token-edge-case/ordering, throughput fields/formula/staggered-arrival, cancelled/timeout exclusion, KV utilization range, scheduler latency validation, prefix cache fields, hit rate computation

## Blocked by / 前置依赖

- [05: Engine Loop & Public API](./05-engine-loop-public-api.md) — metrics collector is called by EngineCore and LLMEngine
