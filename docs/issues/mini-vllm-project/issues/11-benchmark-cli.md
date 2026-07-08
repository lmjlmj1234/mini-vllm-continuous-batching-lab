---
labels: ready-for-agent
Status: ready-for-agent
---

# Issue 11: Benchmark CLI
# Issue 11：基准测试 CLI

## Parent / 父级

[mini-vLLM Project PRD](../PRD.md)

## What was built / 构建内容

A CLI benchmark script (`examples/benchmark.py`) that runs N requests with configurable executor type, token count, and verbosity. Generates a complete metrics report (TTFT, TPOT, throughput, KV utilization) and provides interpretation context based on executor type.

一个 CLI 基准测试脚本（`examples/benchmark.py`），运行 N 个请求，可配置执行器类型、token 数量和输出详细程度。生成完整的指标报告（TTFT、TPOT、吞吐量、KV 利用率），并根据执行器类型提供解读上下文。

## Vertical slice / 垂直切片描述

This slice provides a complete benchmarking tool usable from the command line. It configures the engine, adds requests from sample prompts, drives the engine to completion, and prints a structured benchmark report.

本切片提供一个完整的基准测试工具，可从命令行使用。它配置引擎、从示例 prompt 添加请求、驱动引擎完成运行，并打印结构化的基准测试报告。

## Acceptance criteria / 验收标准

- [x] CLI arguments: `--executor` (fake/qwen), `--requests` (count), `--tokens` (max_new_tokens), `--quiet` (suppress step output)
- [x] Auto-calculates `num_gpu_blocks` based on prompt estimate to avoid OOM
- [x] Loads sample prompts of varying lengths — from "Hello, world!" to multi-sentence technical descriptions
- [x] Runs all requests through `LLMEngine.run_until_done()`
- [x] Prints individual request outputs
- [x] Generates full `MetricsCollector.report()` and `print_report()`
- [x] Prints interpretation summary: TTFT/TPOT meaning, KV utilisation commentary, scheduler latency assessment, OOM warnings
- [x] Works with both fake (zero deps) and Qwen (torch + transformers) executors

## Key code / 核心代码

- `examples/benchmark.py` — CLI benchmark script

## Key tests / 核心测试

- (Manually runnable — no dedicated automated tests for the script itself)

## Blocked by / 前置依赖

- [05: Engine Loop & Public API](./05-engine-loop-public-api.md) — uses LLMEngine
- [06: Metrics Collection](./06-metrics-collection.md) — uses MetricsCollector.report() and print_report()
