#!/usr/bin/env bash
# A/B Experiment - 20260715_163355_6c57ba8
# Command: /mnt/e/mini-vllm-continuous-batching-lab/benchmarks/continuous_batching.py --ab-test --concurrency 2 4 8 --requests 16 --repeats 3 --output-dir benchmark_results
# Timestamp: 2026-07-15T16:43:55
set -x
/mnt/e/mini-vllm-continuous-batching-lab/benchmarks/continuous_batching.py --ab-test --concurrency 2 4 8 --requests 16 --repeats 3 --output-dir benchmark_results
