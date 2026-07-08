#!/usr/bin/env python3
"""mini-vLLM Benchmark — runs requests and prints a metrics report.

Usage::

    # Fake executor (fast, no GPU needed)
    python examples/benchmark.py --executor fake --requests 4

    # Qwen executor (requires torch + transformers)
    python examples/benchmark.py --executor qwen --requests 2 --tokens 16
"""

import argparse
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mini_vllm import Config, LLMEngine


# Sample prompts of varying lengths
_PROMPTS = [
    "Hello, world!",
    "What is the capital of France?",
    "Write a short poem about artificial intelligence.",
    "Explain the concept of attention in transformer models in simple terms.",
    "The quick brown fox jumps over the lazy dog. " * 4,
    "Machine learning is a subset of artificial intelligence that involves "
    "the use of statistical techniques to enable machines to improve with experience. "
    "Deep learning is a further subset that uses neural networks with many layers.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-vLLM Benchmark")
    parser.add_argument(
        "--executor",
        choices=["fake", "qwen"],
        default="fake",
        help="Executor type (default: fake)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=4,
        help="Number of requests to run (default: 4)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=16,
        help="Max new tokens per request (default: 16)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step output",
    )
    args = parser.parse_args()

    print(f"mini-vLLM Benchmark")
    print(f"  executor:     {args.executor}")
    print(f"  requests:     {args.requests}")
    print(f"  max_tokens:   {args.tokens}")
    print()

    # Ensure enough blocks for all requests: each token uses 1 block slot,
    # and each block holds `block_size` tokens.  Estimate ~20 blocks/seq.
    prompt_estimate = sum(len(p) + args.tokens for p in _PROMPTS[:args.requests])
    num_blocks = max(16, prompt_estimate // 4 * 2)

    config = Config(
        executor_type=args.executor,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_num_prefill_tokens=16,
        max_prefill_chunk_size=4,
        block_size=4,
        num_gpu_blocks=num_blocks,
        print_step_events=not args.quiet,
        memory_trace=False,
    )

    engine = LLMEngine(config)

    for i in range(args.requests):
        prompt = _PROMPTS[i % len(_PROMPTS)]
        rid = engine.add_request(prompt, max_new_tokens=args.tokens)
        print(f"  Added request {rid}: prompt={prompt[:40]!r}...  (max_tokens={args.tokens})")

    print(f"\n  Running {args.requests} requests to completion...")
    outputs = engine.run_until_done()

    report = engine.engine_core.metrics_collector.report()
    print(f"  Done in {report['total_time_seconds']:.3f}s"
          f"  (active: {report['active_time_seconds']:.3f}s)\n")

    # Print outputs
    print("=" * 60)
    print("Outputs")
    print("=" * 60)
    for rid, text in outputs.items():
        print(f"  {rid}: {text!r}")

    # Print benchmark report
    engine.engine_core.metrics_collector.print_report()

    # --- Summary comparison with real vLLM ---
    print("Interpretation")
    print("=" * 60)
    report = engine.engine_core.metrics_collector.report()

    if args.executor == "fake":
        print(f"  Fake executor: all metrics are simulation values.")
        print(f"  TTFT={report['avg_ttft_ms']}ms, TPOT={report['avg_tpot_ms']}ms — "
              f"these are CPU arithmetic, not real inference.")
    else:
        p = next(engine.executor._model.parameters(), None)
        device = "GPU" if (p is not None and p.is_cuda) else "CPU"
        print(f"  Qwen2-0.5B real inference metrics.")
        print(f"  TTFT={report['avg_ttft_ms']}ms — time includes model forward pass.")
        print(f"  TPOT={report['avg_tpot_ms']}ms — per-token decode latency on "
              f"{device}.")

    print(f"  KV block utilisation: {report['kv_util_peak_pct']}% peak — "
          f"{'good' if report['kv_util_peak_pct'] > 50 else 'over-provisioned'}")
    print(f"  Scheduler latency: {report['avg_scheduler_latency_ms']}ms — "
          f"{'negligible' if report['avg_scheduler_latency_ms'] < 1 else 'check scheduler'}")

    if report.get("rejected", 0) > 0:
        print(f"  WARNING: {report['rejected']} requests were rejected "
              f"(prompt too long for token budget).")

    print()


if __name__ == "__main__":
    main()
