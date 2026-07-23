#!/usr/bin/env python3
"""Stage Breakdown Profiling Demo.

Demonstrates how to decompose end-to-end LLM serving request latency
into individual stages: queue waiting, scheduler, KV allocation, prefill,
decode, executor forward, KV release, metrics update.

Usage::

    # Fake executor (fast, no GPU needed)
    python examples/demo_stage_breakdown.py --executor fake --requests 16 --tokens 16

    # Qwen executor (requires torch + transformers)
    python examples/demo_stage_breakdown.py --executor qwen --requests 4 --tokens 16
"""

import argparse
import os
import sys
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mini_vllm import Config, LLMEngine


# Sample prompts
_PROMPTS = [
    "Hello, world!",
    "What is the capital of France?",
    "Write a short poem about artificial intelligence.",
    "Explain the concept of attention in transformer models.",
    "The quick brown fox jumps over the lazy dog. " * 4,
    "Machine learning is a subset of artificial intelligence "
    "that involves the use of statistical techniques.",
    "Once upon a time in a land far far away there lived a "
    "brave knight who fought dragons and rescued princesses.",
    "In computer science, a data structure is a data organization, "
    "management, and storage format that enables efficient access "
    "and modification of data.",
    "Python is a high-level, general-purpose programming language. "
    "Its design philosophy emphasizes code readability with the "
    "use of significant indentation via the off-side rule.",
    "Transformers are a type of neural network architecture that "
    "have become the foundation for many state-of-the-art NLP models. "
    "They use self-attention mechanisms to process sequential data.",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mini-vLLM Stage Breakdown Profiling Demo"
    )
    parser.add_argument(
        "--executor",
        choices=["fake", "qwen"],
        default="fake",
        help="Executor type (default: fake)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=16,
        help="Number of requests to run (default: 16)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=16,
        help="Max new tokens per request (default: 16)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    prompt_estimate = sum(
        len(p) + args.tokens for p in _PROMPTS[: max(args.requests, len(_PROMPTS))]
    )
    num_blocks = max(16, prompt_estimate // 4 * 2)

    config = Config(
        executor_type=args.executor,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_num_prefill_tokens=16,
        max_prefill_chunk_size=4,
        block_size=4,
        num_gpu_blocks=num_blocks,
        print_step_events=False,
        memory_trace=False,
    )

    engine = LLMEngine(config)

    # Print experiment configuration
    print("=" * 70)
    print("Stage Breakdown Profiling")
    print("=" * 70)
    print()
    print("  Experiment Configuration")
    print("  " + "-" * 50)
    print(f"  executor:                {args.executor}")
    print(f"  num_requests:            {args.requests}")
    print(f"  max_new_tokens:          {args.tokens}")
    print(f"  block_size:              {config.block_size}")
    print(f"  max_num_seqs:            {config.max_num_seqs}")
    print(f"  max_num_batched_tokens:  {config.max_num_batched_tokens}")
    print(f"  max_prefill_chunk_size:  {config.max_prefill_chunk_size}")
    print(f"  chunked_prefill:         {config.chunked_prefill_enabled}")
    print(f"  num_gpu_blocks:          {config.num_gpu_blocks}")

    # ------------------------------------------------------------------
    # Add requests
    # ------------------------------------------------------------------
    print()
    for i in range(args.requests):
        prompt = _PROMPTS[i % len(_PROMPTS)]
        rid = engine.add_request(prompt, max_new_tokens=args.tokens)
        print(f"  Added request {rid}: prompt_len={len(prompt)}")

    # ------------------------------------------------------------------
    # Run to completion
    # ------------------------------------------------------------------
    print(f"\n  Running {args.requests} requests to completion...")
    t0 = time.time()
    outputs = engine.run_until_done()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.3f}s\n")

    # Print outputs
    print("  " + "-" * 50)
    print("  Generated Outputs")
    print("  " + "-" * 50)
    for rid, text in outputs.items():
        print(f"    {rid}: {text!r}")

    # ------------------------------------------------------------------
    # Stage Breakdown Report
    # ------------------------------------------------------------------
    print()
    engine.profiler.print_report()

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print("\n  Interpretation")
    print("  " + "-" * 50)

    if args.executor == "fake":
        print("  This run used the FAKE executor (pure CPU arithmetic).")
        print("  All timing values reflect Python-level overhead, not real inference.")
        print("  In a real LLM serving setup, executor_forward would be dominated")
        print("  by GPU kernel execution time (attention, FFN, sampling).")
        print()
        print("  What this run demonstrates:")
        print("  - Whether scheduler overhead is negligible vs. executor time")
        print("  - Whether KV cache allocation adds noticeable latency")
        print("  - Whether prefix cache lookups are fast enough")
        print("  - How many engine steps are needed for the workload")
    else:
        print("  This run used the QWEN executor (real Qwen2-0.5B model).")
        device = "GPU" if hasattr(engine.executor, "_model") else "CPU"
        print(f"  Model runs on {device}.")
        print("  executor_forward now includes real model inference time.")
        print("  This gives a meaningful breakdown of scheduler vs. execution.")
        print()
        print("  Note: prefix_cache_lookup and kv_cache_allocation are Python-level")
        print("  overheads. The model's GPU kernel times are included in prefill/decode.")

    print()


if __name__ == "__main__":
    main()
