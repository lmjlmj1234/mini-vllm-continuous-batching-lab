#!/usr/bin/env python3
"""mini-vLLM Benchmark — runs requests and prints a metrics report.

Usage::

    # Default (fake executor, fast, no GPU)
    python examples/benchmark.py

    # Custom workload + scheduler + KV cache config
    python examples/benchmark.py --executor fake --requests 8 --tokens 16 \\
        --max-num-seqs 4 --num-gpu-blocks 128 --quiet

    # Real model (requires torch + transformers + optional GPU)
    python examples/benchmark.py --executor qwen --requests 2 --tokens 32

    # Save metrics as JSON
    python examples/benchmark.py --executor fake --requests 4 --json-output results.json
"""

import argparse
import json
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mini-vLLM Benchmark")
    # --- Workload ---
    parser.add_argument(
        "--executor",
        choices=["fake", "qwen"],
        default="fake",
        help="Executor type (default: fake)",
    )
    parser.add_argument(
        "--requests", type=int, default=4,
        help="Number of requests to run (default: 4)",
    )
    parser.add_argument(
        "--tokens", type=int, default=16,
        help="Max new tokens per request (default: 16)",
    )
    # --- Scheduler ---
    parser.add_argument(
        "--max-num-seqs", type=int, default=4,
        help="Max sequences processed in one step (default: 4)",
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=16,
        help="Max total tokens across all sequences per step (default: 16)",
    )
    parser.add_argument(
        "--max-num-prefill-tokens", type=int, default=16,
        help="Max prefill tokens per step (default: 16)",
    )
    parser.add_argument(
        "--max-prefill-chunk-size", type=int, default=4,
        help="Prompt tokens per prefill chunk (default: 4)",
    )
    parser.add_argument(
        "--decode-first", default=True, action=argparse.BooleanOptionalAction,
        help="Schedule running decode sequences before prefills (default: True)",
    )
    parser.add_argument(
        "--no-chunked-prefill",
        dest="chunked_prefill",
        action="store_false",
        help="Disable chunked prefill (split long prompts across steps)",
    )
    parser.set_defaults(chunked_prefill=True)
    # --- KV Cache ---
    parser.add_argument(
        "--block-size", type=int, default=4,
        help="Tokens per physical KV cache block (default: 4)",
    )
    parser.add_argument(
        "--num-gpu-blocks", type=int, default=None,
        help="Total physical KV cache blocks (auto-calculated if omitted)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=2048,
        help="Maximum prompt length the model can accept (default: 2048)",
    )
    # --- Engine ---
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress step-by-step output",
    )
    parser.add_argument(
        "--memory-trace", action="store_true",
        help="Print detailed BlockAllocator free list at each step",
    )
    parser.add_argument(
        "--json-output", type=str, default=None,
        help="Path to write metrics as JSON (e.g. results.json)",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    errors = []
    for name, value in [
        ("--requests", args.requests),
        ("--tokens", args.tokens),
        ("--max-num-seqs", args.max_num_seqs),
        ("--max-num-batched-tokens", args.max_num_batched_tokens),
        ("--max-num-prefill-tokens", args.max_num_prefill_tokens),
        ("--max-prefill-chunk-size", args.max_prefill_chunk_size),
        ("--block-size", args.block_size),
        ("--max-model-len", args.max_model_len),
    ]:
        if value <= 0:
            errors.append(f"{name} must be > 0, got {value}")
    if args.num_gpu_blocks is not None and args.num_gpu_blocks <= 0:
        errors.append(f"--num-gpu-blocks must be > 0, got {args.num_gpu_blocks}")
    if errors:
        for e in errors:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    # Auto-calculate block count if not specified
    prompt_estimate = sum(len(p) + args.tokens for p in _PROMPTS[:args.requests])
    num_blocks = args.num_gpu_blocks or max(16, prompt_estimate // 4 * 2)

    config = Config(
        executor_type=args.executor,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_prefill_tokens=args.max_num_prefill_tokens,
        max_prefill_chunk_size=args.max_prefill_chunk_size,
        chunked_prefill_enabled=args.chunked_prefill,
        block_size=args.block_size,
        num_gpu_blocks=num_blocks,
        max_model_len=args.max_model_len,
        decode_first=args.decode_first,
        print_step_events=not args.quiet,
        memory_trace=args.memory_trace,
    )

    # --- Print configuration ---
    print("=" * 60)
    print("mini-vLLM Benchmark Configuration")
    print("=" * 60)
    print()
    print("  Workload")
    print("  " + "-" * 50)
    print(f"    executor:       {args.executor}")
    print(f"    requests:       {args.requests}")
    print(f"    max_tokens:     {args.tokens}")
    print()
    print("  Scheduler")
    print("  " + "-" * 50)
    print(f"    max_num_seqs:            {args.max_num_seqs}")
    print(f"    max_num_batched_tokens:  {args.max_num_batched_tokens}")
    print(f"    max_num_prefill_tokens:  {args.max_num_prefill_tokens}")
    print(f"    max_prefill_chunk_size:  {args.max_prefill_chunk_size}")
    print(f"    chunked_prefill:         {config.chunked_prefill_enabled}")
    print(f"    decode_first:            {config.decode_first}")
    print()
    print("  KV Cache")
    print("  " + "-" * 50)
    print(f"    block_size:     {args.block_size}")
    print(f"    num_gpu_blocks: {num_blocks}")
    print()

    # --- Run ---
    engine = LLMEngine(config)

    for i in range(args.requests):
        prompt = _PROMPTS[i % len(_PROMPTS)]
        rid = engine.add_request(prompt, max_new_tokens=args.tokens)
        print(f"  Added request {rid}: prompt={prompt[:40]!r}...  (max_tokens={args.tokens})")

    print(f"\n  Running {args.requests} requests to completion...")
    outputs = engine.run_until_done()

    # --- Metrics ---
    report = engine.engine_core.metrics_collector.report()

    total_time = report["total_time_seconds"]
    active_time = report["active_time_seconds"]
    print(f"  Done in {total_time:.3f}s"
          f"  (active processing: {active_time:.3f}s)\n")

    # Print full report from the collector
    engine.engine_core.metrics_collector.print_report(report)

    # --- Summary ---
    print("Summary")
    print("-" * 60)
    print(f"  total_time_seconds:       {report['total_time_seconds']}")
    print(f"  request_throughput:       {report['throughput_req_per_sec']} req/s")
    print(f"  token_throughput:         {report['throughput_tok_per_sec']} tok/s")
    print(f"  avg_ttft_ms:              {report['avg_ttft_ms']}")
    print(f"  avg_tpot_ms:              {report['avg_tpot_ms']}")
    print(f"  kv_util_peak_pct:         {report['kv_util_peak_pct']}%")
    print(f"  avg_scheduler_latency_ms: {report['avg_scheduler_latency_ms']}")
    print(f"  rejected_requests:        {report['rejected_requests']}")
    print()

    if args.executor == "fake":
        print("  ** These are simulation metrics, not real GPU inference "
              "results. **")
        print("  Use --executor qwen for real model inference timing.")
        print()

    # --- JSON output ---
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Metrics saved to {args.json_output}")
        print()


if __name__ == "__main__":
    main()
