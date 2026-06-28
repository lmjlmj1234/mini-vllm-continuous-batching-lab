#!/usr/bin/env python3
"""Mini-vLLM Continuous Batching — Fake Engine Demo with Memory Trace.

Demonstrates::

    1. Two requests arrive at step 0.
    2. A third request arrives after step 2 (mid-run arrival).
    3. Continuous batching + chunked prefill in action.
    4. Memory Trace mode shows BlockAllocator free list and per-sequence
       BlockTable at each step.
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mini_vllm import Config, LLMEngine


def main() -> None:
    print("=" * 60)
    print("mini-vLLM Continuous Batching — Fake Engine Demo")
    print("=" * 60)

    config = Config(
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_num_prefill_tokens=16,
        max_prefill_chunk_size=4,
        block_size=4,
        num_gpu_blocks=16,
        vocab_size=256,
        memory_trace=True,
    )
    engine = LLMEngine(config)

    print('\n[init] Adding request A (prompt="Hello world", max_new_tokens=8)')
    rid_a = engine.add_request("Hello world", max_new_tokens=8)

    print('[init] Adding request B (prompt="CUDA batching", max_new_tokens=12)')
    rid_b = engine.add_request("CUDA batching", max_new_tokens=12)

    print("\n--- Step 1 ---")
    engine.step()

    print("\n--- Step 2 ---")
    engine.step()

    print('\n[arrival] New request C arrives (prompt="New batch", max_new_tokens=6)')
    rid_c = engine.add_request("New batch", max_new_tokens=6)

    final_outputs = engine.run_until_done()

    print("\n" + "=" * 60)
    print("Final Outputs")
    print("=" * 60)

    for rid, text in final_outputs.items():
        sg = engine.queue.get_by_id(rid)
        seq = sg.seqs[0] if sg and sg.seqs else None
        tokens = seq.num_output_tokens if seq else 0
        print(f"  {rid}: prompt={sg.prompt!r}  output={text!r}  "
              f"(tokens={tokens})")

    for sg in engine.queue.rejected:
        print(f"  {sg.request_id}: prompt={sg.prompt!r}  (REJECTED)")

    # Print benchmark report
    engine.engine_core.metrics_collector.print_report()

    print("\nDone.")


if __name__ == "__main__":
    main()
