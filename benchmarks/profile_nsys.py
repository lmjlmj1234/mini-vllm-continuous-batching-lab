#!/usr/bin/env python3
"""
Nsight Systems profiling script — produces a .nsys-rep file with NVTX range
markers around Phase_Prefill and Phase_Decode for timeline visualization.

Usage:
    nsys profile -t nvtx,cuda -o benchmark_results/nsys_trace -w true \\
        python3 benchmarks/profile_nsys.py

Output:
    benchmark_results/nsys_trace.nsys-rep — open in Nsight Systems GUI.
    Timeline shows Phase_Prefill (first step) followed by Phase_Decode (remaining steps).
    Select Phase_Decode range to inspect GPU kernel timing without Prefill interference.
"""

import os
import sys
import time as time_module

import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mini_vllm import Config, LLMEngine
from mini_vllm.sequence.status import Status

MODEL_PATH = "/mnt/e/vllm_awq_qwen_exp/models/Qwen2.5-0.5B-Instruct"
CONTEXT_LEN = 128
DECODE_TOKENS = 16  # enough decode steps to show in timeline
BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 4096


def build_diverse_prompt(tokenizer, target_len: int) -> str:
    """Build a diverse prompt with unique blocks (no prefix-cache hash collisions)."""
    sentences = [
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms.",
        "Attention allows models to weigh the importance of different input tokens when producing each output token.",
        "Multi-head attention runs multiple attention operations in parallel, capturing different relationships.",
        "Positional encodings give the model information about the order of tokens in the sequence.",
        "Layer normalization stabilizes training by normalizing activations across the feature dimension.",
        "The feed-forward network in each transformer layer applies two linear transformations with a ReLU activation.",
        "Residual connections help gradients flow through deep networks by adding the input to the layer output.",
        "Dropout regularization randomly drops neurons during training to prevent overfitting.",
        "The encoder processes input sequences bidirectionally while the decoder generates output autoregressively.",
        "Scaled dot-product attention computes attention scores by taking the dot product of queries and keys.",
        "Beam search explores multiple candidate sequences during generation to find the most likely output.",
        "The softmax function converts logits into probability distributions over the vocabulary.",
    ]
    text = " ".join(sentences * 10)
    ids = tokenizer.encode(text)
    if len(ids) < target_len:
        last = tokenizer.encode(sentences[-1])
        while len(ids) < target_len:
            ids.extend(last)
    exact = ids[:target_len]
    return tokenizer.decode(exact)


def main():
    print("=== Pre-run Memory ===")
    os.system("free -h | head -2")
    os.system("nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader")

    # ---------- Engine setup ----------
    config = Config(
        executor_type="paged",
        model_path=MODEL_PATH,
        num_gpu_blocks=NUM_GPU_BLOCKS,
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        max_num_prefill_tokens=8192,
        max_prefill_chunk_size=CONTEXT_LEN,
        block_size=BLOCK_SIZE,
        chunked_prefill_enabled=True,
        decode_first=True,
        print_step_events=False,
        memory_trace=False,
        attention_backend="triton",
        request_timeout_s=300,
        gpu_memory_utilization=0.85,
    )
    engine = LLMEngine(config)
    engine_core = engine.engine_core

    tokenizer = engine.executor._tokenizer
    prompt_text = build_diverse_prompt(tokenizer, CONTEXT_LEN)
    actual_len = len(tokenizer.encode(prompt_text))
    print(f"Prompt: {actual_len} tokens")

    # ---------- NVTX-instrumented step ----------
    original_step = engine_core.step
    nvtx = torch.cuda.nvtx

    def nsys_step():
        """EngineCore.step() with NVTX range markers for nsys timeline."""
        engine_core._step_count += 1
        step_num = engine_core._step_count
        engine_core._check_timeouts()

        # Schedule
        result = engine_core._scheduler.schedule()

        only_prefill_seqs = []
        for sg in result.scheduled_prefill_groups:
            for seq in sg.get_unfinished_seqs():
                if seq.status == Status.PREFILL:
                    only_prefill_seqs.append(seq)

        decode_seqs = []
        for sg in result.scheduled_decode_groups:
            decode_seqs.extend(sg.get_unfinished_seqs())

        has_prefill = bool(only_prefill_seqs)
        has_decode = bool(decode_seqs)

        if has_prefill or has_decode:
            bm = engine_core._block_manager
            if has_prefill and bm is not None:
                for seq in only_prefill_seqs:
                    chunk_end = min(
                        seq.prefill_cursor + engine_core._config.max_prefill_chunk_size,
                        len(seq.prompt_token_ids),
                    )
                    for pos in range(seq.prefill_cursor, chunk_end):
                        bm.ensure_block(seq, pos)
            if has_decode and bm is not None:
                for seq in decode_seqs:
                    pos = len(seq.prompt_token_ids) + seq.num_generated_tokens - 1
                    bm.ensure_block(seq, pos)

            model_input = engine_core._input_builder.build(
                prefill_seqs=only_prefill_seqs,
                decode_seqs=decode_seqs,
            )

            torch.cuda.synchronize()

            # ---- NVTX range: Phase_Prefill or Phase_Decode ----
            phase_name = "Phase_Prefill" if has_prefill else "Phase_Decode"
            nvtx.range_push(phase_name)

            model_output = engine_core._executor.execute(model_input)

            torch.cuda.synchronize()
            nvtx.range_pop()

            engine_core._apply_model_output(model_output, only_prefill_seqs, decode_seqs)
            chunk_size = engine_core._config.max_prefill_chunk_size
            for seq in only_prefill_seqs:
                end = min(seq.prefill_cursor + chunk_size, len(seq.prompt_token_ids))
                seq.prefill_cursor = end

        # Cleanup + metrics
        for sg in result.finished_groups:
            for seq in sg.seqs:
                engine_core._executor.cleanup_sequence(seq.seq_id)
                engine_core._metrics.register_sequence(seq)

        bm_stats = engine_core._scheduler.block_manager_stats()
        engine_core._metrics.record_step(
            result, 0.0, 0.0,
            bm_stats["total_blocks"], bm_stats["used_blocks"],
            effective_batch_size=(
                len(result.scheduled_decode_groups) + len(result.scheduled_prefill_groups)
            ),
            running_count=engine_core._scheduler._queue.num_running,
            waiting_count=engine_core._scheduler._queue.num_waiting,
        )

        return result

    engine_core.step = nsys_step

    # ---------- Run ----------
    engine.add_request(prompt_text, max_new_tokens=DECODE_TOKENS)
    t0 = time_module.time()
    while engine.queue.num_waiting > 0 or engine.queue.num_running > 0:
        engine.step()
    elapsed = time_module.time() - t0

    engine_core.step = original_step

    # ---------- Post-run ----------
    print(f"\nProfile run complete: {elapsed:.2f}s wall time")
    print(f"Total engine steps: {engine_core.step_count}")
    print(f"Data captured in nsys trace.")

    print("\n=== Post-run Memory ===")
    os.system("free -h | head -2")
    os.system("nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader")

    # Verify profile output would be correct
    print(f"\nNVTX ranges recorded: Phase_Prefill (first step) + Phase_Decode (next {DECODE_TOKENS} steps)")
    print("Open .nsys-rep in Nsight Systems GUI, filter by 'Phase_' to see ranges.")


if __name__ == "__main__":
    main()
