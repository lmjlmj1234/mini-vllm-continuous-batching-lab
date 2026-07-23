"""
Investigate the 15/16 token mismatch in Reference vs Triton attention backends.

Usage:
    python3 debug_investigate.py --result-dir benchmark_results/continuous_batching_backend_ab/20260715_170705_6c57ba8
"""
import sys, os, json, random, time, torch, gc
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from mini_vllm import Config, LLMEngine, Status
from benchmarks.continuous_batching import _generate_request_prompts, _build_ab_config, _cleanup_gpu

model_path = "/mnt/e/vllm_awq_qwen_exp/models/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

def decode_token(tid):
    """Safely decode a single token ID."""
    try:
        return tokenizer.decode(tid)
    except:
        return f"<{tid}>"

def run_single_backend(requests_data, concurrency, attention_backend, label=""):
    """Run one backend and return output token IDs plus attention logits."""
    config = _build_ab_config(
        concurrency=concurrency,
        model_path=model_path,
        num_gpu_blocks=16384,
        attention_backend=attention_backend,
    )
    engine = LLMEngine(config)
    try:
        for req in requests_data:
            engine.add_request(req["prompt"], max_new_tokens=req["output_length"])
        engine.run_until_done()

        # Extract output tokens
        output_tokens = {}
        for sg_name, sg in engine._queue._finished.items():
            for seq in sg.seqs:
                if seq.status == Status.FINISHED:
                    output_tokens[seq.group_id] = list(seq.output_token_ids)

        return output_tokens
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None
    finally:
        del engine
        _cleanup_gpu()

def compare_token_seqs(ref_tokens, tri_tokens, request_id):
    """Compare two token sequences and report mismatch details."""
    if ref_tokens is None or tri_tokens is None:
        return None

    min_len = min(len(ref_tokens), len(tri_tokens))
    for i in range(min_len):
        if ref_tokens[i] != tri_tokens[i]:
            return {
                "request_id": request_id,
                "position": i,
                "ref_tok": ref_tokens[i],
                "tri_tok": tri_tokens[i],
                "ref_tok_decoded": decode_token(ref_tokens[i]),
                "tri_tok_decoded": decode_token(tri_tokens[i]),
                "ref_len": len(ref_tokens),
                "tri_len": len(tri_tokens),
                "ref_seq": ref_tokens,
                "tri_seq": tri_tokens,
            }
    return None  # exact match

# ============================================================
# Phase 1: Reproduce the exact prompts from the experiment
# ============================================================
print("=" * 70)
print("PHASE 1: Reproduce prompts")
print("=" * 70)

# Regenerate prompts - the experiment used seed=42 with default mixed workload
requests_data = _generate_request_prompts(tokenizer, 16, seed=42)
print(f"Generated {len(requests_data)} requests")
for i, r in enumerate(requests_data):
    print(f"  req-{i:04d}: input={r['actual_input_tokens']}, output={r['output_length']}")

# ============================================================
# Phase 2: Run concurrency=2 with ref_first to reproduce the mismatch
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: Reproduce concurrency=2 mismatch (req-0006)")
print("=" * 70)

concurrency = 2
N_REPEATS = 5
all_mismatches = []

for rep in range(N_REPEATS):
    print(f"\n--- Repeat {rep+1}/{N_REPEATS} (ref_first) ---")

    # Warmup
    warmup_config = _build_ab_config(concurrency=concurrency, model_path=model_path, num_gpu_blocks=16384, attention_backend="reference")
    we = LLMEngine(warmup_config)
    wu = requests_data[:2]
    for req in wu:
        we.add_request(req["prompt"], max_new_tokens=req["output_length"])
    we.run_until_done()
    del we
    _cleanup_gpu()

    # Reference
    ref_tokens = run_single_backend(requests_data, concurrency, "reference", "ref")
    if ref_tokens is None:
        print("Reference FAILED")
        continue

    # Triton
    tri_tokens = run_single_backend(requests_data, concurrency, "triton", "tri")
    if tri_tokens is None:
        print("Triton FAILED")
        continue

    # Compare
    for rid in ref_tokens:
        result = compare_token_seqs(ref_tokens[rid], tri_tokens.get(rid), rid)
        if result is not None:
            all_mismatches.append(result)
            print(f"  MISMATCH: {rid} pos={result['position']} ref={result['ref_tok']}({result['ref_tok_decoded']}) vs tri={result['tri_tok']}({result['tri_tok_decoded']})")

    # Check req-0006 specifically
    r6_ref = ref_tokens.get("req-0006")
    r6_tri = tri_tokens.get("req-0006")
    if r6_ref and r6_tri:
        diff = compare_token_seqs(r6_ref, r6_tri, "req-0006")
        if diff:
            print(f"  req-0006 diff at pos {diff['position']}")
            # Show surrounding tokens
            start = max(0, diff['position'] - 2)
            end = min(len(r6_ref), diff['position'] + 5)
            for p in range(start, end):
                marker = " <--" if p == diff['position'] else ""
                print(f"    pos {p}: ref={r6_ref[p]}({decode_token(r6_ref[p])}) tri={r6_tri[p]}({decode_token(r6_tri[p])}){marker}")

print(f"\nTotal mismatches across {N_REPEATS} runs: {len(all_mismatches)}")
for m in all_mismatches[:5]:
    print(f"  {m['request_id']} pos={m['position']} ref={m['ref_tok']} tri={m['tri_tok']}")

# ============================================================
# Phase 3: Reproduce concurrency=8 mismatch (req-0012)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: Reproduce concurrency=8 mismatch (req-0012)")
print("=" * 70)

concurrency = 8
for rep in range(N_REPEATS):
    print(f"\n--- Repeat {rep+1}/{N_REPEATS} (ref_first) ---")

    # Warmup
    warmup_config = _build_ab_config(concurrency=concurrency, model_path=model_path, num_gpu_blocks=16384, attention_backend="reference")
    we = LLMEngine(warmup_config)
    wu = requests_data[:2]
    for req in wu:
        we.add_request(req["prompt"], max_new_tokens=req["output_length"])
    we.run_until_done()
    del we
    _cleanup_gpu()

    # Reference first
    ref_tokens = run_single_backend(requests_data, concurrency, "reference", "ref")
    if ref_tokens is None:
        continue

    tri_tokens = run_single_backend(requests_data, concurrency, "triton", "tri")
    if tri_tokens is None:
        continue

    for rid in ref_tokens:
        result = compare_token_seqs(ref_tokens[rid], tri_tokens.get(rid), rid)
        if result is not None:
            print(f"  MISMATCH: {rid} pos={result['position']} ref={result['ref_tok']}({result['ref_tok_decoded']}) vs tri={result['tri_tok']}({result['tri_tok_decoded']})")

# ============================================================
# Phase 4: Test Reference with deterministic math SDPA vs Flash SDPA
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: Test Reference with torch.backends.cuda.sdp_kernel math backend")
print("=" * 70)

# Force math backend (deterministic) for SDPA
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    print("SDPA backend set to: math", flush=True)
    # Need to rebuild engine with this context

    warmup_config = _build_ab_config(concurrency=2, model_path=model_path, num_gpu_blocks=16384, attention_backend="reference")
    we = LLMEngine(warmup_config)
    wu = requests_data[:2]
    for req in wu:
        we.add_request(req["prompt"], max_new_tokens=req["output_length"])
    we.run_until_done()
    del we
    _cleanup_gpu()

    ref_math = run_single_backend(requests_data, 2, "reference", "ref(math_sdpa)")

    # Now run triton
    tri = run_single_backend(requests_data, 2, "triton", "tri")

    if ref_math and tri:
        print("\nMath SDPA Reference vs Triton:")
        for rid in ref_math:
            result = compare_token_seqs(ref_math[rid], tri.get(rid), rid)
            if result:
                print(f"  MISMATCH: {rid} pos={result['position']} ref={result['ref_tok']}({result['ref_tok_decoded']}) vs tri={result['tri_tok']}({result['tri_tok_decoded']})")
            else:
                print(f"  MATCH: {rid}")

        # Also compare math vs flash SDPA reference
        ref_flash = run_single_backend(requests_data, 2, "reference", "ref(flash_sdpa)")
        if ref_flash:
            print("\nMath SDPA vs Flash SDPA (both reference):")
            for rid in ref_math:
                result = compare_token_seqs(ref_math[rid], ref_flash.get(rid), rid)
                if result:
                    print(f"  MISMATCH: {rid} pos={result['position']}")
                else:
                    print(f"  MATCH: {rid}")

print("\nDone.")
