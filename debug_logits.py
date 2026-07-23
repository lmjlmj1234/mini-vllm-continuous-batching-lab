"""
Capture logits around the mismatch points for deep investigation.
"""
import sys, os, json, torch, gc, time
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from mini_vllm import Config, LLMEngine, Status
from benchmarks.continuous_batching import _generate_request_prompts, _build_ab_config, _cleanup_gpu

model_path = "/mnt/e/vllm_awq_qwen_exp/models/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
requests_data = _generate_request_prompts(tokenizer, 16, seed=42)

# Patch the engine to capture raw logits at each step
original_step = LLMEngine.step

def patched_step(self):
    result = self._engine_core.step()
    # Capture logits from the model output
    # The model output is stored in _engine_core._last_logits after execute
    if hasattr(self._engine_core, '_last_logits') and self._engine_core._last_logits is not None:
        if not hasattr(self, '_captured_logits'):
            self._captured_logits = []
        self._captured_logits.append({
            'step': self._engine_core.step_count,
            'logits': self._engine_core._last_logits.detach().cpu(),
            'seq_info': self._engine_core._last_seq_info,
        })
    # Capture finished output tokens
    for sg in result.finished_groups:
        for seq in sg.seqs:
            if seq.status == Status.FINISHED:
                text = self._executor.detokenize(seq.output_token_ids)
                self._outputs[seq.group_id] = text
    return result

def run_backend_with_logits(requests_data, concurrency, attention_backend):
    """Run one backend, capturing per-step logits."""
    LLMEngine.step = patched_step

    config = _build_ab_config(
        concurrency=concurrency, model_path=model_path,
        num_gpu_blocks=16384, attention_backend=attention_backend,
    )
    engine = LLMEngine(config)
    engine._captured_logits = []

    try:
        for req in requests_data:
            engine.add_request(req["prompt"], max_new_tokens=req["output_length"])
        engine.run_until_done()

        output_tokens = {}
        for sg_name, sg in engine._queue._finished.items():
            for seq in sg.seqs:
                if seq.status == Status.FINISHED:
                    output_tokens[seq.group_id] = list(seq.output_token_ids)

        return output_tokens, engine._captured_logits
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        del engine
        _cleanup_gpu()
        LLMEngine.step = original_step

# Warmup
warmup_config = _build_ab_config(concurrency=2, model_path=model_path, num_gpu_blocks=16384, attention_backend="reference")
we = LLMEngine(warmup_config)
for req in requests_data[:2]:
    we.add_request(req["prompt"], max_new_tokens=req["output_length"])
we.run_until_done()
del we
_cleanup_gpu()

# Run both backends
print("Running Reference (Flash SDPA)...", flush=True)
ref_tokens, ref_logits = run_backend_with_logits(requests_data, 2, "reference")
print(f"Got {len(ref_tokens)} outputs, {len(ref_logits)} logit steps", flush=True)

# Identify the step where req-0006 generates token at position 43
# req-0006 has output_length=64, so it generates tokens at steps where it's in decode
# Need to match sequence_info to request IDs

r6_ref = ref_tokens.get("req-0006")
print(f"\nreq-0006 outputs: {len(r6_ref)} tokens" if r6_ref else "req-0006 not found")

# Find which step corresponds to position 43 for req-0006
for i, cap in enumerate(ref_logits):
    seq_info = cap['seq_info']
    # seq_info is a tuple of SequenceExecutionInfo
    # Find the decode step that processes req-0006
    if seq_info:
        for si in seq_info:
            if hasattr(si, 'sequence_id') and 'req-0006' in str(si.sequence_id):
                pos = si.cached_len_before if hasattr(si, 'cached_len_before') else -1
                if pos == 43:
                    logits_at_diff = cap['logits']
                    print(f"\nStep {cap['step']}: req-0006 at cached_len_before={pos}")
                    print(f"  Logits shape: {logits_at_diff.shape}")

                    # The logits are [num_tokens, vocab_size] or [1, 1, vocab_size]
                    # Find which sample index corresponds to req-0006
                    sample_idx = si.sample_output_index if hasattr(si, 'sample_output_index') else None
                    print(f"  Sample index: {sample_idx}")

                    if logits_at_diff is not None and sample_idx is not None:
                        if logits_at_diff.dim() == 1:
                            token_logits = logits_at_diff
                        elif logits_at_diff.dim() == 2:
                            token_logits = logits_at_diff[sample_idx]
                        elif logits_at_diff.dim() == 3:
                            token_logits = logits_at_diff[0, sample_idx]
                        else:
                            token_logits = logits_at_diff.flatten()
                            print(f"  Unexpected dims, flattening from {logits_at_diff.shape}")

                        # Top-5 tokens
                        top_vals, top_ids = torch.topk(token_logits, 10)
                        print("\n  Top-10 tokens at divergence point (Reference):")
                        for j in range(10):
                            tok = tokenizer.decode([top_ids[j].item()])
                            print(f"    rank {j}: id={top_ids[j].item()} value={top_vals[j].item():.4f} '{tok}'")

                        # Reference selected token vs this top-1
                        ref_chosen = r6_ref[43]
                        ref_chosen_val = token_logits[ref_chosen].item()
                        top1_val = top_vals[0].item()
                        top2_val = top_vals[1].item() if len(top_vals) > 1 else float('-inf')
                        print(f"\n  Reference chose: id={ref_chosen} value={ref_chosen_val:.4f}")
                        print(f"  Top-1 value: {top1_val:.4f}")
                        print(f"  Top-2 value: {top2_val:.4f}")
                        print(f"  Top-1 vs Top-2 margin: {top1_val - top2_val:.4f}")
                        print(f"  Reference top-1 margin: {top1_val - ref_chosen_val:.4f}")

                        if sample_idx > 0:
                            prev_logits = None
                            for j in range(i-1, -1, -1):
                                if ref_logits[j]['seq_info']:
                                    for si in ref_logits[j]['seq_info']:
                                        if hasattr(si, 'sequence_id') and 'req-0006' in str(si.sequence_id):
                                            prev_logits = ref_logits[j]['logits']
                                            break
                                    if prev_logits is not None:
                                        break

                            if prev_logits is not None:
                                print("\n  Previous step logits (position 42):")
                                if prev_logits.dim() in (1,):
                                    prev_logit_vec = prev_logits
                                elif prev_logits.dim() == 2:
                                    prev_logit_vec = prev_logits[sample_idx - 1 if sample_idx > 0 else 0]
                                else:
                                    prev_logit_vec = prev_logits.flatten()
                                ptop_vals, ptop_ids = torch.topk(prev_logit_vec, 5)
                                for j in range(5):
                                    tok = tokenizer.decode([ptop_ids[j].item()])
                                    print(f"    rank {j}: id={ptop_ids[j].item()} value={ptop_vals[j].item():.4f} '{tok}'")

                    break
        else:
            continue
        break

print("\n\nNow running Triton for logit comparison...", flush=True)

_cleanup_gpu()

tri_tokens, tri_logits = run_backend_with_logits(requests_data, 2, "triton")
print(f"Got {len(tri_tokens)} outputs, {len(tri_logits)} logit steps", flush=True)

r6_tri = tri_tokens.get("req-0006")
print(f"\nreq-0006 outputs: {len(r6_tri)} tokens" if r6_tri else "req-0006 not found")

for i, cap in enumerate(tri_logits):
    seq_info = cap['seq_info']
    if seq_info:
        for si in seq_info:
            if hasattr(si, 'sequence_id') and 'req-0006' in str(si.sequence_id):
                pos = si.cached_len_before if hasattr(si, 'cached_len_before') else -1
                if pos == 43:
                    logits_at_diff = cap['logits']
                    print(f"\nStep {cap['step']}: req-0006 at cached_len_before={pos}")
                    print(f"  Logits shape: {logits_at_diff.shape}")

                    sample_idx = si.sample_output_index if hasattr(si, 'sample_output_index') else None
                    print(f"  Sample index: {sample_idx}")

                    if logits_at_diff is not None and sample_idx is not None:
                        if logits_at_diff.dim() == 1:
                            token_logits = logits_at_diff
                        elif logits_at_diff.dim() == 2:
                            token_logits = logits_at_diff[sample_idx]
                        elif logits_at_diff.dim() == 3:
                            token_logits = logits_at_diff[0, sample_idx]
                        else:
                            token_logits = logits_at_diff.flatten()
                            print(f"  Unexpected dims, flattening from {logits_at_diff.shape}")

                        top_vals, top_ids = torch.topk(token_logits, 10)
                        print("\n  Top-10 tokens at divergence point (Triton):")
                        for j in range(10):
                            tok = tokenizer.decode([top_ids[j].item()])
                            print(f"    rank {j}: id={top_ids[j].item()} value={top_vals[j].item():.4f} '{tok}'")

                        tri_chosen = r6_tri[43] if len(r6_tri) > 43 else -1
                        if tri_chosen >= 0:
                            tri_chosen_val = token_logits[tri_chosen].item()
                            print(f"\n  Triton chose: id={tri_chosen} value={tri_chosen_val:.4f}")

                        top1_val = top_vals[0].item()
                        top2_val = top_vals[1].item() if len(top_vals) > 1 else float('-inf')
                        print(f"  Top-1 value: {top1_val:.4f}")
                        print(f"  Top-2 value: {top2_val:.4f}")
                        print(f"  Top-1 vs Top-2 margin: {top1_val - top2_val:.4f}")
                    break
        else:
            continue
        break

print("\nDone with logit capture.")
