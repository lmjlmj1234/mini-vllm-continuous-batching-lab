"""
Capture logits around the mismatch - direct approach using engine_core.
"""
import sys, os, json, torch, gc, time
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from mini_vllm import Config, LLMEngine, Status
from mini_vllm.engine.engine_core import EngineCore
from benchmarks.continuous_batching import _generate_request_prompts, _build_ab_config, _cleanup_gpu

model_path = "/mnt/e/vllm_awq_qwen_exp/models/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
requests_data = _generate_request_prompts(tokenizer, 16, seed=42)

# Patch EngineCore.step to capture logits
_original_step = EngineCore.step
captured_data = []

def patched_step(self):
    import time, copy
    result = _original_step(self)
    # After _apply_model_output, the logits were consumed.
    # Instead, capture from _apply_model_output
    # The model_output is stored as we save it before _apply_model_output
    if hasattr(self, '_last_raw_logits') and self._last_raw_logits is not None:
        captured_data.append({
            'step': self._step_count,
            'logits': self._last_raw_logits,
            'seq_info': getattr(self, '_last_seq_info', None),
        })
    return result

# Also patch _apply_model_output to save raw logits
_original_apply = EngineCore._apply_model_output
def patched_apply(self, model_output, prefill_seqs, decode_seqs):
    # Save raw logits BEFORE consumption
    # model_output is usually [num_tokens, vocab_size] or similar
    if isinstance(model_output, torch.Tensor):
        self._last_raw_logits = model_output.detach().cpu()
    else:
        self._last_raw_logits = None
    return _original_apply(self, model_output, prefill_seqs, decode_seqs)

def run_backend_with_logits(requests_data, concurrency, attention_backend):
    EngineCore.step = patched_step
    EngineCore._apply_model_output = patched_apply
    global captured_data
    captured_data = []

    config = _build_ab_config(
        concurrency=concurrency, model_path=model_path,
        num_gpu_blocks=16384, attention_backend=attention_backend,
    )
    engine = LLMEngine(config)

    try:
        for req in requests_data:
            engine.add_request(req["prompt"], max_new_tokens=req["output_length"])
        engine.run_until_done()

        output_tokens = {}
        for sg_name, sg in engine._queue._finished.items():
            for seq in sg.seqs:
                if seq.status == Status.FINISHED:
                    output_tokens[seq.group_id] = list(seq.output_token_ids)

        return output_tokens, list(captured_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        del engine
        _cleanup_gpu()
        EngineCore.step = _original_step
        EngineCore._apply_model_output = _original_apply

# Warmup
warmup_config = _build_ab_config(concurrency=2, model_path=model_path, num_gpu_blocks=16384, attention_backend="reference")
we = LLMEngine(warmup_config)
for req in requests_data[:2]:
    we.add_request(req["prompt"], max_new_tokens=req["output_length"])
we.run_until_done()
del we
_cleanup_gpu()

print("Running Reference (Flash SDPA)...", flush=True)
ref_tokens, ref_logits = run_backend_with_logits(requests_data, 2, "reference")
print(f"Got {len(ref_tokens)} outputs, {len(ref_logits)} logit captures", flush=True)

r6_ref = ref_tokens.get("req-0006")
print(f"req-0006 outputs: {len(r6_ref)} tokens" if r6_ref else "req-0006 not found")

# Find the step where req-0006 generates position 43
# The logits tensor shape [num_tokens, vocab_size] where sample indices tell us which position each logit is for
print(f"\nLogit captures: {len(ref_logits)}")

# Analyze the capture structure
for i, cap in enumerate(ref_logits[:3]):
    l = cap['logits']
    print(f"  Capture {i}: step={cap['step']}, logits shape={l.shape}, seq_info={cap['seq_info']}")

# Find step 43 for req-0006 - iterate captures
for i, cap in enumerate(ref_logits):
    l = cap['logits']
    if l.dim() != 2:
        continue
    num_logits = l.shape[0]
    # Each logit corresponds to a token in the sequence at a given step
    # We need to figure out which seq_info maps to which row in logits
    # Since we're not capturing seq_info correctly, let's try a different approach

# Actually let's just compare logits for the specific token positions
# req-0006 has prompt=322 tokens, output_length=64
# Position 43 of output means step where cached_len_before = 43
# Since prefill is chunked, let's find the exact logit

# Simpler approach: just show all captured data structure
if ref_logits:
    l = ref_logits[0]['logits']
    print(f"\nFirst capture shape: {l.shape}, dtype: {l.dtype}")
    # Look at step structure
    for i, cap in enumerate(ref_logits):
        if i < 5 or i > len(ref_logits) - 5:
            print(f"  cap[{i}]: step={cap['step']}, shape={cap['logits'].shape}")

# Now find the token at position 43 output for req-0006 in ref
print(f"\nReference req-0006 tokens around position 43:")
for p in range(40, min(48, len(r6_ref))):
    tok = tokenizer.decode([r6_ref[p]])
    print(f"  pos {p}: id={r6_ref[p]} '{tok}'")

# Also check concurrency=8 for req-0012
print(f"\n\nReference req-0012 (concurrency=8):")
_cleanup_gpu()
_cleanup_gpu()

# Quick check with concurrency=8
warmup_config2 = _build_ab_config(concurrency=8, model_path=model_path, num_gpu_blocks=16384, attention_backend="reference")
we2 = LLMEngine(warmup_config2)
for req in requests_data[:2]:
    we2.add_request(req["prompt"], max_new_tokens=req["output_length"])
we2.run_until_done()
del we2
_cleanup_gpu()

ref8_tokens, ref8_logits = run_backend_with_logits(requests_data, 8, "reference")
r12_ref = ref8_tokens.get("req-0012")
if r12_ref:
    print(f"req-0012 outputs: {len(r12_ref)} tokens")
    for p in range(6, min(14, len(r12_ref))):
        tok = tokenizer.decode([r12_ref[p]])
        print(f"  pos {p}: id={r12_ref[p]} '{tok}'")
    print(f"\nLogit captures for ref8: {len(ref8_logits)}")

print("\nDone.")
