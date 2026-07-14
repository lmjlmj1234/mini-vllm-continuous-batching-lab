"""EngineCore end-to-end tests with real Qwen2.5 model.

Requires a real Qwen2.5 model path. All tests are skipped when
``QWEN_MODEL_PATH`` is not set or model weights are unavailable.

Covers:
1. Single request prefill + 8 decode vs HF step-by-step greedy.
2. Continuous batching: two requests interleaved (decode+prefill).
3. Mixed prefill+decode: new request prefills while existing decodes.
4. Block boundary: sequence crosses a block_size boundary.
"""

import os
import time

import pytest
import torch

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "")


@pytest.fixture(autouse=True)
def _gpu_cleanup():
    """Free GPU memory between tests."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    yield
    gc.collect()
    torch.cuda.empty_cache()



def _has_model():
    if not os.path.exists(MODEL_PATH):
        return False
    for fname in os.listdir(MODEL_PATH):
        if fname.endswith((".safetensors", ".bin")):
            return True
    return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(not _has_model(), reason="Qwen2.5 model weights not available")
def test_e2e_single_prefill_decode():
    """Single request prefill + 8 decode tokens match HF step-by-step."""
    from mini_vllm import LLMEngine, Config

    config = Config(
        model_path=MODEL_PATH,
        executor_type="paged",
        block_size=4,
        num_gpu_blocks=64,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        print_step_events=False,
    )
    engine = LLMEngine(config)
    prompt = "The capital of France is"
    engine.add_request(prompt, max_new_tokens=8)
    outputs = engine.run_until_done()
    req_id = list(outputs.keys())[0]
    our_text = outputs[req_id]

    # HF reference
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda:0")
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device,
    )
    hf.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        past_kv = None
        step_ids = prompt_ids
        generated = []
        for _ in range(8):
            out = hf(input_ids=step_ids, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1).item()
            generated.append(next_id)
            step_ids = torch.tensor([[next_id]], device=device)

    hf_text = tokenizer.decode(generated, skip_special_tokens=True)
    assert our_text == hf_text, (
        f"E2E single: our={our_text!r} hf={hf_text!r}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(not _has_model(), reason="Qwen2.5 model weights not available")
def test_e2e_continuous_batching():
    """Two requests: first prefill+decode, second arrives during decode."""
    from mini_vllm import LLMEngine, Config

    config = Config(
        model_path=MODEL_PATH,
        executor_type="paged",
        block_size=4,
        num_gpu_blocks=64,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        print_step_events=False,
    )
    engine = LLMEngine(config)

    prompt1 = "The capital of France is"
    prompt2 = "The capital of Germany is"

    engine.add_request(prompt1, max_new_tokens=6)

    # Run 3 steps (so req1 has done prefill + some decodes)
    for _ in range(3):
        engine.step()

    # Add second request while first is still decoding
    engine.add_request(prompt2, max_new_tokens=4)

    # Run until all done
    outputs = engine.run_until_done()
    assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"

    # HF reference for both prompts
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda:0")
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device,
    )
    hf.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def hf_generate(prompt, max_tokens):
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            past_kv = None
            step_ids = prompt_ids
            generated = []
            for _ in range(max_tokens):
                out = hf(input_ids=step_ids, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(dim=-1).item()
                generated.append(next_id)
                step_ids = torch.tensor([[next_id]], device=device)
        return tokenizer.decode(generated, skip_special_tokens=True)

    hf_text1 = hf_generate(prompt1, 6)
    hf_text2 = hf_generate(prompt2, 4)

    # Match outputs by checking both prompts
    our_texts = list(outputs.values())
    assert hf_text1 in our_texts, f"Expected {hf_text1!r} in {our_texts}"
    assert hf_text2 in our_texts, f"Expected {hf_text2!r} in {our_texts}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(not _has_model(), reason="Qwen2.5 model weights not available")
def test_e2e_block_boundary():
    """Sequence crosses block_size boundary (crosses into 2nd block)."""
    from mini_vllm import LLMEngine, Config

    config = Config(
        model_path=MODEL_PATH,
        executor_type="paged",
        block_size=4,
        num_gpu_blocks=64,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        print_step_events=False,
    )
    engine = LLMEngine(config)

    # Verify block boundary crossing with the same trustworthy prompt.
    # With block_size=4, 5 prompt tokens + 8 decode = 13 tokens,
    # spanning blocks 0, 1, and 2.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt = "The capital of France is"
    prompt_text = prompt

    # Generate enough tokens to cross multiple block boundaries
    engine.add_request(prompt_text, max_new_tokens=8)
    outputs = engine.run_until_done()
    req_id = list(outputs.keys())[0]
    our_text = outputs[req_id]

    # HF reference
    device = torch.device("cuda:0")
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device,
    )
    hf.eval()

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        past_kv = None
        step_ids = prompt_ids
        generated = []
        for _ in range(8):
            out = hf(input_ids=step_ids, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1).item()
            generated.append(next_id)
            step_ids = torch.tensor([[next_id]], device=device)

    hf_text = tokenizer.decode(generated, skip_special_tokens=True)
    assert our_text == hf_text, (
        f"Block boundary: our={our_text!r} hf={hf_text!r}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(not _has_model(), reason="Qwen2.5 model weights not available")
def test_e2e_mixed_prefill_decode():
    """Mixed prefill+decode: one sequence decodes while another prefills."""
    from mini_vllm import LLMEngine, Config

    config = Config(
        model_path=MODEL_PATH,
        executor_type="paged",
        block_size=4,
        num_gpu_blocks=64,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        print_step_events=False,
    )
    engine = LLMEngine(config)

    prompt1 = "The capital of France is"
    prompt2 = "Germany"

    engine.add_request(prompt1, max_new_tokens=6)

    # Run prefill step (step 1)
    engine.step()
    # Add second request while first is decoding
    engine.add_request(prompt2, max_new_tokens=3)

    # Continue — next step should have decode(req1) + prefill(req2)
    outputs = engine.run_until_done()
    assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda:0")
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device,
    )
    hf.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def hf_generate(prompt, max_tokens):
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            past_kv = None
            step_ids = prompt_ids
            generated = []
            for _ in range(max_tokens):
                out = hf(input_ids=step_ids, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(dim=-1).item()
                generated.append(next_id)
                step_ids = torch.tensor([[next_id]], device=device)
        return tokenizer.decode(generated, skip_special_tokens=True)

    hf_text1 = hf_generate(prompt1, 6)
    hf_text2 = hf_generate(prompt2, 3)

    our_texts = list(outputs.values())
    assert hf_text1 in our_texts, f"Expected {hf_text1!r} in {our_texts}"
    assert hf_text2 in our_texts, f"Expected {hf_text2!r} in {our_texts}"
