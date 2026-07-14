# Milestone B Plan — Qwen2.5 End-to-End ModelRunner

## Overview

Build a complete Qwen2.5 ModelRunner from modular transformer components
(RMSNorm, RoPE, QKV+GQA, SwiGLU MLP, LM head), load HF checkpoint weights
into these custom modules, wire PagedAttention via AttentionBackendRef, and
integrate into EngineCore via PagedExecutor — all under a single unified
layer-per-sequence loop with zero fallback to HF model forward.

---

## 1. Dynamic Config Reading

All dimensions come from `AutoConfig.from_pretrained(model_path)`, read via
`ConfigAdapter` in `mini_vllm/model_runner/config_adapter.py`:

```python
config = AutoConfig.from_pretrained(model_path)
model_config = ModelConfig(
    num_layers=config.num_hidden_layers,
    hidden_size=config.hidden_size,
    num_heads=config.num_attention_heads,
    num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
    head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
    intermediate_size=config.intermediate_size,
    rope_theta=getattr(config, 'rope_theta', 10000.0),
    rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
    vocab_size=config.vocab_size,
    tie_word_embeddings=getattr(config, 'tie_word_embeddings', False),
    max_position_embeddings=getattr(config, 'max_position_embeddings', 32768),
    rope_scaling=getattr(config, 'rope_scaling', None),
)
```

No hardcoded Qwen2.5-0.5B dimensions anywhere.

---

## 2. Weight Loading Mapping

### Module structure: all custom `nn.Module`, no HF module reuse

Modules are plain `nn.Module` subclasses, NOT HF classes. Weight loader copies
weights by name from HF `state_dict`.

### Mapping table (HF key → our module):

| HF state_dict key | Our module / param | Notes |
|---|---|---|
| `model.embed_tokens.weight` | `qwen_model.embed_tokens.weight` | direct copy |
| `model.layers.{i}.input_layernorm.weight` | `qwen_model.layers[i].input_layernorm.weight` | RMSNorm |
| `model.layers.{i}.self_attn.q_proj.weight` | `qwen_model.layers[i].attention.qkv_weight`[0:Q_slice] | fused QKV |
| `model.layers.{i}.self_attn.k_proj.weight` | `qwen_model.layers[i].attention.qkv_weight`[Q_slice:QK_slice] | |
| `model.layers.{i}.self_attn.v_proj.weight` | `qwen_model.layers[i].attention.qkv_weight`[QK_slice:] | |
| `model.layers.{i}.self_attn.o_proj.weight` | `qwen_model.layers[i].attention.o_proj.weight` | direct copy |
| `model.layers.{i}.post_attention_layernorm.weight` | `qwen_model.layers[i].post_attn_layernorm.weight` | RMSNorm |
| `model.layers.{i}.mlp.gate_proj.weight` | `qwen_model.layers[i].mlp.gate_up_weight`[0:G_slice] | fused gate+up |
| `model.layers.{i}.mlp.up_proj.weight` | `qwen_model.layers[i].mlp.gate_up_weight`[G_slice:] | |
| `model.layers.{i}.mlp.down_proj.weight` | `qwen_model.layers[i].mlp.down_proj.weight` | direct copy |
| `model.norm.weight` | `qwen_model.norm.weight` | final RMSNorm |
| `lm_head.weight` | `qwen_model.lm_head.weight` | tied to embed if tie_word_embeddings |

### Fused weight layout

**QKV**: `qkv_weight [Q_sz + K_sz + V_sz, hidden_size]` where
- `Q_sz = num_heads * head_dim`
- `K_sz = num_kv_heads * head_dim`
- `V_sz = num_kv_heads * head_dim`

**Gate+Up**: `gate_up_weight [intermediate_size * 2, hidden_size]` where
- rows [0:intermediate_size) = gate_proj
- rows [intermediate_size:2*intermediate_size) = up_proj

Both fusions done at load time: read three separate HF tensors, `torch.cat` into one.

---

## 3. RoPE (Rotary Position Embedding)

### Implementation: `mini_vllm/model/rotary.py`

Following HF Qwen2 implementation (`Qwen2RotaryEmbedding`):

- **rotary_dim** = `head_dim` (Qwen2.5 always applies to full head_dim, no partial)
- **theta** = 1000000.0 (from config)
- **max_seq_len** = max_position_embeddings (32768 for Qwen2.5-0.5B)

### Prefill/decode position

- **position_ids** come from `ModelInput.block_tables` → for each sequence,
  generate contiguous positions:
  - Prefill: positions[seq_start:seq_end-prefix_len] = range(0, prefill_len) +
    per-seq offset
  - Decode: positions[seq_start] = seq_cached_len (next absolute position)
- Both prefill and decode positions are **absolute** positions in the sequence,
  starting from 0 at the first token.
- For prefix-shared sequences, offset varies per sequence.

### Precomputation

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
# Shape: [head_dim // 2]
```

Cos/sin are computed on demand from `position_ids`:

```python
def apply_rotary_emb(x: Tensor, position_ids: Tensor, cos_cached: Tensor, sin_cached: Tensor) -> Tensor:
    # x: [total_tokens, num_heads, head_dim] (or [total_tokens, num_kv_heads, head_dim])
    # position_ids: [total_tokens]
    cos = cos_cached[position_ids]  # [total_tokens, head_dim//2] → broadcast
    sin = sin_cached[position_ids]
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd  = x_even * sin + x_odd * cos
    return stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
```

### Alignment with HF Qwen2

- Same `inv_freq` formula
- Same cos/sin shape
- Same half-pair rotation pattern
- Verified by `test_rotary.py` against `Qwen2RotaryEmbedding.forward(positions)`

### cos/sin dtype/device

- cos/sin computed in fp32, then cast to model dtype on model device.
- Cached once at init time up to max_seq_len.

---

## 4. Unified Layer Execution Order (every layer, every step)

```text
For each layer i in 0..num_layers-1:

  hidden_states → input_layernorm (RMSNorm)
  → QKV projection (fused qkv_weight @ normed)
  → reshape Q [total_tokens, num_heads, head_dim]
           K [total_tokens, num_kv_heads, head_dim]
           V [total_tokens, num_kv_heads, head_dim]
  → apply RoPE to Q and K (absolute positions)
  → write KV to cache: backend.write_kv_cache(layer_i, k, v, slot_mapping)
  → paged attention:
      if any prefill_tokens:
          backend.prefill_attention(layer_i, q, k, v, attn_metadata, pool, cache_write_k, cache_write_v)
      if any decode_tokens:
          backend.decode_attention(layer_i, q, attn_metadata, pool)
  → output projection (o_proj @ attn_out)
  → + residual: hidden_states = attn_output + residual
  → post_attention_layernorm (RMSNorm)
  → SwiGLU MLP: gate = SiLU(hidden @ gate_up_weight[0:inter])
                 up   = hidden @ gate_up_weight[inter:]
                 mlp_out = gate * up @ down_proj_weight
  → + residual: hidden_states = mlp_out + hidden_states
```

Key invariant: both prefill and decode sequences pass through the **same** layer
loop. No split into prefill-phase / decode-phase.

---

## 5. Mixed Prefill + Decode — One Unified Loop

This is the core architecture requirement. Given `ModelInput` containing both
prefill and decode sequences in the same step:

### Packed tensor layout

Example: seq0 prefill (3 tokens), seq1 prefill (5 tokens), seq2 decode (1 token)

```
input_ids = [t0, t1, t2, | t3, t4, t5, t6, t7, | t8]  # 9 total tokens
positions = [0,  1,  2,  | 0,  1,  2,  3,  4,  | 5]    # absolute positions
```

`ModelInput` fields:
- `input_ids`: [total_tokens] — packed flat
- `positions`: [total_tokens] — packed flat
- `block_tables`: [num_seqs, max_num_blocks_per_seq] — padded
- `slot_mapping`: [total_tokens] — maps each token position to cache slot
- `sample_token_indices`: indices in packed tensor where logits are sampled
  (always the last token of each sequence)
- `attn_metadata`: `AttentionMetadata` with:
  - `prefill_group = AttentionGroup(type="prefill", seq_indices=[0,1],
     token_start=[0,3], token_end=[3,8])`
  - `decode_group = AttentionGroup(type="decode", seq_indices=[2],
     token_start=[8], token_end=[9])`

### What happens per layer

1. **QKV projection**: run on ALL `hidden_states` (all 9 tokens) at once.
2. **RoPE** on Q and K: run on all 9 tokens with respective positions.
3. **Cache write**: `write_to_paged_cache(k, v, slot_mapping)` — writes all 9
   tokens (both prefill and decode) to correct cache slots.
4. **Prefill attention**: `backend.prefill_attention()` uses `attn_metadata`.
   - For seq0 (3 tokens), reads block_table → gathers cached prefix from blocks
     + current tokens (only need current tokens since this is first time seeing
     seq0, cache is empty for it until write above). 方案B: K/V from the
     **just-written cache** → gather uses block_table which now includes the
     tokens written in step 3.
   - For seq1 (5 tokens), same pattern.
5. **Decode attention**: `backend.decode_attention()` uses `attn_metadata`.
   - For seq2 (1 token), gather K/V from cache via block_table (cache now
     contains seq2's previously-written tokens).
6. **Output proj + MLP**: on all tokens together.

### Why this works

The layer loop processes all tokens (prefill + decode) simultaneously through
QKV, RoPE, cache write. The attention step selects which tokens participate in
which attention mode via `AttentionMetadata` groups. This is **not** two model
forwards — it's one forward with differentiated attention per sequence.

### AttentionBackend interface alignment

Current `AttentionBackendRef.write_kv_cache()` takes `[total_tokens, num_heads, head_dim]`
K/V and writes to pool slots via `slot_mapping`. Both prefill and decode tokens
are written in one call.

Current `AttentionBackendRef.decode_attention()` takes the raw Q (per-token)
and reads K/V from cache per block_table. Only decode-group tokens' Q values
are used.

---

## 6. ModelInput Packed Token Layout — Detailed

### From `ModelInputBuilder.build()`:

For each sequence in the step, `ModelInputBuilder` produces:

| Field | Per-sequence | Packed |
|---|---|---|
| `input_ids` | [seq_len] tokens | `cat` into [total_tokens] |
| `positions` | range(0, seq_len) or absolute | `cat` into [total_tokens] |
| `slot_mapping` | [seq_len] slot indices | `cat` into [total_tokens] |
| `block_table` | [num_blocks] → block IDs | pad to [num_seqs, max_blocks] |

### Sequence layout example

```
Seq0 (prefill, 3 tokens, block_size=4):
  input_ids = [101, 102, 103]
  positions = [0, 1, 2]
  slot_mapping = [0, 1, 2]        # block 0, slots 0,1,2
  block_table = [0]               # 1 block
  sample_output_index = 2          # last token

Seq1 (prefill, 5 tokens):
  input_ids = [201, 202, 203, 204, 205]
  positions = [0, 1, 2, 3, 4]
  slot_mapping = [4, 5, 6, 7, 8]  # block 1 full, block 2 slot 0
  block_table = [1, 2]
  sample_output_index = 7          # index in packed tensor

Seq2 (decode, 1 token, already has 6 cached):
  input_ids = [301]
  positions = [6]
  slot_mapping = [11]              # block 2, slot 2
  block_table = [2, 3]
  sample_output_index = 8
```

Packed:
```
input_ids = [101, 102, 103, 201, 202, 203, 204, 205, 301]     # 9 tokens
positions = [0,   1,   2,   0,   1,   2,   3,   4,   6]        # 9 positions
slot_mapping = [0,  1,  2,  4,  5,  6,  7,  8,  11]           # 9 slots
sample_output_indices = [2, 7, 8]                              # 3 samples

AttentionMetadata:
  prefill_group.token_range = [(0, 3), (3, 8)]    # seq0 → input[0:3], seq1 → input[3:8]
  prefill_group.block_tables = [[0], [1, 2]]
  decode_group.token_range = [(8, 9)]              # seq2 → input[8:9]
  decode_group.block_tables = [[2, 3]]
```

---

## 7. PagedExecutor — only `execute(ModelInput) -> ModelRunnerOutput`

### `mini_vllm/executor/paged_executor.py`

```python
class PagedExecutor:
    def __init__(self, config: Config, block_manager):
        self.config = config
        self.device = config.device
        self.attention_backend = AttentionBackendRef()  # or configurable
        self.model_runner = QwenModelRunner(
            model_path=config.model_path,
            attention_backend=self.attention_backend,
            config=config,
            device=self.device,
        )
        # Block manager already allocated pool in EngineCore.__init__
        self.pool = ...  # shared with block_manager

    def execute(self, model_input: ModelInput) -> ModelRunnerOutput:
        logits = self.model_runner.execute_model(model_input)
        # Greedy argmax on sample positions
        sampled_ids = torch.argmax(logits, dim=-1)  # [num_samples]
        return ModelRunnerOutput(
            sampled_token_ids=sampled_ids,
            sampled_sequence_ids=list(range(len(sampled_ids))),
        )

    def tokenize(self, prompt: str) -> list[int]:
        # Uses HF AutoTokenizer.from_pretrained
        ...

    def detokenize(self, token_ids: list[int]) -> str:
        ...
```

No `prefill()` / `decode()` methods. No `past_key_values`. No HF model
forward call. Layer loop is entirely inside `QwenModelRunner.execute_model()`.

---

## 8. Prohibited Patterns

| Prohibited | Why |
|---|---|
| HF `past_key_values` tuple | bypasses paged cache; incompatible with PagedAttention |
| `model.forward(input_ids, past_key_values=...)` | whole-model HF forward; exactly what we're replacing |
| Batch-wide max-KV padding | every seq gets padded to memory bound of longest seq |
| Silent fallback to `QwenExecutor` | defeats purpose of Milestone B |
| `AutoModel.from_pretrained(..., torch_dtype=...)` | only for correctness comparison test, never for execution |
| Automatic online model download in tests | tests must check for local model; `pytest.mark.skipif` |

---

## 9. HF Correctness Alignment (4 levels)

### Level 1: Component alignment (`tests/test_rms_norm.py`, `test_rotary.py`, etc.)

Each custom module's forward is compared against the equivalent HF module:

| Component | HF reference | Tolerance |
|---|---|---|
| RMSNorm | `Qwen2RMSNorm` | atol=1e-6, fp32 |
| Rotary | `Qwen2RotaryEmbedding` | atol=1e-6, fp32 |
| QKV+GQA split | manual HF equivalent | atol=1e-5, fp16 |
| SwiGLU MLP | `Qwen2MLP` | atol=1e-5, fp16 |

### Level 2: Full prefill logits alignment

Compare output of `QwenModelRunner.execute_model(prefill_input)` against
`Qwen2ForCausalLM.forward(input_ids).logits[:, -1, :]` at same prefill tokens.

Tolerance: atol=1.0 for fp16 (compounded through 24 layers of fp16 matmul).

### Level 3: First decode logits alignment

After prefill → argmax greedy sample → feed sampled token as decode input.

Compare `QwenModelRunner.execute_model(decode_input)` logits against
`Qwen2ForCausalLM.forward(input_ids + [sampled], past_key_values=kv_cache).logits[:, -1, :]`.

### Level 4: Multi-step greedy generation

Generate 8+ tokens through EngineCore. Compare token-by-token against HF
`model.generate(max_new_tokens=8, do_sample=False)`.

Must cover at least one block boundary (e.g., block_size=4, generate 8 tokens,
all crossing block boundaries).

All atol=0 (token ID equality) — any divergence is a bug.

---

## 10. Memory Profiling

After ModelRunner is functional:

1. Load the smallest Qwen2.5 model (0.5B, ~900MB fp16)
2. Run one prefill (128 tokens) + 16 decode steps
3. Record: `torch.cuda.max_memory_allocated()` (or CPU `tracemalloc`)
4. Recompute `num_blocks = (total_memory - model_memory) / block_size / 2 / head_dim / num_kv_heads / 2`
5. Verify `KVCachePool.allocate()` uses this block count
6. Verify allocator's watermark matches available blocks
7. Record in `docs/PAGED_ENGINE_IMPLEMENTATION_ROADMAP.md`

---

## 11. Test Categories

| Category | Run condition | Location | Tests |
|---|---|---|---|
| CPU unit | always | `tests/test_rms_norm.py`, `test_rotary.py`, `test_qkv_proj.py`, `test_mlp.py` | shapes, fwd, edge cases |
| GPU component | `torch.cuda.is_available()` | `tests/test_qwen_layer.py`, `test_qwen_model.py` | fwd shapes, small tensor |
| Local model integration | `os.path.exists(model_path)` | `tests/test_weight_loader.py`, `test_qwen_runner.py` | weight loading, logits |
| E2E | model_path + cuda | `tests/test_paged_executor.py` | prefill→decode→output |
| HF correctness | model_path + cuda | `tests/test_hf_alignment.py` | 4-level alignment |

No automatic model download. Integration tests use `--model-path` or env var.

---

## 12. Explicit Non-Goals (out of scope for Milestone B)

- Triton GPU kernel (→ Milestone C)
- CUDA Graph capture/replay (→ Milestone C)
- Prefix Cache (→ Milestone C)
- Tensor Parallel (→ Milestone D)
- Pipeline Parallel (→ Milestone D)
- Preemption (→ Milestone D)
- Sliding Window Attention
- Speculative Decoding / Draft Model
- Chunked Prefill
- Flash Attention / FlashInfer integration
- bfloat16 training mode
- Multi-node inference
- Dynamic batching beyond EngineCore's current Scheduler

---

## 13. Internal Subtasks (B1–B7, single plan approval)

### B1: Config + Weight Mapping (mini_vllm/model/weight_loader.py)
- Extend `ConfigAdapter` to read all Qwen2.5 dimensions (already started)
- Implement `load_qwen_weights()` with fused QKV and gate+up
- Unit test: load a real Qwen2.5-0.5B checkpoint (if available) and verify
  `state_dict` keys match mapping table
- Test: verify fused weight shapes

### B2: Model Layers (mini_vllm/model/)
- Implement: `rms_norm.py`, `rotary.py`, `qkv_proj.py`, `mlp.py`
- `transformer_layer.py`: QwenDecoderLayer assembling above components
- `qwen_model.py`: QwenModel with embed_tokens + N layers + final norm + lm_head
- CPU unit tests for each component (shape, fwd, edge cases)
- GPU tests for single layer forward

### B3: Packed ModelRunner (mini_vllm/model_runner/qwen_runner.py)
- Implement `QwenModelRunner.execute_model(ModelInput) -> Tensor`
- Unified layer loop with mixed prefill+decode attention
- Coordinate with `AttentionBackendRef` interface
- Test with dummy weights: execute_model produces correct logit shapes
- Test both pure-prefill, pure-decode, and mixed ModelInput

### B4: PagedExecutor (mini_vllm/executor/paged_executor.py)
- Implement `PagedExecutor` with `execute() -> ModelRunnerOutput`
- Wire tokenizer (HF AutoTokenizer)
- Greedy sampling (argmax)
- Test: execute returns correct ModelRunnerOutput format

### B5: EngineCore Integration
- Wire `PagedExecutor` into `EngineCore.__init__()` via `config.executor_type`
- Add `executor_type = "paged"` or `"qwen_paged"` to `Config`
- Create `EngineCore` with `PagedExecutor`, run prefill→decode→output
- E2E test: input prompt → output tokens (no crash, correct structure)

### B6: HF Correctness Alignment
- 4-level test suite (`tests/test_hf_alignment.py`)
- Run with real Qwen2.5-0.5B-Instruct weights
- Report: which levels pass, which fail (likely fp16 drift in level 2+)

### B7: Real Memory Profiling
- Profile run with smallest Qwen2.5 model
- Update `num_blocks`, pool size, allocator config
- `peak_runtime_estimate` recorded in roadmap
- Update config defaults if needed

---

## File Change Summary

| Action | File | Purpose |
|---|---|---|
| NEW | `mini_vllm/model/rms_norm.py` | RMSNorm module |
| NEW | `mini_vllm/model/rotary.py` | RoPE module |
| NEW | `mini_vllm/model/qkv_proj.py` | QKV projection + GQA split |
| NEW | `mini_vllm/model/mlp.py` | SwiGLU MLP |
| NEW | `mini_vllm/model/transformer_layer.py` | Single decoder layer |
| NEW | `mini_vllm/model/qwen_model.py` | Full Qwen model |
| NEW | `mini_vllm/model/weight_loader.py` | HF weight loading with fusion |
| NEW | `mini_vllm/model_runner/qwen_runner.py` | QwenModelRunner |
| NEW | `mini_vllm/executor/paged_executor.py` | PagedExecutor |
| MODIFY | `mini_vllm/model_runner/config_adapter.py` | Extend for all Qwen2.5 dims |
| MODIFY | `mini_vllm/model_runner/base.py` | Add `rope_scaling` to ModelConfig |
| MODIFY | `mini_vllm/config.py` | Add `paged` executor_type |
| MODIFY | `mini_vllm/engine/engine_core.py` | Wire PagedExecutor |
| NEW | `tests/test_rms_norm.py` | unit |
| NEW | `tests/test_rotary.py` | unit |
| NEW | `tests/test_qkv_proj.py` | unit |
| NEW | `tests/test_mlp.py` | unit |
| NEW | `tests/test_qwen_layer.py` | GPU component test |
| NEW | `tests/test_qwen_model.py` | GPU component test |
| NEW | `tests/test_weight_loader.py` | local model integration |
| NEW | `tests/test_qwen_runner.py` | local model integration |
| NEW | `tests/test_paged_executor.py` | E2E |
| NEW | `tests/test_hf_alignment.py` | HF correctness 4-level |

---

## Working Agreement

- **No per-module pauses**: write all of B1→B7 in one continuous pass.
- **Only stop for**: architecture deviation from unified loop, interface
  conflict with existing `AttentionBackend`, or inability to maintain
  packed execution.
- **Bug fixes and test adjustments**: fix inline, no pause.
- **Milestone completion report**: one report at end covering all acceptance
  criteria, test counts, HF alignment level, and memory profile.
