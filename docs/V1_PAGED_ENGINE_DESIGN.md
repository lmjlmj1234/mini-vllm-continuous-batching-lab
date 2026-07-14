# V1 Paged Engine — Revised Design Document

> **Phase A output — REVISION 1.** Revised per 14-point review (2026-07-13).
> Design for real PagedAttention, GPU KV cache pool, and batched execution.
> No code has been written yet. This document is the implementation specification.

---

## Section 0: How Each Review Point Is Addressed

| # | Review Point | How Addressed |
|---|-------------|---------------|
| 1 | Unified mixed prefill+decode execution | `Executor` provides single `execute(prefill_seqs, decode_seqs)`. `ModelRunner.execute_model()` runs one Transformer layer loop per step, dispatching per-group to the appropriate attention kernel. See Section 3. |
| 2 | Precise length semantics | Replaced vague "context_lens" with three unambiguous fields: `cached_len_before`, `query_len`, `kv_len_after`. See Section 4. |
| 3 | MetadataBuilder decoupled from BlockManager | New `ModelInputBuilder` class in `engine/input_builder.py`. BlockManager exposes two read-only query methods. See Section 5. |
| 4 | Single BlockTable truth source | `Sequence.block_table` is removed entirely. All consumers read from `BlockManager.get_block_table(seq_id)`. See Section 10. |
| 5 | Separate Chunked Prefill paths | Reference path (per-seq gather + SDPA, tests only). Production GPU path (flattened, block-driven, unified online softmax). See Section 8. |
| 6 | GPU PagedAttention mandatory | GPU PagedAttention is the sole production backend. PyTorch reference exists for test comparison only — no silent fallback. See Section 9. |
| 7 | Corrected memory budget | Removed wrong 1.1M/49M numbers. Profile-run based initialization: run a short forward pass to measure actual free memory, compute block count conservatively. See Section 11. |
| 8 | Remove block_size % num_kv_heads constraint | Spurious constraint removed. Only real constraints: block_size>=1, block_size aligned for GPU shared memory (power of 2 preferred). |
| 9 | Unify dtype (fp16) | Both paged backend and HF reference use fp16. Model weights stay at their native dtype (bf16 for Qwen), activations cast to fp16. |
| 10 | sample_token_indices | LM head only computes logits at positions that need sampling: completed prefill first token + each decode step. Not all prefill tokens. See Section 6. |
| 11 | Complete block lifecycle | Documented: who calls free, when, for all paths (normal finish, cancel, timeout, exception, disconnect). See Section 11. |
| 12 | Deduplicate module responsibilities | Consolidated: KVCacheMetadata removed, AttentionMetadata/ModelInput are single definitions. BlockManager no longer generates tensor metadata. See Section 5. |
| 13 | Revised risk table | GPU kernel crash, unified execution correctness, chunked prefill edge cases, RoPE, resource leak, memory budget all raised to High severity. See Section 14. |
| 14 | Revised completion criteria | Added 8 new criteria: no per-seq HF model call, no Sequence.block_table, MetadatBuilder is separate, reference vs production GPU paths distinguished, Triton kernel tested, LM head selective, release paths documented, no silent fallback from GPU to CPU. See Section 15. |


---

## Section 1: Revised Module Diagram

```
mini_vllm/
├── scheduler/
│   ├── scheduler.py                 ← UNCHANGED (scheduling policy)
│   └── schedule_result.py           ← UNCHANGED (result type)
├── sequence/                        ← UNCHANGED (Sequence, SequenceGroup)
├── cache/
│   ├── allocator.py                 ← ENHANCED (invariant checks, max_blocks)
│   ├── block_table.py               ← ENHANCED (tensor export, no mirror)
│   ├── manager.py                   ← ENHANCED (query methods only: get_block_table, needs_allocation)
│   └── kv_cache_pool.py             ← NEW (GPU tensors, single KV data source)
├── attention/
│   ├── backend.py                   ← NEW (AttentionBackend interface)
│   ├── paged_attention_ref.py       ← NEW (PyTorch reference — TESTS ONLY)
│   └── paged_attention_gpu.py       ← NEW (Triton kernel — PRODUCTION)
├── model_runner/
│   ├── base.py                      ← NEW (ModelRunner interface)
│   ├── config_adapter.py            ← NEW (dynamic config from HF config.json)
│   └── qwen_runner.py               ← NEW (full layer loop, mixed prefill+decode)
├── engine/
│   ├── engine.py                    ← UNCHANGED (public API)
│   ├── engine_core.py               ← ADAPTED (single execute() call, input_builder)
│   ├── input_builder.py             ← NEW (ModelInputBuilder: constructs ModelInput)
│   ├── metrics.py                   ← UNCHANGED
│   └── stage_profiler.py            ← UNCHANGED
├── executor/
│   ├── base.py                      ← ENHANCED (unified execute() replaces prefill()+decode())
│   ├── executor.py                  ← UNCHANGED (FakeModelExecutor)
│   ├── qwen_executor.py             ← PRESERVED (reference, unchanged)
│   └── paged_executor.py            ← NEW (wraps ModelRunner, no _seq_kv)
├── worker/
│   ├── fake_worker.py               ← UNCHANGED
│   └── qwen_worker.py               ← UNCHANGED (returns QwenExecutor)
├── model/
│   └── fake_model.py                ← UNCHANGED
├── config.py                        ← ENHANCED (paged executor type)
└── serving/                         ← UNCHANGED
```

### Key Changes Per Review

| Change | Why |
|--------|-----|
| `engine/input_builder.py` (NEW) | Decouples metadata construction from BlockManager (review #3) |
| `executor/base.py` ENHANCED | `execute()` replaces `prefill()+decode()` (review #1) |
| No `cache/cache_metadata.py` | Deduplicated: only `model_runner/model_input.py` for metadata (review #12) |
| No `attention/metadata.py` | Merged into `model_runner/model_input.py` (review #12) |
| No `worker/paged_worker.py` | PagedExecutor self-contained, PagedWorker not needed (review #12 re-evaluation) |
| `Sequence.block_table` removed | Single truth source in BlockManager._tables (review #4) |


---

## Section 2: Revised Module Responsibilities (Deduplicated)

### Scheduler (UNCHANGED)
- Request lifecycle (waiting/running/finished)
- Token budget (max_num_seqs, max_num_batched_tokens)
- Decode-first scheduling
- Chunked prefill (split long prompts at prefill_cursor)
- KV block admission (calls block_manager.allocate_for_seq)
- Does NOT touch GPU tensors. Outputs ScheduleResult with SequenceGroup lists.

### BlockAllocator (ENHANCED — metadata only, no tensors)
- Free-list management (bool array + ref_counts)
- allocate(N) → List[int] (physical block IDs)
- free(ids): decrement ref_count, return to pool at 0
- increment_ref(pid): for prefix cache sharing
- Callbacks: on_allocate (no-op for pre-allocated pool), on_free (no-op)
- Invariant checks: used + free == total, free block ref_count == 0
- Expose `max_blocks` for GPU pool sizing
- No tensor operations. Pure metadata.

### BlockManager (ENHANCED — read-only query methods, no metadata tensor construction)
- Manages per-seq BlockTable dict (AUTHORITATIVE source)
- Exposes:
  - `get_block_table(seq_id) -> List[int]` — read-only view of block IDs
  - `needs_allocation(seq, position) -> int` — returns physical block ID (allocates if needed)
- allocate_for_seq(seq): creates empty BlockTable
- free(seq_id): pops BlockTable, frees physical blocks via allocator.free()
- probe_prefix_cache(prompt_token_ids) → PrefixCacheProbeResult (for scheduler budget)
- **Does NOT generate slot_mapping, batched tensors, or length arrays**
- **Does NOT sync to Sequence.block_table** (Sequence.block_table removed)

### BlockTable (ENHANCED — no mirror copy)
- `List[BlockTableEntry(physical_block_id, is_shared)]`
- `get_block_ids() -> List[int]` (read-only flat view, no copy to Sequence)
- `get_physical_block(position) -> int | None`
- `export_block_table_tensor(max_blocks: int) -> torch.Tensor` (padded, for GPU kernel)
- `is_shared_at(position) -> bool` (for COW detection)

### KVCachePool (NEW — single KV data source)
- Pre-allocated GPU tensors:
  ```
  key_cache:   [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
  value_cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
  ```
- `get_key_cache(layer_idx) -> Tensor`
- `get_value_cache(layer_idx) -> Tensor`
- Read-only after initialization (writes go through AttentionBackend.write_kv_cache)
- Serves as fallback for prefix cache: prefetch block data for shared sequences

### ModelInputBuilder (NEW — replaces BlockManager's metadata construction role)
- `build(schedule_result, block_manager) -> ModelInput`
- Constructs all tensors needed for one execute_model() call:
  - input_ids (concatenated prefill + decode tokens)
  - positions (absolute positions)
  - slot_mapping (flat, per-token physical slots)
  - block_tables (padded per-sequence)
  - cached_len_before / query_len / kv_len_after for each group
  - sample_token_indices (which positions need LM head)
- Reads block data from BlockManager via `get_block_table(seq_id)`
- Does NOT modify BlockManager state

### AttentionBackend (NEW — interface)
- `allocate_pool(layout) -> KVCachePool` — allocate GPU pool
- `write_kv_cache(layer_idx, key, value, slot_mapping)` — write K/V
- `prefill_attention(...)` — prefill (SDPA + paged prefix gather)
- `decode_attention(...)` — decode (PagedAttention from pool)

### ModelRunner (NEW — single layer loop per step)
- `execute_model(model_input) -> torch.Tensor` — one call per step
- Layer loop handles ALL sequences simultaneously
- Dispatch per attention group to correct AttentionBackend method
- No _seq_kv, no HF past_key_values

### PagedExecutor (NEW — wraps ModelRunner)
- `execute(prefill_seqs, decode_seqs)`:
  1. ModelInputBuilder.build() → ModelInput
  2. ModelRunner.execute_model() → output logits (at sample positions only)
  3. Sample (argmax at sample positions)
  4. Update sequence output tokens and prefill_cursor
- `prepare_block(block_id)`: no-op (pool is pre-allocated)
- `release_block(block_id)`: no-op
- `cleanup_sequence(seq_id)`: no-op (no per-seq state)

### EngineCore (ADAPTED)
- `step()`:
  1. scheduler.schedule() → ScheduleResult
  2. If has_work: executor.execute(prefill_seqs, decode_seqs)  # SINGLE CALL
  3. Cleanup finished sequences

### QwenExecutor (PRESERVED — reference only)
- Unchanged. Uses HF past_key_values, per-sequence model calls.
- Used for reference comparison tests only (Phase 6+).


---

## Section 3: Unified Execution Call Chain

### EngineCore Step Loop

```
LLMEngine.step()
  └→ EngineCore.step()
       ├→ scheduler.schedule() → ScheduleResult
       │    (same 6-phase algorithm, unchanged)
       │
       ├→ ModelInputBuilder.build(schedule_result, block_manager) → ModelInput
       │    ├─ prefill groups → input_ids, positions, slot_mapping, cached_len_before, query_len
       │    ├─ decode groups  → input_ids, positions, slot_mapping, cached_len_before, query_len
       │    └─ sample_token_indices (indices into concatenated input where LM head is needed)
       │
       ├→ executor.execute(model_input)
       │    └→ ModelRunner.execute_model(model_input)
       │         │
       │         # SINGLE LAYER LOOP (one pass, mixed prefill + decode)
       │         ├─ hidden_states = embedding(input_ids)
       │         ├─ for layer_idx in range(num_layers):
       │         │    ├─ residual = hidden_states
       │         │    ├─ hidden_states = input_rms_norm(hidden_states)
       │         │    ├─ q, k, v = qkv_proj(hidden_states)
       │         │    ├─ q, k = rope(q, k, positions)
       │         │    ├─ attention_backend.write_kv_cache(layer_idx, k, v, slot_mapping)
       │         │    │    # All tokens (prefill + decode) write K/V in one shot
       │         │    │
       │         │    ├─ for group in attn_metadata.groups:
       │         │    │    │  # Group = list of (seq_indices, attention_type)
       │         │    │    │  # attention_type ∈ {PREFILL_REF, PREFILL_GPU, DECODE_GPU}
       │         │    │    │
       │         │    │    └─ if group.type == PREFILL_REF:
       │         │    │    │     out = attention_ref.prefill_attention(...)
       │         │    │    └─ if group.type == PREFILL_GPU:
       │         │    │    │     out = attention_gpu.prefill_attention(...)
       │         │    │    └─ if group.type == DECODE_GPU:
       │         │    │    │     out = attention_gpu.decode_attention(...)
       │         │    │    │     # PagedAttention GPU kernel for decode
       │         │    │
       │         │    ├─ hidden_states = output_proj(attention_output) + residual
       │         │    ├─ residual = hidden_states
       │         │    ├─ hidden_states = post_attention_rms_norm(hidden_states)
       │         │    └─ hidden_states = mlp(hidden_states) + residual
       │         │
       │         ├─ hidden_states = final_rms_norm(hidden_states)
       │         ├─ logits = lm_head(hidden_states[sample_token_indices])  # SELECTIVE
       │         └─ return logits  # [num_sample_positions, vocab_size]
       │
       ├─ Sample: argmax per sequence → output_token_ids
       ├─ Update prefill_cursor for prefilling sequences
       ├─ Mark finished sequences in scheduler
       └─ cleanup: block_manager.free(finished), metrics.record

### Key Difference from Original Design

| Aspect | Original | Revised |
|--------|----------|---------|
| Executor interface | `prefill()` + `decode()` as two calls | SINGLE `execute(model_input)` |
| Layer loop | One pass? Two passes? Not specified | One pass, mixed prefill + decode tokens |
| Group dispatch | Implicit | Explicit: each attention group has a type field |
| LM head | Full [num_tokens, vocab_size] | Selective [num_sample_positions, vocab_size] |
| EngineCore → Executor | Two round-trips per step | One round-trip per step |


---

## Section 4: Length Semantics

Three precise fields replace all vague "context_len" / "context_lens" usage.

### Core Definitions

| Field | Symbol | Meaning |
|-------|--------|---------|
| `cached_len_before` | $L_{cache}$ | Number of tokens already in KV cache pool for this sequence BEFORE this step |
| `query_len` | $L_{query}$ | Number of tokens processed THIS step (1 for decode, chunk_size for prefill) |
| `kv_len_after` | $L_{after}$ | Total KV length AFTER this step = `cached_len_before + query_len` |

### Per-Operation Values

| Operation | cached_len_before | query_len | kv_len_after |
|-----------|------------------|-----------|--------------|
| First prefill chunk (no prefix cache) | 0 | chunk_size | chunk_size |
| Chunked prefill (has prefix) | prefill_cursor | chunk_size | prefill_cursor + chunk_size |
| Decode (single step) | len(prompt) + num_generated | 1 | len(prompt) + num_generated + 1 |
| Prefix-cached first chunk | matched_token_count | chunk_size | matched_token_count + chunk_size |

### Position Computation

```
For prefill token i (0-indexed within chunk):
  position = cached_len_before + i
  slot_mapping = block_id * block_size + offset
    where block_id = block_table[position // block_size]
          offset   = position % block_size

For decode:
  position = cached_len_before   # = current total KV length
  slot_mapping = block_id * block_size + offset
```

### KV Pool Slot Index

```
If slot_mapping is a flat index into the pool's [num_blocks * block_size] axis:
  key_cache[layer].view(num_blocks * block_size, num_kv_heads, head_dim)[slot_mapping]
```

### Softmax Semantic

For each attention computation within a group, the softmax normalization is over all tokens in the KV sequence for that sequence:
- Prefill: over ALL `kv_len_after` tokens (prefix + current chunk)
- Decode: over ALL `kv_len_after` tokens (all cached tokens including current write)


---

## Section 5: Revised Metadata Definitions

### ModelInput (single per-step dataclass)

```python
@dataclass
class AttentionGroup:
    """A group of sequences that share the same attention type."""
    seq_indices: List[int]            # indices into the flattened seq list
    attention_type: str               # "prefill_ref" | "prefill_gpu" | "decode_gpu"
    cached_len_before: torch.Tensor   # [num_seqs_in_group]
    query_len: torch.Tensor           # [num_seqs_in_group]
    kv_len_after: torch.Tensor        # [num_seqs_in_group]

@dataclass
class AttentionMetadata:
    groups: List[AttentionGroup]

    # --- Prefill fields (sum of all prefill groups) ---
    prefill_slot_mapping: torch.Tensor    # [num_prefill_tokens]
    prefill_block_tables: torch.Tensor    # [num_prefill_seqs, max_blocks_per_seq]
    prefill_positions: torch.Tensor       # [num_prefill_tokens]

    # --- Decode fields (sum of all decode groups) ---
    decode_block_tables: torch.Tensor     # [num_decode_seqs, max_blocks_per_seq]
    decode_slot_mapping: torch.Tensor     # [num_decode_tokens]
    decode_positions: torch.Tensor        # [num_decode_tokens]

    # --- Shared ---
    block_size: int
    num_kv_heads: int
    head_dim: int

@dataclass
class ModelInput:
    input_ids: torch.Tensor               # [num_batched_tokens] concatenated
    positions: torch.Tensor               # [num_batched_tokens]
    slot_mapping: torch.Tensor            # [num_batched_tokens]
    attn_metadata: AttentionMetadata
    # Indices into input_ids where LM head is needed
    sample_token_indices: torch.Tensor    # [num_sample_positions] (LongTensor)
```

### Construction via ModelInputBuilder

```python
class ModelInputBuilder:
    def __init__(self, block_manager: BlockManager, config: ModelConfig):
        self._block_manager = block_manager
        self._config = config

    def build(
        self,
        prefill_seqs: List[Sequence],
        decode_seqs: List[Sequence],
    ) -> ModelInput:
        """Build ModelInput from scheduled sequences.

        Steps:
        1. For prefill seqs:
           - cached_len_before = seq.prefill_cursor
           - query_len = min(chunk_size, remaining)
           - For each token position: compute slot_mapping via block_manager
           - Concatenate input_ids, positions, slot_mapping
        2. For decode seqs:
           - cached_len_before = len(prompt) + num_generated
           - query_len = 1
           - slot_mapping for the single new token
           - input_id = last output token
           - position = cached_len_before
        3. Build block_table tensors (padded to max_blocks_per_seq)
        4. Build AttentionGroups: prefill_ref group, prefill_gpu group, decode_gpu group
        5. Compute sample_token_indices:
           - For each prefill seq: index of its LAST token if prefill completes this step
           - For each decode seq: index of its single token
        6. Concatenate everything into ModelInput
        """
```

### What Was Removed (Deduplication)

| Removed Item | Reason |
|-------------|--------|
| `cache/cache_metadata.py` | Not needed; all metadata in `model_runner/model_input.py` |
| `attention/metadata.py` | Merged into `model_runner/model_input.py` |
| `KVCacheMetadata` dataclass | Not needed; KVCachePool is self-describing |
| `AttentionMetadata.decode_context_lens` | Replaced by `cached_len_before` in AttentionGroup |
| `AttentionMetadata.prefix_context_lens` | Replaced by `cached_len_before` |
| Separate prefill/decode input_ids | Single concatenated `input_ids` tensor |


---

## Section 6: Selective LM Head (sample_token_indices)

### Problem

For chunked prefill, a single step may process many prefill tokens (e.g., chunk_size=16). Of those 16 tokens, only the LAST token of each sequence needs sampling (to produce the first generated token). Computing the LM head for all 16 prefill tokens wastes computation on intermediate tokens whose logits are never used.

### Solution

The `ModelInput.sample_token_indices` tensor specifies which positions in the concatenated `input_ids` tensor need LM head computation.

```
Example:
  Prefill seq A: chunk of 4 tokens    (positions 0-3 in input_ids)
  Prefill seq B: chunk of 4 tokens    (positions 4-7 in input_ids)
  Decode seq C: 1 token                (position 8 in input_ids)
  Decode seq D: 1 token                (position 9 in input_ids)

  sample_token_indices = [3, 7, 8, 9]
  # Prefill A last token (index 3), Prefill B last token (index 7),
  # Decode C token (index 8), Decode D token (index 9)
```

### Rules

| Scenario | sample_token_indices includes |
|----------|------------------------------|
| Prefill completes this step | Last token of that sequence |
| Prefill continues next step | None (no tokens from this sequence) |
| Decode | The decode token (every decode step needs sampling) |

### Implementation in ModelInputBuilder

```python
for idx, seq in enumerate(prefill_seqs):
    start_idx = prefill_token_offset  # current position in concatenated input_ids
    chunk_len = seq.prefill_chunk_len
    if seq.prefill_cursor + chunk_len >= len(seq.prompt_token_ids):
        # Prefill finishes this step → sample the last token
        sample_indices.append(start_idx + chunk_len - 1)
    prefill_token_offset += chunk_len

for idx, seq in enumerate(decode_seqs):
    sample_indices.append(decode_token_offset + idx)
```

### ModelRunner Integration

```python
# After final RMSNorm:
# hidden_states shape: [num_batched_tokens, hidden_size]
# Select only positions that need logits:
sample_hidden = hidden_states[sample_token_indices]  # [num_sample_positions, hidden_size]
logits = lm_head(sample_hidden)                      # [num_sample_positions, vocab_size]
```


---

## Section 7: Mixed Prefill + Decode Layer Flow

### Single Layer Forward (conceptual)

```python
def forward_layer(
    self,
    layer_idx: int,
    hidden_states: torch.Tensor,    # [num_batched_tokens, hidden_size]
    positions: torch.Tensor,        # [num_batched_tokens]
    attn_metadata: AttentionMetadata,
    kv_cache_pool: KVCachePool,
) -> torch.Tensor:
    # --- 1. Pre-attention ---
    residual = hidden_states
    hidden_states = self.input_layernorm(layers[layer_idx])(hidden_states)

    # --- 2. QKV projection ---
    qkv = self.qkv_proj(layers[layer_idx])(hidden_states)
    q, k, v = qkv_split(qkv)  # Q: [bt, num_heads, head_dim], K/V: [bt, num_kv_heads, hd]

    # --- 3. RoPE ---
    q, k = apply_rotary_emb(q, k, positions, self.rope_cache)

    # --- 4. KV Cache Write (all tokens) ---
    self.attention_backend.write_kv_cache(
        layer_idx, k, v, model_input.slot_mapping
    )

    # --- 5. Per-Group Attention ---
    attn_output = torch.zeros_like(q)  # [num_batched_tokens, num_heads, head_dim]

    for group in attn_metadata.groups:
        group_q = q[group.seq_indices]
        if group.type == "prefill_ref":
            # Reference path: gather prefix KV from pool, concat, SDPA
            out = self.attn_ref.prefill_attention(
                layer_idx, group_q, k, v,
                group, kv_cache_pool
            )
        elif group.type == "prefill_gpu":
            # GPU PagedAttention prefill (unified online softmax)
            out = self.attn_gpu.prefill_attention(
                layer_idx, group_q, k, v,
                group, kv_cache_pool
            )
        elif group.type == "decode_gpu":
            # PagedAttention decode (read-only from pool)
            out = self.attn_gpu.decode_attention(
                layer_idx, group_q,
                group, kv_cache_pool
            )
        attn_output[group.seq_indices] = out

    # --- 6. Post-attention ---
    hidden_states = self.output_proj(layers[layer_idx])(attn_output) + residual
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(layers[layer_idx])(hidden_states)
    hidden_states = self.mlp(layers[layer_idx])(hidden_states) + residual

    return hidden_states
```

### Key Properties

| Property | Implementation |
|----------|---------------|
| Single layer loop | Yes — `for layer_idx in range(num_layers)` wraps all sequences |
| Mixed mode | All tokens go through QKV + RoPE + KV write together |
| Attention dispatch | Per-group, based on group.type |
| No per-seq Python loop | All operations are batched tensors |
| No HF past_key_values | KV write goes to pool via slot_mapping |

### Why write_kv_cache BEFORE attention?

The write happens before attention for both prefill and decode:
- **Prefill**: attention needs the just-written chunk tokens to be available for the full attention computation (prefix + chunk).
- **Decode**: attention needs the single new token's K/V to be in the pool (for consistency with the online softmax accumulation).

This is the same ordering as vLLM and other production systems.


---

## Section 8: Chunked Prefill — Two Paths

### 8.1 Reference Path (paged_attention_ref.py — TESTS ONLY)

For test comparison against QwenExecutor. Not used in production.

Algorithm per sequence:
```
1. Gather existing prefix K/V from pool via block_table
   prefix_k = gather(pool.key_cache[layer], block_table, up to cached_len_before tokens)
   prefix_v = gather(pool.value_cache[layer], block_table, up to cached_len_before tokens)

2. Concatenate with current chunk K/V:
   all_k = cat([prefix_k, current_chunk_k], dim=0)  # [kv_len_after, num_kv_heads, head_dim]
   all_v = cat([prefix_v, current_chunk_v], dim=0)

3. Compute SDPA with causal mask:
   attn_output = sdpa(q, all_k, all_v, causal_mask=True)
   # Causal mask only applied to current chunk portion
   # Prefix portion is fully visible (no mask)
```

Implementation via `torch.nn.functional.scaled_dot_product_attention`:
```python
q = q.unsqueeze(0)                          # [1, num_heads, query_len, head_dim]
k = all_k.unsqueeze(0)                      # [1, num_kv_heads, kv_len_after, head_dim]
v = all_v.unsqueeze(0)
attn = F.scaled_dot_product_attention(q, k, v, is_causal=(query_len > 1))
# is_causal=True gives upper-triangular mask within the query window
```

### 8.2 Production GPU Path (paged_attention_gpu.py — MANDATORY)

For production prefill. Avoids per-sequence gather. Uses block-table driven access + online softmax over all tokens.

Algorithm per group of sequences:
```
Inputs:
  - Q: [num_tokens_in_group, num_heads, head_dim]  (flattened chunk tokens of all seqs)
  - group.cached_len_before: [num_seqs]
  - group.query_len: [num_seqs]
  - block_tables: [num_seqs, max_blocks]
  - kv_cache_pool: the global pool

For each sequence s in group:
  seq_tokens = slice Q at offset determined by query_len prefix sum
  q_head_dim = head_dim same as decode
  
  # Phase 1: Process PREFIX tokens (already in pool)
  # Same as decode attention: read blocks via block_table, online softmax
  # For each block in block_table[s]:
  #   block_k, block_v = pool[block_id]
  #   # Only read up to cached_len_before % block_size for last block
  #   valid_tokens = min(block_size, cached_len_before[s] - block_idx * block_size)
  #   scores = Q @ block_k[:valid_tokens] / sqrt(head_dim)
  #   Online softmax accumulate
  
  # Phase 2: Process CURRENT CHUNK tokens (just written via write_kv_cache)
  # Read back the chunk's own K/V from pool
  # For each chunk token position within the same sequence:
  #   scores = Q @ K_chunk_transposed / sqrt(head_dim), causal mask applied
  #   Online softmax continue
  
  # Phase 3: Final softmax normalization
  output = o / d  # From online softmax accumulator
```

**Key invariant**: The online softmax spans ALL `kv_len_after` tokens (prefix + chunk) in one unified pass, not two separate softmaxes. This matches the mathematical definition.

### 8.3 When to Use Each Path

| Scenario | Path | Reason |
|----------|------|--------|
| Unittest: prefill correctness | Reference | Simple, debuggable |
| Unittest: chunked vs full equivalence | Reference | Deterministic SDPA |
| Integration test: PagedExecutor | GPU | Must test production path |
| Benchmark / Real inference | GPU | Performance |
| query_len == 1 (single-token prefill) | GPU | Falls through to decode-like kernel |

### 8.4 Validation

```python
# Test: reference and GPU paths produce identical results for same input
# (for all attention types except the GPU decode path):
ref_output = reference_path(q, k, v, group)
gpu_output = gpu_path(q, k, v, group)
assert_allclose(ref_output, gpu_output, atol=1e-3)
```


---

## Section 9: GPU PagedAttention as Mandatory Backend

### Architecture

```
AttentionBackend (abstract interface)
  ├── AttentionBackendRef
  │    ├── Uses SDPA (gather + concat, per-group)
  │    └── __init__.py tie: only importable when "paged_attention_ref" mode set
  │
  └── AttentionBackendGPU (DEFAULT, MANDATORY for production)
       ├── allocate_pool() → KVCachePool
       ├── write_kv_cache(layer, k, v, slot_mapping)
       ├── prefill_attention(...)  # GPU Triton kernel
       └── decode_attention(...)   # GPU Triton kernel
```

### Selection Logic

```python
class AttentionBackend:
    @staticmethod
    def create(
        config: ModelConfig,
        backend_type: str = "gpu",  # default is GPU
    ) -> "AttentionBackend":
        if backend_type == "ref":
            return AttentionBackendRef(config)
        elif backend_type == "gpu":
            return AttentionBackendGPU(config)
        else:
            raise ValueError(f"Unknown backend: {backend_type}")
```

- `backend_type="gpu"` is the DEFAULT
- `backend_type="ref"` is for test comparison ONLY
- NO automatic fallback from GPU to ref
- If GPU backend init fails (e.g., no CUDA), the error propagates — no silent degrade

### GPU Backend Guard

```python
class AttentionBackendGPU:
    def __init__(self, config: ModelConfig):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU PagedAttention backend requires CUDA. "
                "Use backend_type='ref' for CPU testing."
            )
        # Verify Triton is importable
        try:
            import triton  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "GPU PagedAttention backend requires Triton. "
                "Install with: pip install triton"
            )
```

### AttentionBackend Interface (Revised)

```python
class AttentionBackend(ABC):
    @abstractmethod
    def allocate_pool(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> KVCachePool:
        ...

    @abstractmethod
    def write_kv_cache(
        self,
        layer_idx: int,
        key: torch.Tensor,      # [num_tokens, num_kv_heads, head_dim]
        value: torch.Tensor,    # [num_tokens, num_kv_heads, head_dim]
        slot_mapping: torch.Tensor,  # [num_tokens] (long)
    ) -> None:
        """Scatter write K/V to cache pool at specified slots."""
        ...

    @abstractmethod
    def decode_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,            # [num_decode_tokens, num_heads, head_dim]
        attn_metadata: AttentionMetadata,
        kv_cache_pool: KVCachePool,
    ) -> torch.Tensor:
        """Paged decode attention. All sequences processed together."""
        ...

    @abstractmethod
    def prefill_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,            # [num_prefill_tokens, num_heads, head_dim]
        key: torch.Tensor,              # [num_prefill_tokens, num_kv_heads, head_dim]
        value: torch.Tensor,            # [num_prefill_tokens, num_kv_heads, head_dim]
        attn_metadata: AttentionMetadata,
        kv_cache_pool: KVCachePool,
    ) -> torch.Tensor:
        """Prefill attention with paged prefix."""
        ...
```


---

## Section 10: Single BlockTable Truth Source

### Problem

Current code:
```python
# BlockManager._tables: Dict[str, BlockTable]  ← authoritative
# Sequence.block_table: List[int]               ← mutable mirror copy
# After each ensure_block():
#   seq.block_table = manager._tables[seq.seq_id].get_block_ids()
```

This dual source can diverge if one is modified without the other.

### Solution: Remove Sequence.block_table Entirely

```python
# Sequence class (NO block_table field):
@dataclass
class Sequence:
    seq_id: str
    group_id: str
    prompt_token_ids: List[int]
    output_token_ids: List[int]
    sampling_params: SamplingParams
    status: SequenceStatus
    prefill_cursor: int           # only used by scheduler/builder
    num_generated_tokens: int
    ...  # no block_table field
```

### All Consumers Switch to BlockManager

| Consumer | Current Access | Revised Access |
|----------|---------------|----------------|
| BlockManager.ensure_block() | Writes seq.block_table after mutation | Writes self._tables[seq_id] only |
| BlockManager.free() | Reads seq.seq_id to lookup | Uses seq.seq_id to lookup |
| ModelInputBuilder (new) | Reads seq.block_table for slot mapping | Calls block_manager.get_block_table(seq_id) |
| Metrics/Profiler | Reads seq.block_table to compute blocks | Calls block_manager.get_block_table(seq_id) |

### BlockManager Query API

```python
class BlockManager:
    def get_block_table(self, seq_id: str) -> List[int]:
        """Read-only view of block IDs for a sequence."""
        table = self._tables.get(seq_id)
        if table is None:
            return []
        return table.get_block_ids()  # Read-only: returns copy (cheap, O(num_blocks))
```

### Trade-off

`get_block_table()` returns a copy of the block ID list. For typical sequences (tens to low hundreds of blocks), this is O(100) int copy per call, which is negligible compared to GPU kernel time. If profiling shows this is a bottleneck, we can switch to a shared reference or expose the internal list directly.


---

## Section 11: Memory Budget & Block Lifecycle

### 11.1 Memory Budget Calculation

**Removed**: All hardcoded large numbers (1.1M, 49M, 37k blocks) from the original design.

**Replaced with**: Profile-run based initialization.

#### Initialization Flow

```python
def compute_num_gpu_blocks(
    model: nn.Module,
    config: ModelConfig,
    device: torch.device,
    gpu_memory_utilization: float = 0.90,
) -> int:
    # 1. Measure free memory before model load
    free_before, total = torch.cuda.mem_get_info(device)

    # 2. Load model weights (done outside this function)
    # After load, measure used:
    free_after, _ = torch.cuda.mem_get_info(device)
    weights_memory = free_before - free_after

    # 3. Run a short warmup forward pass
    dummy_input = torch.randint(0, config.vocab_size, (1, 1), device=device)
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize()

    # 4. Measure peak usage after warmup
    free_final, _ = torch.cuda.mem_get_info(device)
    runtime_overhead = free_after - free_final  # CUDA context, activations peak

    # 5. Compute available memory for KV pool
    available = free_final * gpu_memory_utilization
    # Or more conservatively:
    # available = (total - weights_memory - runtime_overhead) * gpu_memory_utilization

    # 6. KV bytes per block
    bytes_per_block = (
        config.num_layers
        * 2  # key + value
        * config.block_size
        * config.num_kv_heads
        * config.head_dim
        * 2  # fp16 = 2 bytes
    )

    num_blocks = int(available / bytes_per_block)
    return max(num_blocks, 1)  # at least 1 block
```

#### Estimated Numbers for RTX 3060 (12 GB)

```
Total memory:      12,288 MB
Model weights:      ~1,024 MB (Qwen2.5-0.5B fp16)
CUDA + runtime:     ~1,500 MB
Available for KV:   ~9,764 MB (before utilization factor)

With utilization=0.90:
  usable = ~8,788 MB → equivalent to ~46k blocks of size 16
  (This is a back-of-envelope check. Actual value depends on runtime measurement.)
```

**Critical**: The actual block count is determined at runtime by `compute_num_gpu_blocks()`, not by static calculation. The 46k number is illustrative only.

### 11.2 Block Lifecycle: Complete Release Ownership

| Event | Who Calls free() | When | What Happens |
|-------|-----------------|------|-------------|
| **Normal finish** | EngineCore.step() | After execute() returns, for sequences in ScheduleResult.finished_groups | `block_manager.free(seq_id)` → allocator decrements ref_counts, blocks return to free pool |
| **Scheduler timeout** | EngineCore._check_timeouts() | At start of each step() | `scheduler.free_seq(seq_id)` → triggers block_manager.free(seq_id) |
| **Scheduler reject** | EngineCore.step() | After schedule() returns rejected_groups | `block_manager.free(seq_id)` for rejected sequences (already allocated in allocate_for_seq) |
| **Client cancel** | LLMEngine.cancel_request() | On demand | `scheduler.free_seq(seq_id)` → `engine_core._cleanup_seq(seq_id)` → block_manager.free(seq_id) |
| **HTTP disconnect** | StreamManager.on_disconnect() | Connection drop | `LLMEngine.cancel_request(request_id)` → same path as cancel |
| **Exception in execute()** | EngineCore.step() (try/finally) | Exception handler | Iterate all running seqs: `block_manager.free(seq_id)` |
| **Graceful shutdown** | LLMEngine.shutdown() | Engine stop | Iterate all remaining seqs: `block_manager.free(seq_id)` |

#### Code Sketch

```python
def _cleanup_sequence(self, seq_id: str):
    """Central cleanup point for all paths."""
    self.block_manager.free(seq_id)
    # No executor cleanup needed (no _seq_kv in PagedExecutor)

def step(self):
    try:
        # 1. Timeouts (can produce finished groups indirectly)
        self._check_timeouts()

        # 2. Schedule
        result = self.scheduler.schedule()

        # 3. Rejections
        for sg in result.rejected_groups:
            for seq in sg.seqs:
                self._cleanup_sequence(seq.seq_id)

        # 4. Execute (if any work)
        if result.num_batched_tokens > 0:
            model_input = self.input_builder.build(
                result.scheduled_prefill_groups,
                result.scheduled_decode_groups,
            )
            outputs = self.executor.execute(model_input)
            self._process_outputs(outputs)

        # 5. Finish cleanup
        for sg in result.finished_groups:
            for seq in sg.seqs:
                self._cleanup_sequence(seq.seq_id)
                self.metrics.register_sequence(seq)

    except Exception:
        # 6. Exception recovery
        for seq_id in list(self.block_manager._tables.keys()):
            self._cleanup_sequence(seq_id)
        raise

    finally:
        self.metrics.record_step(...)
```


---

## Section 12: KV Cache Tensor Layout & ConfigAdapter

### 12.1 Pool Shape

```
key_cache:   [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
value_cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
dtype: torch.float16
```

### 12.2 GPU PagedAttention Kernel (Triton) — Revised

```python
def paged_attention_decode(
    query,              # [num_decode_seqs, num_heads, head_dim]
    key_cache,          # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache,        # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables,       # [num_decode_seqs, max_blocks_per_seq] (torch.long, -1 padded)
    context_lens,       # [num_decode_seqs] (torch.long) = cached_len_before + query_len
    scale: float,       # 1 / sqrt(head_dim)
    block_size: int,
    num_kv_heads: int,
    num_query_heads: int,
    max_context_len: int,  # max over batch, for loop bound
) -> torch.Tensor:     # [num_decode_seqs, num_heads, head_dim]
```

One kernel launch for the entire decode batch. Grid: (num_decode_seqs, num_query_heads). Each program processes one (seq, q_head) pair.

### 12.3 Prefill GPU Kernel (Triton)

For prefill, the GPU kernel must handle multiple query tokens per sequence (query_len > 1). This is more complex than decode (single query). The kernel processes the prefix tokens from the pool (block-by-block, same as decode) and then the chunk tokens.

#### First Version Strategy

Phase 1 (V1.0): Use the reference SDPA path for prefill. The GPU kernel is decode-only. This avoids the complexity of a multi-query prefill Triton kernel in the first iteration.

Phase 2 (V1.1): Implement a full Triton prefill kernel that handles both prefix + chunk in one online-softmax pass.

This is acceptable because:
- Decode is the throughput-critical path (runs every step)
- Prefill is less frequent (only when new requests arrive)
- The reference prefill path still uses GPU (SDPA), just not a custom kernel

**No silent fallback**: The code explicitly chooses the reference path for prefill; it does NOT fall back from a failed GPU kernel attempt.

### 12.4 ConfigAdapter

```python
@dataclass
class ModelConfig:
    model_type: str           # "qwen2"
    num_layers: int
    hidden_size: int
    num_heads: int            # Q heads
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    hidden_act: str           # "silu"
    intermediate_size: int
    tie_word_embeddings: bool
    dtype: torch.dtype        # model native dtype (bf16 for Qwen2.5)
    activation_dtype: torch.dtype  # torch.float16 (for computation)
```

All layers read from ModelConfig. ConfigAdapter reads from HF config.json dynamically.


---

## Section 13: Streamlined File List & Phase Plan

### 13.1 Complete File Inventory

#### NEW files (10)

| File | Purpose |
|------|---------|
| `cache/kv_cache_pool.py` | GPU tensor pool allocation |
| `attention/backend.py` | AttentionBackend interface |
| `attention/paged_attention_ref.py` | PyTorch reference attention (tests only) |
| `attention/paged_attention_gpu.py` | Triton PagedAttention kernel (production) |
| `model_runner/base.py` | ModelRunner interface |
| `model_runner/config_adapter.py` | Dynamic model config from HF |
| `model_runner/qwen_runner.py` | Qwen-specific layer loop |
| `engine/input_builder.py` | ModelInputBuilder (decoupled from BlockManager) |
| `executor/paged_executor.py` | PagedExecutor wrapping ModelRunner |
| `tests/test_paged_attention.py` | PagedAttention test suite |

#### ENHANCED files (5)

| File | Changes |
|------|---------|
| `cache/allocator.py` | Add invariant checks, max_blocks exposure |
| `cache/block_table.py` | Add tensor export, remove mirror sync |
| `cache/manager.py` | Simplify to read-only query methods |
| `executor/base.py` | Replace prefill()+decode() with execute() |
| `config.py` | Add paged executor type, backend selection |

#### ADAPTED files (1)

| File | Changes |
|------|---------|
| `engine/engine_core.py` | Use ModelInputBuilder + single executor.execute() |

#### UNCHANGED files (all others)

| Category | Files |
|----------|-------|
| Scheduler | scheduler.py, schedule_result.py |
| Sequence | sequence.py, sequence_group.py, status.py, sampling_params.py |
| Engine | engine.py, metrics.py, stage_profiler.py |
| Serving | All files in serving/ |
| Workers | fake_worker.py, qwen_worker.py |
| Executor | executor.py (FakeModelExecutor), qwen_executor.py (reference) |
| Model | fake_model.py |
| Prefix cache | prefix_cache.py (DISABLED by default) |

#### REMOVED (relative to original design proposal)

| File | Rationale |
|------|-----------|
| `cache/cache_metadata.py` | Merged into model_runner/model_input.py |
| `attention/metadata.py` | Merged into model_runner/model_input.py |
| `model_runner/model_input.py` | Content lives in `model_runner/base.py` as ModelInput dataclass |
| `worker/paged_worker.py` | Not needed: PagedExecutor is self-contained |

### 13.2 Revised Phase Plan (10 phases, was 12)

| Phase | Name | Files | Dependencies |
|-------|------|-------|--------------|
| **1** | **Metadata & Interfaces** | `input_builder.py`, `backend.py`, `base.py`, `config_adapter.py`, + enhanced `allocator.py`, `block_table.py`, `manager.py`, `executor/base.py` | None |
| **2** | **GPU KV Pool** | `kv_cache_pool.py` | Phase 1 (interfaces) |
| **3** | **Cache Write Reference** | `paged_attention_ref.py` (write + prefill SDPA) | Phase 2 (pool) |
| **4** | **PagedAttention Ref** | `paged_attention_ref.py` (decode paged read) | Phase 3 |
| **5** | **Triton Decode Kernel** | `paged_attention_gpu.py` (decode only) | Phase 4 (for comparison) |
| **6** | **Qwen ModelRunner** | `qwen_runner.py` (single layer, single seq first) | Phase 1, 5 |
| **7** | **Full Prefill** | `qwen_runner.py` (full prefill + paged K/V write) | Phase 6 |
| **8** | **Batched Decode** | `qwen_runner.py`, `paged_executor.py` | Phase 7 |
| **9** | **Chunked Prefill** | `paged_attention_ref.py`, `paged_attention_gpu.py` | Phase 8 |
| **10** | **Integration** | `engine_core.py`, `config.py` | Phase 9 |

Each phase produces working, tested code. No phase depends on future phases.


---

## Section 14: Revised Risk Table

| Risk | Severity | Mitigation |
|------|----------|------------|
| **GPU kernel crash / OOB memory access** | **HIGH** | Extensive unit tests with random data; Triton auto-tuning; bounds-check every index; use PyTorch reference as comparison |
| **Unified execution correctness** (mixed prefill+decode in single layer loop) | **HIGH** | Phase 6 tests single-sequence first; Phase 7-8 add batching; compare hidden states and logits against QwenExecutor |
| **Chunked prefill edge cases** | **HIGH** | Test non-divisible prompt lengths (19, chunk=8); test single-token chunk; test block_size boundaries; compare full vs chunked for exact equivalence |
| **RoPE integration mismatch** | **HIGH** | Use same cos/sin precomputation as HF; test single decoder layer against HF output |
| **GQA head mapping error** (14 Q heads → 2 KV heads, 7:1 ratio) | **HIGH** | Verify kv_head = q_head // ratio formula; test with batch of identical inputs and compare attention output |
| **Resource leak** (blocks not freed on cancel/timeout/exception/disconnect) | **HIGH** | Centralized cleanup in _cleanup_sequence(); try/finally in step(); test all failure paths with fault injection |
| **Memory budget too aggressive** | **HIGH** | Profile-run based initialization; conservative gpu_memory_utilization=0.90; test with OOM scenarios |
| **Triton kernel too slow on RTX 3060** | **MEDIUM** | Optimize block sizes per GPU; if decode kernel is slower than reference SDPA, use reference path for prefill (acceptable as stated in 12.3) |
| **Block table export format mismatch** | **MEDIUM** | Validate tensor shape, dtype, padding (-1 sentinel) before kernel launch |
| **Prefix cache interference** (stale blocks, uninitialized KV data) | **MEDIUM** | Disabled by default with explicit guard. Only enable after COW implementation is verified |
| **Existing tests break** | **LOW** | FakeModelExecutor unchanged; BlockManager API changes are additive (query methods + old path removed); all 214 tests must pass after each phase |
| **ModelInputBuilder blocks per-step latency** | **LOW** | Pure Python + torch.tensor creation; expected <0.1ms for batch=4; measure and optimize if needed |

### Risk Changes from Original

| Risk | Original Severity | Revised Severity | Why |
|------|------------------|-----------------|-----|
| GPU kernel crash | Medium → **HIGH** | Triton kernel is now MANDATORY production path, not optional |
| Chunked prefill | High → **HIGH** (unchanged) | — |
| RoPE integration | High → **HIGH** (unchanged) | — |
| Resource leak | Not listed → **HIGH** | New risk from review item #11 |
| Memory budget | Medium → **HIGH** | Wrong numbers removed; profile-based init is new and untested |
| GPU kernel slow | Low → **MEDIUM** | Kernel is mandatory; no fallback means performance must be acceptable |


---

## Section 15: Revised Completion Criteria

Original 16 criteria are preserved. 8 new criteria added (marked **NEW**).

### Core Functionality (must all be true)

1. K/V data lives in GPU physical block pool (not _seq_kv dict)
2. BlockAllocator block IDs determine real tensor addresses (block_id → pool index)
3. BlockTable tensors are kernel input (not metadata-only)
4. Slot mapping determines new K/V write positions
5. PagedAttention reads from discrete physical blocks (not contiguous padding)
6. **No batch-wide KV padding** — each sequence uses only its own blocks
7. Paged backend does NOT use HF past_key_values
8. Different context lengths decoded in same batch (no padding to max_len)
9. Physical blocks can be non-contiguous (block_table handles indirection)
10. **One ModelRunner execution handles multiple requests simultaneously**
11. Blocks can be freed and reused (ref_count correctly tracks lifecycle)
12. Output matches QwenExecutor reference (argmax tokens identical for same input)
13. Chunked prefill correctly reads paged prefix and produces identical output to single-shot full prefill

### No Per-Seq HF Model Call

14. **NEW**: No per-sequence Python loop around model() in the paged execution path
15. **NEW**: Single `execute()` call per EngineCore.step() (not separate prefill()+decode())

### Metadata Cleanliness

16. **NEW**: Sequence.block_table field is removed (single truth source in BlockManager)
17. **NEW**: ModelInputBuilder is a separate class, not BlockManager responsibility

### Attention Backend

18. **NEW**: Reference and production GPU attention paths are distinct — reference is importable only for tests
19. **NEW**: GPU PagedAttention kernel (Triton) is tested against reference for both prefill and decode
20. **NEW**: No silent fallback from GPU to CPU/reference — if GPU backend fails, error propagates

### LM Head Efficiency

21. **NEW**: sample_token_indices is used — LM head is only computed at positions that need sampling

### Resource Lifecycle

22. **NEW**: All release paths are documented and tested: normal finish, cancel, timeout, exception, disconnect, shutdown

### Existing Tests

23. All 214+ existing tests pass (FakeModelExecutor path unchanged)
24. New test suite (paged_attention, paged_executor, chunked_prefill) runs green

### Documentation

25. Documentation clearly states what's implemented vs planned


---

## Section 16: Architecture Decisions Still Requiring User Confirmation

The following decisions are NOT finalized in this revision and require explicit user approval before implementation begins.

### A. Prefill GPU Kernel (V1.0 vs V1.1)

**Question**: In V1.0, should the prefill path use the reference implementation (SDPA per-seq gather + concat) for simplicity, deferring the full Triton prefill kernel to V1.1?

**Trade-off**:
- Using SDPA for prefill in V1.0: faster to ship, still GPU-accelerated, but the prefill step has a per-group gather loop (not fully batched)
- Full Triton prefill in V1.0: more complex, longer development time, but fully batched (no per-group Python loop in attention)

**Recommendation**: Use SDPA for prefill in V1.0. The decode path is the throughput-critical path. Prefill runs less frequently.

### B. Prefix Cache: Disabled vs Removed

**Question**: Should the prefix cache mechanism be (a) disabled by default but present in code, or (b) removed entirely and re-added later?

**Current approach**: (a) Disabled by default with guard. Keeps the code structure for future re-enablement after COW is implemented.

**Alternative**: (b) Remove prefix_cache.py entirely. It has known risks (stale entries, block registered before KV write). Re-add from git history when needed.

**Recommendation**: Option (a) — disabled by default. The code is stable and tested. Removing and re-adding is churn.

### C. Dtype Strategy: Full fp16 vs Mixed Weight Precision

**Question**: Should the paged backend cast model weights to fp16 at load time, or keep bf16 weights and cast activations to fp16 for the KV path?

**Chosen**: Keep weights at their native precision (bf16 for Qwen2.5). KV cache pool is fp16. During the forward pass, activations are cast to fp16 when written to the KV cache. QKV projections use the native weight dtype.

**User to confirm**: Is this acceptable, or should the entire model be forced to fp16?

### D. PagedWorker Necessity

**Question**: As re-evaluated per review #12, should there be a separate PagedWorker, or is the existing QwenWorker sufficient with a `backend_type` parameter?

**Current position**: PagedWorker is NOT needed. The QwenWorker already provides model loading and device setup. Adding a `backend_type` parameter to QwenWorker and having it return PagedExecutor when `backend_type="paged"` is simpler and avoids code duplication.

### E. BlockManager API Boundary

**Question**: The design separates metadata construction (ModelInputBuilder) from BlockManager. But BlockManager still needs to expose block allocation on-demand (needs_allocation). Is this too much coupling?

**Alternative**: Have the scheduler pre-allocate all needed blocks before execute(), and ModelInputBuilder only reads existing blocks. This would make BlockManager a read-only query interface during execute().

**Recommendation**: Keep needs_allocation as a BlockManager method. It is a read-or-allocate pattern (idempotent), not a state mutation during execution. The allocation happens during ModelInputBuilder.build(), which runs BEFORE executor.execute().

### F. Test Strategy: GPU Test Requisites

**Question**: Can GPU tests (Phase 5+ Triton kernel) use the RTX 3060 directly, or should CI run on CPU with the reference backend?

**Recommendation**: GPU tests run on the development machine (RTX 3060) during development. CI tests use the reference backend on CPU for cross-checks. A small subset of GPU tests can be tagged for GPU CI when available.

---

**This revision is complete. Pausing for confirmation. No coding has begun.**

