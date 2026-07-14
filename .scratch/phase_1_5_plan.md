# Phase 1.5: Unified Execute Wiring (REVISION 1)

## Overview

Wire `ModelInputBuilder` + unified `execute()` into `EngineCore.step()`,
replacing the separate `executor.prefill()` / `executor.decode()` call
path for the paged executor. Old executors (Fake, Qwen) keep their
existing APIs as backward-compatible wrappers.

**Key constraint (user-verified):** All executors share the same
`execute(model_input: ModelInput) -> ModelRunnerOutput` signature.
No executor receives Python `Sequence` objects. The `ModelInput` carries
a lightweight read-only execution snapshot (`SequenceExecutionInfo`) for
the FakeExecutor's simulation needs.

---

## 1. ModelRunnerOutput (`mini_vllm/model_runner/base.py`)

```python
@dataclass(frozen=True)
class ModelRunnerOutput:
    sampled_token_ids: List[int]
    sampled_sequence_ids: List[str]   # maps each sampled token to its sequence
```

Explicit mapping from sample → sequence_id. EngineCore iterates this
and updates each `Sequence` by its `seq_id`. No implicit ordering.

---

## 2. SequenceExecutionInfo (`mini_vllm/model_runner/base.py`)

```python
@dataclass(frozen=True)
class SequenceExecutionInfo:
    sequence_id: str
    phase: str                             # "prefill" or "decode"
    query_start: int                       # cached_len_before (absolute start position)
    query_len: int                         # number of tokens in this step
    cached_len_before: int                 # same as query_start, explicit
    kv_len_after: int                      # cached_len_before + query_len
    sample_output_index: Optional[int]     # index into sample_token_indices, or None
```

Placed **inside** `ModelInput`:

```python
@dataclass
class ModelInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    attn_metadata: AttentionMetadata
    sample_token_indices: torch.Tensor
    sequence_info: Tuple[SequenceExecutionInfo, ...]  # one per sequence, prefill then decode
```

Constraints:
- Read-only snapshot, immutable (`frozen=True`)
- Executor never touches `Sequence` objects
- FakeExecutor uses `sequence_id` and `phase` to drive its simulation
- PagedExecutor can ignore `sequence_info` entirely
- EngineCore writes back to `Sequence` by matching `sequence_id` in output

---

## 3. Executor Protocol (`mini_vllm/executor/base.py`)

Change `execute()` return type from `None` to `ModelRunnerOutput`:

```python
def execute(self, model_input: ModelInput) -> ModelRunnerOutput: ...
```

Same signature for all executors. No `Sequence` args.
Old `prefill()` / `decode()` kept for Qwen backward compat.

---

## 4. FakeModelExecutor.execute() (`mini_vllm/executor/executor.py`)

New `execute(model_input: ModelInput) -> ModelRunnerOutput`:

1. Iterates `model_input.sequence_info` to determine which sequences to simulate
2. For `phase="prefill"`: writes prompt tokens to fake KV at positions
   `[query_start, ..., query_start + query_len - 1]`
3. For sequences where `sample_output_index is not None`:
   - Completing prefill: produces first token via `_model.prefill_token()`
   - Decode: reads KV, produces next token via `_model.decode_token()`
4. Returns `ModelRunnerOutput(sampled_token_ids, sampled_sequence_ids)`

**Internal state management:** FakeExecutor maintains a dict
`self._sim_state: Dict[str, int]` mapping `sequence_id → last_output_token`
so it can produce deterministic tokens for decode without touching Sequence
objects.

Old `prefill()` / `decode()` remain unchanged for Qwen compat.

---

## 5. EngineCore (`mini_vllm/engine/engine_core.py`)

Changes to `step()`:

```python
# After collecting prefill_seqs and decode_seqs (same as today)
model_input = self._input_builder.build(prefill_seqs, decode_seqs)

if model_input.input_ids.numel() > 0:
    output = self._executor.execute(model_input)
    self._apply_model_output(output, prefill_seqs, decode_seqs)
```

New `_apply_model_output()`:
```python
def _apply_model_output(self, output, prefills, decodes):
    # Build lookup: sequence_id → Sequence
    seq_map = {}
    for s in prefills:
        seq_map[s.seq_id] = (s, "prefill", True)   # (seq, phase, may_be_first_token)
    for s in decodes:
        seq_map[s.seq_id] = (s, "decode", False)

    for token_id, seq_id in zip(output.sampled_token_ids, output.sampled_sequence_ids):
        entry = seq_map.get(seq_id)
        if entry is None:
            continue
        seq, phase, is_first = entry
        if phase == "prefill" and is_first:
            # Completing prefill — set first output token
            seq.output_token_ids = [token_id]
            seq.num_generated_tokens = 1
            seq.first_token_time = time.time()
            seq.status = Status.RUNNING
        else:
            # Decode — append
            seq.output_token_ids.append(token_id)
            seq.num_generated_tokens += 1
```

Wire `ModelInputBuilder` in `__init__`:
```python
from ..engine.input_builder import ModelInputBuilder
self._input_builder = ModelInputBuilder(block_manager, config)
```

---

## 6. ModelInputBuilder (`mini_vllm/engine/input_builder.py`)

Add `sequence_info` construction to `build()`:

```python
seq_info = []
for meta in prefill_meta:
    seq_info.append(SequenceExecutionInfo(
        sequence_id=meta.seq.seq_id,
        phase="prefill",
        query_start=meta.cached_len_before,
        query_len=meta.query_len,
        cached_len_before=meta.cached_len_before,
        kv_len_after=meta.cached_len_before + meta.query_len,
        sample_output_index=...  # index if prefill completes, else None
    ))
for meta in decode_meta:
    seq_info.append(SequenceExecutionInfo(
        sequence_id=meta.seq.seq_id,
        phase="decode",
        query_start=meta.cached_len_before,
        query_len=1,
        cached_len_before=meta.cached_len_before,
        kv_len_after=meta.cached_len_before + 1,
        sample_output_index=...  # always sampled
    ))
```

Length assertions (data invariant):
- For each prefill info: `kv_len_after == cached_len_before + query_len`
- For each decode info: `query_len == 1`, `kv_len_after == cached_len_before + 1`

Decode cached_len_before semantics (verified):
```
output_token_ids[-1] was written to cache in the PRIOR step at position = cached_len_before - 1
This step's NEW token sits at position cached_len_before (== prompt_len + num_generated_tokens)
Its KV will be written after this step completes, making kv_len_after = cached_len_before + 1
```

---

## 7. Backend Naming (`mini_vllm/config.py` + `mini_vllm/attention/backend.py`)

| Old | New |
|-----|-----|
| `backend_type` field | `attention_backend` field |
| Value `"paged"` | Value `"triton"` |
| Value `"ref"` | Value `"reference"` |
| `create(..., backend_type=...)` | `create(..., backend=...)` |

No tests refer to `backend_type` directly — safe rename.

---

## 8. Token/Cache Length State

**Prefill chunk:**
```
cached_len_before = prefill_cursor
query_len = chunk_size  (capped by max_prefill_chunk_size)
positions = [cached_len_before, ..., cached_len_before + query_len - 1]
kv_len_after = cached_len_before + query_len
KV written at positions [cached_len_before, ..., kv_len_after - 1]
```

**Decode (each step):**
```
cached_len_before = prompt_len + num_generated_tokens  ← tokens already in KV cache
query_len = 1
position = cached_len_before  ← absolute position of the input token THIS step
kv_len_after = cached_len_before + 1

output_token_ids[-1]  ← this is the token at position (cached_len_before - 1)
                          It was written to KV in the PRIOR step.
                          It is the INPUT to this decode step.

NEW token at position = cached_len_before
  Its KV is written DURING this step (after attention computes it).
  It becomes output_token_ids[-1] in the NEXT step.
```

**Assertions** in EngineCore:
- For prefill groups: `kv_len_after == cached_len_before + query_len` per group
- For decode groups: `query_len == 1`

---

## 9. Tests (`tests/test_engine_core_unified.py`)

12 tests using spy/wrapper:

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_execute_called_once` | Spy executor: execute() called exactly once per step |
| 2 | `test_mixed_batch_single_input` | Both prefill+decode in same ModelInput |
| 3 | `test_incomplete_prefill_no_sample` | Unfinished chunk: sample_output_index is None |
| 4 | `test_completed_prefill_first_token` | Completing prefill → first token → Status.RUNNING |
| 5 | `test_decode_appends_token` | Decode step appends next token |
| 6 | `test_sample_sequence_mapping` | Sampled sequence_ids map back to correct Sequences |
| 7 | `test_empty_batch_no_execute` | Empty input_ids → execute() not called |
| 8 | `test_fake_executor_behavior_preserved` | Old engine integration tests pass |
| 9 | `test_qwen_compat_path` | QwenExecutor prefill()/decode() unchanged |
| 10 | `test_old_prefill_decode_not_called` | Spy: paged path doesn't call legacy methods |
| 11 | `test_length_assertions` | cached/query/kv-after consistency per group |
| 12 | `test_backend_naming` | attention_backend accepts valid values |

---

## 10. Files Modified

| File | Change |
|------|--------|
| `mini_vllm/model_runner/base.py` | Add ModelRunnerOutput, SequenceExecutionInfo |
| `mini_vllm/executor/base.py` | execute() return type → ModelRunnerOutput |
| `mini_vllm/executor/executor.py` | Add `execute()` using sequence_info snapshot |
| `mini_vllm/engine/engine_core.py` | Unified call path with _apply_model_output |
| `mini_vllm/engine/input_builder.py` | Build sequence_info tuple in ModelInput |
| `mini_vllm/config.py` | backend_type → attention_backend |
| `mini_vllm/attention/backend.py` | Rename params + values |
| `mini_vllm/__init__.py` | Export ModelRunnerOutput, SequenceExecutionInfo |

## 11. Files Created

| File | Content |
|------|---------|
| `tests/test_engine_core_unified.py` | 12 phase-1.5 tests |

## 12. Out of Scope (explicitly excluded)

- GPU KV Cache allocation
- Triton kernels
- Attention math implementation
- Qwen ModelRunner
- Old benchmarks
- Phase 2 work
- Scheduled tasks
