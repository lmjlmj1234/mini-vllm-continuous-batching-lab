# Testing Guide — mini-vLLM

## 1. Test System Overview

**135 unit and integration tests, all passing.**  
Run time: ~3 seconds with fake executor.

```
tests/
├── test_request.py              # ~12 tests — Data structures
├── test_kv_cache_manager.py     # ~13 tests — BlockAllocator, BlockTable, BlockManager
├── test_prefix_cache.py         # ~20 tests — PrefixCache, ref counts, sharing, probe
├── test_scheduler.py            # ~12 tests — Scheduler (admit, decode-first, chunked)
├── test_engine.py               #  ~9 tests — LLMEngine integration (run_until_done, OOM)
├── test_serving_layer.py        # ~20 tests — ServingLayer (generate, stream, rate limit)
├── test_fault_injection.py      # ~15 tests — Queue overflow, block exhaustion, timeout storm
├── test_stage_profiler.py       # ~15 tests — StageProfiler (record, report, reset, exception)
└── test_metrics.py              # ~39 tests — Metrics semantics (TTFT, TPOT, throughput, KV, scheduler)
```

### Test Layers

```
┌─────────────────────────────────────────────────────┐
│  test_metrics.py  — Metrics semantics & correctness  │
├─────────────────────────────────────────────────────┤
│  test_fault_injection.py  — Production failure modes  │
├─────────────────────────────────────────────────────┤
│  test_serving_layer.py  — HTTP/stream lifecycle     │
├─────────────────────────────────────────────────────┤
│  test_engine.py  — Full engine loop (integration)   │
├─────────────────────────────────────────────────────┤
│  test_scheduler.py  — Scheduling decisions           │
├─────────────────────────────────────────────────────┤
│  test_kv_cache_manager.py / test_prefix_cache.py    │
│  — KV cache allocation, prefix cache, ref counts    │
├─────────────────────────────────────────────────────┤
│  test_request.py  — Data structures (Sequence, etc.)│
└─────────────────────────────────────────────────────┘
```

---

## 2. What Each Test File Covers

### `test_request.py` — Data Structures
- `Status` enum values and finished logic
- `SamplingParams` defaults and custom values
- `Sequence` initial state, lifecycle (WAITING→PREFILL→RUNNING→FINISHED), `to_dict()`
- `SequenceGroup` create_sequence, is_finished, get_unfinished_seqs, empty group

### `test_kv_cache_manager.py` — Block Allocation
- `BlockTable`: add_block, get_physical_block, clear, position→block mapping
- `BlockAllocator`: allocate, free, OOM (return None), callbacks, stats
- `BlockManager`: allocate_for_seq (empty start), ensure_block (on-demand), free, OOM during execution

### `test_prefix_cache.py` — Prefix Cache
- `PrefixCache`: insert, lookup, lookup_span, insert_span, hash determinism, partial block
- Ref-counting: allocate sets ref=1, increment_ref, free decrements, double-free safety
- Sharing: same prefix → shared blocks, ref count increases, free one doesn't release
- Partial prefix: only matching prefix blocks are shared
- Probe: read-only (no ref count change), stale entries (ref=0) excluded
- Scheduler integration: cache hit reduces prefill budget

### `test_scheduler.py` — Scheduling Logic
- Admit waiting → prefill → decode → finish lifecycle
- Token counting (num_prefill_tokens, num_decode_tokens)
- Budget limits (max_num_seqs, max_num_batched_tokens)
- Decode-first: running decode sequences occupy slots before prefill
- Chunked prefill: long prompt split across steps
- Partial prefill: not yet finished sequences stay in PREFILL, not DECODE
- On-demand: no blocks allocated at admission time

### `test_engine.py` — Engine Integration
- Full run_until_done with multiple requests
- Mid-arrival merge (new request arrives while others running)
- OOM during execution (RuntimeError)
- ScheduleResult fields exist
- KV tracking (allocated_blocks > 0, kv_tokens_written > 0)

### `test_serving_layer.py` — HTTP Serving
- Generate (non-stream), empty prompt rejection, max_tokens=0
- Streaming basic, token accumulation
- Rate limiting (RPM)
- Admission control (PROMPT_TOO_LONG, QUEUE_OVERFLOW, BLOCK_EXHAUSTED)
- Stream Manager (acquire/release, max streams)
- Cancel (running request, non-existent)
- Timeout (auto-cancel)
- Metrics endpoint (JSON format, after rejection, after cancel)
- Disconnect lifecycle (abort mid-stream, blocks freed, no double-count)

### `test_fault_injection.py` — Production Failures
- Queue overflow: full queue → QUEUE_OVERFLOW, after drain → accepts
- Block exhaustion: no free blocks → BLOCK_EXHAUSTED, no crash, blocks recovered
- Stream exhaustion: max streams → TOO_MANY_STREAMS, release → recovers
- Timeout storm: 4 requests timeout, all blocks freed, metrics updated
- Cancel storm: 4 requests cancelled, all blocks freed, ref_count integrity
- Prefix cache stale entry: freed block not falsely used, new request re-creates
- Admission under block pressure: 10 blocks / 100 requests → all BLOCK_EXHAUSTED, no-admission → OOM crash

### `test_metrics.py` — Metrics Semantics
Detailed in Section 3 below.

### `test_stage_profiler.py` — Stage Profiler
- Single stage recording, multiple records (count/total/avg/max)
- Context manager, start/end API, reset, record_raw
- Exception handling (corrupted stage state survives exception)
- Integration with EngineCore (engine_step_total, total_requests, total_engine_steps)
- Cancellation doesn't crash profiler

---

## 3. Metrics Computation Formulas

For detailed formulas and semantic definitions of all metrics (TTFT, TPOT, Throughput, KV Block Utilization, Scheduler Latency, Prefix Cache Hit Rate), see [`docs/Metrics.md`](./Metrics.md). The formulas documented there are the canonical reference.

Key rules enforced by the test suite:
- **TPOT denominator**: `num_output_tokens - 1` (the first token's latency is captured by TTFT)
- **Only FINISHED sequences** contribute to TTFT, TPOT, and throughput (cancelled/timeout excluded)
- **Throughput uses wall-clock time** (`last_finish_time - first_arrival_time`), not per-request latency
- **KV utilization** range: [0, 100]; values above 80 indicate high pressure

---

## 4. How to Run Tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_metrics.py -v
python3 -m pytest tests/test_scheduler.py -v
python3 -m pytest tests/test_fault_injection.py -v

# Run a specific test class
python3 -m pytest tests/test_metrics.py::TestTTFT -v

# Run a specific test method
python3 -m pytest tests/test_metrics.py::TestTTFT::test_ttft_field_exists_and_non_negative -v

# Run with short traceback (cleaner output)
python3 -m pytest tests/ --tb=short

# Run quietly (summary only)
python3 -m pytest tests/ -q

# Stop on first failure
python3 -m pytest tests/ -x
```

---

## 5. How to Run Demos

```bash
# Basic engine demo — two requests + mid-arrival
python3 examples/demo_fake_engine.py

# Benchmark — multiple requests with metrics output
python3 examples/benchmark.py --executor fake --requests 4

# Stage breakdown profiling
python3 examples/demo_stage_breakdown.py --executor fake --requests 16 --tokens 16

# Benchmark with Qwen real model (requires torch + transformers + ~1GB disk)
python3 examples/benchmark.py --executor qwen --requests 2 --tokens 16
```

### Interpreting Demo Output

**`demo_fake_engine.py`** — Shows step-by-step scheduling events:
```
[step 1]
  waiting: [req-0001(prompt=13)]
  running: [req-0000(PREFILL,cursor=4,gen=0)]
  scheduled prefill: [req-0000]  prefill_tokens=4
  scheduled decode: [—]
  KV blocks allocated: 1/16
```
This shows:
- req-0000 is in PREFILL, cursor at 4 (4 tokens done out of ~13)
- req-0001 is still waiting (not yet admitted)
- KV blocks: 1 used out of 16

**`benchmark.py`** — Shows summary with TTFT/TPOT/Throughput:
```
  TTFT (avg/min/max):    0.31 / 0.31 / 0.31 ms
  TPOT (avg/min/max):    0.09 / 0.08 / 0.10 ms
  Throughput:            1.26 req/s,  20.25 tok/s
```
**Important**: With `--executor fake`, these numbers are **pure CPU overhead** (no real inference). They do not represent real model performance.

**`demo_stage_breakdown.py`** — Shows stage-level timing breakdown:
```
  stage                        count   total_ms    avg_ms     max_ms     pct
  ---------------------------------------------------------------------------
  scheduler_step                  14      2.10     0.1502    0.2439    37.0%
  engine_step_total               14      5.68     0.4055    0.5911   100.0%
  decode                          12      1.22     0.1015    0.1364    21.5%
  executor_forward                14      1.67     0.1194    0.1833    29.4%
  prefill                          8      0.54     0.0673    0.0797     9.5%
  ...
```
The `pct` column shows each stage's percentage of `engine_step_total`.  
**With the fake executor**, these numbers are Python-level wall clock, not GPU time.

---

## 6. What These Tests Prove

✅ **Scheduling logic** — admit, prefill, decode, finish lifecycle is correct  
✅ **Budget management** — max_num_seqs and max_num_batched_tokens are respected  
✅ **Chunked prefill** — long prompts are split and executed step by step  
✅ **Decode-first priority** — decode sequences take precedence over prefill  
✅ **KV block allocation** — on-demand allocation works, blocks are freed on finish  
✅ **Prefix cache** — hash-based sharing, ref counting, stale entry detection  
✅ **Request queue** — waiting → running → finished/rejected lifecycle  
✅ **Metrics** — TTFT/TPOT/Throughput formulas are implemented correctly  
✅ **Metrics integrity** — cancelled/timeout requests don't inflate completed counts  
✅ **Fault handling** — OOM, queue overflow, timeout, cancel, disconnect all clean up resources  
✅ **Streaming** — SSE-style token streaming, poll until done, disconnect cleanup  

## 7. What These Tests Do NOT Prove

❌ **Real model inference performance** — The fake executor does CPU arithmetic, not GPU inference  
❌ **GPU kernel-level performance** — No CUDA kernels, no Nsight Systems, no PyTorch profiler  
❌ **End-to-end latency on real hardware** — Numbers from fake executor are meaningless for real-world latency  
❌ **Production-scale throughput** — Tests run with tiny configs (4 blocks, 16 tokens) for speed  
❌ **Memory pressure under real workloads** — Fake KV cache is a Python dict, not GPU memory  
❌ **Multi-GPU or distributed inference** — Not implemented  
❌ **Long-context correctness** — Not tested beyond single-chunk scenarios  
❌ **Tokenizer correctness** — Fake tokenizer uses ASCII mod, not a real tokenizer  

**Bottom line**: These tests verify that the **scheduling architecture, KV block management, request lifecycle, metrics accounting, and failure recovery** logic is correct. They do **not** measure or validate real model performance. Use the Qwen executor for real inference testing, and use Nsight Systems / PyTorch profiler for GPU kernel-level profiling.
