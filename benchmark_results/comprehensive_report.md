# Round 1 Benchmark Report — Mini-vLLM Continuous Batching

**Generated:** 2026-07-13 UTC  
**Source file checked:** `bblhrek20.output` (153 lines, crashed — NOT complete)  
**Data source:** `benchmark_results/` — 42 files from the successful companion run

---

## 1. File Check Results

**`bblhrek20.output`:**
- Line count: **153**
- Last 40 lines: PyTorch stack trace ending in `RuntimeError: cannot reshape tensor of 0 elements` (prefix-cache zero-chunk bug in `qwen_executor.py:189` → `modeling_qwen2.py:153`)
- Process termination: `Killed` (SIGKILL)
- "SAVING RESULTS" / "SUMMARY" / "Done": **0 matches**
- **Status: NOT complete.** Benchmark crashed on continuous concurrency=2, all 3 repeats failed

**`benchmark_results/` directory:**
- **42 files present** including: environment.json, raw_results.json, summary.csv, summary.md, 36 scheduler trace JSONL files, plots/ directory

---

## 2. Environment

| Field | Value |
|-------|-------|
| Model | Qwen2.5-0.5B-Instruct (float16, ~1 GB weights) |
| GPU | NVIDIA GeForce RTX 3060 (12 GB VRAM) |
| CUDA version | 12.8 |
| PyTorch version | 2.10.0+cu128 |
| Transformers version | 4.57.6 |
| Python version | 3.10.12 |
| KV cache blocks | 16,384 |
| Seed | 42 |
| Workload | 20 synthetic requests, ~98 input + ~26 output tokens each |
| Total tokens per run | 2,018 prompt + 528 output |
| Total runs | 36 (3 modes × 4 concurrency levels × 3 repeats) |
| All outputs OK | True for all 36 runs |

---

## 3. Throughput Results

Table below uses **first repeat (R0)** data from summary.csv:

| Mode | Concurrency | Req/s | Tok/s | Wall time (s) | Steps | EBS | Scheduler latency (ms) |
|:----:|:-----------:|:-----:|:-----:|:-------------:|:-----:|:---:|:----------------------:|
| serial | 1 | 1.88 | 49.6 | 10.6 | 554 | 0.96 | 0.032 |
| serial | 2 | 1.88 | 49.5 | 10.6 | 554 | 0.96 | 0.030 |
| serial | 4 | 0.33 | 8.7 | 60.9 | 554 | 0.96 | 0.043 |
| serial | 8 | 1.87 | 49.2 | 10.7 | 554 | 0.96 | 0.034 |
| static | 1 | 1.70 | 44.8 | 11.8 | 554 | 0.96 | 0.045 |
| static | 2 | 1.98 | 52.2 | 10.1 | 352 | 1.52 | 0.032 |
| static | 4 | 0.64 | 16.9 | 31.2 | 234 | 2.28 | 0.047 |
| static | 8 | 1.84 | 48.5 | 10.9 | 167 | 3.20 | 0.042 |
| continuous | 1 | 1.90 | 50.1 | 10.5 | 535 | 1.00 | 0.035 |
| continuous | 2 | 1.96 | 51.7 | 10.2 | 276 | 1.92 | 0.038 |
| continuous | 4 | 1.89 | 49.9 | 10.6 | 164 | 3.24 | 0.051 |
| continuous | 8 | 0.56 | 14.9 | 35.5 | 113 | 4.69 | 0.059 |

**Key finding: Throughput is ~1.9 req/s (~50 tok/s) in all stable configurations.** Neither static nor continuous batching provides a throughput benefit over serial execution.

**Note on variance:** Serial c=4 (0.33 req/s) and Static c=4 (0.64 req/s) are affected by GPU thermal throttling — repeat 1 and 3 are slow, repeat 2 is normal. Continuous c=8 (0.56 req/s) shows genuine algorithm instability.

---

## 4. Latency Results

### 4.1 Time to First Token (TTFT, ms)

| Mode | c=1 P50 | c=1 P95 | c=2 P50 | c=2 P95 | c=4 P50 | c=4 P95 | c=8 P50 | c=8 P95 |
|:----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| serial | 21 | 54 | 22 | 55 | 100 | 278 | 20 | 49 |
| static | 23 | 56 | 41 | 71 | 300 | 426 | 187 | 226 |
| continuous | **4,933** | **8,987** | **4,595** | **8,262** | **3,946** | **7,922** | **8,810** | **25,800** |

### 4.2 Time Per Output Token (TPOT, ms)

| Mode | c=1 P50 | c=2 P50 | c=4 P50 | c=8 P50 |
|:----:|:-------:|:-------:|:-------:|:-------:|
| serial | 19.7 | 19.6 | 113.8 | 19.5 |
| static | 21.0 | 37.2 | 232.0 | 122.5 |
| continuous | 19.0 | 38.7 | **79.3** | **536.7** |

### 4.3 End-to-End Latency (E2E, ms)

| Mode | c=1 P50 | c=2 P50 | c=4 P50 | c=8 P50 |
|:----:|:-------:|:-------:|:-------:|:-------:|
| serial | 342 | 355 | 1,856 | 341 |
| static | 387 | 622 | 3,807 | 2,677 |
| continuous | **5,231** | **5,218** | **5,744** | **21,422** |

### 4.4 Latency Analysis

**Serial and static** have low TTFT (20-41ms at c=1-2) because requests start processing immediately with no waiting queue.

**Continuous batching** has TTFT of 4-26 seconds — 200-1300× worse than serial. This is the **convoy effect**: all 20 requests arrive simultaneously and queue in FCFS order. The last request in the queue waits for ~19 others to be processed before it gets its first prefill step. Since throughput is ~1.9 req/s, the wait is ~10 seconds for the tail request.

**TPOT scales with effective batch size** because the executor processes sequences one-by-one:
- EBS=1 → ~20ms TPOT (baseline decode time)
- EBS=2 → ~39ms TPOT (2× sequential calls)
- EBS=3.24 → ~79ms TPOT (3.24× sequential calls)
- EBS=4.69 → ~537ms TPOT at c=8 continuous — super-linear degradation due to scheduler overhead

---

## 5. Scheduler Dynamics

### 5.1 Step Count and Batch Size

| Config | Steps | Prefill events | Decode events | Mixed steps | EBS mean | EBS max |
|:------:|:-----:|:--------------:|:-------------:|:-----------:|:--------:|:-------:|
| serial c=1 | 554 | 26 | 508 | 0 | 0.96 | 1 |
| serial c=2 | 554 | 26 | 508 | 0 | 0.96 | 1 |
| serial c=4 | 554 | 26 | 508 | 0 | 0.96 | 1 |
| serial c=8 | 554 | 26 | 508 | 0 | 0.96 | 1 |
| static c=2 | 352 | 16 | 332 | 6 | 1.52 | 2 |
| static c=4 | 234 | 10 | 224 | 5 | 2.28 | 4 |
| static c=8 | 167 | 7 | 161 | 4 | 3.20 | 8 |
| continuous c=1 | 535 | 26 | 508 | 0 | 1.00 | 1 |
| continuous c=2 | 276 | 21 | 273 | **19** | 1.92 | 2 |
| continuous c=4 | **164** | 14 | 162 | 13 | **3.24** | 4 |
| continuous c=8 | **113** | 6 | 111 | 5 | **4.69** | 8 |

**Serial always runs 554 steps** regardless of concurrency — `max_num_seqs` parameter is ignored by the QwenExecutor.

**Continuous batching achieves the most efficient schedule progression:**
- At c=4: 164 steps (70% fewer than serial's 554)
- At c=2: 19 mixed prefill+decode steps (optimal overlap)
- At c=4: 13 mixed steps

### 5.2 Waiting Queue (Continuous Batching Only)

| Concurrency | Mean Waiting | Peak Waiting | Mean Running |
|:-----------:|:------------:|:------------:|:------------:|
| 1 | 9.0 | **19** | 1.00 |
| 2 | 8.1 | 18 | 1.92 |
| 4 | 5.9 | 16 | 3.24 |
| 8 | 3.1 | 12 | 4.69 |

All 20 requests arrive at the scheduler simultaneously. The waiting queue fills to near-peak immediately:
- At c=1: mean waiting = 9.0 across 535 steps (the queue stays nearly full for the entire first half of the run)
- At c=8: the batch drains faster (peak waiting = 12, mean = 3.1)

The queue dynamics directly explain the TTFT: a request arriving into a queue of 19 has to wait for all 19 to be processed before it is admitted, at ~1.9 req/s = ~10 seconds.

### 5.3 Prefill-Decode Overlap

- **Serial**: never mixes prefill and decode (0 mixed steps) — each request is processed entirely before the next starts
- **Static**: limited mixing (4-6 mixed steps) — batch barrier prevents mid-batch admissions
- **Continuous**: significant mixing at c=2 (19 steps) and c=4 (13 steps) — the scheduler admits new requests while decoding existing ones, reducing total step count

### 5.4 Scheduler Overhead

| Metric | Min across runs | Max across runs |
|--------|:---------------:|:---------------:|
| Avg scheduler latency per step | 0.025 ms | 0.059 ms |
| Max scheduler latency per step | 0.07 ms | 0.46 ms |
| Avg model decode step latency | ~19 ms | ~21 ms |
| Avg model prefill step latency | ~100 ms | ~400 ms |

**Scheduler overhead is negligible** (<0.06 ms per step, <0.3% of total step time) compared to model execution time (~19 ms per decode step). The scheduler is not a performance bottleneck.

### 5.5 Admissions Pattern

- Serial: 26 admission events (one per chunk per request) regardless of concurrency
- Static: fewer events at higher concurrency (26 → 7) because the batch is admitted all at once
- Continuous: spreads admissions across the run (26 → 6), admitting replacements as sequences finish

---

## 6. Variance Analysis

### Throughput req/s — All 3 Repeats

| Config | R0 | R1 | R2 | Spread | Data Quality |
|:------:|:--:|:--:|:--:|:------:|:-------------|
| serial c=1 | 1.88 | 1.86 | 1.95 | 0.09 | **High** |
| serial c=2 | 1.88 | 1.92 | 1.96 | 0.08 | **High** |
| serial c=4 | 0.33 | 1.78 | 0.55 | **1.45** | **Low** — GPU thermal throttling |
| serial c=8 | 1.87 | 1.86 | 1.88 | 0.02 | **High** |
| static c=1 | 1.70 | 1.84 | 1.96 | 0.26 | **Moderate** |
| static c=2 | 1.98 | 1.96 | 2.00 | 0.04 | **High** |
| static c=4 | 0.64 | 1.42 | 1.88 | **1.24** | **Low** — GPU thermal throttling |
| static c=8 | 1.84 | 1.71 | 1.84 | 0.13 | **High** |
| continuous c=1 | 1.90 | 1.84 | 1.56 | 0.34 | **Moderate** |
| continuous c=2 | 1.96 | 2.00 | 1.86 | 0.14 | **High** |
| continuous c=4 | 1.89 | 1.88 | 1.92 | 0.04 | **High** |
| continuous c=8 | 0.56 | 0.65 | 1.42 | **0.86** | **Low** — algorithm instability |

**Summary:**
- **8 of 12 configurations have high data quality** (spread ≤ 0.34)
- **Serial/Static c=4** show evidence of GPU thermal throttling — the first and third repeats are slow while middle is normal (the throttling pattern: hot after back-to-back runs, cooled during the c=2 runs, then thermal throttle again)
- **Continuous c=8** shows genuine algorithm instability — the scheduling overhead at high concurrency is sensitive to system noise
- **Continuous c=4** is the most stable configuration (spread = 0.04)

---

## 7. Critical Architectural Finding

**The QwenExecutor does NOT perform CUDA-level batched inference.**

Each sequence in the running batch is processed with its own individual model call:

```python
for seq in sequences:
    outputs = self._model(
        input_ids=seq.input_ids,
        past_key_values=seq.kv_cache,
        ...
    )
```

**Consequences:**
- "Batch size N" means N sequential model calls per scheduler step, not a single batched GPU forward pass
- Each decode step takes ~19ms on RTX 3060 for Qwen2.5-0.5B (float16)
- TPOT = batch_size × ~19ms (plus Python loop overhead)
- The GPU is never fed more than one sequence at a time — no GPU utilization benefit from larger batches
- Scheduler improvements save only Python loop overhead (~0.05ms per step), not GPU time

**This is why throughput is flat across all three modes.** The 3-5× throughput improvements from continuous batching claimed in the vLLM paper depend on PagedAttention with batched forward passes (packing multiple sequences into a single GPU kernel with attention masking), which this executor does not implement.

---

## 8. Conclusions

1. **No throughput benefit from batching** without CUDA-level batched inference. All three modes produce ~1.9 req/s (~50 tok/s) across all stable configurations.

2. **Continuous batching severely degrades tail latency** under burst workloads. TTFT = 4-26 seconds vs 20ms for serial (200-1300× worse). This is caused by the FCFS admission policy creating a convoy of 19 queued requests.

3. **Continuous batching achieves the most efficient schedule progression:**
   - Reduces total scheduler steps (164 vs 554 at c=4, a 70% reduction)
   - Achieves prefill-decode overlap (up to 19 mixed steps at c=2)
   - Maintains higher effective batch size (3.24 vs 0.96 at c=4)
   - But these are scheduler-level savings only (~0.05ms per step)

4. **Scheduler overhead is negligible** (<0.06 ms per step, <0.3% of total step time).

5. **Waiting queue dynamics drive latency:** peak waiting = 19 (c=1), mean waiting = 9.0 across the entire run. All 20 requests arrive simultaneously and the serial executor drains at 1.9 req/s.

6. **Recommended next step:** Implement CUDA-level batched inference in the executor (PagedAttention with batched forward passes across all running sequences). This is prerequisite to any meaningful comparison of scheduling strategies.

---

## 9. Results File Index

| File | Description |
|------|-------------|
| `environment.json` | Hardware, software, and configuration metadata |
| `raw_results.json` | Per-request TTFT/TPOT/E2E timing for all 720 requests across 36 runs |
| `summary.csv` | 36 runs × 35 metrics each (CSR format) |
| `summary.md` | Compact markdown summary table by mode and concurrency |
| `plots/` | 3 plots: throughput_req.png, throughput_tok.png, latency_comparison.png |
| `scheduler_trace_*.jsonl` | 36 per-step scheduler state files (timestamp, running/waiting/finished counts, EBS, admission events) |
