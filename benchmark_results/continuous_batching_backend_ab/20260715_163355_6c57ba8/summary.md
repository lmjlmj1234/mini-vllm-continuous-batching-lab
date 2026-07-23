# Continuous Batching Backend A/B Experiment

Reference Attention vs Triton Paged Decode Attention

- **Model**: Qwen2.5-0.5B-Instruct
- **GPU**: NVIDIA GeForce RTX 3060
- **PyTorch**: 2.10.0+cu128
- **dtype**: float16
- **Git commit**: 6c57ba85485933e7f91a5523e59a269581981045
- **Working tree dirty**: True
- **Timestamp**: 2026-07-15T16:33:56
- **Requests per run**: 16
- **GPU blocks**: 16384

---

## Throughput Comparison

| Concurrency | Order | Ref Req/s | Tri Req/s | Speedup | Ref Tok/s | Tri Tok/s | Speedup | Uplift % | TPOT Reduction % | Correct |
|---|---|---|---|---|---|---|---|---|---|---|

## Latency Comparison

| Concurrency | Ref TPOT50 | Tri TPOT50 | Ref TPOT95 | Tri TPOT95 | Ref TTFT50 | Tri TTFT50 | Ref E2E50 | Tri E2E50 |
|---|---|---|---|---|---|---|---|---|

## GPU Memory

| Concurrency | Ref Peak (MB) | Tri Peak (MB) |
|---|---|---|

## Correctness

**Overall Correctness: PASS**

## Workload

| Metric | Value |
|---|---|
| Total requests | 16 |
| Concurrency levels | [2, 4, 8] |
| Repeats | 3 |
| Workload | default_mixed |

