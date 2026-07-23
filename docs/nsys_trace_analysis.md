# Nsight Systems Trace Analysis

**File**: `benchmark_results/nsys_trace.nsys-rep`
**Capture command**: `nsys profile -t nvtx,cuda -o benchmark_results/nsys_trace -w true -f true python3 -m benchmarks.profile_nsys`
**Date**: 2026-07-15
**Tool**: Nsight Systems CLI 2023.4.4.54

---

## Q1: CUDA Stream Utilization — how many streams are active, and is any stream overlap visible?

**Answer**: 7 CUDA streams detected (IDs 6–12), but stream 6 carries ~90%+ of all workload. Streams 7–12 show only sporadic activity from CCCL/CUB helper operations (radix sort, segmented sort, select, reduce — 384 instances each across the trace).

**Consequence**: No stream-level overlap is visible. The engine executes serially on the default stream. Any attempt at asynchronous scheduling (CPU-GPU pipeline) would need to introduce multi-stream parallelism, which currently does not exist.

---

## Q2: Synchronization Pattern — how many sync calls per step, and where do they come from?

**Answer**: 
- Total `cudaStreamSynchronize`: 1,782 calls (190.6 ms total wall time)
- During decode: 1,134 calls (~75 per step)
- `cudaDeviceSynchronize`: 32 calls (~2 per step)
- Sync overhead: ~0.3% of trace wall time

**Pattern**: Every decoder layer issues sync calls. The `ModelRunner.execute_model()` dispatches kernels layer-by-layer and syncs after each layer. This serializes the entire forward pass: no kernel from layer N+1 can begin on GPU until layer N's kernels finish and the CPU confirms completion.

**Implication for async scheduling**: These sync calls are the fundamental blocker. Removing them would require splitting GPU work submission from CPU work (sampling, scheduler) onto separate streams/threads.

---

## Q3: Kernel Launch Overhead — what fraction of step time is spent launching kernels vs. executing them?

**Answer**: Within the current trace, we only have CUDA API timing, not GPU kernel execution timing. The available data:

- Per decode step: ~1,134 kernel launches via `cudaLaunchKernel`
- Avg launch latency: 14.3 μs per call
- Total launch overhead per step: ~16.2 ms of CPU-side wall time
- Total decode step wall time (including nsys overhead): ~72 ms

On the CPU timeline, launch overhead occupies ~22% of the step's CPU-side time (16.2 ms / 72 ms). However, this is misleading — `cudaLaunchKernel` is asynchronous and does not block GPU execution. The real impact can only be assessed with GPU kernel execution timelines (not captured in this trace).

**Without kernel execution data**, we cannot distinguish between:
- GPU is fast and CPU launch overhead dominates → async scheduling helps
- GPU is the bottleneck → async scheduling doesn't help

The Phase 2 CUDA Event profiling (97.6% GPU occupancy) strongly suggests GPU is the bottleneck at batch=1, making launch overhead irrelevant.

---

## Q4: Memcpy Pattern — what is transferred, in which direction, and how often?

**Answer**:
- Total `cudaMemcpyAsync`: 3,081 (2,142 during decode)
- H2D transfers: ~870 (avg 30.2 μs each)
- D2H transfers: ~1,272 (avg 30.2 μs each)

**Content (inferred from code flow)**:
- **H2D**: Input token IDs per step ~0.5 KB, KV cache updates per layer: ~16 × 2 block entries = 32 blocks, each ~64 KB, total ~2 MB per step.
- **D2H**: Logits output: shape `[1, 151936]` (Qwen2.5-0.5B vocab), FP32 = ~608 KB per step.

The H2D/D2H ratio (~1:1.5) reflects the architecture: small input payload (1 token) + large output payload (full vocab distribution for sampling) + KV cache staging.

---

## Q5: NVTX Phase Duration Ratio — what is the decode-to-prefill ratio?

**Answer**:
- Phase_Prefill: 1,766.68 ms (includes GPU initialization + Triton kernel compilation + actual prefill)
- Phase_Decode (15 steps): 1,044 ms aggregate, mean 72.33 ms per step

**Ratio**: 1,044 / 1,767 ≈ 0.59× (decode aggregate vs. single prefill+init)

**Important caveat**: The prefill number is inflated by ~1.5s of initialization overhead (cuLibraryLoadData, cuModuleLoadData, cuKernelGetFunction). Pure prefill time is estimated at ~250 ms based on min_step timing patterns. This makes the prefill-to-decode ratio approximately 250:72 ≈ 3.5:1 for batch=1, which is within expected range for a 0.5B model with 128-token input.

---

## Q6: Memory Allocation — are runtime allocations present during steady-state decode?

**Answer**: Yes, but minimal:
- `cudaMalloc`: 209 calls across all phases, but only ~4 per decode step
- `cudaFree`: 1 call total (entirely negligible)
- `cudaHostAlloc`: 3 calls total
- `cuModuleLoadData`: 2 calls (initialization only)

Most allocations happen during initialization and Triton kernel compilation (Phase_Prefill). During steady-state decode, allocations are limited to KV cache block growth (when new blocks are allocated for new tokens). This is expected behavior — the BlockManager allocates new physical blocks as the sequence lengthens.

**KV cache memory**: At 128 tokens with batch_size=1, the KV cache uses ~2 MB per layer × 16 layers ≈ 32 MB, negligible for a 3.21 GB GPU allocation.

---

## Q7: Kernel Launch Distribution — are there any hotspots or tail steps?

**Answer**:
- 15 decode steps captured
- Mean: 72.33 ms, Min: 50.54 ms (step 2), Max: 191.59 ms (step 3)
- Standard deviation: ~36 ms (high variance)

**Tail analysis**:
- Step 3 (191.59 ms) is 2.6× the mean — a clear outlier
- Step 7 (156.55 ms) and step 15 (97.65 ms) are also above average
- The remaining 12 steps cluster in the 50–80 ms range

**Likely causes for Step 3 tail**:
1. Triton JIT compilation of a new kernel variant (first encounter of a specific attention shape or block size)
2. KV cache block allocation causing a synchronous `cudaMalloc` stall
3. Memory defragmentation or TLB miss on first access to newly allocated blocks

The variance suggests that the engine is not in a purely steady state — occasional compilation or allocation events perturb decode latency by 2-3×. This is relevant for production serving, where SLOs would need headroom for these spikes.

---

## Summary of Findings

| Dimension | Finding | Async Scheduling Impact |
|-----------|---------|------------------------|
| Stream utilization | Single stream, serial execution | Full rewrite needed for pipeline parallelism |
| Sync calls | 75 syncs/step, every layer | Root cause of serialization |
| Kernel launch overhead | ~16 ms/step (CPU side) | Irrelevant if GPU is bottleneck |
| Memcpy | ~143 transfers/step, 30 μs avg | Overlap possible but gain limited |
| Allocation | Minimal during steady state | Not a concern |
| Tail latency | Step 3 is 2.6× mean (191 ms) | Needs investigation for production SLOs |

**Key limitation**: This trace lacks GPU kernel execution timelines. All conclusions about GPU utilization are inferred from Phase 2 CUDA Event profiling (97.6% GPU at batch=1) rather than directly measured from the nsys trace.
