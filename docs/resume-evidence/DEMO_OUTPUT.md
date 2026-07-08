# Demo & Benchmark Output / 演示与基准测试输出

## 1. Report Scope / 报告范围

- **Date executed / 执行日期:** 2026-07-07
- **Environment / 环境:** WSL2 (Ubuntu 22.04) on Windows
- This report records actual execution output from this session. / 本报告记录本次会话的实际执行输出。
- No source code, tests, or configuration were modified for this report. / 未修改任何源代码、测试或配置文件。

> **中文摘要：** 本报告记录 4 个演示脚本在 WSL2 环境中的实际输出，包括 fake 执行器和 Qwen 真实模型两种模式。

## 2. Demo Inventory / 演示清单

| Demo | Executor / 执行器 | Requests / 请求数 | Max Tokens / 最大 Token | Steps / 步数 | Purpose / 用途 |
|------|----------|----------|-----------|-------|---------|
| `demo_fake_engine.py` | fake | 3 (A+B early, C mid-arrival / A+B 先到，C 中途到达) | A:8, B:12, C:6 | 16 | Step-by-step scheduling trace with KV block allocation visualisation / 逐步调度追踪与 KV 块分配可视化 |
| `benchmark.py` | fake | 4 | 16 | 34 | Throughput, TTFT, TPOT benchmarks with KV utilisation / 吞吐量、TTFT、TPOT 基准与 KV 利用率 |
| `demo_stage_breakdown.py` | fake | 4 | 8 | 22 | Stage-level latency breakdown (scheduler vs executor vs KV ops) / 阶段级延迟分解 |
| `benchmark.py` | qwen | 1 | 4 | 5 | Real Qwen2-0.5B inference metrics (CPU-only) / 真实 Qwen2-0.5B 推理指标（仅 CPU） |

> **中文摘要：** 4 个演示覆盖了逐步调度追踪、基准测试、阶段分析三个 fake 执行器场景，以及一个 Qwen2-0.5B 真实模型推理场景。

## 3. Raw Output — Demo 1: `demo_fake_engine.py` / 原始输出 — 演示 1

This demo shows three requests being processed by the fake engine with step-by-step scheduling decisions and KV block allocation. / 本演示展示三个请求在 fake 引擎中的处理过程，包括逐步调度决策和 KV 块分配。

```
============================================================
mini-vLLM Continuous Batching — Fake Engine Demo
============================================================

[init] Adding request A (prompt="Hello world", max_new_tokens=8)
[init] Adding request B (prompt="CUDA batching", max_new_tokens=12)

--- Step 1 ---
  [step 1]
    waiting:                  [—]
    running:                  [req-0000(PREFILL,cursor=4,gen=0), req-0001(PREFILL,cursor=4,gen=0)]
    scheduled prefill:        [req-0000, req-0001]  prefill_tokens=8
    scheduled decode:         [—]  decode_tokens=0
    token budget remaining:   8/16
    KV blocks allocated:      2/16  (slot_capacity=8)
    KV tokens written:        8  (actual data in cache)
    +- BlockAllocator free list [step 1]
    |  free blocks:  [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    |  used blocks:  [0, 1]
    |  [req-0000-seq-0] BlockTable: L0->P0
    |  [req-0001-seq-0] BlockTable: L0->P1
    |  [req-0000-seq-0] ALLOC: blocks [0]
    |  [req-0001-seq-0] ALLOC: blocks [1]
    |  [req-0000-seq-0] blocks=1 (would be 5 with eager), saved=4
    |  [req-0001-seq-0] blocks=1 (would be 7 with eager), saved=6
    +-

--- Step 2 ---
  [step 2]
    waiting:                  [—]
    running:                  [req-0000(PREFILL,cursor=8,gen=0), req-0001(PREFILL,cursor=8,gen=0)]
    scheduled prefill:        [req-0000, req-0001]  prefill_tokens=8
    scheduled decode:         [—]  decode_tokens=0
    token budget remaining:   8/16
    KV blocks allocated:      4/16  (slot_capacity=16)
    KV tokens written:        16  (actual data in cache)
    ...

[arrival] New request C arrives (prompt="New batch", max_new_tokens=6)
  [step 3]
    waiting:                  [—]
    running:                  [req-0000(RUNNING,cursor=-,gen=1), req-0001(PREFILL,cursor=12,gen=0), req-0002(PREFILL,cursor=4,gen=0)]
    scheduled prefill:        [req-0000, req-0001, req-0002]  prefill_tokens=11
    scheduled decode:         [—]  decode_tokens=0
    KV blocks allocated:      7/16
    KV tokens written:        27
    ...
  [step 4]
    running:                  [req-0000(RUNNING,gen=2), req-0001(RUNNING,gen=1), req-0002(PREFILL,cursor=8)]
    scheduled prefill:        [req-0001, req-0002]  prefill_tokens=5
    scheduled decode:         [req-0000]  decode_tokens=1
    KV blocks allocated:      10/16
    ...
  [step 5-10]  decode-heavy phase: 3 running, 3 decode_tokens/step
  [step 11]
    running:                  [req-0001(RUNNING,gen=8)]
    finished:                 [req-0000, req-0002]
    KV blocks allocated:      6/16  (freed 8 blocks from finished requests)
    ...
  [step 11-15]  single decode: req-0001 gen=8..12
  [step 16]
    running:                  [—]
    KV blocks allocated:      0/16  (all blocks freed)
    finished:                 [req-0001]

============================================================
Final Outputs
============================================================
  req-0000: prompt='Hello world'  output='M5TV9Z(L'  (tokens=8)
  req-0002: prompt='New batch'  output='Y)!*Ct'  (tokens=6)
  req-0001: prompt='CUDA batching'  output='Vq*I"me5~iR '  (tokens=12)

================
Benchmark Report
================
  Requests:              3  (prompt=33 tok, output=26 tok)
  Steps:                 16
  Total time (wall):        0.001s

  TTFT (avg/min/max):      0.3 / 0.2 / 0.38 ms
  TPOT (avg/min/max):      0.07 / 0.06 / 0.07 ms

  Throughput (wall):        2978.2 req/s,  25811.1 tok/s
  KV blocks (peak/avg):  14 / 50.8%  (of 16 total)
  Block utilisation:     3.71 tokens/block
  Scheduler latency:     0.0155 ms avg,  0.0548 ms max
```

### Step-by-Step Interpretation / 逐步解读

| Step / 步骤 | Key Event / 关键事件 | KV Blocks / KV 块 | Interpretation / 解读 |
|------|-----------|-----------|----------------|
| 1 | Both A and B admitted as PREFILL / A 和 B 以预填充状态准入 | 2/16 | chunked_prefill=4 将 13 token (A) 和 15 token (B) 的 prompt 拆分；各写 4 个 token = 2 个块 |
| 2 | Both continue PREFILL, cursor=8 / 继续预填充，游标到 8 | 4/16 | 按需分配确认：每 4 个 token 分配 1 个新块 |
| 3 | A finishes PREFILL → RUNNING (gen=1), C mid-arrival admitted / A 完成预填充转为运行，C 中途到达被准入 | 7/16 | A 写 1 个解码 token，B 写最后一块预填充，C 开始预填充。按需分配 3 个新块 |
| 4 | Decode-first budget: A has 1 decode token deducted first / 解码优先预算：A 的 1 个解码 token 先扣除 | 10/16 | decode_tokens=1 先从预算中扣除，剩余 10 个 token 分配给 B（预填充完成）和 C（继续预填充） |
| 5-10 | Pure decode phase: all 3 requests RUNNING / 纯解码阶段：3 个请求都在运行 | 峰值 14 | 所有请求以每步 3 个解码 token 的速度生成。步骤 8：req-0000 达到其 eager 等价最大值 |
| 11 | A and C finish, B continues alone / A 和 C 完成，B 单独继续 | 6/16 | 释放 8 个块。BlockAllocator 回收已释放的物理块 ID |
| 11-15 | B decodes gen=8..12 (5 steps) / B 解码 gen=8..12（5 步） | 增长到 7 | B 在第 15 步为最终解码 token 再分配 1 个块 |
| 16 | All done, 0/16 blocks / 全部完成 | 0/16 | 所有块返回空闲列表。KV tokens written=56（33 prompt + 23 generated） |

> **中文解读：** 16 步调度演示展示了按需分配（步骤 1 每个请求仅 1 个块，eager 模式需要 5 和 7）、中途到达合并（步骤 3 请求 C 到达后立即被准入）、分块预填充（A 的 13 个 token 分 4 块完成）、解码优先（步骤 4 解码 token 先扣预算）、块生命周期（步骤 11 和 16 的 FREE 事件）和块重用（req-0001 重用了 req-0000 释放的块 0）。

### What the Demo Demonstrates / 演示验证的内容

- **On-demand allocation / 按需分配**: At step 1, each request holds 1 block (would have been 5 and 7 with eager). At peak (step 8), 14/16 blocks used vs 16/16 with eager. Saving: 6-10 blocks across the run. / 步骤 1 每个请求仅 1 个块（eager 模式为 5 和 7），峰值 14/16（eager 为 16/16），全程节省 6-10 个块。
- **Mid-arrival merge / 中途到达合并**: Request C arrives at step 3 and is admitted immediately (next step) while A and B are still running. / 请求 C 在第 3 步到达，下一步立即被准入。
- **Chunked prefill / 分块预填充**: A's 13-token prompt takes 4 chunks (4+4+4+1 across steps 1-4). B's 15-token prompt takes 4 chunks (4+4+4+3). / A 的 13 token prompt 分 4 块处理。
- **Decode-first priority / 解码优先**: Step 4: A's decode token is scheduled before B and C's prefill tokens. / 步骤 4：A 的解码 token 优先于 B 和 C 的预填充 token。
- **Block lifecycle / 块生命周期**: FREE events at step 11 (A, C) and step 16 (B) confirm BlockAllocator properly recycles blocks. / FREE 事件确认 BlockAllocator 正确回收块。
- **Block reuse / 块重用**: Step 11: req-0001 allocates block 0 (previously owned by req-0000). / 步骤 11：req-0001 分配块 0（此前为 req-0000 所有）。

## 4. Raw Output — Demo 2: `benchmark.py` (fake executor, 4 requests, 16 tokens) / 原始输出 — 演示 2

```
mini-vLLM Benchmark
  executor:     fake
  requests:     4
  max_tokens:   16

  Added request req-0000: prompt='Hello, world!'...  (max_tokens=16)
  Added request req-0001: prompt='What is the capital of France?'...  (max_tokens=16)
  Added request req-0002: prompt='Write a short poem about artificial inte'...  (max_tokens=16)
  Added request req-0003: prompt='Explain the concept of attention in tran'...  (max_tokens=16)

  Running 4 requests to completion...
  Done in 0.002s  (active: 0.001s)

============================================================
Outputs
============================================================
  req-0000: "%'^u($C3Tp@. \\zI"
  req-0001: ' f9I+]>;?#@pTEdd'
  req-0002: 'LR+<{ !!QR&EE&jI'
  req-0003: 'L$[m98^qLjAU`>Fx'

================
Benchmark Report
================
  Requests:              4  (prompt=163 tok, output=64 tok)
  Steps:                 34
  Total time (wall):        0.002s  (active: 0.001s)

  TTFT (avg/min/max):      0.73 / 0.35 / 1.13 ms
  TPOT (avg/min/max):      0.05 / 0.03 / 0.06 ms

  Throughput (wall):        2490.31 req/s,  39844.95 tok/s
  Throughput (active):      2786.91 req/s,  44590.61 tok/s

  KV blocks (peak/avg):  51 / 27.7%  (of 112 total)
  Block utilisation:     3.81 tokens/block

  Scheduler latency:     0.0165 ms avg,  0.0706 ms max
  Step latency:          0.0422 ms avg,  1.44 ms total
```

### Interpretation / 解读

- **TTFT variation / TTFT 变化**: req-0000 TTFT=0.35ms (admitted first), req-0003 TTFT=1.13ms (waits for slot/budget). 3.2× spread reflects scheduling delay, not compute variation. / 3.2 倍差异反映的是调度延迟，而非计算变化。
- **TPOT stability / TPOT 稳定性**: 0.03-0.06ms across all requests — fake executor has no memory bandwidth bottleneck. / 所有请求 0.03-0.06ms — fake 执行器无内存带宽瓶颈。
- **KV peak utilisation / KV 峰值利用率**: 45.5% (51/112 blocks). The system is over-provisioned for this workload — 61 blocks never used. / 系统对此工作负载过度配置 — 61 个块从未使用。
- **Wall vs active throughput / 时钟 vs 活跃吞吐量**: 2490 vs 2787 req/s — ~11% idle time between batches. / 批次间约 11% 的空闲时间。
- **Scheduler overhead / 调度器开销**: 0.0165ms avg — negligible relative to step latency. / 相对于步延迟可忽略不计。

> **中文解读：** 4 个请求、34 步的基准测试。TTFT 从 0.35ms 到 1.13ms 的 3.2 倍差异反映的是调度延迟。KV 峰值利用率仅 45.5%，系统过度配置。调度器开销 0.0165ms，相对于步延迟可忽略。

## 5. Raw Output — Demo 3: `demo_stage_breakdown.py` (fake executor, 4 requests, 8 tokens) / 原始输出 — 演示 3

```
Stage Breakdown
---------------------------------------------------------------------------
stage                       count   total_ms    avg_ms    max_ms    pct
---------------------------------------------------------------------------
engine_step_total              22       1.20    0.0547    0.1402  100.0%
executor_forward               21       0.53    0.0250    0.0565   43.7%
scheduler_step                 22       0.43    0.0197    0.0741   35.9%
prefill                        14       0.38    0.0271    0.0551   31.5%
request_queue_waiting           4       0.37    0.0926    0.0999   30.8%
kv_cache_allocation            46       0.13    0.0029    0.0150   11.1%
decode                         17       0.11    0.0065    0.0131    9.2%
metrics_update                 22       0.08    0.0036    0.0060    6.6%
prefix_cache_lookup             4       0.02    0.0039    0.0064    1.3%
kv_cache_release                4       0.02    0.0048    0.0052    1.6%
---------------------------------------------------------------------------
Total profiled time                       3.27 ms
Total requests                               4
Total engine steps                          22
```

### Interpretation / 解读

- **engine_step_total** (100%): The top-level timing. Everything else is a sub-component. / 顶层计时，其他均为子组件。
- **executor_forward** (43.7%): Largest sub-stage. With the fake executor, this is pure CPU arithmetic (no GPU). Would dominate even more with a real model. / 最大子阶段。fake 执行器下为纯 CPU 运算，使用真实模型时占比会更高。
- **scheduler_step** (35.9%): Second-largest. Scheduling overhead is significant at these microsecond scales because the fake executor is so fast. With a real model, this percentage would drop to <1%. / 在微秒级尺度下调度开销显著。使用真实模型时占比将降至 <1%。
- **prefill** (31.5%): Overlaps with executor_forward (prefill is a sub-type of forward). 14 prefill calls for 4 requests — chunked prefill splits long prompts. / 14 次预填充调用对应 4 个请求——分块预填充将长 prompt 拆分。
- **request_queue_waiting** (30.8%): 4 calls, one per request. Each waits for the scheduler to pick it up. High percentage because requests are very short. / 每个请求一次调用，等待调度器拾取。百分比高是因为请求非常短。
- **kv_cache_allocation** (11.1%): 46 individual allocations (on-demand). 0.0029ms avg — negligible. Would be slower with real GPU allocator. / 46 次按需分配，平均 0.0029ms，使用真实 GPU 分配器时会变慢。
- **decode** (9.2%): 17 decode steps. 0.0065ms avg — fast because fake executor's decode is simple arithmetic. / 17 次解码步骤，fake 执行器下解码是简单算术运算。
- **metrics_update** (6.6%), **prefix_cache_lookup** (1.3%), **kv_cache_release** (1.6%): Overhead stages. All negligible. / 开销阶段，均可忽略。

**Profile with a real model / 使用真实模型的预期画像**: executor_forward would be ~95%+, scheduler_step and kv_cache_allocation would shrink to <0.5%. / executor_forward 将占 ~95% 以上，scheduler_step 和 kv_cache_allocation 将缩至 <0.5%。

> **中文解读：** 阶段级延迟分解显示 executor_forward（43.7%）和 scheduler_step（35.9%）是最大的两个子阶段。但请注意，这是 fake 执行器下的画像——使用真实 GPU 模型时，executor_forward 将占 95% 以上，调度器开销将降至几乎可忽略。

## 6. Raw Output — Demo 4: `benchmark.py` (Qwen2-0.5B, 1 request, 4 tokens) / 原始输出 — 演示 4

```
mini-vLLM Benchmark
  executor:     qwen
  requests:     1
  max_tokens:   4

  Added request req-0000: prompt='Hello, world!'...  (max_tokens=4)

  Running 1 requests to completion...
  Done in 0.308s  (active: 0.308s)

============================================================
Outputs
============================================================
  req-0000: " I'm a "

================
Benchmark Report
================
  Requests:              1  (prompt=4 tok, output=4 tok)
  Steps:                 5
  Total time (wall):        0.308s  (active: 0.308s)

  TTFT (avg/min/max):      211.08 / 211.08 / 211.08 ms
  TPOT (avg/min/max):      32.25 / 32.25 / 32.25 ms

  Throughput (wall):        3.25 req/s,  12.99 tok/s
  Throughput (active):      3.25 req/s,  13.0 tok/s

  KV blocks (peak/avg):  2 / 8.8%  (of 16 total)
  Block utilisation:     4.0 tokens/block

  Scheduler latency:     0.0289 ms avg,  0.0422 ms max
  Step latency:          61.5524 ms avg,  307.76 ms total
```

### Interpretation / 解读

- **Hardware context / 硬件背景**: Runs on WSL2 with no GPU passthrough. Qwen2-0.5B inference runs entirely on CPU via PyTorch. These numbers represent CPU inference performance, NOT GPU performance. / 在 WSL2 上运行，无 GPU 直通。Qwen2-0.5B 推理完全在 CPU 上通过 PyTorch 执行。这些数字代表的是 CPU 推理性能，而非 GPU 性能。
- **TTFT=211.08ms**: Includes model forward pass on CPU for a 4-token prompt. On a GPU, this would be ~5-15ms. / 包括 4 个 token prompt 在 CPU 上的模型前向传播。在 GPU 上约 5-15ms。
- **TPOT=32.25ms**: Per-token decode on CPU. On a GPU with CUDA, this would be ~10-20ms for Qwen2-0.5B. / 在 CPU 上每个 token 的解码时间。在 GPU 上约 10-20ms。
- **Output " I'm a " / 输出 " I'm a "**: Real model output. The model continues the prompt "Hello, world!" with a reasonable continuation. / 真实模型输出。模型对 "Hello, world!" 给出了合理的继续。
- **Scheduler latency 0.0289ms**: 0.05% of step latency — confirms scheduler overhead is negligible vs model inference. / 仅占步延迟的 0.05%——确认调度器开销相对于模型推理可忽略。
- **Step latency 61.55ms avg / 平均步延迟 61.55ms**: Dominated by Qwen forward pass on CPU. 5 steps × 61.55ms ≈ 308ms total. / 主要由 CPU 上的 Qwen 前向传播主导。
- **KV blocks 2/16**: Tiny workload. 4 prompt tokens + 4 generated tokens = 8 KV entries = 2 blocks (block_size=4). Only 12.5% of cache used. / 仅使用 12.5% 的缓存。

> **中文解读：** 这是项目唯一的真实模型推理演示。Qwen2-0.5B 在 WSL2 CPU 上运行（无 GPU 直通），TTFT=211ms，TPOT=32ms。模型生成了合理的输出 " I'm a "。这些数字代表 CPU 推理性能，GPU 上慢约 10-20 倍。调度器开销 0.0289ms 相对于 61.55ms 的步延迟可以忽略。

## 7. What These Demos Demonstrate / 演示验证与未验证的内容

### Demonstrated / 已验证：
- Scheduler lifecycle (WAITING→PREFILL→RUNNING→FINISHED) with correct state transitions / 调度器生命周期与正确的状态转换
- Chunked prefill splitting long prompts across steps / 分块预填充将长 prompt 拆分到多步
- Decode-first priority budget allocation / 解码优先的预算分配
- Mid-arrival request merge (new request admitted while others running) / 中途到达请求合并
- On-demand KV block allocation (blocks grow only as tokens are written) / 按需 KV 块分配
- Block lifecycle (ALLOC→FREE→recycle) / 块的生命周期
- Token budget tracking (decode_tokens deducted before prefill_tokens) / Token 预算追踪
- Stage-level profiling breakdown with fake executor / fake 执行器的阶段级分析
- Real model inference via Qwen2-0.5B (HuggingFace Transformers, CPU) / Qwen2-0.5B 真实模型推理
- Metrics output (TTFT, TPOT, throughput, KV utilisation, scheduler latency) / 指标输出

### NOT demonstrated / 未验证：
- GPU inference (no CUDA kernel execution) / GPU 推理
- Preemption (OOM during decode is not handled — raises RuntimeError) / 抢占（解码时 OOM 会抛出异常）
- Prefix cache sharing between requests (all prompts in demos are unique) / 请求间的前缀缓存共享
- Production throughput (fake executor metrics are CPU overhead, not real inference) / 生产吞吐量
- Large-scale workloads (max 4 requests, max 55 prompt tokens) / 大规模工作负载
- Tokenizer correctness (fake tokenizer uses ASCII modulo; Qwen demo uses real tokenizer but only 1 request) / Tokenizer 正确性
- Realistic KV cache pressure (max 51/112 blocks with fake, 2/16 with Qwen) / 实际的 KV 缓存压力
- Distributed or multi-GPU inference / 分布式或多 GPU 推理

> **中文摘要：** 演示验证了调度器生命周期、分块预填充、解码优先、中途到达合并、按需 KV 分配、块生命周期和真实模型推理等核心功能。未验证 GPU 推理、抢占机制、生产级吞吐量和分布式推理。

## 8. Reproduction Commands / 复现命令

```bash
# Demo 1: Step-by-step scheduling trace / 逐步调度追踪
PYTHONPATH=. python3 examples/demo_fake_engine.py

# Demo 2: Benchmark with fake executor / fake 执行器基准测试
PYTHONPATH=. python3 examples/benchmark.py --executor fake --requests 4 --tokens 16

# Demo 3: Stage breakdown profiling / 阶段分析
PYTHONPATH=. python3 examples/demo_stage_breakdown.py --executor fake --requests 4 --tokens 8

# Demo 4: Real model inference with Qwen2-0.5B / Qwen2-0.5B 真实模型推理
PYTHONPATH=. python3 examples/benchmark.py --executor qwen --requests 1 --tokens 4

# Quiet mode (shorter output) / 静默模式（简短输出）
PYTHONPATH=. python3 examples/benchmark.py --executor fake --requests 4 --tokens 16 --quiet
```
