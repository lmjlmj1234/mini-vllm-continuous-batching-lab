# Metrics Framework

## 设计原则

**统一收集，集中报告。**

Metrics 不在各个模块中乱打日志。所有数据通过 `MetricsCollector` 统一采集，在 Engine 运行结束后生成结构化的 Benchmark Report。

```
EngineCore.step()
    ↓
Scheduler.schedule()  ← 记录 scheduler_latency
    ↓
Executor.prefill()    ← 记录 first_token_time（在 Sequence 上）
Executor.decode()     ← 记录 output_tokens（在 Sequence 上）
    ↓
MetricsCollector.record_step()  ← 记录 per-step 数据
    ↓
(全部请求完成)
    ↓
MetricsCollector.report()  ← 生成最终报告
```

---

## 指标详解

### 1. TTFT — Time to First Token

**定义**: 从请求到达（`arrival_time`）到第一个输出 token 生成（`first_token_time`）的时间。

**公式**: `TTFT = first_token_time - arrival_time`

**为什么重要**:

TTFT 反映了系统的**响应速度**。用户发送一条消息后，多久能看到第一个字？

TTFT 的组成：
- **等待时间**：请求在队列中等待被 Scheduler 调度的时间
- **Prefill 时间**：Executor 处理 prompt tokens 的时间

**反映的设计质量**:

| 组件 | 对 TTFT 的影响 |
|------|---------------|
| Scheduler | decode-first 策略会优先处理 decode，可能延迟新请求的 prefill 调度。如果 TTFT 过高，说明 Scheduler 的 prefill 优先级可能太低 |
| Executor | Prefill 速度直接影响 TTFT。Qwen2-0.5B 的 prefill 比 FakeModel 慢几个数量级 |
| Token Budget | `max_num_prefill_tokens` 限制每步的 prefill token 数，过小会导致长 prompt 的 TTFT 变长 |
| Chunked Prefill | 将长 prompt 分块处理，每个块产生一个 TTFT 延迟。块越小，TTFT 越小（但代价是更多的调度步骤） |

**在真实 vLLM 中**: TTFT 是服务等级目标（SLO）的关键指标。在线服务通常要求 TTFT < 500ms。

---

### 2. TPOT — Time Per Output Token

**定义**: 生成每个输出 token 的平均时间，从第一个 token 之后开始计算。

**公式**: `TPOT = (finish_time - first_token_time) / num_output_tokens`

**为什么重要**:

TPOT 反映了系统的**解码吞吐能力**。用户看到第一个字后，多久能看到下一个字？

TPOT 的组成：
- **Decode 时间**：模型每次 forward pass 生成一个 token 的时间
- **调度开销**：每步 Scheduler 的时间摊销到每个 token 上

**反映的设计质量**:

| 组件 | 对 TPOT 的影响 |
|------|---------------|
| Executor | Decode forward pass 的速度是 TPOT 的下限。Qwen2-0.5B 在 CPU 上的解码可能在 100ms+ |
| Scheduler | 如果 decode-first 导致每步只有 1 个 decode token（其他预算给了 prefill），TPOT 仍然很低，但整体 throughput 受影响 |
| Batch Size | 同时 decode 的序列越多，每个 token 的调度开销摊销越低，TPOT 越稳定 |

**在真实 vLLM 中**: TPOT 影响流式体验。在线服务通常要求 TPOT < 50ms。

---

### 3. Throughput — 吞吐量

**定义**: 系统在单位时间内处理的请求数和 token 数。

**公式**:
```
Throughput(req/s) = total_finished_requests / total_time
Throughput(tok/s) = total_output_tokens / total_time
```

**为什么重要**:

Throughput 反映了系统的**总体处理能力**。在给定硬件上能处理多少请求？

**反映的设计质量**:

| 组件 | 对 Throughput 的影响 |
|------|--------------------|
| Scheduler | Continuous batching 的核心优势：通过 decode-first 和 token budget 最大化每步的 token 处理量。如果 CPU 利用率低（token budget 总用不满），说明 Scheduler 策略有优化空间 |
| Executor | 模型 forward pass 的速度是 throughput 的上限 |
| BlockManager | On-demand 分配减少了不必要的 block 占用，使得更多序列可以同时运行，提高 throughput |

**在真实 vLLM 中**: Throughput 是离线批处理场景的优化目标。在线服务需要在 TPOT SLO 约束下最大化 throughput。

---

### 4. KV Utilization — KV Cache 使用率

**定义**: 已分配的物理 block 占总 block 池的比例。

**公式**: `KV_util = used_blocks / total_blocks * 100%`

**为什么重要**:

KV Utilization 反映了**内存管理效率**。分配了 16 个 block，实际用了几个？

**反映的设计质量**:

| 组件 | 对 KV Utilization 的影响 |
|------|------------------------|
| BlockManager | On-demand 分配只在需要时才分配 block，初始使用率低，随着生成增长。如果 peak 使用率远低于 100%，说明 num_gpu_blocks 设置过大 |
| Scheduler | 并发运行的序列数越多，block 使用率越高。如果 peak 使用率低，说明 Scheduler 的 max_num_seqs 限制过紧 |
| Executor | 每个 token 写操作调用 ensure_block()，触发 on-demand 分配 |

**在真实 vLLM 中**: 这是显存管理的关键指标。在真实的 GPU 上，KV Cache 占用了大部分显存。过高的利用率意味着 OOM 风险，过低的利用率意味着浪费。

---

### 5. Block Utilization — Block 使用效率

**定义**: 平均每个 block 存储了多少个 token 的 KV 数据。

**公式**: `Block_util = total_tokens / num_allocated_blocks`

**为什么重要**:

Block Utilization 反映了**空间利用效率**。每个 block 有 `block_size` 个 slot，实际用了几个？

**理想情况**: 每个 block 正好存满 `block_size` 个 token，除了最后一个 block。

**反映的设计质量**:

| 组件 | 对 Block Utilization 的影响 |
|------|---------------------------|
| BlockManager | On-demand 分配的特点：逐 block 分配，只分配需要的数量。最后一个 block 可能不满。利用率接近 block_size（例如，对于 block_size=4，利用率接近 4） |
| 分配策略 | Eager 分配（一次性分配所有需要的 block）会有大量浪费。On-demand 分配只有在 block 写满时才会分配下一个 block |

**在真实 vLLM 中**: Block 利用率反映 PagedAttention 的内存效率。真实 vLLM 的 block 利用率通常在 60-80%（因为每个 block 不完全填满时有内存碎片）。

---

### 6. Scheduler Latency — 调度延迟

**定义**: 每次 `scheduler.schedule()` 调用所花费的时间。

**为什么重要**:

Scheduler latency 反映了**调度算法的开销**。如果 Scheduler 本身就需要 10ms，而模型 forward pass 只要 50ms，那调度开销就占了总时间的 17%。

**反映的设计质量**:

| 组件 | 对 Scheduler Latency 的影响 |
|------|---------------------------|
| Scheduler | 调度算法的复杂度是 O(num_waiting + num_running)。随着等待队列增长，调度时间线性增长 |
| 设计优劣 | Continuous batching 的调度逻辑是否高效？每次遍历队列的开销是否可控？ |

**在真实 vLLM 中**: Scheduler latency 通常远小于模型推理时间（μs 级 vs ms 级）。如果 Scheduler latency 成为瓶颈，说明调度算法需要优化。

---

## 指标之间的关系

```
高 KV Utilization  + 高 Block Utilization  = 内存高效
低 TTFT            + 低 TPOT               = 响应迅速
高 Throughput      + 低 Scheduler Latency  = 系统高效
```

这些指标互相制约：
- **TTFT vs Throughput**：decode-first 策略优先保证 decode，但这会延迟新请求的 prefill（增加 TTFT），但提高了总吞吐量
- **TPOT vs Throughput**：增大 batch size 通常提高 throughput 但可能降低单个 TPOT（因为需要等待更长序列完成）
- **KV Utilization vs Block Utilization**：更多并发序列提高 KV 使用率，但每个序列的最后一个 block 不满，降低 block 利用率

---

## 如何使用 MetricsCollector

```python
from mini_vllm import Config, LLMEngine

config = Config(executor_type="fake", max_new_tokens=16)
engine = LLMEngine(config)

# 添加请求
engine.add_request("Hello, world!", max_new_tokens=8)
engine.add_request("How are you?", max_new_tokens=12)

# 运行到完成
engine.run_until_done()

# 获取 metrics 报告
metrics = engine.engine_core.metrics_collector
report = metrics.report()
metrics.print_report(report)
```

输出示例：
```
=============================
Benchmark Report
=============================

  Requests:              2  (prompt=16 tok, output=20 tok)
  Steps:                 7
  Total time:            0.003s

  TTFT (avg/min/max):    1.12 / 0.76 / 1.48 ms
  TPOT (avg/min/max):    0.15 / 0.12 / 0.18 ms

  Throughput:            714.29 req/s,  7142.86 tok/s

  KV blocks (peak/avg):  5 / 31.2%  (of 16 total)
  Block utilisation:     3.67 tokens/block

  Scheduler latency:     0.0107 ms avg,  0.0305 ms max
  Step latency:          0.4546 ms avg,  1.66 ms total
```
