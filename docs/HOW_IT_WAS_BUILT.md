# mini-vLLM 项目搭建全过程 — 面试说明

> 本文档基于当前代码（176 tests passing）、docs、examples、git history 和模块依赖关系，按阶段说明项目如何从零构建。
> 
> 项目只有 2 个 git commit，以下按**模块依赖的递进关系**说明，而非严格时间线。

---

## 1. 初始目标和技术边界

### 目标
从零复现 vLLM 核心架构（Continuous Batching + PagedAttention + Prefix Cache），深入理解 LLM Serving 引擎的工作机理。不是生产级推理引擎，而是**教育项目**——让你能在单线程中单步 trace 调度和内存分配的全过程。

### 技术边界
- **不写 CUDA / Triton kernel** — PagedAttention 的 GPU kernel 层不在范围内
- **不做 GPU 显存管理** — 用 Python dict 模拟 KV cache
- **不做分布式推理** — 单进程单线程
- **第一版不做 Prefix Cache** — 后续迭代引入
- **第一版不做 Chunked Prefill** — 后续迭代引入
- **第一版不做 HTTP 服务层** — 后续 iteration 引入

### 约束
- 纯 Python 3.10+，零外部依赖（fake executor 模式）
- 所有组件可单步调试
- Fake Model 用 ASCII 算术模拟 Transformer，用 Python dict 模拟 KV cache

---

## 2. 第一版最小闭环包含哪些模块

按依赖关系自底向上构建：

### 数据模型层
- **Status** — WAITING(0) / PREFILL(1) / RUNNING(2) / FINISHED(3) / REJECTED / CANCELLED 状态枚举
- **SamplingParams** — max_tokens, temperature, top_p, top_k 的 dataclass
- **Sequence** — 核心数据载体。持有 token 缓冲区、状态机、KV block table、计时戳
- **SequenceGroup** — 用户请求的封装。持有 prompt、采样参数、子序列
- **RequestQueue** — 4 池队列（waiting/running/finished/rejected），全部存 SequenceGroup

### 配置层
- **Config** — 集中管理所有可调参数，dataclass 设计

### KV Cache 管理层
- **BlockAllocator** — 底层物理块池，free list 管理，支持回调
- **BlockTable** — 逻辑 → 物理块映射（PagedAttention 核心思想）
- **BlockManager** — 上层协调器。管理 per-sequence BlockTable，封装分配逻辑

### 调度层
- **Scheduler** — 初见最简单的 schedule()：Finish → Decode → Admit 三步
- **ScheduleResult** — 调度结果的结构化报告

### 执行层
- **FakeModelExecutor** — 用 Python dict 模拟 KV cache。prefill 写 KV、decode 读 KV 影响输出
- **FakeWorker** — Worker 工厂，创建 executor

### 引擎层
- **EngineCore** — 内循环：schedule → prefill → decode → cleanup
- **LLMEngine** — 对外 API：add_request, step, run_until_done

### 设计决策：LLMEngine / EngineCore 拆分
EngineCore 是内循环（持有 scheduler + executor），LLMEngine 是对外 API（持有所有组件 + 输出捕获）。这 mirror vLLM 的生产架构——EngineCore 可以在后台线程中持续运行，LLMEngine 处理请求入队和结果收集。

---

## 3. Scheduler 是什么时候加入的？解决什么问题？

Scheduler 从一开始就存在——它是 Continuous Batching 的核心。但初版的 Scheduler 非常简单：

```
Phase 1: Finish — 检查 running 组，完成的释放 block、移入 finished 池
Phase 2: Decode — 剩余 running 组成为 decode batch
Phase 3: Admit — 从 waiting 池取出能放下的请求，创建 Sequence，分配 KV block，标记 PREFILL
```

### 解决的问题
Continuous Batching 和 Static Batching 的区别：

```
Static Batching:
  Batch 1: [A: 4 tok, B: 4 tok, C: 4 tok] → 4 steps, 全部完成
  Batch 2: [D: 256 tok, E: 4 tok, F: 4 tok] → E 和 F 等 D 跑完 256 步
  → E 和 F 的 P99 延迟 = 256 steps

Continuous Batching:
  Step 1: prefill A, decode [B, C] (B,C 是上一批没完成的)
  Step 2: decode A, prefill E (E 刚进来)
  Step 3: decode A, decode E, prefill F
  → E 和 F 很快就开始生成 token
  → 但每步 budget 被 A 的 decode 占用一部分
```

### 后续演进
在 Phase 2 中，Scheduler 从 3 个阶段扩展到 6 个阶段：

1. **Finish** — 检查完成的序列
2. **Categorize** — 分为 decode、prefill_continue、running
3. **Decode-First Budget** — decode 先消耗预算
4. **Chunked-Prefill Continue** — 未完成的 prefill 继续
5. **Admit New** — 准入新请求（含 prefix cache probe）
6. **Token Counts** — 汇总统计

---

## 4. KV Cache / BlockManager / Prefix Cache / Metrics（简述）

For detailed documentation on KV cache architecture, see [`docs/Memory_Manager.md`](./Memory_Manager.md).
For prefix cache design, see [`docs/PrefixCache.md`](./PrefixCache.md).
For metrics formulas, see [`docs/Metrics.md`](./Metrics.md).

Sections 4–6 of the original document (KV Cache architecture, Prefix Cache design, Metrics formulas) have been
removed from this build-narrative document because they duplicate the dedicated docs linked above. The
build-order logic remains: data model → KV cache → scheduler → executor → engine → metrics → profiling → serving.

## 7. Stage Breakdown Profiling 的设计

### 加入动机
端到端的 TTFT/TPOT 只能告诉你 "慢了多少"，不能告诉你 "慢在哪里"。Stage Breakdown 将 serving request 的延迟拆解为 10 个独立阶段：

### 可拆的阶段
| 阶段 | 来源 | 反映 |
|------|------|------|
| `request_queue_waiting` | EngineCore 记录 | 队列等待时间 |
| `scheduler_step` | schedule() 调用 | 调度算法 overhead |
| `prefix_cache_lookup` | probe_prefix_cache | 前缀探测 overhead |
| `kv_cache_allocation` | ensure_block | 物理块分配时间 |
| `prefill` | executor.prefill() | prompt 处理时间 |
| `decode` | executor.decode() | 逐 token 生成时间 |
| `executor_forward` | 合并 prefill+decode | 总体推理时间 |
| `kv_cache_release` | BlockManager.free | 块释放时间 |
| `metrics_update` | record_step | 指标采集 overhead |
| `engine_step_total` | 整个 step | 基准，其他 % 基于此 |

### 不可拆的阶段
- GPU kernel 级分解（attention、FFN、sampling）
- PyTorch / CUDA kernel overlap
- CPU-GPU 同步开销
- Tokenizer 耗时（在 step loop 外）

### 实现
- 纯 Python `time.time()` + context manager
- `StageProfiler` 独立类，`record(stage)` context manager
- `record_raw(stage, duration_s)` 直接记录
- 支持 start/end 对称 API
- 自动 bottleneck hint（如 "scheduler_step dominates"）

### 验证策略
15 个测试覆盖：单 stage record、聚合统计（count/total/avg/max）、exception 处理、empty report、reset、start/end API、EngineCore 集成、cancel 后不 crash。

---

## 8. Serving Layer / Fault Injection 的设计

### 架构
```
POST /generate
  ↓
AdmissionControl.check()
  ├── prompt_too_long?     → PROMPT_TOO_LONG
  ├── queue_overflow?      → QUEUE_OVERFLOW
  └── block_exhausted?     → BLOCK_EXHAUSTED
  ↓
RateLimiter.check()
  ├── RPM gate
  └── TPM gate
  ↓
StreamManager.try_acquire()     [if stream=True]
  └── TOO_MANY_STREAMS
  ↓
Engine.add_request() → run_until_done()
  ↓
StreamManager.release()
Response
```

### 为什么需要多层守卫
```
没有 Admission Control：
  1 个 10k-token prompt 进入 Scheduler
  → chunked prefill 2500 步
  → 阻塞所有 decode 2500 步
  → KV cache 最终耗尽 → OOM crash

有 Admission Control：
  1 个 10k-token prompt 在 admission 阶段被 BLOCK_EXHAUSTED
  → 零引擎开销
```

### Disconnect 生命周期

这是 serving 层最复杂的部分。关键问题：**客户端断开连接 ≠ GPU 资源释放**。

```
Client 断开
  → TCP socket EOF
  → Serving 层知道 client 断开
  → 但 Engine 的 SequenceGroup 仍在 _running
  → BlockManager 的 ref_count > 0
  → Scheduler 持续分配 token budget
  → GPU 持续计算 token，没人消费
  → 资源泄漏直到 sequence 自然完成（max_tokens）
```

**解决方案**：`generate_stream_safe` 包装器 + generator close 自动 cleanup：

```python
def generate_stream_safe(self, prompt, max_tokens):
    try:
        while not finished:
            token_text, gen_count, finished, finish_reason = ...
            yield (token_text, ...)
    finally:
        # 不管因为什么退出（disconnect、exception、正常完成）
        # 都确保 _abort + release
        if not finished:
            self._abort(engine_rid, tracking_id)
        finally:
            self._stream_manager.release(tracking_id)
```

### 验证的异常路径

| 场景 | 测试 | 验证点 |
|------|------|--------|
| 队列满 | test_queue_overflow_rejects_new | QUEUE_OVERFLOW 不 crash |
| 队列 drain 后恢复 | test_queue_accepts_after_drain | 状态残留不阻止新请求 |
| Block 耗尽 | test_block_exhaustion_rejects_new | BLOCK_EXHAUSTED |
| 极端 block 短缺 | test_block_exhaustion_does_not_crash | 1 block 也不 crash |
| 请求完成后 block 回收 | test_blocks_recovered_after_finish | block 回到 free pool |
| Stream 并发上限 | test_stream_exhaustion_rejects | max_streams 保护 |
| Timeout Storm | test_timeout_releases_blocks | 4 请求全部 timeout 后 block 全释放 |
| Cancel Storm | test_cancel_ref_count_integrity | 共享 block 的 ref_count 全部归零 |
| Disconnect 资源清理 | test_streaming_disconnect_comprehensive | 同时验证 queue/block/table/stream/metrics 全部清空 |
| Double Abort | test_streaming_disconnect_no_double_abort | metrics 不重复计数 |
| 无 admission control 的 OOM | test_10blocks_no_admission_control_crashes | 证明 admission control 的必要性 |

验证不变量：`cancel/success 后 all ref_counts == 0 → 无泄漏`。

---

## 9. pytest 最新结果

```
PYTHONPATH=. python3 -m pytest tests/ -q --tb=short
176 passed, 1 warning in 3.00s
```

### 测试文件覆盖

| 文件 | 数量 | 覆盖内容 |
|------|------|---------|
| `test_request.py` | ~12 | Status 枚举、SamplingParams、Sequence 生命周期、SequenceGroup 状态、get_unfinished_seqs |
| `test_kv_cache_manager.py` | ~13 | BlockTable add/get/clear、Allocator allocate/free/OOM/callback/stats、Manager on-demand/free/stats |
| `test_prefix_cache.py` | ~20 | PrefixCache insert/lookup/span、hash 确定性、ref_count 递增/递减/双释放安全、共享语义、probe 只读、stale entry、Scheduler 集成 |
| `test_scheduler.py` | ~12 | Admit/Prefill/Decode/Finish 生命周期、max_num_seqs、decode-first、chunked prefill、ignored_reasons |
| `test_engine.py` | ~9 | run_until_done、mid-arrival merge、OOM、ScheduleResult 字段、KV 统计 |
| `test_serving_layer.py` | ~20 | Generate 非流式/流式、RPM 限制、Admission Control、Stream Manager、Cancel、Timeout、Metrics 端点、Disconnect |
| `test_fault_injection.py` | ~15 | Queue overflow、Block exhaust、Stream exhaust、Timeout storm、Cancel storm、Stale entry、Admission block pressure |
| `test_metrics.py` | ~39 | TTFT 字段/顺序/多请求、TPOT 分母/单 token 排除/min/avg/max、Throughput wall-clock vs active、cancel/timeout 不污染、KV util、Scheduler latency、Prefix cache metrics、Stage profiler metrics、Serving counters、Metrics endpoint |
| `test_stage_profiler.py` | ~15 | record/report/reset、aggregation、context manager exception、start/end、percentage、EngineCore integration、cancel stability |

---

## 10. Demo 怎么跑

### demo_fake_engine.py
```bash
python examples/demo_fake_engine.py
```
**证明什么**：
- 3 个请求经过 continuous batching（请求 A → B 一起到，C 中途到达）
- Chunked prefill 将长 prompt 分步处理
- Memory trace 模式展示每步的 BlockAllocator free list 和 per-sequence BlockTable
- On-demand 分配对比 Eager 分配的节约量

**输出示例**：每步打印 waiting/running/scheduled prefill/scheduled decode/KV blocks，最后打印 Benchmark Report。

### benchmark.py
```bash
python examples/benchmark.py --executor fake --requests 4
python examples/benchmark.py --executor qwen --requests 2 --tokens 16
```
**证明什么**：
- 结构化 Metrics 报告（TTFT/TPOT/Throughput/KV Util/Scheduler Latency）
- Fake vs Qwen 双 executor 模式
- KV block utilisation 分析（>50% 为 good，<50% 为 over-provisioned）

### demo_stage_breakdown.py
```bash
python examples/demo_stage_breakdown.py --executor fake --requests 16 --tokens 16
```
**证明什么**：
- 端到端延迟可拆解为 10 个独立阶段
- 每阶段提供 count/total_ms/avg_ms/max_ms/percent_of_total
- 自动 bottleneck hint 识别瓶颈阶段
- Fake vs Qwen 模式对比 Python overhead vs 真实推理

---

## 11. Claude Code 辅助了什么，我做了什么

### Claude Code 辅助生成的部分

1. **测试代码生成** — 根据功能模块的接口自动生成 test template，包括 test class 定义、fixture setup、基本 test case 结构。这是最典型的应用场景。

2. **文档框架** — README.md、docs/ 目录下的架构文档（Phase1_Architecture.md、Scheduler.md、Memory_Manager.md、PrefixCache.md 等）的大量初稿由 Claude 生成，我负责 review 和修正。

3. **配置样板** — `pyproject.toml`、`.gitignore` 等基础设施文件。

4. **代码审查** — Claude 帮助 diff review，发现潜在的 edge case（如 double-free 安全、ref_count 边界条件、TPOT 分母处理）。

### 我（开发者）负责的工程判断

1. **模块拆分决策** — 决定 SequenceGroup/Sequence 的分离粒度（mirror vLLM but not over-engineered）、决定 BlockAllocator/BlockManager/BlockTable 三层的职责划分。

2. **Scheduler 6-Phase 架构** — 从最初 3 阶段演进到 6 阶段的完整设计。Decode-first、Chunked Prefill、Token Budget Model 都是我设计的，Claude 不参与算法决策。

3. **On-Demand 分配对比 Eager 的切换** — 这是架构级 decision。分析了两种策略的 trace 数据后决定替换。

4. **Two-Phase Probe + Allocate** — Prefix Cache 的这个核心设计模式（probe 只读、allocate 才改 ref_count）是我设计的。Claude 帮助完善了缓存一致性逻辑。

5. **Metrics 口径定义** — TTFT/TPOT 公式、分母选择、wall-clock vs active time 的区分、cancel/timeout 排除逻辑。这些涉及到"什么算正常的语义问题"。

6. **Fault Injection 测试的场景设计** — 哪些异常路径需要覆盖（cancel storm、timeout storm、disconnect lifecycle、no-admission OOM 证明）。

7. **架构文档的准确性审查** — Claude 生成的文档中 incorrect 或 misleading 的描述需要逐条修正。

**总结**：Claude 是 coding assistant + documentation accelerator + test template generator。所有架构决策、算法设计、质量门禁（code review 后是否 merge）由我完成。

---

## 12. 当前项目不能证明什么

1. **GPU 推理性能** — Fake executor 的 TTFT/TPOT 是纯 CPU 算术时间，不代表真实推理性能

2. **GPU Kernel 级优化** — 没有 CUDA kernel、没有 Nsight Systems、没有 PyTorch profiler

3. **真实硬件端到端延迟** — Fake executor 的数字对真实部署毫无意义

4. **生产规模吞吐** — 测试在小配置下运行（4 blocks、16 tokens）

5. **GPU 显存压力** — Fake KV cache 是 Python dict，不是 GPU 显存

6. **多 GPU / 分布式推理** — 未实现

7. **长上下文正确性** — 未测试超长 prompt

8. **Tokenizer 正确性** — Fake tokenizer 用 ASCII mod，不是真正的 tokenizer

9. **真实 Preemption / Swap** — OOM 时直接 raise RuntimeError，没有 vLLM 的 swap-out + restore

10. **Prefix Cache LRU Eviction** — 当前无淘汰策略，生产系统需要

11. **Copy-on-Write** — 架构预留了扩展点（BlockTableEntry.is_shared、ref_count），但未实现 COW

12. **并发引擎步骤** — 引擎是单线程同步运行，不是真正的异步 pipeline

---

## 2 分钟面试回答

### "这个项目你是如何一步一步搭建的？使用了哪些方法？"

---

**第一段：问题定义**

"我当时的目标不是做生产系统，而是一个清晰的教育复现——用纯 Python 实现 vLLM 的 Continuous Batching、PagedAttention 和 Prefix Cache 核心架构，零外部依赖，能在单线程中单步调试。

技术边界很明确：不做 CUDA kernel、不做 GPU 显存管理、不做分布式。用 ASCII 算术模拟 Transformer，用 Python dict 模拟 KV Cache。

这么做的好处是：所有调度算法和内存管理逻辑都可以用 pdb 走通，不用和 GPU 驱动打交道。"

---

**第二段：架构演进**

"我从数据模型层开始构建。最底层是 Sequence 和 SequenceGroup——Sequence 持有 token 缓冲区、状态机和 KV block table；SequenceGroup 封装用户请求。

然后构建了三层 KV 缓存架构：
- 底层是 BlockAllocator（物理块池，free list + 引用计数）
- 中间是 BlockManager（协调器，按需分配）
- 顶层是 BlockTable（逻辑→物理映射）

关键决策是把 Eager 预分配改为 On-Demand 按需分配——block 只在写入时才分配，而不是 admission 时就全部预留。文档中的 trace 对比显示能节省约 60% 的显存空间。

Scheduler 从最初的 3 阶段（Finish→Decode→Admit）演化到 6 阶段，加入了 Decode-First 优先级、Chunked Prefill、Token Budget 模型。Decode-First 保证正在生成的用户不会因为新的 prefill 请求而卡顿。

Prefix Cache 是后来加入的，核心设计是 Two-Phase Probe + Allocate——Scheduler 在计算 budget 前只读地探测缓存，知道了 uncached 长度后再做调度决策，最后 admission 时实际 attach 共享 block。Ref Count 保护共享 block 的 UAF。"

---

**第三段：质量保障**

"测试覆盖是逐步积累的。当前 176 个测试覆盖了每个阶段——从基础的 Sequence 生命周期（4 个测试），到 KV Cache 管理层（13 个测试），到 Prefix Cache 的 ref_count 语义（20 个测试），到 Scheduler 的 decode-first 和 chunked prefill（12 个测试），到 Engine 集成（9 个测试），到 Serving Layer 的 HTTP 生命周期（20 个测试），到故障注入场景（15 个测试），到 Metrics 公式的精确语义验证（39 个测试），再到 Stage Profiler 的计时准确性（15 个测试）。

其中 Metrics 测试是最有价值的——验证了 TPOT 用 num_output_tokens - 1 做分母、cancel 请求不污染 throughput、throughput 分母包含 idle gap 而非 per-request latency 这些细节。

Fault Injection 测试覆盖了 cancel storm、timeout storm、disconnect lifecycle 和没有 admission control 时的 OOM crash。核心不变量是：cancel/success 后所有 block 的 ref_count 必须归零。"

---

**第四段：工程方法与总结**

"方法论上分三层：
- **我独立完成**：架构设计（模块拆分、调度算法、Two-Phase Prefix Cache、On-Demand 分配）、Metrics 口径决策、异常路径场景设计
- **Claude Code 辅助**：测试模板生成、文档初稿、diff review 发现 edge case
- **Claude + 我协作**：Claude 生成代码我 review，发现的问题修正后迭代

这个项目的价值不在于性能，而在于你可以在 commit-by-commit 的粒度上理解 vLLM 的调度器在做什么、BlockManager 为什么需要 ref_count、Prefix Cache 如何降低 TTFT。它把一套工业级的 serving 架构拆解成可单步调试的 Python 代码——这是我面试时最能讲清楚的项目。"

---

## 附录：vLLM 对应关系速查表

For a complete module-by-module mapping, see [`docs/VLLM_Mapping.md`](./VLLM_Mapping.md). <!--
Key correspondences: LLMEngine↔LLMEngine, EngineCore↔EngineCore, Scheduler.schedule()↔Scheduler.schedule(),
ScheduleResult↔SchedulerOutputs, BlockAllocator↔BlockAllocator (with ref_count additions),
BlockManager↔BlockSpaceManager (with prefix cache integration),
BlockTable↔BlockTable (with is_shared flag), PrefixCache↔PrefixCache/BlockPrefixMgr (no LRU),
FakeModelExecutor↔no direct equivalent, MetricsCollector↔StatLogger (centralized vs distributed),
StageProfiler↔no direct equivalent (educational tool).
-->
