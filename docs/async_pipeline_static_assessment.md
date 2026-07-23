# Async Pipeline — Static Code Assessment

> **Phase 1 — Static Code Review Only. No profiling data, no runtime execution.**
>
> **Do not cite any performance numbers. No benchmark was run.**

---

## 1. 当前 Engine 单步真实调用链

```
LLMEngine.step()
  └─ EngineCore.step()
       ├─ _check_timeouts()                        # CPU — 遍历 running/waiting 队列
       ├─ Scheduler.schedule()                     # CPU — 调度决策
       ├─ ensure_block() 循环                       # CPU — BlockManager 按需分配
       ├─ ModelInputBuilder.build()                # CPU + H2D — 构建输入张量
       ├─ Executor.execute(model_input)            # GPU — 统一 prefill+decode 前向
       │    └─ QwenModelRunner.execute_model()
       │         ├─ embed_tokens                   # GPU
       │         ├─ for each layer:
       │         │    ├─ RMSNorm + QKV proj         # GPU
       │         │    ├─ RoPE                       # GPU
       │         │    ├─ write_kv_cache             # GPU (scatter write)
       │         │    ├─ prefill_attention          # GPU (SDPA)
       │         │    ├─ decode_attention           # GPU (paged attention)
       │         │    ├─ o_proj + post_attention    # GPU
       │         ├─ final norm + lm_head            # GPU
       │         └─ gather sample positions → logits
       ├─ _apply_model_output()                    # CPU — 写回 sampled tokens
       ├─ cleanup_sequence() / metrics             # CPU
       └─ MetricsCollector.record_step()           # CPU
```

**关键特征**：每个 `EngineCore.step()` 内部所有阶段**严格串行**，无任何重叠。

---

## 2. 当前是否存在 CPU Scheduler 与 GPU Execute 重叠？

**否。完全不存在。**

`engine_core.py:66-174` 中 `step()` 方法的执行流程是：

1. `scheduler.schedule()` — 纯 CPU（Phase 1-6）
2. `ensure_block()` 循环 — 纯 CPU
3. `ModelInputBuilder.build()` — CPU + H2D 拷贝
4. `torch.Tensor` 构造 + `.to(device)` — H2D 同步
5. `executor.execute(model_input)` — 整个 GPU 前向
6. `_apply_model_output()` — CPU（写回序列状态）
7. metrics — CPU

`torch.cuda.synchronize()` **未显式调用**，但由于第 3 步构造张量时有 PyTorch 隐式同步，且第 6 步的 CPU 操作在第 5 步完成后才会遇到 CUDA 同步点，因此 GPU 和 CPU 在单步粒度上无重叠。

---

## 3. 当前是否存在后台 Engine Loop、Future、线程池、多 CUDA Stream 或 asyncio？

| 机制 | 存在？ | 说明 |
|------|--------|------|
| **后台 Engine Loop** | 否 | `run_until_done()` 是 `while True: step()` 同步循环 (`engine.py:129-138`) |
| **Future / concurrent.futures** | 否 | 全项目无 `ThreadPoolExecutor`、`ProcessPoolExecutor`、`Future` 导入 |
| **线程池** | 否 | 无 `threading` 导入，无后台工作线程 |
| **多 CUDA Stream** | 否 | 全项目无 `torch.cuda.Stream` 或 `torch.cuda.StreamContext` |
| **asyncio** | 否 | `api_server.py` 中 `poll_stream()` 是同步 yield，非 async def |
| **torch.compile** | 否 | 无 `torch.compile` 或 `torch.jit` |
| **CUDA Graph** | 否 | 全项目无 `torch.cuda.CUDAGraph` 引用 |

**结论**：当前纯单线程同步，无任何并行机制。

---

## 4. Continuous Batching、SSE Streaming 和 Async Scheduling 分别实现到什么程度？

### 4.1 Continuous Batching

**已实现**，位于 `Scheduler.schedule()` (`scheduler.py:79-248`)。

完整的分阶段调度器包含：
- **Phase 1**: Finish check — 检查 running 序列是否达到 `max_tokens`
- **Phase 2**: Categorize — 拆分为 decode 组和 prefill-continue 组
- **Phase 3**: Decode-first budget — 先扣 decode 预算，剩余给 prefill
- **Phase 4**: Chunked-prefill continue — 继续未完成的 prefill
- **Phase 5**: Admit new — 带前缀缓存感知的新请求准入
- **Phase 6**: Token counts & debug reason

支持 `static_batch_mode`（所有 running 完成才准入新请求）作为对比基线。

**缺失**：无动态抢占（preemption），无权重组调度优先级（如按等待时间升权）。

### 4.2 SSE Streaming

**部分实现**，位于 `api_server.py:139-256`。

- `generate_stream()` — 使用 callback 方式，返回 `(engine_rid, tracking_id, error)` 三元组
- `poll_stream()` — 每个 step 轮询一次，提取新生成的 token，返回文本片段
- `generate_stream_safe()` — 带 disconnect 保护的 generator，`finally` 块确保资源释放
- `StreamManager` — 限制最大并发流数量（`max_streams`）

**不是真正的异步 SSE**。客户端依然通过同步轮询获得新 token，没有 `asyncio` event loop。HTTP 层面的 SSE 格式 (`format_sse_event()`) 已实现，但缺乏异步 HTTP server 集成（如 FastAPI/aiohttp）。

### 4.3 Async Scheduling

**未实现**。当前每个 `EngineCore.step()` 内部 CPU 阶段和 GPU 阶段完全串行。

辅助文件 `benchmark_async_pipeline_profile.py` 中包含了一个 `timed_step()` 包装器，将 step 内的各阶段分离计时并计算 `cpu_gap_ms`（GPU 之间的 CPU 间隙），但这是**测量工具而非实现**。该脚本的目标是量化理论上可隐藏的 CPU 时间，而非实际实现管道重叠。

---

## 5. Scheduler 中哪些工作理论上可以与 GPU Execute 重叠？

### 可以重叠（不依赖当前步 GPU 输出）

| 工作 | 理由 | 重叠难度 |
|------|------|----------|
| **Phase 5: Admit new waiting groups** | 新请求准入只读 `prompt_token_ids` 和前缀缓存，不依赖任意 decode 输出 | 低 |
| **Phase 4: Chunked-prefill continue 的预算计算** | 只读 `prefill_cursor`，不依赖输出 | 低 |
| **为下一步构建 ModelInput (部分)** | decode 序列的 position/slot 可用当前信息提前计算；但 `sample_token_indices` 和 prefill 部分依赖输出 | 中 |
| **为下一步的 ensure_block()** | 若提前知道下步的 decode 序列，可提前分配 | 低 |

### 部分可重叠

| 工作 | 理由 | 重叠难度 |
|------|------|----------|
| **Phase 3: Decode-first budget（部分预算计算）** | token 计数本身不依赖 GPU；但 "哪些序列是 decode"的判定发生在 finish check 后 | 低 |
| **前缀缓存探测** | 只读操作，但仅对新请求有意义 | 低 |

### 不能重叠（严格依赖当前步 GPU 输出）

| 工作 | 依赖 | 原因 |
|------|------|------|
| **Phase 1: Finish check** | 需要 `num_generated_tokens` | 判定是否达到 `max_tokens` |
| **Phase 3: Decode group 确认** | 需要当前步 output token | 判定 prefill 完成后转到 RUNNING 还是继续 PREFILL |
| **`_apply_model_output()`** | 需要 `sampled_token_ids` | 写回序列的 output token |
| **Phase 4 的 cursor 推进** | 依赖 `prefill_cursor` 更新 | 在 _apply_model_output 后执行 |

---

## 6. 哪些工作依赖当前 Decode 输出，不能提前执行？

| 工作 | 依赖的具体数据 | 代码位置 |
|------|---------------|----------|
| **Finish check** | `seq.num_generated_tokens`（由 `_apply_model_output` 更新） | `scheduler.py:92` |
| **prefill → RUNNING 状态转换** | sampled token ID（决定是否完成 prefill） | `engine_core.py:208-210` |
| **下一个 decode 步的 input token** | 上一步 sampled 的 token ID（`output_token_ids[-1]`） | `input_builder.py:100` |
| **prefill 的 sample_indices** | 上一步的 prefill_cursor 更新结果 | `input_builder.py:81-82` |
| **E2E 延迟计算** | finish_time（= finish 后 `time.time()`） | `scheduler.py:95` |
| **TTFT 计算** | first_token_time（= 首次 apply_output 时记录） | `engine_core.py:210` |

核心依赖链条：**GPU forward → sampled tokens → sequence state update → 下一轮的 finish check + input 构建**。打破这个依赖链是实现 async scheduling 的最大挑战。

---

## 7. 当前代码结构实现 Async Scheduling 需要改动哪些模块？

### 7.1 核心改动

| 模块 | 改动 | 性质 |
|------|------|------|
| **`engine_core.py`** | 将 `step()` 拆为 `step_pre_gpu()` + `step_post_gpu()`；引入 CUDA Stream；加入双缓冲 `ModelInput`；管理调度流水线 | **高**（重构） |
| **`executor/base.py`** | `execute()` 改为接受 stream 参数，返回 future/event，不阻塞 | **高**（接口变更） |
| **`scheduler/scheduler.py`** | Phase 1（finish check）与 Phase 2-6 分离；支持"预调度"（predictable decode 序列提前调度） | **中**（重构） |
| **`engine/input_builder.py`** | 支持为已知 decode 序列提前构建 `ModelInput` 的一部分（position/slot mapping） | **中**（扩展） |

### 7.2 配套改动

| 模块 | 改动 | 性质 |
|------|------|------|
| **`engine/engine.py`** | `step()` 可能需要新的非阻塞接口 | **低** |
| **`serving/api_server.py`** | `poll_stream()` / `generate_stream_safe()` 适配新的步进模式 | **低** |
| **`model_runner/qwen_runner.py`** | `execute_model()` 支持 CUDA Stream 参数（`torch.cuda.StreamContext`） | **低** |
| **`attention/paged_attention_gpu.py`** | 验证 Triton kernel 的 stream 安全性（默认应安全） | **低**（验证） |

### 7.3 新增组件

| 组件 | 用途 |
|------|------|
| **PipelineManager** | 管理 CPU-GPU 双缓冲状态（current/next `ModelInput`） |
| **CUDA Stream 池** | 分离 H2D 传输流和计算流 |
| **事件/完成回调机制** | GPU 完成后的轻量通知 |

### 7.4 改动量预估

- **核心改动**: 约 300-500 行
- **配套改动**: 约 100-200 行  
- **新增组件**: 约 200-400 行

总计约 **600-1100 行**，且需要一次模块接口不兼容的重构（`executor.execute()` 签名变更）。

---

## 8. 当前代码结构实现 CUDA Graph 需要解决哪些动态 Shape、Buffer 和地址稳定性问题？

CUDA Graph 要求**静态图**（所有 kernel launch、内存地址、张量 shape 必须在 capture 时固定）。当前架构的每个动态维度都构成阻碍：

### 8.1 动态 Shape 问题

| 动态维度 | 来源 | 每步变化 | 影响 |
|----------|------|----------|------|
| **总 token 数** | `ModelInput.input_ids.shape[0]` | 随 prefill chunk 和 decode 数量变化 | **图必须重新 capture** |
| **prefill token 数** | prefill 序列分 chunk 后不定长 | 每步不同 | 分离 prefill/decode 路径后仍需处理 |
| **decode 序列数** | 取决于 schedule 和 finish | 每步可能不同 | batch size 不固定 |
| **block table 行数** | = decode 序列数 | 同上 | `decode_block_tables` shape 变化 |
| **block table 列数**（max_blocks） | 取决于最长序列的已分配块数 | 序列增长时变化 | 列数不固定 |
| **slot_mapping** | 每 token 一个物理槽 | prefill 时变化 | 长度动态 |
| **position 张量** | = token 数 | 动态 | 同 input_ids |
| **prefill attention 的 Q/K/V** | prefill 为每个 token 产生 | chunk 大小变化 | 无法静态化 |

### 8.2 缓冲区和地址稳定性问题

| 问题 | 原因 | 影响 |
|------|------|------|
| **KV cache 指针变化**（方案B） | write-first 方案中，cache write 的 slot 每步不同 | kernel 输入地址不固定 |
| **block table 内容变化** | 序列增长时新增块 | 张量数据变化，但 graph 只 capture 指针，可以容忍数据变化 |
| **Paged attention 的 `seq_start_loc`** | 取决于序列长度分布 | 在 triton kernel 中作为参数传递 |
| **`max_blocks` padding 变化** | 最长序列块数变化 | 列数变化，图必须重新 capture |

### 8.3 可行路径

CUDA Graph 捕获可以通过以下方式减少图重建频率：

1. **预分配最大 buffer**：`input_ids`、`positions`、`slot_mapping` 分配最大容量（`max_num_batched_tokens`），用 `num_tokens` 掩码控制实际计算量
2. **固定 block table 尺寸**：使用全局 `max_blocks` 固定列数，不足补 -1
3. **分离 prefill 和 decode 为独立图**：prefill 路径形状变化太大难以捕获；decode 路径相对稳定
4. **解码-only 图捕获**：decode 模式时，序列数不变时可捕获，新序列加入时重捕获

**当前 Triton 后端** (`paged_attention_gpu.py`) 已经使用 `seq_start_loc` 等动态参数，但未使用统一 CUDA Graph API。捕获需要在 `QwenModelRunner.execute_model()` 外层包装。

### 8.4 改动量预估

CUDA Graph 需要更彻底的架构修改（主要是 buffer 管理），预估 **800-1500 行**，且对模型路由器的改动多于对调度器的改动。

---

## 9. Async Scheduling 与 CUDA Graph 比较

| 维度 | Async Scheduling | CUDA Graph |
|------|------------------|------------|
| **预期收益来源** | 隐藏 CPU 调度/输入构建时间，提高 GPU 利用率；多个 decode 步之间管道重叠 | 消除 Python 端 kernel launch 开销和 kernel 间同步；提高短序列 decode 的 batch 效率 |
| **收益上限** | `~max(cpu_gap, gpu_time)` — 收益受 CPU 和 GPU 中较慢者限制；CPU 越快，收益越小 | 固定 overhead 节省（~50-500μs/step），对长序列 prefill 收益微；对短 decode 步的收益相对明显 |
| **实现复杂度** | **中**（~600-1100 行）。需拆分 step、引入 stream、双缓冲管理 | **高**（~800-1500 行）。需固定所有动态 buffer、处理图重建、管理图池 |
| **正确性风险** | **中**。双缓冲引入竞态条件（预构建的 ModelInput 与下一轮状态冲突） | **低-中**。CUDA Graph 语义确定性高，但动态 shape 的 fallback 路径可能引入边界错误 |
| **与当前项目契合度** | **高**。当前架构已有明确的前后阶段分离，拆分 step 的改动相对局部。可直接复用现有 `benchmark_async_pipeline_profile.py` 的拆分思路 | **低**。CUDA Graph 对理解 vLLM 核心架构帮助有限，且需要大量与"展示关键抽象"无关的 buffer 管理代码 |
| **架构展示价值** | **高**。展示对 LLM serving 系统深度的理解 — CPU-GPU 管道重叠、调度器解耦、并发控制。可引出的深层话题很多（调度策略、背压、watermark） | **中**。CUDA Graph 更多是 GPU 底层优化技巧 |
| **对当前 profiling 数据的依赖** | **高**。需要知道 CPU gap 的真实大小来判断投入产出比 | **中**。即使没有精确 profile，CUDA Graph 在批量 decode 场景的收益是已知的 |
| **可增量实现** | **可**。可以先在 fake executor 上验证调度重叠逻辑，再迁移到真实 GPU | **难**。必须一次性解决所有动态维度才能得到可用结果 |

### 总结建议

基于当前代码结构和项目目标，**Async Scheduling 的性价比更高**：

1. 当前项目的 CPU 阶段（scheduler + input build）与 GPU 阶段完全串行，这是最明显的等待浪费
2. CPU gap 的量级很可能远大于 kernel launch overhead
3. Async Scheduling 的改动更干净
4. CUDA Graph 对 buffer 管理的额外复杂度会模糊项目核心重点

---

## 10. 下一阶段最小化 Profile 应该只测哪些指标？

### 10.1 测量什么

仅使用 `StageProfiler`（已集成到 EngineCore），不加载模型、不使用 GPU：

```python
# 例：fake executor + StageProfiler
config = Config(executor_type="fake", device="cpu")
```

| 指标 | 意义 | 如何获取 |
|------|------|----------|
| `scheduler_step` 各阶段时间分布 | 调度器内 Phase 1-6 的 CPU 时间 | `StageProfiler.report()` |
| `executor_forward` 耗时 | 模拟 GPU 执行时间（fake executor 用 sleep 模拟） | `StageProfiler.report()` |
| `engine_step_total` | 单步总时间 | `StageProfiler.report()` |
| 单步内 scheduler 占比 | 判断 CPU 调度是否占总时间显著部分 | `StageProfiler._print_bottleneck_hint()` |
| `kv_cache_allocation` 时间 | ensure_block 的开销 | `StageProfiler.report()` |
| `metrics_update` 时间 | 度量收集开销 | `StageProfiler.report()` |

### 10.2 不测量什么

- 不测量 TPOT/TTFT/E2E（需要真实模型）
- 不测量 GPU kernel 时间
- 不测量 CUDA 事件时间差
- 不测量 H2D 传输时间
- 不测量多个并发度

### 10.3 测试场景

| 场景 | 请求数 | 目的 |
|------|--------|------|
| 1 请求短生成 | 1, max_tokens=16 | 基准单步时间 |
| 4 请求并发 | 4, max_tokens=16 | 调度阶段压力 |
| 长 prompt + 短生成 | 1, prompt=256t, gen=1 | prefill 阶段的 scheduler 开销 |
| 混合到达 | 4 交错到达 | 连续批处理的调度器热路径 |

### 10.4 输出

```
# 预期输出格式（来自 StageProfiler.print_report()）

  Stage Breakdown
  ----------------------------------------------------
  stage                        count  total_ms    avg_ms    max_ms    pct
  ----------------------------------------------------
  engine_step_total               24   123.45    5.1438   12.3456  100.0%
  scheduler_step                  24    12.34    0.5143    1.2345   10.0%
  executor_forward                24   111.11    4.6296   11.1111   90.0%
  ...
```

上述 profile 使用 fake executor（超轻量），无需 GPU、无需模型权重、不会 OOM。

---

## 静态初步结论

1. **当前调度器和执行器完全串行** — CPU 阶段和 GPU 阶段之间无任何重叠，`EngineCore.step()` 内的所有工作在同一线程中顺序执行。

2. **Continuous Batching 已完整实现** — 六阶段调度器（finish → categorize → decode-first budget → chunked prefill continue → admit new → accounting）实现完整。

3. **流水线重叠的主要瓶颈是 CPU 调度/输入构建与 GPU 执行的串行化** — 调度器输出影响输入构建，输入构建影响 GPU 输入，三者当前是一个原子步。

4. **依赖链限制了重叠空间** — finish check 和 apply_output 严格依赖当前步 GPU 输出，不可提前执行。但在 decode 密集阶段（序列数稳定时），下步调度的大部分信息是预先可知的。

5. **当前代码结构适合引入 Async Scheduling** — 模块边界清晰（Scheduler/EngineCore/Executor），拆分 `step()` 的改动相对独立且可增量实现。

6. **CUDA Graph 在当前阶段优先级低** — 投入产出比差，且可能使代码复杂度偏离项目焦点。

> **本报告不包含任何性能数据或性能推测。所有结论基于源码静态分析。**
