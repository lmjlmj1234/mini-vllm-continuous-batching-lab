# Async Pipeline — Priority Decision Report

> **Synthesis of Phase 1 (static code review) and Phase 2 (minimal GPU profile).**
>
> No additional code was run in producing this document.

---

## Data Basis

| Source | Key finding |
|--------|-------------|
| `async_pipeline_static_assessment.md` | Engine step is fully serial; no concurrency mechanisms; Continuous Batching complete; Async Scheduling and CUDA Graph unimplemented |
| `minimal_async_profile.json` | GPU forward = **37.59 ms avg (97.6%)**, CPU gap = **923 us avg (2.4%)**, theoretical max speedup = **1.024x** |
| `minimal_async_profile.md` | Input build dominates CPU gap (848 us / 923 us = 92%), scheduler is negligible (31 us) |

Profile condition: Qwen2.5-0.5B, batch_size=1, context_length=128, triton attention backend, 10 recorded decode steps after 3 warmup steps.

---

## 1. Async Scheduling 是否值得做？

**当前配置（batch_size=1）下：不值得。**

- GPU forward 占 step 的 **97.6%**，CPU gap 仅 **2.4%**（923 us）
- 理论最大加速比 **1.024x**，隐藏效率取 70% 后实际约 **1.017x**
- 增加双缓冲、CUDA Stream、PipelineManager 的复杂度（~600-1100 行）与 ~2% 的收益不成比例

**但在更高 batch size 下可能改变结论：**

- Scheduler 开销随 running 序列数线性增长（`O(N)` 的 finish check + block table query）
- Input_build 开销也随 token 数增长（tensor 构造、H2D 拷贝）
- 每步需要 schedule 的序列组越多，前缀缓存探测开销越大
- 在 batch_size=8 或 16 时，CPU gap 可能增长到 step 的 10-20%

**结论**：以当前数据，不建议立即实现。应先在 batch_size=4/8 下验证 CPU gap 的扩增趋势，再决策。

---

## 2. 它属于 P0、P1 还是 P2？

| 级别 | 定义 | 对应 |
|------|------|------|
| **P0** | 阻塞性缺失，不实现则项目功能不完整 | — |
| **P1** | 重要功能，有明确收益，应尽快实现 | — |
| **P2** | 锦上添花，有收益但不是核心目标 | **Async Scheduling** |

**归类为 P2**。理由：

- 没有它引擎依然能完整执行 Continuous Batching 的全流程
- 项目的核心功能已由现有模块覆盖（Scheduler、Paged Attention、KV Cache、Executor）
- 当前数据不支撑"显著性能提升"的承诺
- 但代码结构已预留拆分空间（`step()` 内各阶段的边界清晰），未来实现成本可控

---

## 3. CUDA Graph 是否优先级更高？

**否。同样属于 P2，且优先级略低于 Async Scheduling。**

比较维度：

| | Async Scheduling | CUDA Graph |
|--|------------------|------------|
| 当前收益空间 | 2.4% (batch=1) | 更小（kernel launch overhead << 2.4%） |
| 实现复杂度 | 600-1100 行 | 800-1500 行 |
| 对架构目标的贡献 | 展示系统级 pipeline 设计 | 展示底层 GPU 优化技巧 |
| 工程价值 | 高（调度器解耦、并发控制） | 中（CUDA Graph API 知识） |
| 与现有架构的契合度 | 高（模块边界清晰） | 低（动态 shape 广泛） |
| 可增量验证 | 可用 fake executor 先验证逻辑 | 必须一次性解决所有动态维度 |

CUDA Graph 的核心场景（短序列、小 batch、高 launch 频率）在当前 profiling 数据中没有体现——GPU forward 37.6 ms 中，matmul 计算占绝对主导，launch overhead 占比可忽略。

---

## 4. 如果项目只增加一个异步或 Pipeline 优化，应选哪个？

**建议选择：SSE Streaming 异步化（Async HTTP Server）。**

这不是"传统意义"的 pipeline 优化，但这三个候选方案中：

| 候选 | 预期收益 | 复杂度 | 工程价值 |
|------|----------|--------|----------|
| Async Scheduling | 2-5% 性能 | 中 | 高 |
| CUDA Graph | <1% 性能 | 高 | 中 |
| **Async HTTP (SSE)** | 新能力 | 低 | 中 |

Async HTTP 不是性能优化而是功能完善——当前 `generate_stream_safe()` 使用同步轮询，无法在 FastAPI/uvicorn 中提供真正的非阻塞 SSE。加上 async handler 后：

- 可以同时服务多个慢速客户端而不阻塞 worker 线程
- 展示从 HTTP → admission → engine 的端到端异步链路
- 改动量小（仅 serving 层，不改 engine 核心）
- 可自然过渡到"如何背压"、"如何连接池管理"等设计问题

如果坚持在 engine 层做 pipeline 优化，则 **Async Scheduling** 优于 CUDA Graph。

---

## 5. 哪个方案更偏性能收益？

### CUDA Graph

CUDA Graph 是纯性能方案。它不改变系统架构，不增加新能力，只降低已有路径的开销。收益是确定性的（只要图能捕获，launch overhead 一定归零），但上界低。

### Async Scheduling

Async Scheduling 既是性能方案也是架构方案。在性能层面，它隐藏 CPU 延迟；在架构层面，它引入流水线状态管理（双缓冲、事件通知）。即使收益仅为 2-5%，增加的架构抽象本身有助于理解生产级系统的设计。

**性能收益比较（batch_size=1）：**

| 方案 | 理论收益上限 | 置信度 |
|------|-------------|--------|
| Async Scheduling | 2.4% step time | 高（profile 直接测量） |
| CUDA Graph | ~0.1-0.5% step time | 低（估算，未测量 launch overhead） |

**在 batch_size=8 场景两者收益都可能增长，但 CUDA Graph 增长更快**（launch overhead 与 batch size 成正比），需要 profile 数据验证。

---

## 6. 哪个方案更偏架构完整度？

### Async Scheduling

引入：
- **PipelineManager**：管理双缓冲状态机（current `ModelInput` vs next `ModelInput`）
- **CUDA Stream 分离**：H2D 传输流 vs 计算流，可独立调度
- **完成通知机制**：GPU kernel completion callback 触发下一轮调度
- **阶段解耦**：将 `EngineCore.step()` 拆为 `schedule()` + `prepare_input()`（CPU）+ `execute()`（GPU）+ `finalize()`（CPU），每个阶段可以独立测试

这些抽象展示了生产级 LLM serving 系统如何处理 CPU-GPU 流水线。

### CUDA Graph

不改变架构，只改变执行路径。核心改动在 `QwenModelRunner.execute_model()` 外层包装，在内层使用 `torch.cuda.CUDAGraph.make_graphed_callables()` 或类似机制。对架构完整度贡献有限。

---

## 7. 各方案的最小实现范围

### Async Scheduling 最小实现

```
mini_vllm/engine/
  ├── engine_core.py      # [修改] step() 拆为 step_pre_gpu / step_post_gpu
  ├── pipeline_manager.py # [新增] 双缓冲 ModelInput 管理
  └── input_builder.py    # [修改] 增量构建下一轮的 input（仅 decode 部分）

mini_vllm/executor/
  └── base.py             # [修改] execute() 接受 stream 参数（可选）

mini_vllm/scheduler/
  └── scheduler.py        # [修改] schedule() 支持提前执行 Phase 4+5（跳过 finish check）
```

不涉及：attention backend、model runner、serving layer。

### CUDA Graph 最小实现

```
mini_vllm/model_runner/
  ├── qwen_runner.py      # [修改] execute_model() 外层 CUDA Graph capture wrapper
  └── graph_manager.py    # [新增] 图池管理、输入输出别名管理

mini_vllm/cache/
  └── pool.py              # [修改] 固定 KV cache 地址（图捕获期间）

mini_vllm/engine/
  └── input_builder.py      # [修改] 使用预分配最大 buffer 替代动态分配
```

### SSE Async HTTP 最小实现

```
serving/
  ├── api_server.py       # [修改] 添加 async def generate endpoint
  └── sse.py              # [修改] 支持 AsyncGenerator 接口
```

不涉及 engine 层任何修改。

---

## 8. 各方案的验收指标

### Async Scheduling

| 指标 | 通过标准 | 测量方式 |
|------|----------|----------|
| CPU-GPU 重叠 | scheduler + input_build start after GPU launch | CUDA Event 时间戳验证 |
| 无竞态 | 连续 1000 步无 token 顺序错误、无采样错位 | 确定性 seed + seq_id 验证 |
| 加速比 | step time 减低 > 预期 hideable 的 50% | StageProfiler 对比 |
| 测试通过 | 现有 test_engine_e2e.py 全部通过 | pytest |
| 单步正确性 | 输出 token 与串行版本逐序列逐 token 一致 | HF alignment test |

### CUDA Graph

| 指标 | 通过标准 | 测量方式 |
|------|----------|----------|
| 图捕获成功 | 首次 warmup step 无 re-capture 异常 | Python 异常监控 |
| 输出一致 | 图路径与非图路径输出 diff=0 | 逐 token 对比 |
| decode 加速 | 纯 decode 阶段 step time 降低 > 5% | 仅在高 batch 下有意义 |
| 动态 shape 回退 | 任意动态 shape 自动 fallback 到 eager 模式 | 功能测试 |

### SSE Async HTTP

| 指标 | 通过标准 | 测量方式 |
|------|----------|----------|
| 非阻塞 | 同时 4 个慢速流不阻塞第 5 个请求 | concurrency test |
| 断连清理 | disconnect 后 1s 内释放所有 engine 资源 | stream_manager active 计数 |
| E2E 正确 | streaming 输出 = 非 streaming 输出 | 逐 token 对比 |

---

## 9. 推荐执行顺序

```
Phase A [高优先级] — 功能补全
  └─ SSE Streaming 异步化（FastAPI / uvicorn async handler）
      ├─ 收益：解锁真正的非阻塞 SSE
      ├─ 工作量：~200-300 行（仅 serving 层）
      └─ 工程权重：中-高

Phase B [中等优先级] — 为 pipeline 铺路
  └─ batch_size=4/8 条件下的最小 profile 扩增
      ├─ 收益：确认 CPU gap 是否随 batch 扩张到值得优化的水平
      ├─ 工作量：~50 行配置修改
      └─ 决策门：若 batch=4 时 CPU gap > 5% → 进入 Phase C

Phase C [低优先级, 门控] — Async Scheduling
  ├─ 前置条件：Phase B 确认 CPU gap > 5%
  ├─ 收益：取决于 Phase B 数据，当前预估 2-5%
  ├─ 工作量：~600-1100 行
  └─ 替代方案：若 CPU gap 仍 < 5%，完全放弃本方向

Phase D [不推荐单独做] — CUDA Graph
  ├─ 前置条件：None（在任何阶段都可独立启动）
  ├─ 收益：低（batch_size=1 时可忽略）
  ├─ 工作量：~800-1500 行
  └─ 建议：仅当项目重点转向 GPU 底层优化时考虑
```

---

## 10. 哪些结论仍需进一步 Profile 才能确认

| 待确认点 | 当前状态 | 需要的 Profile | 优先级 |
|----------|----------|----------------|--------|
| **batch_size=4/8 CPU gap 扩增幅度** | 仅在 batch=1 测量过 | fake executor + 多请求并发 profile | **高**（决策门） |
| **Input_build 中 H2D vs CPU 时间占比** | 混合测量（848 us 总） | `torch.cuda.Event` 分离 H2D 同步时间 | **中** |
| **Scheduler 在更高 concurrency 下是否会成为瓶颈** | batch=1 仅 31 us | batch=4/8 的 scheduler 时间分布 | **中** |
| **CUDA kernel launch overhead** | 未测量 | CUDA Graph replay vs eager 的 kernel time 对比 | **低** |
| **编解码（tokenize/detokenize）是否与 engine 路径重叠** | 未测量 | tokenize + detokenize 时间 | **低** |

---

## 总结

```
优先级梯队：

  P1 — SSE Streaming 异步化          [功能完整度]
  P2 — batch_size 扩增 profile        [数据完备性]
  P3 — Async Scheduling               [收益待确认]
  P4 — CUDA Graph                     [收益低 + 复杂度高]
```

**核心结论**：在 batch_size=1 的 profile 数据下，Async Scheduling 的 2.4% 理论收益不支撑其 ~800 行的实现投入。建议在补全 SSE 异步化后，**先在 batch_size=4/8 下验证 CPU gap 是否随并发增长到 5-10%**，再决定是否进入 Async Scheduling。

CUDA Graph 不推荐在当前项目阶段独立实现。
