# Test Report / 测试报告

## 1. Report Scope / 报告范围

- **Project / 项目:** mini-vLLM Continuous Batching Lab — reference reimplementation of vLLM core architecture
- **Branch / 分支:** `main`
- **Commit:** `5338f1e` (serving layer: HTTP disconnect lifecycle management + fault injection tests)
- **Test date / 测试日期:** 2026-07-07
- **Environment / 环境:** WSL2 (Ubuntu 22.04) on Windows
- **This report records only the actual execution results from this session. / 本报告仅记录本次会话的实际执行结果。**
- No source code, tests, or configuration were modified for this report. / 未修改任何源代码、测试或配置文件。

> **中文摘要：** 本报告记录的是在 WSL2 环境中运行项目 176 个测试的真实结果。数据来自实际执行，非虚构。

## 2. Environment / 环境

| Item / 项目 | Value / 值 |
|------|-------|
| OS / 操作系统 | Ubuntu 22.04.5 LTS (WSL2, kernel 6.6.87.2-microsoft-standard-WSL2) |
| Python | 3.10.12 |
| pip | 26.1.2 |
| Git branch / 分支 | main |
| Git commit | `5338f1e` |
| Working tree / 工作树 | Modified (docs additions, source changes staged/unstaged from prior development session) / 已修改（文档新增、代码变更处于暂存/未暂存状态） |
| Test command / 测试命令 | `PYTHONPATH=. python3 -m pytest -q` |

> **中文摘要：** 测试环境为 WSL2 Ubuntu 22.04，Python 3.10.12，Git 提交 5338f1e。测试通过 `PYTHONPATH=. python3 -m pytest -q` 运行。

## 3. Test Inventory / 测试清单

Nine test files exist under `tests/`: / `tests/` 目录下有 9 个测试文件：

| File / 文件 | Test Count / 测试数 | Domain / 领域 |
|------|----------------------|--------|
| `tests/test_request.py` | 12 | Sequence lifecycle, Status enum, SamplingParams, SequenceGroup / 序列生命周期、状态枚举、采样参数、序列组 |
| `tests/test_kv_cache_manager.py` | 12 | BlockTable, BlockAllocator allocate/free/OOM/callbacks, BlockManager on-demand/free |
| `tests/test_prefix_cache.py` | 30 | PrefixCache insert/lookup/probe, ref_count, sharing semantics, stale entry, scheduler integration |
| `tests/test_scheduler.py` | 12 | Admit/prefill/decode/finish lifecycle, decode-first, chunked prefill, ignored reasons |
| `tests/test_engine.py` | 9 | Run-until-done, mid-arrival merge, OOM, ScheduleResult fields, KV tracking |
| `tests/test_serving_layer.py` | 28 | Non-stream/stream generate, RPM rate limit, admission control, cancel, timeout, disconnect lifecycle |
| `tests/test_fault_injection.py` | 15 | Queue overflow, block exhaustion, stream exhaustion, timeout storm, cancel storm, stale prefix entry, admission block pressure |
| `tests/test_metrics.py` | 39 | TTFT/TPOT/Throughput formulas, cancelled/timeout exclusion, KV utilization, scheduler latency, prefix cache metrics, stage profiler metrics, serving counters |
| `tests/test_stage_profiler.py` | 15 | Record/report/reset, context manager, start/end API, EngineCore integration, cancel stability |
| **Total / 总计** | **176** | |

> **中文摘要：** 9 个测试文件涵盖数据结构、KV 缓存、前缀缓存、调度器、引擎集成、服务层、故障注入、指标语义和阶段分析器。共 176 个测试。

## 4. Raw pytest Output / Pytest 原始输出

```
$ PYTHONPATH=. python3 -m pytest -q
........................................................................ [ 40%]
........................................................................ [ 81%]
................................                                         [100%]
=============================== warnings summary ===============================
../../../home/mxtia/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:65
  /home/mxtia/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:65: FutureWarning:
  The pynvml package is deprecated. Please install nvidia-ml-py instead.
  If you did not install pynvml directly, please report this to the maintainers
  of the package that installed pynvml for you.
    import pynvml  # type: ignore[import]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
176 passed, 1 warning in 2.97s
```

- **Exit code / 退出码:** 0
- **Passed / 通过:** 176
- **Failed / 失败:** 0
- **Skipped / 跳过:** 0
- **Errors / 错误:** 0
- **Warnings / 警告:** 1 (pynvml deprecation warning from torch/cuda import — unrelated to project code / 与项目代码无关的 torch 第三方库弃用警告)

Additionally, `pytest --collect-only -q` confirmed 176 tests collected. / 另外通过 `--collect-only` 确认收集到 176 个测试。

> **中文摘要：** 176 个测试全部通过，0 失败，0 跳过，0 错误。仅 1 个来自 torch 的第三方库弃用警告，与项目代码无关。运行时间 2.97 秒。

## 5. Test Result Interpretation / 测试结果解读

### What the tests confirm (correctness): / 测试验证的内容（正确性）：

- **Data structures** (test_request.py): Status enum lifecycle, SamplingParams defaults/customs, Sequence state machine (WAITING→PREFILL→RUNNING→FINISHED/REJECTED/CANCELLED), SequenceGroup management. / **数据结构**：状态枚举生命周期、采样参数、序列状态机、序列组管理。
- **KV cache allocation** (test_kv_cache_manager.py): BlockTable add/get/clear mapping, BlockAllocator allocate/free/OOM return None/callbacks/stats, BlockManager on-demand allocation (ensure_block at boundary only, empty on admission), OOM during execution. / **KV 缓存分配**：块表映射、分配器分配/释放/OOM、管理器按需分配、执行时 OOM。
- **Prefix cache** (test_prefix_cache.py): Hash-based insert/lookup/span, ref_count semantics (allocate sets ref=1, increment_ref, free decrements, double-free guard), shared block semantics (same rehash prefix shared, ref_count increases with sharer, free one sharer does NOT release), read-only probe (no ref change), stale entry exclusion (ref=0 entries not returned), scheduler integration (cache hit reduces prefill budget). / **前缀缓存**：基于哈希的插入/查找、引用计数语义、共享块语义、只读探测、过期条目排除、调度器集成。
- **Scheduler** (test_scheduler.py): Admit→prefill→decode→finish lifecycle, max_num_seqs budget, decode-first priority (running decode occupies slots before prefill), chunked prefill (long prompt split across steps), ignored reasons (NO_TOKEN_BUDGET, MAX_NUM_SEQS_LIMIT, etc.). / **调度器**：准入→预填充→解码→完成生命周期、预算限制、解码优先、分块预填充、忽略原因。
- **Engine integration** (test_engine.py): Multiple requests to completion via run_until_done, mid-arrival merge (new request while others running), OOM raised as RuntimeError, ScheduleResult fields present, KV tracking. / **引擎集成**：多请求运行至完成、中途到达合并、OOM 抛出异常、调度结果字段、KV 追踪。
- **Serving layer** (test_serving_layer.py): Non-stream generate success/empty prompt rejection/max_tokens=0, SSE streaming basic and token accumulation, RPM rate limit rejection and window reset, admission control (PROMPT_TOO_LONG, QUEUE_OVERFLOW, BLOCK_EXHAUSTED), stream manager max count, cancel running/non-existent, timeout auto-cancel, metrics endpoint JSON, disconnect lifecycle (abort mid-stream, blocks returned, no double-count, comprehensive cleanup). / **服务层**：非流式生成、SSE 流式、速率限制、准入控制、取消、超时、断连生命周期。
- **Fault injection** (test_fault_injection.py): Queue overflow reject and recovery after drain, block exhaustion (reject, no crash, blocks recovered after finish), stream exhaustion (reject and slot recovery), timeout storm (all blocks freed, metrics updated), cancel storm (all blocks freed, ref_count integrity verified), stale prefix cache entry (not falsely used, new request re-creates), admission under block pressure (100 long requests all rejected without crash, no-admission control causes OOM crash as expected). / **故障注入**：队列溢出恢复、块耗尽无崩溃、流耗尽恢复、超时风暴资源清理、取消风暴引用计数完整性、过期前缀缓存条目不会误用、块压力下的准入控制行为。
- **Metrics semantics** (test_metrics.py): TTFT field non-negative and first_token_time ≥ arrival_time, TPOT denominator = num_output_tokens - 1, single-token excluded, TPOT min/avg/max ordering, throughput formula consistency, wall-clock vs active-time distinction, cancelled/timeout excluded from completed counts, KV utilization range [0,100], scheduler latency fields, prefix cache hit rate, stage profiler metrics (percent of total sums to ~100%, engine_step_total is largest, empty profiler no crash, exception in context manager handled), serving counters separate. / **指标语义**：TTFT/TPOT 公式正确性、吞吐量计算、已完成请求排除规则、KV 利用率范围、调度器延迟、前缀缓存命中率、阶段分析器指标、服务计数器。
- **Stage profiler** (test_stage_profiler.py): Single/multiple stage recording with count/total/avg/max, context manager exception handling, empty report, reset, start/end API, percent of total calculation, EngineCore integration, cancel stability (profiler not crashed by cancel). / **阶段分析器**：单/多阶段记录、上下文管理器异常处理、空报告、重置、开始/结束 API、总计百分比计算、EngineCore 集成、取消稳定性。

### What the tests do NOT prove: / 测试无法证明的内容：

- **Production readiness / 生产就绪性**: No load tests, no concurrency stress tests, no long-running stability tests. / 无负载测试、无并发压力测试、无长期稳定性测试。
- **GPU/CUDA correctness / GPU/CUDA 正确性**: Tests run entirely on fake executor (CPU arithmetic). No CUDA kernel is tested. / 测试全部在 fake 执行器上运行，未测试 CUDA 内核。
- **Real model inference / 真实模型推理**: No test uses the real Qwen executor. All metrics values are simulated. / 无测试使用真实 Qwen 执行器，所有指标值均为模拟。
- **Production vLLM equivalence / 生产级 vLLM 等价性**: Tests verify mini-vLLM's own semantics, not equivalence to vLLM's production behavior. / 测试验证的是 mini-vLLM 自身的语义，而非与 vLLM 生产行为的等价性。
- **No unverified bugs / 无未验证的 bug**: 176 passing tests do not guarantee absence of bugs in uncovered paths. / 176 个通过测试不能保证未覆盖路径中不存在 bug。
- **Token boundary correctness / Token 边界正确性**: Real tokenizer behavior is not tested (fake tokenizer uses ASCII modulo arithmetic). / 未测试真实的 tokenizer 行为。

> **中文摘要：** 测试覆盖了状态转换、资源生命周期、调度策略、请求生命周期和指标计算等核心逻辑的正确性。故障注入套件系统验证了资源耗尽和取消场景下的恢复能力。但所有测试均在 fake 执行器上运行，无法证明 GPU 性能、真实模型推理或生产级 vLLM 等价性。

## 6. Failures or Unverified Items / 失败或未验证项

**All 176 tests passed.** No failures were observed. / **全部 176 个测试通过。** 未观察到任何失败。

**Unverified areas (tested but not executed in this session): / 未验证的领域（已编写测试但本次未执行）：**
- The Qwen executor (`mini_vllm/executor/qwen_executor.py`) has no dedicated automated tests. Its correctness was verified only through manual benchmark runs in this session. / Qwen 执行器没有专用的自动化测试，其正确性仅通过本次会话的手动基准运行验证。
- No coverage data was collected (coverage tool not configured in pyproject.toml). / 未收集覆盖率数据。

> **中文摘要：** 全部 176 个测试通过。Qwen 执行器缺少专用自动化测试，仅通过手动基准测试验证。项目未配置覆盖率工具。

## 7. Reproduction Commands / 复现命令

```bash
# Run all tests / 运行全部测试
PYTHONPATH=. python3 -m pytest -q

# Run all tests with verbose output / 详细输出
PYTHONPATH=. python3 -m pytest -v

# Run all tests with short traceback / 简短回溯
PYTHONPATH=. python3 -m pytest --tb=short

# Run single test file / 运行单个测试文件
PYTHONPATH=. python3 -m pytest tests/test_metrics.py -v

# Collect only (list tests without running) / 仅收集（列出测试但不运行）
PYTHONPATH=. python3 -m pytest --collect-only -q
```

> **中文摘要：** 使用 `PYTHONPATH=. python3 -m pytest -q` 运行全部 176 个测试。支持单个文件测试和 `--collect-only` 模式。

## 8. Conclusion / 结论

- **All 176 tests pass** with zero failures, zero skipped, and one unrelated third-party deprecation warning. / **176 个测试全部通过**，零失败、零跳过、一个无关的第三方库弃用警告。
- **9 test files** cover data structures, KV cache allocation, prefix cache, scheduler, engine integration, serving layer, fault injection, metrics semantics, and stage profiler. / **9 个测试文件**涵盖数据结构、KV 缓存分配、前缀缓存、调度器、引擎集成、服务层、故障注入、指标语义和阶段分析器。
- **Test scope / 测试范围**: Correctness of state transitions, resource lifecycle, scheduling policy, request lifecycle, and metrics accounting. The fault injection suite adds systematic recovery verification for resource exhaustion and cancellation scenarios. / 状态转换、资源生命周期、调度策略、请求生命周期和指标计算正确性。故障注入套件增加了资源耗尽和取消场景的系统恢复验证。
- **Not tested / 未测试**: GPU performance, real model inference, production-scale concurrency, CUDA kernel correctness, real vLLM equivalence. All metrics from tests are simulated values from the fake executor. / GPU 性能、真实模型推理、生产级并发、CUDA 内核正确性、vLLM 等价性。所有指标均为 fake 执行器的模拟值。

> **中文摘要：** 结论：176 个测试全部通过，覆盖项目核心逻辑的正确性。故障注入套件是该测试体系的亮点。主要局限是所有测试均在 fake 执行器上运行，不涉及 GPU 或真实模型推理。
