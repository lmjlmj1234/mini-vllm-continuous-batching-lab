# Limitations — mini-vLLM Continuous Batching Lab / 项目限制说明

## 1. Positioning / 定位

| Aspect / 方面 | Limitation / 限制 |
|--------|-----------|
| **Intended use / 预期用途** | Reference reimplementation of vLLM's core architecture. NOT production-ready. / vLLM 核心架构的参考复现。**非生产就绪。** |
| **Completeness / 完整性** | Implements ~30% of vLLM's feature surface (scheduler, KV cache, prefix cache, metrics, serving). Does NOT implement: preemption, swapping, CUDA kernels, tensor parallelism, pipeline parallelism, quantization, speculative decoding, multi-LoRA, beam search, guided decoding, etc. / 实现了 vLLM 约 30% 的功能面。未实现：抢占、交换、CUDA 内核、张量并行、流水线并行、量化、推测解码等。 |
| **Code maturity / 代码成熟度** | Built in iterative phases over development sessions. Some modules carry phase-specific scaffolding (e.g., memory trace debug output, stage profiler for development profiling) that would be removed or trimmed for production. / 在多次开发会话中迭代构建。部分模块带有阶段特定的脚手架代码。 |

> **中文摘要：** 本项目是 vLLM 核心架构的参考复现，实现了约 30% 的功能面。**不适用于生产环境。** 核心缺失功能包括抢占、交换、自定义 CUDA 内核和分布式推理。

## 2. Model Execution / 模型执行

| Limitation / 限制 | Detail / 细节 | Impact / 影响 |
|-----------|--------|--------|
| **Single model / 单一模型** | Only Qwen2-0.5B executor is implemented. No Llama, GPT, or other model support. / 仅实现了 Qwen2-0.5B 执行器。 | Cannot test with different model architectures. / 无法测试不同模型架构。 |
| **No GPU acceleration / 无 GPU 加速** | WSL2 environment has no CUDA device passthrough. Qwen executor runs on CPU via PyTorch. / WSL2 环境无 GPU 直通。 | TTFT=211ms, TPOT=32ms for 0.5B model — ~10-20× slower than GPU. / 比 GPU 慢约 10-20 倍。 |
| **Lazy model loading / 延迟模型加载** | Model and tokenizer are loaded once at QwenExecutor construction. No model offloading, no multi-GPU sharding. / 在 QwenExecutor 构造时一次加载。 | Loads full model into memory. 0.5B model with float32 ≈ 2GB RAM. / 0.5B 参数模型占用约 2GB 内存。 |
| **Greedy sampling only / 仅贪心采样** | `_sample_token()` uses `argmax` — greedy decoding only. No temperature, top-k, top-p, min-p, or beam search. / 使用 argmax 贪心解码。 | Cannot reproduce diverse/varied outputs. / 无法复现多样化的输出。 |
| **HuggingFace transformers wrapper / HF Transformers 封装** | Relies on `transformers`'s native `past_key_values` for KV cache management, not a custom PagedAttention kernel. / 依赖 transformers 原生 KV 缓存管理。 | vLLM's key innovation (PagedAttention) is not implemented. / vLLM 的核心创新 PagedAttention 未实现。 |
| **No attention mask optimization / 无注意力掩码优化** | `_build_attention_mask()` uses a simple all-ones mask. No sliding window, no sparse attention, no ALiBi. / 使用简单的全 1 掩码。 | Limited to full-attention models only. / 仅支持全局注意力模型。 |
| **Fake executor divergence / Fake 执行器差异** | Fake executor uses ASCII modulo arithmetic for token generation. / Fake 执行器使用 ASCII 模算术生成 token。 | Passing tests with fake executor do not guarantee correct model inference. / Fake 执行器上的测试不能保证真实模型推理的正确性。 |

> **中文摘要：** 仅支持 Qwen2-0.5B 一个模型，无 GPU 加速（WSL2 无 GPU 直通），仅贪心采样，使用 transformers 原生 KV 缓存而非 PagedAttention。Fake 执行器与真实模型的差异意味着测试通过不能保证推理正确性。

## 3. KV Cache — Known Gaps vs vLLM / KV 缓存 — 与 vLLM 的已知差距

| Feature / 功能 | mini-vLLM | vLLM Production / vLLM 生产版 | Gap Analysis / 差距分析 |
|---------|-----------|-----------------|--------------|
| **Allocation strategy / 分配策略** | On-demand (ensure_block at write time) / 按需分配 | On-demand with over-commitment / 按需分配 + 超卖 | Functional match for basic case / 基本场景功能匹配 |
| **Preemption / 抢占** | Not implemented. OOM raises `RuntimeError`. / 未实现 | Swap victim blocks to CPU, restore on re-schedule / 将受害者块交换到 CPU | **Critical gap.** No recovery from OOM. / **严重差距。** 无法从 OOM 中恢复 |
| **Copy-on-Write / 写时复制** | `increment_ref()` implemented but COW path never exercised / API 已实现但未使用 | Used when diverging from shared prefix blocks / 从共享前缀块分叉时使用 | API present, no real usage / API 就绪但无实际使用 |
| **Swapping / 交换** | Not implemented. No CPU block allocator. / 未实现 | Blocks swapped to CPU when GPU memory is full / GPU 内存满时交换到 CPU | Missing entirely / 完全缺失 |
| **Block size / 块大小** | Fixed at `block_size=4` / 固定为 4 | Configurable (typically 16) / 可配置（通常 16） | block_size=4 is unrealistic for production / 4 的块大小不适用于生产 |
| **KV cache storage / KV 缓存存储** | Python dict / PyTorch past_key_values | Custom CUDA-managed block-level cache / 自定义 CUDA 管理块级别缓存 | Not memory-accurate. Not PagedAttention blocks. / 非内存精确，非 PagedAttention 块 |
| **Prefix cache hash / 前缀缓存哈希** | Simple `hash(tuple(tokens))` — collision probability nonzero / 简单 Python hash | Hash-based with collision detection / 带碰撞检测的哈希 | No hash collision handling / 无哈希碰撞处理 |
| **Prefix cache stale entry / 前缀缓存过期条目** | Not evicted until hash collision or manual clear / 不主动驱逐 | Evicted promptly on block free / 块释放时立即驱逐 | Minor leak — entries accumulate / 小泄漏——条目会累积 |
| **Block utilisation tracking / 块利用率追踪** | `kv_tokens_written / (allocated_blocks * block_size)` | Similar metric internally / 内部有类似指标 | Functionally equivalent / 功能等价 |

> **中文摘要：** 与 vLLM 的核心差距：**无抢占机制**（OOM 直接抛异常）、无 CPU 交换、无 PagedAttention（使用 transformers 原生 KV 缓存）、无哈希碰撞处理、过期缓存条目不主动驱逐。按需分配策略在基本场景上与 vLLM 功能匹配。

## 4. Scheduler / 调度器

| Limitation / 限制 | Detail / 细节 |
|-----------|--------|
| **Single priority level / 单优先级** | Decode-first is the only priority rule. No request-level priority, no QoS tiers, no SLA guarantees. / 解码优先是唯一的优先级规则。无请求级优先级、无 QoS 层级、无 SLA 保障。 |
| **No preemption / 无抢占** | Running requests cannot be preempted. If a high-priority request arrives and all slots are full, it waits. / 运行中的请求不能被抢占。高优先级请求只能在所有槽位满时等待。 |
| **No swap-out policy / 无换出策略** | No victim selection for OOM recovery. / 无 OOM 恢复的受害者选择策略。 |
| **No fairness / 无公平性** | Long-running decode requests can occupy the batch indefinitely. No per-request time limits within the scheduler. / 长时间运行的解码请求可无限占用批次。调度器内无请求级时间限制。 |
| **Fixed max_num_seqs / 固定预算上限** | Budget limits are static Config values. No dynamic adjustment based on model capacity or observed memory pressure. / 预算限制为静态配置值。不根据模型容量或内存压力动态调整。 |
| **Chunked prefill overhead / 分块预填充开销** | chunk_size=4 is small. Long prompts (30+ tokens) may take 8+ steps of partial prefill before producing a single output token. / chunk_size=4 太小，长 prompt 可能需要 8+ 步部分预填充才能产出第一个输出 token。 |
| **No iterative prefill improvement / 无迭代预填充优化** | vLLM's later versions improve chunked prefill with multi-batch and speculative prefill techniques. mini-vLLM has no such optimizations. / vLLM 后续版本改进了分块预填充，mini-vLLM 无此类优化。 |

> **中文摘要：** 调度器实现了核心 6 阶段算法和解码优先策略，但缺乏优先级层级、抢占机制、公平性保障和动态预算调整。分块预填充的 chunk_size=4 对于长 prompt 效率较低。

## 5. Serving Layer / 服务层

| Limitation / 限制 | Detail / 细节 |
|-----------|--------|
| **Single-process, single-thread / 单进程单线程** | All requests processed synchronously in a single `step()` loop. No async processing, no thread pool, no asyncio event loop integration. / 所有请求在单一步循环中同步处理。无异步处理、无线程池、无 asyncio 集成。 |
| **uvicorn/ASGI only / 仅支持 uvicorn** | Runs via `uvicorn` in serving layer. No gunicorn, no multiprocess, no deployment orchestration. / 通过 uvicorn 运行。无 gunicorn、无多进程、无部署编排。 |
| **No authentication / 无认证** | No API keys, no JWT, no TLS. Rate limiting uses RPM/TPM counters (in-memory, reset on server restart). / 无 API 密钥、无 JWT、无 TLS。速率限制使用内存计数器。 |
| **No request routing / 无请求路由** | Single endpoint `/v1/completions`. No `/v1/chat/completions`, no model routing. / 单一端点。无聊天端点、无模型路由。 |
| **Streaming via SSE only / 仅 SSE 流式** | Server-Sent Events for streaming. No WebSocket, no chunked transfer encoding fallback. / 服务器发送事件。无 WebSocket。 |
| **No health check endpoint / 无健康检查端点** | No `/health`, `/ping`, `/readyz` endpoints for load balancer integration. / 无负载均衡集成端点。 |
| **No graceful shutdown / 无优雅关闭** | Requests in flight are lost on server stop. No drain, no connection draining. / 服务器停止时丢失正在处理的请求。 |
| **No distributed rate limiting / 无分布式速率限制** | RPM/TPM counters are per-process. In multi-instance deployment, each instance has its own counters. / 计数器是进程级别的。多实例部署下各实例有独立计数器。 |

> **中文摘要：** 服务层实现了 HTTP/SSE、速率限制、取消、超时和断连处理，但架构是单进程单线程的。缺少认证、健康检查、优雅关闭和分布式速率限制等生产级功能。

## 6. Prefix Cache / 前缀缓存

| Limitation / 限制 | Detail / 细节 |
|-----------|--------|
| **Hash collision / 哈希碰撞** | Uses Python's built-in `hash()` — collisions are possible and unhandled. / 使用 Python 内置 hash()，碰撞可能发生且未处理。 |
| **No eviction / 无驱逐** | Stale entries (hash → stale PID) remain in the dictionary after the block is freed. Only excluded at lookup time. Accumulates over time. / 过期条目在字典中累积，仅在查询时排除。 |
| **Consecutive-only matching / 仅连续匹配** | Prefix matches must be from block 0 consecutively. First miss breaks the chain. No suffix matching, no fuzzy matching. / 必须从块 0 开始连续匹配，首次未命中即中断。无后缀匹配、无模糊匹配。 |
| **Prompt-only / 仅缓存 prompt** | Only prompt tokens are cached. Generated tokens are never cached for future prefix reuse. / 仅缓存 prompt token。生成的 token 从不缓存。 |
| **No prefix-aware scheduling / 无前缀感知调度** | Scheduler does not prioritise requests that share a prefix (e.g., system prompt). / 调度器不会优先处理共享前缀的请求。 |
| **No cache warming / 无缓存预热** | No mechanism to pre-populate the cache with known prefixes. / 没有预填充已知前缀的机制。 |

> **中文摘要：** 前缀缓存实现了基于哈希的共享和引用计数，但存在 Python 哈希碰撞风险、无驱逐策略、仅连续匹配、不缓存生成 token、无前缀感知调度、无缓存预热等限制。

## 7. Metrics / 指标

| Limitation / 限制 | Detail / 细节 |
|-----------|--------|
| **No persistent metrics store / 无持久化指标存储** | Metrics exist only as in-memory `MetricsCollector` state. No Prometheus, no Grafana, no logging to disk. / 指标仅存在于内存中。无 Prometheus、无 Grafana、无磁盘日志。 |
| **No histogram export / 无直方图导出** | TTFT/TPOT distributions are reported as only avg/min/max. No p50/p90/p99/histogram. / 仅报告 avg/min/max。无 p50/p90/p99 或直方图。 |
| **Throughput window / 吞吐量窗口** | Throughput is computed over the entire engine lifetime. No sliding window, no per-interval throughput. / 吞吐量在整个引擎生命周期上计算。无滑动窗口。 |
| **No latency breakdown per stage / 无逐阶段延迟分解** | Stage profiler gives aggregate timing but does not assign latency to individual requests or model layers. / 阶段分析器提供聚合计时。 |
| **No CUDA metrics / 无 CUDA 指标** | No GPU utilisation, no CUDA kernel time, no memory bandwidth utilisation. / 无 GPU 利用率、无 CUDA 内核时间、无内存带宽利用率。 |
| **No request-level tracing / 无请求级追踪** | No per-request span/trace ID linking scheduling, execution, and serving stages. / 无请求级跨度/追踪 ID。 |

> **中文摘要：** 指标系统实现了 TTFT、TPOT、吞吐量和 KV 利用率等核心公式的正确性验证，但缺少持久化存储（Prometheus/Grafana）、分位数直方图（p50/p99）、CUDA 指标和请求级追踪。

## 8. Testing / 测试

| Limitation / 限制 | Detail / 细节 |
|-----------|--------|
| **No GPU tests / 无 GPU 测试** | All 176 tests run on fake executor. GPU-specific issues are not covered. / 全部 176 个测试在 fake 执行器上运行。未覆盖 GPU 相关问题。 |
| **No load tests / 无负载测试** | No concurrency stress testing, no long-running stability testing, no throughput degradation testing. / 无并发压力测试、无长期稳定性测试、无吞吐量衰减测试。 |
| **No regression benchmarks / 无回归基准** | No benchmark suite to detect performance regressions across commits. / 无检测性能回退的基准套件。 |
| **Fake tokeniser / Fake Tokenizer** | Tests with fake executor use ASCII modulo tokenisation. Token boundary, BPE, and special token issues are not tested. / 使用 ASCII 模 Tokenization，未测试 Token 边界、BPE 和特殊 Token。 |
| **Qwen executor untested / Qwen 执行器无自动化测试** | `qwen_executor.py` has no dedicated automated tests. Verified only through manual benchmark runs. / 仅通过手动基准测试验证。 |
| **No preemption tests / 无抢占测试** | OOM raises RuntimeError — no preemption to test. / OOM 抛出异常——无抢占可测试。 |
| **No swap tests / 无交换测试** | CPU swapping not implemented — no swap to test. / 未实现 CPU 交换。 |
| **No integration tests with real model / 无真实模型集成测试** | No test runs with actual Qwen2-0.5B inference. / 未使用真实 Qwen2-0.5B 推理运行测试。 |
| **No coverage measurement / 无覆盖率测量** | Coverage tool not configured in `pyproject.toml`. Code coverage percentage is unknown. / 未配置覆盖率工具。代码覆盖率未知。 |

> **中文摘要：** 176 个测试覆盖了核心逻辑的正确性，故障注入套件是该测试体系的亮点。但所有测试均在 fake 执行器上运行，不涉及 GPU、负载测试或真实模型推理。Qwen 执行器缺少自动化测试，项目未配置覆盖率工具。

## 9. Documentation / 文档

| Limitation / 限制 | Detail / 细节 |
|-----------|--------|
| **Redundancy / 内容冗余** | Multiple docs files contain overlapping content. Partially cleaned up in a prior deduplication pass but some overlap remains. / 多个文档包含重叠内容。已部分清理但仍有重叠。 |
| **Bilingual inconsistency / 双语不一致** | Some source files have Chinese comments alongside English comments. Mix of languages in code comments is a maintainability concern. / 部分源文件混用中英文注释，影响可维护性。 |
| **No API reference / 无 API 参考文档** | No auto-generated API docs (Sphinx/pydoc). `CLAUDE.md` serves as the developer guide. / 无自动生成的 API 文档。 |
| **Stale references / 过时引用** | `Testing_Guide.md` still mentions "135 unit and integration tests" (now 176). Scheduling phase documentation may lag behind code changes. / 测试指南仍引用了旧测试数 135（现为 176）。 |

> **中文摘要：** 项目文档较为丰富，但存在内容冗余、中英双语不一致、缺乏自动生成的 API 文档等问题。部分文档中的测试计数和调度阶段描述可能滞后于代码变更。

## 10. Benchmark / 基准测试

| Limitation / 限制 | Detail / 细节 |
|-----------|--------|
| **Fake executor metrics are meaningless / Fake 执行器指标无意义** | TTFT=0.3ms, TPOT=0.07ms from fake executor represent CPU overhead, not model performance. / 代表 CPU 开销，非模型性能。 |
| **Qwen CPU metrics are not representative / Qwen CPU 指标不具代表性** | WSL2 CPU inference is 10-20× slower than GPU. Do not use these numbers for performance claims. / WSL2 CPU 推理比 GPU 慢 10-20 倍。 |
| **Single-replica only / 仅单副本** | All benchmarks run a single engine instance. No throughput scaling data, no horizontal scaling analysis. / 所有基准测试运行单一引擎实例。 |
| **Tiny workload size / 极小工作负载** | Max 4 requests, max 55 prompt tokens, max 16 generated tokens. No large-scale benchmark data. / 最多 4 个请求，最大 55 个 prompt token。 |
| **No warmup / 无预热** | First request after engine start includes model loading time. No warmup iterations to stabilise measurements. / 引擎启动后的首次请求包含模型加载时间。 |

> **中文摘要：** Fake 执行器的指标仅代表 CPU 开销，Qwen CPU 推理比 GPU 慢 10-20 倍。所有基准测试为单副本、小工作负载（最多 4 请求）、无预热。**这些数字不应用于性能声明。**

## 11. Summary / 总结

| Area / 领域 | Status / 状态 | Production Gap / 生产级差距 |
|------|--------|----------------|
| Scheduler / 调度器 | Core algorithm implemented (6-phase, decode-first, chunked prefill) / 核心算法已实现 | No preemption, no priority, no swap / 无抢占、无优先级、无交换 |
| KV Cache / KV 缓存 | Three-layer architecture with on-demand allocation / 三层架构+按需分配 | No PagedAttention, no swap, OOM crashes / 无 PagedAttention、无交换、OOM 崩溃 |
| Prefix Cache / 前缀缓存 | Hash-based sharing with ref counting / 基于哈希的共享+引用计数 | No eviction, no collision handling, no prefix-aware scheduling / 无驱逐、无碰撞处理、无前缀感知调度 |
| Serving Layer / 服务层 | HTTP/SSE with rate limiting, cancel, timeout, disconnect / HTTP/SSE+速率限制+取消+超时+断连 | Single-process, no auth, no health check, no graceful shutdown / 单进程、无认证、无健康检查、无优雅关闭 |
| Model Execution / 模型执行 | Qwen2-0.5B via HuggingFace | Single model, CPU-only, greedy sampling / 单一模型、仅 CPU、贪心采样 |
| Metrics / 指标 | TTFT/TPOT/Throughput/KV utilisation | No persistent store, no histograms, no CUDA metrics / 无持久化存储、无直方图、无 CUDA 指标 |
| Testing / 测试 | 176 tests, fault injection suite / 176 测试+故障注入套件 | No GPU tests, no load tests, no regression benchmarks / 无 GPU 测试、无负载测试、无回归基准 |
| Documentation / 文档 | Extensive bilingual docs / 广泛的双语文档 | Some redundancy, some stale counts / 部分冗余、部分过时计数 |

**Bottom line / 总结：** mini-vLLM faithfully reimplements vLLM's scheduling and KV cache architecture as a reference implementation. It is not suitable for production inference. The largest gaps vs vLLM are: no preemption (swap-to-CPU), no custom PagedAttention kernel (uses transformers native KV cache), no GPU execution path, no distributed inference, and no production serving features. / mini-vLLM 忠实地复现了 vLLM 的调度和 KV 缓存架构，作为参考实现。**不适用于生产推理。** 与 vLLM 的最大差距包括：无抢占机制、无自定义 PagedAttention 内核、无 GPU 执行路径、无分布式推理、无生产级服务功能。
