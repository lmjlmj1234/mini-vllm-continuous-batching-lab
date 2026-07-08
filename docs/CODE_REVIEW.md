# Code Review: HEAD vs Working Tree (Source Files)
# 代码审查：HEAD 与工作树（源码文件）

- **Reviewer:** Claude Code (two-axis review)
- **Fixed point:** HEAD (`5338f1e`)
- **Scope:** `mini_vllm/`, `tests/`, `examples/` — uncommitted changes since HEAD
- **Diff size:** ~2729 lines

- **审查者：** Claude Code（双轴审查）
- **基准点：** HEAD（`5338f1e`）
- **范围：** `mini_vllm/`、`tests/`、`examples/` — 自 HEAD 以来的未提交变更
- **差异大小：** 约 2729 行

---

## Standards / 代码规范

**Axis:** Does the code conform to documented coding standards and avoid common code smells?
**Standards source:** No `CODING_STANDARDS.md` exists in this repo — review falls back to the Fowler smell baseline (Refactoring, ch.3).

**审查轴：** 代码是否符合文档化编码规范，是否避免了常见代码坏味？
**规范来源：** 仓库中不存在 `CODING_STANDARDS.md` —— 审查回退到 Fowler 坏味基线（《重构》第 3 章）。

### 1. Speculative Generality — `_STAGE_LABELS` identity map / 投机泛型——`_STAGE_LABELS` 恒等映射

`stage_profiler.py` defines a `_STAGE_LABELS` dictionary where every key maps to itself — a complete identity with `_STAGES`. This structure is used exactly once (in `print_report()`) and adds no semantic value.

`stage_profiler.py` 定义了一个 `_STAGE_LABELS` 字典，其中每个键都映射到自身——与 `_STAGES` 完全一致。该结构仅被使用一次（在 `print_report()` 中），不提供任何语义价值。

```python
_STAGE_LABELS = {
    "request_queue_waiting": "request_queue_waiting",
    "scheduler_step": "scheduler_step",
    ...
```

If human-readable labels are ever needed, add them then. Delete for now.

如果将来需要人类可读的标签，到那时再添加。现在先删除。

### 2. Primitive Obsession — string-typed stage names / 原始类型痴迷——字符串类型的阶段名称

`StageProfiler.record()`, `start()`, `end()`, and `record_raw()` all accept raw strings. `_records` is initialized via `setdefault`, so any string — including typos like `"scheduler_stepp"` — silently creates an orphan entry that never appears in the report.

`StageProfiler.record()`、`start()`、`end()` 和 `record_raw()` 都接受裸字符串。`_records` 通过 `setdefault` 初始化，因此任何字符串——包括拼写错误如 `"scheduler_stepp"` ——都会静默创建一个孤儿条目，永远不会出现在报告中。

A `Literal` type alias, `Enum`, or initializing `_records` from the fixed `_STAGES` set would eliminate this path.

使用 `Literal` 类型别名、`Enum` 或从固定的 `_STAGES` 集合初始化 `_records` 可以消除这条路径。

### 3. Shotgun Surgery — profiling circuit wiring / 散弹式修改——性能分析器布线

Adding StageProfiler touches 5 core modules: `cache/manager.py` (constructor param + 3 probe points), `engine/engine.py` (construction, property), `engine/engine_core.py` (wrapping entire `step()` body in context managers), and two `__init__.py` files. `engine_core.py`'s `step()` is rewritten to embed 4 `with self._profiler.record(...)` scopes.

添加 StageProfiler 涉及 5 个核心模块：`cache/manager.py`（构造函数参数 + 3 个探测点）、`engine/engine.py`（构造、属性）、`engine/engine_core.py`（用 context manager 包裹整个 `step()` 函数体），以及两个 `__init__.py` 文件。`engine_core.py` 的 `step()` 被重写以嵌入 4 个 `with self._profiler.record(...)` 作用域。

A decorator wrapping `step()`, or an event system that broadcasts hooks to subscribers, could have confined the change to fewer files.

一个包裹 `step()` 的装饰器，或一个向订阅者广播钩子的事件系统，本可以将修改限制在更少的文件中。

### 4. Duplicated Code — executor fallback logic / 重复代码——执行器回退逻辑

`_write_to_kv` (and `_read_from_kv`) contain the same `_block_manager is not None` guard pattern in both `executor.py` and `qwen_executor.py`. The same `ensure_block` call is repeated when `block_manager` is available, and the same `seq.block_table[...]` fallback when it isn't. Extract to a mixin or default method on the executor protocol.

`_write_to_kv`（和 `_read_from_kv`）在 `executor.py` 和 `qwen_executor.py` 中都包含相同的 `_block_manager is not None` 守卫模式。当 `block_manager` 可用时重复调用 `ensure_block`，不可用时使用相同的 `seq.block_table[...]` 回退。提取为 executor 协议上的 mixin 或默认方法。

### 5. Speculative Generality — unused metrics counters / 投机泛型——未使用的指标计数器

`MetricsCollector` adds `_rpm_rejected` / `_tpm_rejected` counters via `count_rpm_rejected()` / `count_tpm_rejected()` setters, exposed in the report as `"rpm_rejected"` / `"tpm_rejected"`. These counters are never called by any code path in the diff — neither `engine_core.py` nor the serving layer wires them. Added but unconnected state.

`MetricsCollector` 通过 `count_rpm_rejected()` / `count_tpm_rejected()` 设置器添加了 `_rpm_rejected` / `_tpm_rejected` 计数器，在报告中暴露为 `"rpm_rejected"` / `"tpm_rejected"`。这些计数器在 diff 中从未被任何代码路径调用——`engine_core.py` 和服务层都没有连接它们。已添加但未连接的状态。

### 6. Mysterious Name — `rpm` / `tpm` / 神秘的名称——`rpm` / `tpm`

The abbreviations `rpm_rejected` / `tpm_rejected` are never expanded or documented. No docstring explains what "RPM" (requests per minute) or "TPM" (tokens per minute) means, and since the counters drive no behavior, readers cannot infer from usage.

缩写 `rpm_rejected` / `tpm_rejected` 从未被展开或文档化。没有文档字符串解释 "RPM"（每分钟请求数）或 "TPM"（每分钟 token 数）的含义，而且由于这些计数器不驱动任何行为，读者无法从使用中推断出来。

---

## Spec / 规范符合度

**Axis:** Does the code faithfully implement the PRD?
**Spec source:** `docs/issues/mini-vllm-project/PRD.md` (three-phase architecture, 15 issues)

**审查轴：** 代码是否忠实地实现了 PRD？
**规范来源：** `docs/issues/mini-vllm-project/PRD.md`（三阶段架构、15 个 issue）

### (a) PRD requirements that are missing or partially implemented / PRD 要求缺失或部分实现的内容

**1. `make_block_allocator_callbacks()` is dead code in the executor protocol.**
**1. `make_block_allocator_callbacks()` 是 executor 协议中的死代码。**

`executor/base.py` defines `make_block_allocator_callbacks()` (lines 68-72), and both `FakeModelExecutor` and `QwenExecutor` implement it. But the actual wiring in `engine.py` (lines 59-63) is done directly:

`executor/base.py` 定义了 `make_block_allocator_callbacks()`（第 68-72 行），`FakeModelExecutor` 和 `QwenExecutor` 都实现了它。但实际接线在 `engine.py`（第 59-63 行）中直接完成：

```python
allocator.set_callbacks(on_allocate=self._executor.prepare_block, ...)
```

The method is never called. The PRD does not prescribe this method specifically, but the Executor Protocol (PRD §Implementation Decisions) defines "KV callbacks" as part of the uniform interface — an unused public method makes the protocol contract misleading.

该方法从未被调用。PRD 并未具体规定此方法，但 Executor 协议（PRD §实现决策）将 "KV 回调" 定义为统一接口的一部分——未使用的公共方法使协议契约具有误导性。

**2. `_check_timeouts()` accesses scheduler internals across API boundary.**
**2. `_check_timeouts()` 跨越 API 边界访问调度器内部属性。**

`engine_core.py` accesses `self._scheduler._config.request_timeout_s` and `self._scheduler._queue.running` — both private attributes of sibling modules. The PRD states: "serving only accesses the engine through `LLMEngine.add_request()`, `step()`, and `cancel_request()`" (PRD §Further Notes). Timeout enforcement is service governance; co-locating it inside EngineCore couples core engine to serving concerns unnecessarily.

`engine_core.py` 访问了 `self._scheduler._config.request_timeout_s` 和 `self._scheduler._queue.running`——都是兄弟模块的私有属性。PRD 声明："服务层仅通过 `LLMEngine.add_request()`、`step()` 和 `cancel_request()` 访问引擎"（PRD §补充说明）。超时强制终止是服务治理功能；将其放在 EngineCore 内部不必要地将核心引擎与服务关注点耦合在一起。

### (b) Behavior in the diff not asked for (scope creep) / diff 中超出规范要求的行为（范围蔓延）

**1. `memory_trace` config field and `_print_memory_trace()`.**
**1. `memory_trace` 配置字段和 `_print_memory_trace()`。**
PRD User Story #29 asks for "memory tracing (allocated/freed blocks per step, free list dump, block table dump)". The actual implementation adds per-sequence commentary like "if pre-allocated, blocks=X, saved=Y" — exceeding the "tool to debug KV cache" brief. Minor, but the level of detail goes beyond what the PRD describes.

PRD 用户故事 #29 要求"内存追踪（每步分配/释放的块、空闲列表转储、块表转储）"。实际实现添加了逐序列的注释，如"if pre-allocated, blocks=X, saved=Y"——超出了"用于调试 KV 缓存的工具"的简要描述。程度轻微，但详细程度超出了 PRD 的描述范围。

**2. `SamplingParams` fields `top_p`, `top_k`, `stop_token_ids`, `stop_strings`.**
**2. `SamplingParams` 字段 `top_p`、`top_k`、`stop_token_ids`、`stop_strings`。**

PRD lists `SamplingParams` as a type but does not define its schema. These four fields are added but unused — the fake executor never reads them, so they parse and store without effect.

PRD 将 `SamplingParams` 列为一个类型但未定义其模式。这四个字段已添加但未被使用——假执行器从不读取它们，因此它们被解析和存储但没有任何效果。

### (c) Implemented but inconsistent with the PRD / 已实现但与 PRD 不一致

**1. `cancel_request()` bypasses `RequestQueue` public API.**
**1. `cancel_request()` 绕过了 `RequestQueue` 的公共 API。**

`engine_core.py` (lines 133-160) directly mutates `RequestQueue` internals: `self._scheduler._queue._running.pop()`, `_waiting.pop()`, `_finished[rid] = sg`. `RequestQueue` has public methods for state transitions (`mark_running()`, `mark_finished()`, `mark_rejected()`), but cancel does not use them. The same applies to block freeing — it correctly calls `self._scheduler._block_manager.free(seq.seq_id)`, matching the PRD's `BlockManager.free(seq_id)` contract, but the surrounding queue state transitions bypass the queue's own API.

`engine_core.py`（第 133-160 行）直接修改了 `RequestQueue` 的内部结构：`self._scheduler._queue._running.pop()`、`_waiting.pop()`、`_finished[rid] = sg`。`RequestQueue` 具有状态转换的公共方法（`mark_running()`、`mark_finished()`、`mark_rejected()`），但取消操作没有使用它们。块释放方面同理——它正确调用了 `self._scheduler._block_manager.free(seq.seq_id)`，与 PRD 的 `BlockManager.free(seq_id)` 契约一致，但周围的队列状态转换绕过了队列自身的 API。

**2. `StageProfiler.record()` return type annotation is wrong.**
**2. `StageProfiler.record()` 返回类型注解错误。**

```python
@contextmanager
def record(self, stage: str) -> None: ...
```

`@contextmanager` converts the generator function to a context manager — the return type should be `Generator` or `Iterator`, not `None`. This does not affect runtime (the decorator handles the protocol), but it misdescribes the contract for stubs and type checkers.

`@contextmanager` 将生成器函数转换为上下文管理器——返回类型应为 `Generator` 或 `Iterator`，而不是 `None`。这不影响运行时（装饰器处理了协议），但它错误地描述了存根和类型检查器的契约。

**3. Test imports from root-level `serving/`, not `mini_vllm/serving/`.**
**3. 测试从根级 `serving/` 导入，而非 `mini_vllm/serving/`。**

PRD states: "The serving layer (`mini_vllm/serving/`) is a separate package" (PRD §Implementation Decisions). But the PRD itself uses wording suggesting `mini_vllm/serving/`, while the actual package appears to live at root `serving/` in some places. `test_metrics.py` imports `from serving.api_server import ServingLayer`, which depends on the calling PYTHONPATH. The inconsistency between the PRD's documented path and actual structure could cause import failures.

PRD 声明："服务层（`mini_vllm/serving/`）是一个独立的包"（PRD §实现决策）。但实际包在某些地方看起来位于根目录的 `serving/`。`test_metrics.py` 导入 `from serving.api_server import ServingLayer`，这取决于调用方的 PYTHONPATH。PRD 记录的路径与实际结构之间的不一致可能导致导入失败。

---

## Summary / 总结

| Axis | Findings | Worst issue |
|------|----------|-------------|
| **Standards / 规范** | 6 smells (1 Speculative Generality, 1 Primitive Obsession, 1 Shotgun Surgery, 1 Duplicated Code, 1 Speculative Generality, 1 Mysterious Name) / 6 个代码坏味 | **Shotgun Surgery / 散弹式修改** — profiling touches 5 modules when a decorator or event system could confine it to 1 |
| **Spec / 规范符合度** | 2 missing/partial, 2 scope creep, 3 inconsistencies / 2 个缺失/部分实现、2 个范围蔓延、3 个不一致 | **API boundary violation / API 边界违反** — `_check_timeouts()` and `cancel_request()` reach into scheduler internals instead of using public interfaces |
