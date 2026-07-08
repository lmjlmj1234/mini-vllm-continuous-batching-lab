# Executor 架构设计

## 核心架构

```
LLMEngine
  └── Worker (owns model + tokenizer)
        └── Executor (runs forward passes)
              ├── tokenize / detokenize
              ├── prefill  (prompt → KV cache → first token)
              └── decode   (KV cache → next token)
```

在真实 vLLM 中，`Worker` 负责管理 GPU 设备与模型加载，`ModelRunner`（类似我们的 `Executor`）负责实际的 forward pass。在我们的简化版本中：

- **Worker**: 创建并持有 Executor，是 Engine 的入口
- **Executor**: 实现 `tokenize`、`detokenize`、`prefill`、`decode` 方法，是 EngineCore 的直接调用对象

---

## Executor 职责

| 职责 | 方法 | 说明 |
|------|------|------|
| 文本编码 | `tokenize(prompt) → List[int]` | 将用户输入转为 token IDs |
| 文本解码 | `detokenize(token_ids) → str` | 将输出 token IDs 转为可读文本 |
| 预填充 | `prefill(sequences)` | 处理 prompt tokens，写入 KV Cache，产出第一个 token |
| 解码 | `decode(sequences)` | 读取 KV Cache，生成下一个 token，写入新 KV |
| Block 生命周期 | `prepare_block(block_id)` | BlockAllocator 分配新 block 时创建 KV 存储 |
| Block 生命周期 | `release_block(block_id)` | BlockAllocator 释放 block 时清理 KV 存储 |
| 序列清理 | `cleanup_sequence(seq_id)` | 序列完成后清理 per-sequence KV cache |
| 统计 | `get_kv_stats() → dict` | KV Cache 使用情况报告 |

Executor 不关心**调度策略**（什么时候调度哪个序列）、不关心**内存管理策略**（什么时候分配/释放 block）。它只负责：
1. 接收 Scheduler 安排好的序列
2. 执行模型 forward pass
3. 管理 KV Cache 数据的物理存储

## Worker 职责

Worker 的职责比 Executor 更简单：

- 在 `__init__` 中创建 Executor 实例（加载模型和 tokenizer）
- 通过 `get_executor()` 返回 Executor 给 Engine

在真实 vLLM 中，Worker 的职责包括：
- 管理 GPU 设备（`torch.cuda.set_device`）
- 初始化分布式通信（NCCL）
- 加载模型权重
- 分配 KV Cache 显存

在我们的设计中，这些细节被简化到 Worker 内部，Engine 不需要知道这些。

---

## FakeExecutor 与 QwenWorker 对比

### FakeModelExecutor

```python
class FakeModelExecutor:
    def __init__(self, config, block_manager=None):
        self._model = FakeModel(vocab_size=config.vocab_size)
        self._kv_cache: Dict[int, List[int]] = {}  # 只存整数

    def tokenize(self, prompt):
        return [ord(c) % vocab_size for c in prompt]

    def prefill(self, sequences):
        for seq in sequences:
            # 简单的整数运算模拟 KV write
            for pos in range(start, end):
                self._write_to_kv(seq, pos, seq.prompt_token_ids[pos])

            # 简单的整数运算模拟 first token
            first_token = (last_token * 3 + 1) % vocab_size
            seq.output_token_ids = [first_token]
```

### QwenExecutor

```python
class QwenExecutor:
    def __init__(self, config, block_manager=None):
        self._model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
        self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        self._seq_kv: Dict[str, past_key_values] = {}  # 存储真实张量

    @torch.no_grad()
    def prefill(self, sequences):
        for seq in sequences:
            input_ids = torch.tensor([token_ids])
            outputs = self._model(
                input_ids=input_ids,
                past_key_values=self._seq_kv.get(seq.seq_id),
                use_cache=True,
            )
            self._seq_kv[seq.seq_id] = outputs.past_key_values

            next_token = torch.argmax(outputs.logits[0, -1, :]).item()
            seq.output_token_ids = [next_token]
```

### 关键区别

| 维度 | FakeModelExecutor | QwenExecutor |
|------|------------------|--------------|
| 模型 | `FakeModel` (纯整数运算) | Qwen2-0.5B (真实 Transformer) |
| Tokenizer | `ord(c) % vocab_size` | HuggingFace AutoTokenizer |
| KV Cache | `Dict[int, List[int]]` | Transformers past_key_values (张量) |
| Prefill | 简单循环写 KV | 真实 forward pass |
| Decode | 数学公式模拟 | 真实自回归生成 |
| 性能 | 微秒级 | 秒级（受模型大小影响） |

### 接口一致性

尽管内部实现完全不同，两者对外暴露的方法签名完全一致：

```python
# 任何 Executor 的实现都必须满足这个接口
class Executor(Protocol):
    def tokenize(self, prompt: str) -> List[int]: ...
    def detokenize(self, token_ids: List[int]) -> str: ...
    def prefill(self, sequences: List[Sequence]) -> None: ...
    def decode(self, sequences: List[Sequence]) -> None: ...
    def prepare_block(self, block_id: int) -> None: ...
    def release_block(self, block_id: int) -> None: ...
    def cleanup_sequence(self, seq_id: str) -> None: ...
    def get_kv_stats(self) -> Dict[str, int]: ...
```

---

## 为什么接口设计重要

### 1. 关注点分离

接口定义了明确的边界：

```
Scheduler → schedule() → [prefill_groups, decode_groups]
                                 ↓
Executor  → prefill(seqs) / decode(seqs)
```

Scheduler 不关心 Executor 如何计算。Executor 不关心 Scheduler 的策略。它们通过 `Sequence` 对象（token IDs、状态）通信，不需要互相知道内部实现。

### 2. 可替换性

因为接口一致，我们可以：

- 在开发测试时用 `FakeModelExecutor`（快、可预测、不需要 GPU）
- 在生产部署时用 `QwenExecutor`（真实模型推理）
- 未来可以加入 `VLLMExecutor`（使用 vLLM 的 PagedAttention）

所有替换只需要改变 `Config.executor_type`，不需要修改 Scheduler、BlockManager、EngineCore。

### 3. 可测试性

```python
def test_scheduler_chunked_prefill():
    # 使用 FakeModelExecutor，不需要 GPU
    config = Config(executor_type="fake")
    engine = LLMEngine(config)
    engine.add_request("test", max_new_tokens=4)
    result = engine.step()
    assert result.num_prefill_tokens == 4
```

Scheduler 的测试完全不依赖模型实现细节。

---

## 为什么模型替换时 Scheduler 不需要修改

这是本架构最关键的观察：

**Scheduler 是策略层，Executor 是实现层。**

Scheduler 回答的问题是：
- 哪个序列组应该被调度？（等待 → 运行）
- 这个步骤处理多少 token？（token budget）
- 优先处理 prefill 还是 decode？（decode-first）

Scheduler 做出这些决策时，**完全不涉及模型计算**。它只操作元数据：
- `seq.status`（PREFILL / RUNNING / FINISHED）
- `seq.prefill_cursor`（还有多少 prompt tokens 未处理）
- `seq.num_generated_tokens`（已经生成了多少 token）
- `seq.sampling_params.max_tokens`（目标长度）

Executor 回答的问题是：
- 给定 prompt tokens，下一个 token 是什么？
- KV Cache 应该如何更新？

当从 `FakeModelExecutor` 切换到 `QwenExecutor` 时：
- `schedule()` 的输入输出完全不变
- Scheduler 依然返回 `ScheduleResult`，包含 `scheduled_prefill_groups` 和 `scheduled_decode_groups`
- EngineCore 依然做 `executor.prefill(prefill_seqs)` 和 `executor.decode(decode_seqs)`
- BlockAllocator/BlockManager 的 `ensure_block()` 调用模式不变

**唯一需要修改的地方**：Engine 的构造函数，选择使用哪个 Worker/Executor。这属于"组合根"（Composition Root）的职责，不属于业务逻辑变更。

### 真实 vLLM 中的对应关系

For a complete module-by-module mapping, see [`docs/VLLM_Mapping.md`](../docs/VLLM_Mapping.md). The key correspondence is: `Executor` (Protocol) ↔ `ModelRunner` abstract interface, `FakeModelExecutor` ↔ no equivalent (pure testing), `QwenExecutor` ↔ `GPUModelRunner` / `CPUModelRunner`.
