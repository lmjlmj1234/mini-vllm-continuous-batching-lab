# Phase 3 计划：Cache Write Reference

## 目标

创建一个独⽴的纯 PyTorch cache write 函数，根据 `slot_mapping` 将每 token 的 K/V 张量逐 slot 写入 `KVCachePool` 的 per-layer cache tensor。

只做写入，不做 attention，不接 executor，不涉及 Qwen 模型。

---

## 接口设计

单层、无状态、纯函数风格。多层由上层逐层调⽤，Phase 3 不承担 layer orchestration。

### 建议 API

```python
def write_to_paged_cache(
    key: torch.Tensor,          # [num_tokens, num_kv_heads, head_dim]
    value: torch.Tensor,        # [num_tokens, num_kv_heads, head_dim]
    key_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_dim]
    slot_mapping: torch.Tensor, # [num_tokens], dtype=torch.long
    block_size: int,
) -> None
```

**为什么不是 `KVCachePool` + `layer_idx`？**

- 保持函数纯粹，不依赖 `KVCachePool` dataclass；
- 上层（executor、AttentionBackend）负责把 `pool.key_caches[layer]` 传过来；
- 测试可以直接创建 `torch.empty()` tensor，无需实例化 pool；
- 避免 Phase 3 处理 model-layer loops。

### 建议文件位置

`mini_vllm/cache/cache_write.py` — 只包含这一个函数。

---

## Shape 和 dtype 契约

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `key` | `[num_tokens, num_kv_heads, head_dim]` | 与 `key_cache` 一致 | 要写入的 K tensor |
| `value` | `[num_tokens, num_kv_heads, head_dim]` | 与 `key_cache` 一致 | 要写入的 V tensor |
| `key_cache` | `[num_blocks, num_kv_heads, block_size, head_dim]` | `torch.float16` / `bfloat16` / `float32` | pool 的 key 侧 |
| `value_cache` | `[num_blocks, num_kv_heads, block_size, head_dim]` | 与 `key_cache` 一致 | pool 的 value 侧 |
| `slot_mapping` | `[num_tokens]` | `torch.long` | 每个 token 对应一个 flat slot 索引 |

**`num_tokens`** = prefill 时为 chunk 内所有 token 数（可能 > 1）；decode 时为 batch size（每 seq 1 token）。

---

## slot_mapping 语义

### 映射公式

```
block_id = slot // block_size          # → index into dim 0 of key_cache / value_cache
block_offset = slot % block_size       # → index into dim 2 of key_cache / value_cache
```

### `slot_mapping[i]` 与 `key[i] / value[i]` 的对应关系

- `slot_mapping[i]` 指定 `key[i]` 和 `value[i]` 写入 cache 的目标 slot；
- `key[i]` 和 `value[i]` **总是写入同一个 slot**；
- `i` 等同在 prefill token 序列中的位置，或 decode batch 中的序号。

### slot == -1 的处理语义

`slot == -1` 表示**此 token 不需要写入 KV cache**。

在当前 Phase 2 架构中，prefill chunk 的所有 token 都写入 cache。但未来 Phase 9 (Chunked Prefill) 在 cache hit 时，部分 prefix token 不需要重新写入。

因此必须支持 `-1` 作为**跳过标记**：

```
if slot == -1:
    continue  # 跳过此 token，不写入 pool
```

这样 Phase 3 接口不需要在上层再加一层过滤逻辑。

### 重复 slot 的处理

**允许。** 如果两个 token 的 `slot_mapping` 相同：

```python
slot_mapping = [5, 5]
key[0] = A, key[1] = B   # 顺序写入 slot 5
```

最终值是 `key[1]` / `value[1]`（最后写入者获胜）。

这在正常操作中不会发生（一个 slot 对应一个 unique token position），但不需要禁止——直观语义就是"最后写入获胜"。测试应覆盖。

### 非连续 physical blocks

**完全支持。** `slot_mapping` 中的 `slot` 经过 `//` 和 `%` 分解后直接索引 dim 0。只要 `block_id < key_cache.shape[0]`，block IDs 不需要连续。

例如 `block_size=4` 时：

```
slot_mapping = [20, 21, 22, 23, 80, 81, 82, 83]
# → block_ids = [5, 5, 5, 5, 20, 20, 20, 20]  # 非连续，完全合法
```

### 未写 slot 的 sentinel 保护

**保持不变。** 函数只写 `slot_mapping` 指定的位置。其他所有位置不受影响。

---

## 实现（pseudocode）

```python
def write_to_paged_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    # --- Validation ---
    num_tokens, num_kv_heads, head_dim = key.shape
    assert value.shape == key.shape
    assert key_cache.shape == value_cache.shape
    assert key_cache.shape[1:] == (num_kv_heads, block_size, head_dim)
    assert len(slot_mapping) == num_tokens
    assert key.dtype == key_cache.dtype
    assert value.dtype == value_cache.dtype
    assert key.device == key_cache.device
    assert value.device == value_cache.device

    for t in range(num_tokens):
        slot = slot_mapping[t].item()
        if slot == -1:
            continue  # skip — prefix token already cached

        block_id = slot // block_size
        block_offset = slot % block_size

        # Write K: [num_kv_heads, head_dim] → [block_id, :, block_offset, :]
        key_cache[block_id, :, block_offset, :] = key[t]

        # Write V: same slot
        value_cache[block_id, :, block_offset, :] = value[t]
```

**论证为什么逐 token 循环是对的：**

- `key[t]` 是 `[num_kv_heads, head_dim]`，直接用 `key_cache[block_id, :, block_offset, :]` 索引；
- PyTorch 的 advanced indexing 在 `:` (slice) 处不会 expand 不存在的 dim；
- 不会意外 flatten `key` 或 `key_cache`；
- Reference 实现优先正确性 > 性能；Phase 5 (Triton kernel) 会做 batched write。

---

## 错误处理

| 条件 | 错误类型 | 消息 |
|------|----------|------|
| `key.shape != value.shape` | `ValueError` | key shape {key.shape} != value shape {value.shape} |
| `key.shape[0] != len(slot_mapping)` | `ValueError` | mismatch num_tokens |
| `key.shape[1] != key_cache.shape[1]` (kv_heads) | `ValueError` | mismatch num_kv_heads |
| `key.shape[2] != key_cache.shape[3]` (head_dim) | `ValueError` | mismatch head_dim |
| `key_cache.shape != value_cache.shape` | `ValueError` | cache shape mismatch |
| `key.dtype != key_cache.dtype` | `ValueError` | dtype mismatch |
| `key.device != key_cache.device` | `ValueError` | device mismatch |
| `slot >= num_blocks * block_size` (越界) | `IndexError` | slot {slot} exceeds max slot {max_slot} |
| `slot < -1` (负值但不是 -1) | `ValueError` | invalid slot {slot} (must be >= -1) |
| `block_size <= 0` | `ValueError` | block_size must be > 0 |

越界 slot 的检查需要**显式验证**——因为 PyTorch 在 out-of-bounds advanced indexing 时可能只给 warning 然后 clamp，不做 `IndexError`。所以必须 `if slot >= total_slots: raise IndexError(...)`。

---

## 测试文件

`tests/test_cache_write.py` — 15 个测试：

| # | 测试名 | 场景 | key 条件 |
|---|--------|------|----------|
| 1 | `test_single_token_decode` | 1 token，已知 slot，写入后读回匹配 | CPU + fp16 |
| 2 | `test_multi_token_prefill` | 1 seq, 8 tokens 跨 2 block (block_size=4)，验证全部 | CPU + fp16 |
| 3 | `test_batched_multi_sequence` | 2 seq，各 1 token，不同 slot，分别验证 | CPU + fp16 |
| 4 | `test_multi_layer` | 3 层，各写入相同 slot，每层值不同，独立验证 | CPU + fp16 |
| 5 | `test_block_boundary` | token 3 在 block A 末尾 + token 4 在 block B 开头 | CPU + fp16 |
| 6 | `test_non_contiguous_blocks` | block IDs = [5, 17, 3] 而非 [0,1,2] | CPU + fp16 |
| 7 | `test_repeated_slot_overwrite` | 同一 slot 写入两次，最终值为第二次 | CPU + fp16 |
| 8 | `test_block_reuse_overwrite` | 先填满 block，释放后重新分配不同 seq，覆盖写入 | CPU + fp16 |
| 9 | `test_unwritten_slots_unchanged` | 写前 clone snapshot，写 2 slot，验证其余不变 | CPU + fp16 |
| 10 | `test_kv_independence` | 同 slot：K=ones, V=zeros，验证独立 | CPU + fp16 |
| 11 | `test_slot_negative_one_skipped` | slot=[-1, 5, -1, 8]，只写 slot 5 和 8 | CPU + fp16 |
| 12 | `test_slot_out_of_range` | slot >= total_slots → IndexError | CPU + fp16 |
| 13 | `test_slot_less_than_negative_one` | slot=-2 → ValueError | CPU + fp16 |
| 14 | `test_shape_dtype_device_errors` | 6 个 submode：shape、kv_heads、head_dim、dtype、device、block_size | CPU |
| 15 | `test_inplace_does_not_create_new_tensor` | 写入后 `key_cache.data_ptr()` 不变 | CPU |

---

## 文件修改清单

| 文件 | 操作 |
|------|------|
| `mini_vllm/cache/cache_write.py` | **新建** — `write_to_paged_cache()` 函数 |
| `mini_vllm/cache/__init__.py` | 导出 `write_to_paged_cache` |
| `mini_vllm/__init__.py` | 导出 `write_to_paged_cache` |
| `tests/test_cache_write.py` | **新建** — 15 个测试 |

---

## 本阶段明确禁止

- ❌ Executor wiring（不修改任何 executor）
- ❌ Attention 数学（不实现 attention 计算）
- ❌ Triton kernel
- ❌ Qwen 模型加载
- ❌ Chunked Prefill 集成
- ❌ 修改 EngineCore / Engine / Scheduler
- ❌ 修改 `AttentionBackend` 具体实现
- ❌ 依赖 `KVCachePool` 以外的 Phase 2 组件

---

## 验收条件

1. 写入后从 cache 读回的值与原值逐元素匹配；
2. 多次 prefill token 跨 block 边界写入全部正确；
3. 多 sequence batch 写入不互相污染；
4. 重复 slot 最后写入者获胜；
5. slot=-1 的 token 被跳过；
6. 越界/负值 slot 报错；
7. shape、dtype、device 不匹配报错；
8. 未写 slot 保持原样；
9. `data_ptr` 不变（in-place write）；
10. 全部 262+15=277 测试通过。
