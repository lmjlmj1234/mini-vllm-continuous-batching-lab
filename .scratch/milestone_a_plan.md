# Milestone A 计划：PagedAttention Correctness（修订版）

## 概述

一次完成从 cache gather 到 PagedAttention 参考实现的全链路，包含 decode/prefill attention、GQA、causal mask、attention scale。最终与连续 PyTorch SDPA 逐元素对齐。

---

## 核心设计决定

### 缓存写入与 Attention 顺序：方案 B

**写入优先，再 gather 全部 KV 做 attention**。即模型层先写本轮 K/V 到 cache，再调 attention read。

```
decode 流程：
  1. QKV projection → (q, k, v)
  2. write_to_paged_cache(slot_mapping[current_decode_token])
  3. decode_attention(q) → gather P+1 token 的 KV 从 cache → 做 attention

prefill 流程：
  1. QKV projection → (q, k, v)
  2. write_to_paged_cache(slot_mapping[P..P+Q-1])
  3. prefill_attention(q, key, value, ...) → gather P+Q token 的 KV 从 cache → 做 attention
     （key/value 参数预留为 future Triton fused path 使用，ref 实现走 cache gather）
```

这样 decode 和 prefill 的 gather 入口完全一致：总是从 cache 读 `kv_len` 个 token。

### gather_paged_kv 的参数

```python
def gather_paged_kv(
    key_cache: torch.Tensor,       # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor,     # same
    block_table: List[int],        # 物理 block ID 列表
    num_tokens: int,               # 要读取的 token 总数 = kv_len
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

`num_tokens` 语义清晰：读这么多 token。调用方传：
- decode：`num_tokens = cached_len_before + 1 (= kv_len_after)`
- prefill：`num_tokens = cached_len_before + query_len (= kv_len_after)`

不再有 `cached_len_before` 和 `kv_len_after` 混淆。

### Causal Mask：显式构造 offset-aware mask

**不再使用 `is_causal=True`**，因为 SDPA 内置的 `is_causal=True` 在 `q_len < kv_len` 时产生错误的 mask（它假设 query position 0 对应 key position 0，但在 prefill with prefix 时 query position 0 对应 key position P）。

显式构造：

```python
# batch_size=1 for per-seq loop
# q_len = Q, kv_len = P + Q
causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
# shape: [Q, P+Q], bool matrix
# mask[i, j] = True iff key_position[j] <= query_position[i]

attn_output = F.scaled_dot_product_attention(
    q_tokens, full_k, full_v,  # each with batch=1
    attn_mask=causal_mask,
    is_causal=False,
    scale=scale,
)
```

**mask 属性验证**：

| 场景 | P | Q | mask[i,j] = key[j] <= query[i] = (P+i) | 预期 |
|------|---|---|----------------------------------------|------|
| Full prefill | 0 | N | j ≤ i | 标准因果 |
| Chunk: q0 | 8 | 2 | j ≤ 8 | 可见 0..8 |
| Chunk: q1 | 8 | 2 | j ≤ 9 | 可见 0..9 |
| Decode | 8 | 1 | j ≤ 8 | 可见全部 0..8 |

---

## 文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| NEW | `mini_vllm/cache/cache_read.py` | `gather_paged_kv()` 函数 |
| NEW | `mini_vllm/attention/paged_attention_ref.py` | `AttentionBackendRef` 全实现 |
| MOD | `mini_vllm/attention/__init__.py` | 导出 `AttentionBackendRef` |
| NEW | `tests/test_paged_attention_ref.py` | ~25 个测试 |

---

## 模块设计

### 1. `cache_read.py` — gather_paged_kv

```python
def gather_paged_kv(
    key_cache: torch.Tensor,       # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor,     # same
    block_table: List[int],        # physical block IDs
    num_tokens: int,               # number of KV tokens to gather
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather contiguous K/V sequences from paged cache.
    
    Iterates logical blocks 0, 1, 2, ... mapping each to physical block
    through block_table. Each block contributes up to block_size tokens.
    The last block may be partial (num_tokens not aligned to block_size).
    
    Returns:
        key_out:   [num_tokens, num_kv_heads, head_dim]
        value_out: [num_tokens, num_kv_heads, head_dim]
    """
    remaining = num_tokens
    key_parts, value_parts = [], []
    logical_idx = 0
    
    while remaining > 0:
        physical_id = block_table[logical_idx]
        tokens_this_block = min(block_size, remaining)
        
        # 从 cache 读: [num_kv_heads, block_size, head_dim] -> slice [:tokens_this_block]
        k_block = key_cache[physical_id, :, :tokens_this_block, :]
        v_block = value_cache[physical_id, :, :tokens_this_block, :]
        
        # 转置: [num_kv_heads, tokens, head_dim] -> [tokens, num_kv_heads, head_dim]
        key_parts.append(k_block.permute(1, 0, 2))
        value_parts.append(v_block.permute(1, 0, 2))
        
        remaining -= tokens_this_block
        logical_idx += 1
    
    if num_tokens == 0:
        n_kv_heads, head_dim = key_cache.shape[1], key_cache.shape[3]
        return (
            torch.empty(0, n_kv_heads, head_dim, dtype=key_cache.dtype, device=key_cache.device),
            torch.empty(0, n_kv_heads, head_dim, dtype=key_cache.dtype, device=key_cache.device),
        )
    
    return torch.cat(key_parts, dim=0), torch.cat(value_parts, dim=0)
```

### 2. `paged_attention_ref.py` — AttentionBackendRef

```python
class AttentionBackendRef(AttentionBackend):
    """Pure-PyTorch PagedAttention reference.
    
    方案 B：写入优先，attention 只从 cache gather 全部 KV。
    每序列逐 token loop — reference 实现，正确性优先。
    """
    
    def __init__(self, config: ModelConfig):
        self._config = config
        self._pool: Optional[KVCachePool] = None
        self._block_size: int = 0
    
    def allocate_pool(self, ...) -> KVCachePool:
        pool = KVCachePool.allocate(...)
        self._pool = pool
        self._block_size = pool.block_size
        return pool
    
    def write_kv_cache(self, layer_idx, key, value, slot_mapping):
        pool = self._pool
        write_to_paged_cache(
            key, value,
            pool.key_caches[layer_idx],
            pool.value_caches[layer_idx],
            slot_mapping,
            pool.block_size,
        )
```

#### decode_attention 实现

```python
def decode_attention(self, layer_idx, query, attn_metadata, pool):
    num_decode = query.shape[0]
    if num_decode == 0:
        return query
    
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = pool.num_kv_heads
    n_repeats = num_heads // num_kv_heads
    scale = head_dim ** -0.5
    
    decode_group = next(g for g in attn_metadata.groups 
                        if g.attention_type == "decode_gpu")
    
    output_parts = []
    for i in range(num_decode):
        seq_idx = decode_group.seq_indices[i]
        cached = decode_group.cached_len_before[i].item()
        kv_len = cached + 1  # existing + this decode token
        
        block_row = attn_metadata.decode_block_tables[seq_idx]
        block_ids = [int(b) for b in block_row if b != -1]
        
        k, v = gather_paged_kv(
            pool.key_caches[layer_idx],
            pool.value_caches[layer_idx],
            block_ids, kv_len, pool.block_size,
        )
        k = k.repeat_interleave(n_repeats, dim=1)
        v = v.repeat_interleave(n_repeats, dim=1)
        
        q = query[i].unsqueeze(0)  # [1, num_heads, head_dim]
        k = k.unsqueeze(0)         # [1, kv_len, num_heads, head_dim]
        v = v.unsqueeze(0)
        
        # For Q=1, the single query at position P should see ALL keys.
        # Mask: [1, kv_len] all True.
        attn_output = F.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False,
        )
        
        # is_causal=False + no attn_mask = full attention = correct for Q=1
        output_parts.append(attn_output)
    
    return torch.cat(output_parts, dim=0)
```

**NOTE:** decode 的 Q=1，不需要 causal mask（第 i 个 query 就一个 token，看不到自己之后的任何 token）。全 attention mask（没有 mask）就是正确的。

但等一下 —— 对于 batch decode，每个 decode token 的位置不同。`is_causal=False` + `attn_mask=None` 意味着 `q` 可以 attend 全部 `kv`。由于每个 decode token 的 `k`/`v` 已经包含了它自己（通过方案 B cache write），如果允许 attend 全部，那当前 decode token 会 attend 到自己的 KV，这是正确的。但会不会 attend 到 batch 中其他 seq 的 KV？不会，因为每个 seq 的 K/V 是单独 gather 的。

所以 decode 用 `is_causal=False` 不加 mask 就是对的。

#### prefill_attention 实现

```python
def prefill_attention(self, layer_idx, query, key, value, attn_metadata, pool):
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = pool.num_kv_heads
    n_repeats = num_heads // num_kv_heads
    scale = head_dim ** -0.5
    
    output_parts = []
    token_offset = 0
    
    for group in attn_metadata.groups:
        if group.attention_type not in ("prefill_gpu", "prefill_ref"):
            continue
        
        for local_i in range(len(group.seq_indices)):
            cached = group.cached_len_before[local_i].item()
            q_len = group.query_len[local_i].item()
            kv_len = cached + q_len
            
            block_row = attn_metadata.prefill_block_tables[
                group.seq_indices[local_i]
            ]
            block_ids = [int(b) for b in block_row if b != -1]
            
            # Gather ALL KV from cache (方案 B：已写了当前 chunk)
            full_k, full_v = gather_paged_kv(
                pool.key_caches[layer_idx],
                pool.value_caches[layer_idx],
                block_ids, kv_len, pool.block_size,
            )
            full_k = full_k.repeat_interleave(n_repeats, dim=1)
            full_v = full_v.repeat_interleave(n_repeats, dim=1)
            
            # Offset-aware causal mask
            # query_positions = P, P+1, ..., P+Q-1
            # key_positions   = 0, 1, ..., P+Q-1
            # mask[q_i, k_j] = key_positions[k_j] <= query_positions[q_i]
            device = query.device
            query_pos = torch.arange(cached, cached + q_len, device=device)
            key_pos = torch.arange(kv_len, device=device)
            causal_mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
            # shape [Q, P+Q] = [Q, kv_len]
            
            q_tokens = query[token_offset:token_offset + q_len]
            q_tokens = q_tokens.unsqueeze(0)
            fk = full_k.unsqueeze(0)
            fv = full_v.unsqueeze(0)
            
            attn_output = F.scaled_dot_product_attention(
                q_tokens, fk, fv,
                attn_mask=causal_mask,
                is_causal=False,
                scale=scale,
            )
            
            output_parts.append(attn_output.squeeze(0))
            token_offset += q_len
    
    if not output_parts:
        return query
    return torch.cat(output_parts, dim=0)
```

---

## 测试设计

测试思路：每个测试手动构造已知 K/V cache，执行 write + gather + attention，与 contiguous SDPA reference 逐元素对齐（FP16 atol=1e-3）。

### gather_paged_kv 测试

| # | 测试 | 场景 |
|---|------|------|
| 1 | `test_gather_full_block` | 1 block，读全部 block_size token |
| 2 | `test_gather_partial_block` | 1 block，读部分 token |
| 3 | `test_gather_multi_block` | 3 blocks，读 10 token (4+4+2) |
| 4 | `test_gather_non_contiguous_blocks` | block_ids=[7, 3, 15] |
| 5 | `test_gather_zero_tokens` | num_tokens=0 → 空 tensor |
| 6 | `test_gather_round_trip` | write → gather → 逐元素一致 |

### decode attention 测试（与 contiguous SDPA 对齐）

| # | 测试 | P | Q | 验证 |
|---|------|---|---|------|
| 7 | `test_decode_single_seq` | 4 | 1 | decode token 正确 |
| 8 | `test_decode_multi_seq_diff_context` | [4, 8, 12] | 均为 1 | 不同 context length |
| 9 | `test_decode_block_boundary` | 7 (跨 block: 4+3) | 1 | KV 跨 block 边界 |
| 10 | `test_decode_non_contiguous_blocks` | 8 (block_ids=[6, 11]) | 1 | 非连续物理块 |
| 11 | `test_decode_gqa` | num_heads=8, kv_heads=2 | 任意 | repeat_interleave 展开正确 |
| 12 | `test_decode_partial_last_block` | 5 (block_size=4, 1st block 满, 2nd 1/4) | 1 | partial block |
| 13 | `test_decode_sentinel_not_read` | 只写 4 token, kv_len=4 | — | 未初始化区域不被读 |

### prefill attention 测试（offset-aware causal mask 验证）

| # | 测试 | P | Q | 验证 |
|---|------|---|---|------|
| 14 | `test_prefill_full_prompt` | 0 | 6 | P=0, Q=6, 标准因果 |
| 15 | `test_prefill_chunk_with_prefix` | 8 | 2 | P>0, Q>1, offset mask |
| 16 | `test_prefill_chunk_longer` | 8 | 3 | P>0, Q>1, 更长的 Q |
| 17 | `test_prefill_chunk_q1` | 8 | 1 | P>0, Q=1（等效 decode） |
| 18 | `test_prefill_multi_seq` | [0, 8] | [4, 2] | 不同 prefix length |
| 19 | `test_prefill_future_token_leakage` | 8 | 3 | 构造未来 token 极大值，确认 q1 未见 future |

### attention scale & mask 对比

| # | 测试 | 验证 |
|---|------|------|
| 20 | `test_attention_scale` | scale = 1/sqrt(head_dim) |
| 21 | `test_causal_mask_hand_crafted` | 与手写 attention mask 的 contiguous SDPA 对齐 |

### ref backend 集成测试

| # | 测试 | 验证 |
|---|------|------|
| 22 | `test_backend_write_kv_cache` | write_kv_cache 委托到 write_to_paged_cache |
| 23 | `test_backend_factory` | AttentionBackend.create(backend="reference") 返回 AttentionBackendRef |

### contiguous reference 对齐方法

所有 attention 测试执行步骤：

```python
# 1. 构造随机 K/V 并写入 paged cache
write_to_paged_cache(k, v, key_cache, value_cache, slots, block_size)

# 2. 做 paged attention
output_paged = backend.decode_attention(0, query, attn_meta, pool)

# 3. 构造 contiguous K/V 作为 ground truth
kv_contiguous = gather_paged_kv(key_cache, value_cache, block_ids, kv_len, block_size)
# 或用随机生成但已知的连续 K/V

# 4. 做 contiguous SDPA
contiguous_k = kv_contiguous[0].repeat_interleave(n_repeats, dim=1).unsqueeze(0)
contiguous_v = kv_contiguous[1].repeat_interleave(n_repeats, dim=1).unsqueeze(0)
q = query_for_this_seq.unsqueeze(0)

if Q == 1:
    # Decode: full attention (Q=1, no causal mask needed)
    ref = F.scaled_dot_product_attention(q, contiguous_k, contiguous_v,
                                          scale=scale, is_causal=False)
else:
    # Prefill: offset-aware causal mask
    mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
    ref = F.scaled_dot_product_attention(q, contiguous_k, contiguous_v,
                                          attn_mask=mask, is_causal=False,
                                          scale=scale)

# 5. 逐元素对齐
assert torch.allclose(output_paged, ref.squeeze(0), atol=1e-3)
```

---

## 本里程碑禁止

- ❌ 不实现 Triton kernel
- ❌ 不加载 Qwen 模型
- ❌ 不修改 EngineCore 主执行链
- ❌ 不修改 executor
- ❌ 不修改 scheduler
- ❌ 不修改 `AttentionBackend.create()` 以外的 factory

## 验收条件

1. gather_paged_kv 从正确物理 block 读取正确数量 token
2. Cache round-trip: write → gather → read 值与写入一致
3. **方案 B**: decode/prefill 先写 cache 再 gather，不使用 `is_causal=True`
4. **显式 offset-aware causal mask**: `mask[i,j] = key_pos[j] <= query_pos[i]`
5. Decode (Q=1): 无 mask，全 attention，当前 token 可见全部已有 KV
6. Full prefill (P=0): Q>1，标准因果
7. Chunked prefill (P>0, Q>1): offset mask，q0 可见 0..P，q(Q-1) 可见 0..P+Q-1
8. Future-token leakage test: future 位置设极大值，验证 q_i 未见 k_{>P+i}
9. GQA repeat_interleave 正确
10. 非连续 block IDs 不影响结果
11. partial last block 正确处理
12. Attention scale = 1/sqrt(head_dim)
13. 未写 sentinel 不被读取
14. AttentionBackendRef 通过 factory create
15. 全部 ~310 tests pass
