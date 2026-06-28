# Prefix Cache

## 设计动机

### 为什么需要 Prefix Cache

在真实推理部署中，大量请求共享相同的**前缀**：

- **System Prompt**: "You are a helpful assistant. Answer concisely."
- **Few-shot Examples**: "Q: What is X? A: X is ... Q: What is Y? A:"
- **Chat History**: 多轮对话中，每次新的用户输入前面都带着完整历史

如果每个请求都从零开始 prefill 这一段内容，意味着：

1. **重复计算**: GPU 反复计算相同的注意力矩阵
2. **浪费显存带宽**: 相同的 KV Cache 数据被反复写入
3. **增加 TTFT**: 每个新请求都要等待 prefill 完成

Prefix Cache 的回答是：**相同块 → 共享，不重复计算**。

---

### Prefix Cache 如何降低 TTFT

TTFT = 等待时间 + Prefill 时间

没 Prefix Cache：
```
请求 A: [system_prompt | few_shot  | user_query] → prefill 全部 200 token → 200ms
请求 B: [system_prompt | few_shot  | user_query'] → prefill 全部 200 token → 200ms
```

有 Prefix Cache：
```
请求 A: [system_prompt | few_shot  | user_query] → prefill 200 token → 200ms
                    ↓ hash → cache
请求 B: [system_prompt | few_shot  | user_query'] → prefill 50 token → 50ms
请求 B 的 system_prompt + few_shot (150 token) 共享 A 的 KV Cache
```

TTFT 降低量正比于共享前缀的长度。共享 75% 的前缀 → TTFT 降低约 75%。

**但这不只是 TTFT 问题**。共享前缀的请求不消耗 GPU 计算资源去重复 prefill，这意味着 GPU 可以把资源用于处理更多新请求。Throughput 也随之提高。

---

## 架构设计

### 核心关系图

```
Sequence A ("The capital of France is Paris")
Sequence B ("The capital of France is London")
              (block_size=4)

Sequence A:
  Tokens: [   The capital of     |  France is   P   |       aris         ]
          ┌───────────────────┐┌───────────────────┐┌───────────────────┐
Block      Block 0            │ Block 1            │ Block 2            │
(Logical)  L0                 │ L1                 │ L2                 │
          └────────┬──────────┘└────────┬──────────┘└────────┬──────────┘
                   │                    │                    │
         ┌─────────▼──────────┐  ┌──────▼──────────┐  ┌─────▼───────────┐
BlockTable A:  [     0,           |        2,          |        4         ]
              shared(P0)          exclusive            exclusive
                                   ↓                    ↓
         ┌─────────▼──────────┐  ┌──────▼──────────┐  ┌─────▼───────────┐
Physical    Block 0 (ref=2)      Block 2 (ref=1)      Block 4 (ref=1)
Blocks:   ┌───────────────────┐┌───────────────────┐┌───────────────────┐
          │   KV: [k1,v1,...] ││   KV: [k5,v5,...] ││   KV: [k9,v9,...] │
          └───────────────────┘└───────────────────┘└───────────────────┘

Sequence B:
  Tokens: [   The capital of     |  France is   L   |       ondon         ]
          ┌───────────────────┐┌───────────────────┐┌───────────────────┐
Block      Block 0            │ Block 1            │ Block 2            │
(Logical)  L0                 │ L1                 │ L2                 │
          └────────┬──────────┘└────────┬──────────┘└────────┬──────────┘
                   │                    │                    │
         ┌─────────▼──────────┐  ┌──────▼──────────┐  ┌─────▼───────────┐
BlockTable B:  [     0,           |        3,          |        5         ]
              shared(P0)            exclusive(COW未来)   exclusive
                   │                    │                    │
                   ▼                    ▼                    ▼
Physical    Block 0 (ref=2)      Block 3 (ref=1)      Block 5 (ref=1)
Blocks:   ┌───────────────────┐┌───────────────────┐┌───────────────────┐
          │   KV: [k1,v1,...] ││   KV: [k6,v6,...] ││   KV: [k10,v10,.] │
          └───────────────────┘└───────────────────┘└───────────────────┘


Prefix Cache (hash → physical_block_id):
┌─────────────────────────────────────────────┐
│  hash([The capital of ])  →  Block 0        │  ← 由 A 注册，B 共享
│  hash([France is P])      →  Block 2        │  ← A 独有
│  hash([aris])             →  Block 4        │  ← A 独有
│  hash([France is L])      →  Block 3        │  ← B 独有
│  hash([ondon])            →  Block 5        │  ← B 独有
└─────────────────────────────────────────────┘
```

---

### 关键概念

#### 1. Ref Count（引用计数）

每个物理 block 有一个引用计数。这是 Prefix Cache 能够工作的基础：

```
Physical Block 0: ref_count = 2
  → 被 Sequence A 引用（prefix hit，allocate 时设 ref=1）
  → 被 Sequence B 引用（cache hit，increment_ref 后 ref=2）

释放流程：
  Sequence A 结束 → BlockManager.free("A") → allocator.free([0]) → ref 2→1
  Sequence B 结束 → BlockManager.free("B") → allocator.free([0]) → ref 1→0 → 真正释放
```

**为什么需要 Ref Count**：
- 没有 ref count，无法知道 block 是否还被其他 sequence 使用
- 没有 ref count，Sequence A 结束后 Block 0 会被释放，Sequence B 还在用，导致 UAF（use-after-free）
- Ref count 是 Copy-on-Write 的基础：写共享 block 前，检查 ref_count > 1 → 需要 COW

#### 2. Shared Block

Shared Block 是一个物理 block 被多个 BlockTable 同时引用。它的核心约束：

- **只读**: 共享的 block 不能被写入新数据
- **不变性**: 共享 block 内的 KV 数据在共享期间不变
- **生命周期**: 被所有引用者释放后才真正释放

BlockTable 通过 `is_shared` 标记识别共享 entry。未来 COW 时，这个标记触发写时复制流程。

#### 3. Prefix Hash

每个逻辑 block 的内容（prompt token IDs）通过 hash 函数映射为一个整数：

```python
hash = hash(tuple(block_tokens))
```

两个 sequence 的相同 block 内容 → 相同 hash → 命中 cache。

约束：
- hash 碰撞理论上可能导致错误共享
- 对教育实现容忍，生产环境使用更健壮的 hash（如 xxhash）
- Hash 仅针对 prompt 内容（decode 产生的 token 会进入未缓存的新 block）

---

### 数据流

```
Step N:
  1. Scheduler.schedule()
     → Phase 5: admit waiting group
       → create_sequence(seq_id)
       → BlockManager.allocate_for_seq(seq)
         → compute block hashes from seq.prompt_token_ids
         → PrefixCache.lookup(hash) for each block
         → for matching blocks: increment_ref + add_shared_block
         → shared_count = number of consecutive matched blocks
         → remaining blocks allocated on-demand during prefill
       → seq.status = PREFILL
  
  2. EngineCore: executor.prefill(seqs)
     → for each sequence, for each position:
       → _write_to_kv(position, token)
         → is_block_shared(seq, position)?
           YES → return immediately (data already exists)
           NO  → ensure_block(seq, position)
                  → block already in table? return PID
                  → need new block?
                     → PrefixCache.lookup(hash)? 
                       YES → increment_ref + add_shared_block
                       NO  → allocate() + add_block + insert(hash, pid)
                  → write token KV data to block

Step N+1:
  → Another sequence with same prefix admitted
  → allocate_for_seq finds cache hits → shares blocks
```

---

### 与 BlockTable 的配合

Shared Block 必须和 BlockTable 配合的根本原因：

**BlockTable 是 sequence 访问物理 block 的唯一入口。**

Sequence 不知道物理 block 的绝对地址。它只知道："我的第 i 个逻辑 block 对应物理 block X"。

当 prefix cache 共享一个 block 时：
1. BlockManager 在 `allocate_for_seq` 中将共享 block 添加到 BlockTable
2. BlockTable entry 标记 `is_shared=True`
3. Executor 通过 `is_block_shared()` 检查要不要写入
4. 未来 COW 时，executor 替换 BlockTable 中的 entry

没有 BlockTable，sequence 无法访问物理 block。没有 prefix cache，BlockTable 只能指向独占 block。两者配合才实现了"共享无感"：

```python
# Executor 中的代码 —— 不关心 block 是共享还是独占
pid = block_manager.ensure_block(seq, position)  # 返回物理 block ID
if not block_manager.is_block_shared(seq, position):
    kv_cache[pid].write(...)  # 只有独占 block 才写
```

---

### 未来扩展到 Copy-on-Write

#### 为什么需要 Copy-on-Write

共享的 block 是**只读**的。但 Decode 阶段，Sequence 会生成新 token，写入新的 KV Cache。

问题：如果 Sequence A 和 B 共享 Block 0，在 Decode 时，A 需要写入 Block 0 的下一个 token 位置——但 Block 0 是共享的。

**Copy-on-Write 解决这个问题**：

```
1. Sequence A 需要写入 Block 0 中某个 token position
2. Executor 检查 Block 0 是否 shared → is_shared_at(position) → True
3. Allocate 一个新的物理 block (Block 0')
4. 将 Block 0 的现有 KV 数据复制到 Block 0'
5. 将 Sequence A 的 BlockTable entry（逻辑 Block 0）指向 Block 0'
6. Decrement Block 0 的 ref_count
7. 现在：
   - Sequence B 仍然指向 Block 0（共享，ref=1）
   - Sequence A 指向 Block 0'（独占，ref=1）
8. Sequence A 写入新数据到 Block 0'
```

#### 当前设计中的 COW 扩展点

```python
@dataclass
class BlockTableEntry:
    physical_block_id: int
    is_shared: bool = False  # <-- COW 触发器

class BlockTable:
    def is_shared_at(self, token_position: int) -> bool:
        """Executor 用这个检测是否需要 COW"""
        ...
```

BlockAllocator 的 ref count 已经支持 COW 的引用管理：

```python
# 未来 COW 实现伪代码
def copy_on_write(seq, position):
    old_pid = table.get_physical_block(position)
    old_ref = allocator.get_ref_count(old_pid)
    
    if old_ref > 1:
        # Block is shared → need COW
        new_pid = allocator.allocate(1)[0]
        copy_kv_data(old_pid, new_pid)
        table.replace_entry(logical_idx, new_pid, is_shared=False)
        allocator.free([old_pid])  # decrement ref
        return new_pid
    else:
        # Block is exclusive → safe to write
        return old_pid
```

不需要修改 BlockAllocator、PrefixCache 和 BlockManager 的核心逻辑——这是设计的关键质量。

---

## 代码结构

| 文件 | 新增/修改 | 职责 |
|------|----------|------|
| `cache/prefix_cache.py` | **新增** | Hash-based Prefix Cache (lookup/insert、ProbeResult) |
| `cache/allocator.py` | **修改** | 添加 ref_count 跟踪和 increment_ref() |
| `cache/block_table.py` | **修改** | 添加 BlockTableEntry (is_shared flag), add_shared_block() |
| `cache/manager.py` | **修改** | allocate_for_seq 和 ensure_block 集成 prefix cache + probe_prefix_cache() |
| `scheduler/scheduler.py` | **修改** | Phase 5 admit 调用 probe_prefix_cache，使用 uncached tokens 计算 budget |
| `scheduler/schedule_result.py` | **修改** | 新增 cached_token_count / num_uncached_prefill_tokens / matched_block_count |
| `engine/metrics.py` | **修改** | 新增 prefix cache 指标（total_cached_tokens, hit rate） |
| `executor/executor.py` | **修改** | _write_to_kv 跳过共享 block 的写入 |

---

## Scheduler-aware Prefix Cache

### 为什么 Scheduler 需要感知

在第一版设计中，Prefix Cache 是 BlockManager 内部的透明优化。Scheduler 完全不知道 cache 的存在。这意味着：

```
Scheduler budget 计算（旧版）：
  prompt_len = len(tokens)  →  this_chunk = min(prompt_len, chunk_size)
  → 全部 token 占用 budget

实际执行（有 cache）：
  N 个 block 是共享的 → KV 数据已存在 → 不需要计算
  → 但 Scheduler 仍然为这些 token 扣除了 budget
  → 浪费调度机会
```

**问题**：当 prefix cache 命中时，实际 prefill 计算量小于 Scheduler 假设的值。这导致：
- Token budget 被浪费（已缓存的 token 不需要计算，但仍然占用预算）
- Admission 决策过于保守（因为预算假设计算了更多 token）
- Chunked prefill 长度过大（缓存的部分不需要分块）

**修复**：Scheduler 应该知道多少 prompt tokens 已缓存，只对未缓存的部分做预算计算。

### 两阶段设计

```
Phase 1: Probe（只读查询）

Scheduler.schedule(), Phase 5 Admit:
  probe = BlockManager.probe_prefix_cache(prompt_token_ids)
  uncached = prompt_len - probe.cached_token_count
  this_chunk = min(uncached, chunk_size)

  → 只检查，不修改 ref_count
  → Scheduler 基于 uncached tokens 做 budget 决策

Phase 2: Allocate（实际分配）

Scheduler.schedule(), Phase 5 Admit (继续):
  seq = create_sequence(...)
  BlockManager.allocate_for_seq(seq)  ← 真正的 attach + ref_count++
  seq.prefill_cursor = probe.cached_token_count  ← 跳过缓存部分
```

**为什么分离**：
1. **关注点分离**：Scheduler 是策略层（知道"有多少缓存"），BlockManager 是实现层（知道"如何共享 block"）
2. **只读安全**：Probe 不修改状态，可以安全地多次调用，不会因 probe 导致内存错误
3. **状态一致**：Allocate 在 budget 确认后才执行，避免了"查了不用"的资源浪费

### Scheduler 的变化

**预算计算**（Phase 5 Admit）：

```python
# 旧：所有 prompt tokens 计入预算
prompt_len = len(sg.prompt_token_ids)
this_chunk = min(prompt_len, chunk_size)

# 新：只有未缓存 token 计入预算
probe = block_manager.probe_prefix_cache(sg.prompt_token_ids)
uncached = prompt_len - probe.cached_token_count
this_chunk = min(uncached, chunk_size)
```

**Prefill Cursor**（跳过已缓存部分）：

```python
# 旧：从头开始
seq.prefill_cursor = 0

# 新：跳过已缓存的 block
seq.prefill_cursor = probe.cached_token_count
```

**ScheduleResult 记录**：

```python
result.cached_token_count += probe.cached_token_count  # 缓存命中数
result.num_uncached_prefill_tokens += this_chunk       # 实际计算量
result.matched_block_count += probe.matched_block_count  # 共享 block 数
```

### 完整数据流

```
Step N:
  Scheduler.schedule()
    Phase 5 (Admit):
      1. BlockManager.probe_prefix_cache(tokens)
         → 只读：计算 hash，查 cache，验证 ref_count > 0
         → 返回 PrefixCacheProbeResult {block_count, token_count, pids}
      
      2. uncached = prompt_len - probe.cached_token_count
         this_chunk = min(uncached, chunk_size)
      
      3. budget check: this_chunk > prefill_budget?
         → NO → admit (uncached 部分很小，预算充足)
         → YES → ignore/reject (uncached 部分太大)
      
      4. BlockManager.allocate_for_seq(seq)
         → 实际 attach：increment_ref, add_shared_block
         → 与 probe 结果一致（但自己重新检查，防止竞态）
      
      5. seq.prefill_cursor = probe.cached_token_count

  EngineCore: executor.prefill(seqs)
    → prefill 从 cursor 位置开始
    → cursor 已跳过所有缓存 token
    → 只处理未缓存的部分
    → 写新 KV，注册新 block 到 cache
```

### 测试验证

关键测试场景：

| 测试 | 预期 |
|------|------|
| Probe 不修改 ref_count | 多次 probe 后 ref_count 不变 |
| allocate_for_seq 修改 ref_count | probe→allocate 后 ref_count 增加 |
| Stale cache 不计入 cached count | 所有 ref 释放后 probe 返回 0 |
| Cache hit 减少 budget 消耗 | 同 prefix 的第二请求消耗更少预算 |
| 部分匹配只缓存匹配 block | 不同 suffix 的第一个不同 block 后不再共享 |
| Metrics 报告缓存命中率 | report() 包含 total_cached_tokens 和 hit_rate |

---

## 与真实 vLLM 的对应

| mini-vLLM | 真实 vLLM | 差异 |
|-----------|----------|------|
| `PrefixCache` | `BlockPrefixMgr` | 真实 vLLM 使用 LRU 缓存，支持 eviction |
| `PrefixCacheProbeResult` | Scheduler 内部 prefix 匹配结果 | 真实 vLLM 在 Scheduler 内直接计算，无显式 probe 对象 |
| `probe_prefix_cache()` | Scheduler 的 `_get_cached_prefix_len()` | 概念相同：查 cache 但不修改 ref |
| Ref Count | `BlockAllocator.refcount` | 完全一致 |
| `BlockTableEntry.is_shared` | `LogicalTokenBlock` | 真实 vLLM 在 block 粒度跟踪共享 |
| `allocate_for_seq` + cache | Scheduler `PrefixCacheHit` | 真实 vLLM 在 scheduler 内做 prefix matching |
| Scheduler 预算使用 uncached tokens | Scheduler `num_unmatched_tokens` | 完全一致的设计决策 |
| `prefill_cursor = cached_token_count` | `seq.data.num_computed_tokens` | 语义相同：跳过已计算部分 |
| `result.cached_token_count` | 无直接对应（vLLM 在日志中 tracking） | 教育实现更显式 |
| `matched_block_count` | `num_matched_blocks` | 概念相同 |
| Hash function | `hash_request_time` | 真实 vLLM 使用更复杂的 hash 避免碰撞 |
| COW 扩展点 | `copy-on-write` | 真实 vLLM 在 `BlockSpaceManager` 中实现 |

主要差异：
- 真实 vLLM 在 Scheduler 层做 prefix 匹配（决定能跳过多少 prefill）
- 我们的 probe 是 BlockManager 的方法，Scheduler 调用它——接口更清晰
- 真实 vLLM 有缓存淘汰（LRU），我们暂未实现
- 真实 vLLM 的 matched_block_count 用于跳过 prefill（set_block_manager），我们的 cached_token_count 用于设置 prefill_cursor

---

## Review Questions

### 基础理解

**Q1**: Prefix Cache 为什么能降低 TTFT？

> A: 相同前缀的 prompt token 对应的 KV Cache block 只需计算一次，后续请求共享已有 block。TTFT 中 prefill 时间正比于需要**新计算**的 token 数。共享前缀越长，需要新计算的 token 越少，TTFT 越小。

**Q2**: 为什么需要引用计数（Ref Count）？

> A: 当一个 block 被多个 sequence 共享时，某个 sequence 结束不能简单地释放 block —— 需要确认是否还有其他 sequence 在使用。Ref count 跟踪这一信息，只在 ref_count 降为 0 时真正释放 block。没有 ref count，要么 UAF（提前释放），要么内存泄漏（从不释放）。

**Q3**: Shared Block 为什么必须和 BlockTable 配合？

> A: BlockTable 是 sequence 访问物理 block 的唯一路径。共享 block 的本质是多个 BlockTable 包含同一个 physical_block_id。BlockTable 的 `is_shared` 标记让 executor 知道是否应该写入数据。没有 BlockTable，sequence 不知道哪些 block 是共享的、session 不知道自己的逻辑到物理映射。

### 设计权衡

**Q4**: 如果两个 sequence 同时到达，有相同的 prefix，为什么不共享？

> A: 因为 prefix cache 在第一个 sequence 的 ensure_block 执行后才注册。如果两个 sequence 在同一个 scheduler step 中被 admit，allocate_for_seq 时 cache 仍是空的。这是正确的行为——跨 step 共享，同 step 不共享。真实 vLLM 中也有类似的限制。

**Q5**: Hash 碰撞会怎样？如何缓解？

> A: Hash 碰撞 = 不同的 token 内容映射为相同的 hash → 错误地共享了实际上不同的 KV 数据 → 模型输出错误。缓解：使用更健壮的 hash（xxhash），或在 hash 匹配后做 full token comparison。教育实现中接受碰撞风险。

**Q6**: 为什么注册在 allocate 时而不是 block 写满时？

> A: 因为 prompt tokens 在 sequence 创建时就完全已知。Block 注册在分配时刻（写数据之前），另一个 sequence 看到 cache 命中时 block 可能尚未写完，但只要写入顺序正确（先写的 sequence 先处理），读取时数据已就绪。这个简化在教育场景中是可接受的。

### 架构理解

**Q7**: 如果切换到 QwenExecutor，Prefix Cache 是否还有效？

> A: Prefix Cache 的核心——hash 匹配、block 共享、ref count——在 BlockAllocator/BlockManager 层面，不依赖 executor 类型。但 QwenExecutor 存储 KV 为 per-seq 张量（past_key_values），不是 block 粒度的显存。因此 block 级共享的 KV 数据需要额外的桥接逻辑才能被模型读取。教育实现中 FakeModelExecutor 完全支持 prefix cache，QwenExecutor 需要额外的 KV 数据整合工作。

**Q8**: Copy-on-Write 如何从当前设计扩展？

> A: (1) Executor 检测 `is_block_shared() = True` → (2) 分配新 block → (3) 复制 KV 数据 → (4) 替换 BlockTable entry → (5) 写新数据 → (6) 释放旧 block 的引用。BlockAllocator 的 ref_count 已支持引用管理，BlockTableEntry.is_shared 已标记共享 block，COW 只需要 executor 层实现写时检测和 block 替换逻辑。

**Q9**: Prefix Cache 可能引入哪些正确性问题？

> A: (1) Hash 碰撞 → 数据错误。(2) 共享 block 被写入 → 数据覆盖，影响所有 sharer。我们的实现通过 `is_block_shared()` + `_write_to_kv` 跳过写入来防止 (2)。(3) Stale cache entry（block 被所有引用释放后，cache 中 hash→pid 仍存在）→ 通过检查 `get_ref_count(pid) > 0` 在共享前验证来处理。

### 真实 vLLM

**Q10**: 真实 vLLM 的 Prefix Cache 与我们的实现有什么本质区别？

> A: (1) 真实 vLLM 的 prefix matching 发生在 Scheduler 层，决定"这个序列可以跳过多少 prefill token"。(2) 真实 vLLM 使用 LRU 淘汰策略管理缓存容量。(3) 真实 vLLM 的 PagedAttention kernel 直接支持 block 级 KV 读取，因此共享 block 的 KV 数据可以被直接用于注意力计算——不需要像 FakeModelExecutor 那样"跳过写入"。(4) 真实 vLLM 的 hash 更健壮（包含 seq ID、token ID、lora ID 等信息）。
