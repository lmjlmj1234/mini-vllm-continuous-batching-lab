# Architecture Review: mini-vLLM

> **用途**：面试复习 / 架构回顾
> **设计原则**：Architecture = Who owns what responsibility
> **与真实 vLLM 的关系**：核心概念一致，实现细节简化

---

## 1. 整体架构图

```
 ┌─────────────────────────────────────────────────────────┐
 │                    LLMEngine                             │
 │  add_request → RequestQueue (waiting/running/finished)   │
 └────────────────────────┬────────────────────────────────┘
                          │ step()
                          ▼
 ┌─────────────────────────────────────────────────────────┐
 │                  EngineCore.step()                       │
 │                                                          │
 │  ┌──────────────────────────────────────────────────┐   │
 │  │               Scheduler.schedule()                 │   │
 │  │                                                    │   │
 │  │  Phase 1: Finish  ─── check running groups         │   │
 │  │  Phase 2: Categorize ── decode vs prefill-continue │   │
 │  │  Phase 3: Decode-first budget                      │   │
 │  │  Phase 4: Chunked-prefill continue                 │   │
 │  │  Phase 5: Admit new ── probe cache → budget → alloc│   │
 │  │  Phase 6: Token counts + debug_reason              │   │
 │  └──────────────────────┬───────────────────────────┘   │
 │                         │ ScheduleResult                │
 │  ┌──────────────────────▼───────────────────────────┐   │
 │  │              Executor                               │   │
 │  │  prefill(seqs)  ─── write KV from prefill_cursor   │   │
 │  │  decode(seqs)   ─── read KV → produce next token   │   │
 │  │  cleanup(seqs)  ─── release executor state          │   │
 │  └──────────────────────┬───────────────────────────┘   │
 │                         │                                │
 │  ┌──────────────────────▼───────────────────────────┐   │
 │  │           BlockManager (on-demand)                  │   │
 │  │  probe_prefix_cache() ── read-only hash lookup     │   │
 │  │  allocate_for_seq() ── attach shared blocks + ref++ │   │
 │  │  ensure_block() ── allocate/physical or use cache  │   │
 │  │  free() ── decrement ref, release at ref==0        │   │
 │  └──────────────────────┬───────────────────────────┘   │
 │                         │                                │
 │  ┌──────────────────────▼───────────────────────────┐   │
 │  │          PrefixCache (hash → physical_block_id)     │   │
 │  │  lookup(hash) ── find cached block                 │   │
 │  │  insert(hash, pid) ── register new block           │   │
 │  └──────────────────────────────────────────────────┘   │
 │                                                          │
 └─────────────────────────────────────────────────────────┘
                          │
                          ▼
                    MetricsCollector.report()
                    TTFT / TPOT / Throughput / Prefix Cache hit rate
```

### Responsibilities

| 组件 | 职责 | 一句话 |
|------|------|--------|
| **LLMEngine** | 用户接口 + 组件组装 | 接收请求，驱动循环，输出结果 |
| **EngineCore** | Step 编排 | schedule → prefill → decode → cleanup → metrics |
| **Scheduler** | 决策层：谁在什么时候做什么 | 基于 token budget + decode first + chunked prefill 做 admission 和优先级 |
| **ScheduleResult** | Scheduler 的输出契约 | 告诉 EngineCore 哪些 group 做 prefill、哪些 decode、缓存命中统计 |
| **Executor** | 执行层：实际做 prefill/decode | 管理 KV cache slab，调用 BlockManager 做 on-demand allocation |
| **BlockManager** | KV cache 逻辑管理层 | 管理 BlockTable、Prefix Cache 集成、ref count 生命周期 |
| **BlockAllocator** | KV cache 物理层 | 分配/释放物理 block，跟踪 ref count |
| **PrefixCache** | Hash 索引 (hash → physical_block_id) | 提供 block 级别复用，减少重复 prefill 计算 |
| **RequestQueue** | 四池队列 | waiting / running / finished / rejected |

**架构核心**：不是数据流是职责边界。Scheduler 不知道模型细节，Executor 不知道调度策略，BlockManager 把物理分配和逻辑 cache 分开。

---

## 2. Runtime

### Engine Step

每个 `step()` 是一个完整的 schedule + execute 循环：

```
EngineCore.step()
  ├── Scheduler.schedule()        → 决策（纯逻辑，零模型调用）
  ├── Executor.prefill(seqs)      → 写新 KV，产生 first token
  ├── Executor.decode(seqs)       → 读 KV，产下一个 token
  ├── Executor.cleanup(seqs)      → 释放 executor 资源
  └── MetricsCollector.record()   → 收集 step 指标
```

### Continuous Batching

**定义**：每一步都重新调度所有 seqs（running + waiting），不是等一个 batch 全部做完再处理下一个。

vs Static Batch：

| Static Batch | Continuous Batching |
|---|---|
| 收集 N 个请求 → 一起前向 → 等所有生成完 → 释放 | 每步调度：decode 的继续 decode，新请求的做 prefill |
| GPU 利用率锯齿状（满载 / 空闲交替） | GPU 利用率平滑 |
| 延迟 = batch 中最后一个完成的耗时 | 延迟 = 各自独立的 pipeline 时间 |
| 快速请求等慢速请求 | 解码优先保证交互性 |

**为什么 Continuous Batching 更好？**：
1. GPU 利用率不浪费——没有"等所有请求完成"的空闲期
2. 延迟更低——decode 优先确保已开始的请求快速产出 token
3. 吞吐更高——每一步都在做有用计算（prefill 或者 decode），不会因为 batch 中的 straggler 阻塞

### Decode First

**为什么 Engine Step 不是固定时间而是事件驱动？**：

因为在 Continuous Batching 中，step 的耗时由**当前 step 的负载**决定：
- 某个 step 可能只有 decode（轻量，快）
- 某个 step 可能有大量 prefill（重量，慢）
- 固定时间 = 浪费 GPU 或 来不及处理

事件驱动 = "上一步做完立即开始下一步"，没有固定间隙。

### Chunked Prefill

一个长 prompt 不一次做完，而是分成多个 chunk 分布在多个 step 中：

```
Step 0: chunk 0-3  (prefill 4 tokens) → 还有 12 tokens 没处理
Step 1: chunk 4-7  (prefill another 4)
Step 2: chunk 8-11
Step 3: chunk 12-15 → prefill 完成，产 first token
Step 4: decode → decode → decode ...
```

Preemptible： chunk 不会锁死 GPU 太久，decode 序列不会被饿死。

---

## 3. Scheduler

### Request Queue

```
waiting ──→ running ──→ finished
    │                     ↑
    └──→ rejected ───────┘
```

- **waiting**：新到的请求，还没被调度器受理
- **running**：已受理，在做 prefill 或 decode
- **finished**：所有 seqs 已完成
- **rejected**：prompt 太长，被拒绝

### Scheduler Phases

```
Phase 1: Finish  ─── 检查 running groups 谁生成了足够 token
Phase 2: Categorize ─ 分成 decode / prefill-continue 两组
Phase 3: Decode-first budget ─ decode 先扣预算
Phase 4: Chunked-prefill continue ─ 继续未完成的 prefill
Phase 5: Admit new ── 新请求：probe cache → 算 uncached → check budget → 分配
Phase 6: Token counts + debug_reason
```

### Key Concepts

**Token Budget**：

```
max_num_batched_tokens = 16    # 一步最多处理 16 个 token
max_num_prefill_tokens = 16    # prefill 单独上限
max_num_seqs = 4               # 一步最多 4 个 seq

剩余预算计算：
  1. decode 先扣（每个 decode seq 扣 1 token）
  2. prefill-continue 扣
  3. 新请求 admission 用剩下的

实际（prefix cache）：Scheduler 用 uncached_tokens 而不是 prompt_len 算预算
  uncached = prompt_len - probe.cached_token_count
  this_chunk = min(uncached, chunk_size)
```

**Decode Priority**：为什么 decode 优先？因为 decode 是串行的——每个 decode seq 每步必须产一个 token，否则用户的交互延迟会直接累积。Prefill 可以等，decode 不能等。

### 设计问题

**为什么 Decode 优先？**
1. **延迟敏感**：decode 直接影响用户体验（TPOT），prefill 只影响 TTFT
2. **不可暂停**：decode seq 如果这步不做，用户感知到的是 "卡住了"；prefill 延迟一个 step 是无感的
3. **轻量**：decode 每个 seq 只消耗 1 个 token 预算，优先扣了不影响 prefill 太多

**为什么 Token Budget 必须存在？**
1. **防止 OOM**：一次性 prefill 太多 token 会耗尽 KV cache block
2. **公平性**：防止一个长 prompt 占满整个 step，饿死其他 seq
3. **可预测性**：每步处理不超过 `max_num_batched_tokens` 个 token，step latency 可预测

**为什么 Scheduler 不直接执行模型？**
1. **关注点分离**：Scheduler 做策略（谁做什么），Executor 做执行（模型前向）
2. **可测试性**：调度逻辑可以单独测试，不需要加载模型
3. **可插拔**：FakeExecutor（测试用）和 QwenExecutor（真实模型）共用同一套调度器

---

## 4. Memory Manager

### KV Cache: 为什么不用连续内存？

传统 Transformer inference：

```
PagedAttention (本实现)：
  ┌────┬────┬────┬────┐        ┌────┬────┬────┬────┐
  │B0  │ B1 │    │    │        │ B0 │    │ B2 │ B3 │
  └────┴────┴────┴────┘        └────┴────┴────┴────┘
  seq-A 逻辑视图             物理 block pool（碎片化 OK）
```

连续 KV Cache（传统方案）：

```
  ┌────┬────┬────┬────┐
  │ A0 │ A1 │ A2 │ A3 │  ← 必须连续分配
  └────┴────┴────┴────┘
  分配一次就不能变 → 如果预留过多 → 浪费
                     如果预留过少 → OOM
```

**PagedAttention** 借鉴操作系统分页管理：
- 逻辑上连续（BlockTable 给 seq 的视角是连续的 logical block）
- 物理上离散（实际 block 可以不连续存放）
- 按需分配（不需要预留未来空间）

### Block

| 概念 | 说明 |
|------|------|
| **Physical Block** | BlockAllocator 管理的一块固定大小存储（block_size 个 token 的 KV） |
| **Logical Block** | Sequence 视角的连续 block 编号（0, 1, 2, ...） |
| **BlockTable** | 逻辑 block → 物理 block 的映射表 |
| **BlockSize** | 每个 block 容纳的 token 数（本实现：4） |

### BlockAllocator

```
BlockAllocator:
  _free:      [True, True, False, True, False, ...]   # free list
  _ref_counts:[0,    0,    1,     0,    1,    ...]     # ref count

  allocate(N)  → 找 N 个 free block，设 ref_count=1
  free(pids)   → 每个 pid 的 ref_count -= 1，==0 时归还 free pool
  increment_ref(pid) → ref_count += 1（prefix cache 共享用）
```

### BlockTable

```
seq-A 的 BlockTable:
  [L0→P0, L1→P3, L2→P5]     # 逻辑连续，物理离散

seq-B 的 BlockTable（共享前缀）:
  [L0→P0(shared), L1→P3(shared), L2→P7]  # 前两个 block 与 A 共享
```

### On-demand Allocation

**为什么需要？**

传统方案：create_sequence 时 allocate(prompt_len / block_size) 个 block，未来再补。

问题：prompt 生成的长度不确定——你无法提前知道 decode 阶段需要多少 block。

On-demand：`ensure_block(seq, position)` 在**写入 KV 时**分配 block：

```
prefill cursor 推进：
  pos 0  → ensure_block(seq, 0)  → 没有 block → allocate → 返回 pid
  pos 1  → ensure_block(seq, 1)  → 还在 block 0 范围内 → 直接返回已有 pid
  pos 4  → ensure_block(seq, 4)  → 跨 block 边界 → allocate
  ...

decode 阶段：
  pos 20 → ensure_block(seq, 20) → 需要新 block → allocate
  
最终：分配的 block 数量 = 实际需要的数量，0 浪费
```

---

## 5. Prefix Cache

### 核心机制

```
prompt tokens: [A, B, C, D, E, F, G, H]  block_size=4
                      ↓ compute_block_hashes
hashes:        [h0(A,B,C,D), h1(E,F,G,H)]
                      ↓
PrefixCache:
  lookup(h0) → physical_block_id?  → 命中 → 共享（ref_count++）
  lookup(h1) → physical_block_id?  → 命中 → 共享
                                → 未命中 → 正常分配 → insert(h1, new_pid)
```

### 共享过程

```
Request A (第一个到达):
  [A, B, C, D, E, F, G, H]
  → allocate blocks P0, P1
  → insert(h0→P0), insert(h1→P1)
  → KV: P0=[A,B,C,D]  P1=[E,F,G,H]

Request B (第二个到达，相同 prefix [A,B,C,D]):
  [A, B, C, D, X, Y, Z, ...]
  → probe: h0→P0(ref=1>0) → 匹配, h1 ≠ hash(X,Y,Z,?) → 只从 L0 开始匹配
  → allocate_for_seq: P0(ref=1→2, shared), 然后 新 block P2 for [X,Y,Z,...]
  → prefill cursor = 4（跳过已缓存的前 4 个 token）
  → KV write 只写 pos 4+（写入 P2）
```

### RefCount

**为什么需要 RefCount？**

```
Request B 共享了 P0（与 A 共同持有）。当 A 结束时：
  free(A) → P0.ref_count: 2→1  → P0 仍然存活（B 还在用）
  
当 B 结束时：
  free(B) → P0.ref_count: 1→0  → P0 归还 free pool

如果没有 ref_count：
  A 结束时 P0 被释放 → B 还在用 → use-after-free
```

### Stale Entry

```
Request A 使用 P0, P1
Request B 共享 P0

A 结束 → free(A) → P0.ref=2→1 (仍然存活), P1.ref=1→0 (释放)
                    PrefixCache 中仍有 h1→P1 的条目（stale）
                    
Request C 进来：
  probe: h0→P0(ref=1>0) → 匹配
         h1→P1(ref=0) → stale → 不匹配 → 截断

// 核心：probe 检查 ref_count > 0，stale 条目不会计入 cached_token_count
```

### 为什么 Prefix Cache 不直接复制 KV？

复制 KV = 内存翻倍 + 计算翻倍 = 失去了 cache 的意义。

共享 block 的核心优势：**零复制**。第二份请求不需要为共享部分分配内存、不需要写 KV、不需要做注意力计算。只需要在 BlockAllocator 中 increment_ref。

### 为什么需要 Shared Block？

```
没有 Shared Block（完整复制）：
  Request B 需要 P0' = copy(P0), P1' = copy(P1)  → 2 倍内存 + 2 倍写入

有 Shared Block：
  Request B 直接 read P0, P1（ref 增加了 1）    → 0 额外内存 + 0 额外写入
```

当两个 seq 共享 P0 时，Executor 的 `_write_to_kv()` 直接跳过：

```python
def _write_to_kv(self, seq, token_position, token_id):
    if self._block_manager.is_block_shared(seq, token_position):
        return  # 数据已存在，跳过写入
    # 否则正常分配 + 写入
```

---

## 6. Scheduler-aware Prefix Cache

### 为什么 Scheduler 需要感知？

第一版 Prefix Cache 是 BlockManager 内部的透明优化：

```
旧方案（Scheduler 意识不到 cache）：
  Scheduler 看 prompt_len = 16 → 扣 16 个 token 预算
  但实际有 8 个 token 已缓存，只需要算 8 个
  → 浪费了 8 个 token 的预算空间 → 本来可以多 admit 一个请求
```

新方案：Scheduler **先查缓存再算预算**。

### 命中后 Token Budget 变化

```
有 Prefix Cache：
  prompt_len=16, cached_token_count=8
  uncached_tokens = 16 - 8 = 8
  this_chunk = min(8, chunk_size=4) = 4
  → 只消耗 4 个 token 预算

无 Prefix Cache：
  prompt_len=16, cached_token_count=0
  uncached_tokens = 16
  this_chunk = min(16, 4) = 4（但要连续 4 个 step 才能做完）
  → 每步消耗 4 个 token 预算
```

### 命中后 Admission 变化

```
有 Cache：
  16 tokens 请求，8 已缓存 → 只用 8 个 uncached budget
  如果 max_prefill_budget=10 → 可以 admit（8 <= 10）
  
无 Cache：
  16 tokens 请求 → 需要 16 个 budget
  如果 max_prefill_budget=10 → 不能 admit（16 > 10）
  
→ 同样的硬件配置，有 cache 可以 admit 更多请求
→ 被拒绝的长 prompt 因为 cache 命中也可以被接受
```

### 命中后 Chunked Prefill 变化

```
有 Cache：
  prompt_len=16, cached_token_count=12
  prefill_cursor = 12（跳过前 12 个 token）
  → 1 个 step 做完剩下 4 个 token
  → TTFT 大幅缩短

无 Cache：
  prompt_len=16, cached_token_count=0
  prefill_cursor = 0（从开头开始）
  → chunk_size=4 → 需要 4 个 step 才能做完
  → TTFT 更长
```

### 数据流

```
Scheduler Phase 5 (Admit):
  1. probe = BlockManager.probe_prefix_cache(prompt_tokens)
     → 只读，不修改 ref_count
  
  2. uncached = len(prompt_tokens) - probe.cached_token_count
  
  3. budget check: this_chunk <= prefill_budget ?
     → YES → continue
     → NO  → ignore/reject
  
  4. allocate_for_seq(seq) → 真正 attach 共享 block，increment_ref
  
  5. seq.prefill_cursor = probe.cached_token_count
     → Executor 的 prefill loop 从 cursor 开始，跳过缓存部分
```

---

## 7. 项目与真实 vLLM 对比

### Already Implemented

| 特性 | mini-vLLM | 真实 vLLM | 一致？ |
|------|-----------|-----------|--------|
| **Continuous Batching** | EngineCore.step() 循环 | LLMEngine.step() 循环 | ✓ 概念一致 |
| **Scheduler** | 6-phase 调度 | Scheduler.schedule() | ✓ 逻辑一致 |
| **Decode First** | decode 先扣 budget | same | ✓ |
| **Chunked Prefill** | max_prefill_chunk_size | max_num_batched_tokens | ✓ |
| **Token Budget** | max_num_batched_tokens | same | ✓ |
| **BlockManager** | BlockTable + BlockAllocator | BlockSpaceManager | ✓ |
| **BlockAllocator** | ref_count, free list, allocate/free | same | ✓ |
| **Prefix Cache** | Hash → pid, block-level sharing | BlockPrefixMgr | ✓ 但 ours 无 LRU |
| **probe_prefix_cache** | Scheduler-aware probe | _get_cached_prefix_len | ✓ |
| **PagedAttention 核心** | 逻辑 block → 物理 block 映射 | same | ✓ |
| **On-demand Allocation** | ensure_block at write time | same | ✓ |
| **Metrics** | TTFT, TPOT, Throughput | same | ✓ |
| **Qwen Worker** | HuggingFace Transformers Qwen2 | vLLM 支持 Qwen2 | ✓ 简化版 |

### NOT Implemented（不实现的原因）

| 特性 | 为什么省略 |
|------|-----------|
| **Multi-GPU / Tensor Parallel** | 教育模型只需理解单卡架构；多卡是工程优化，不改变设计 |
| **Distributed Serving** | 无分布式需求（只跑本地 single node） |
| **PagedAttention CUDA Kernel** | 用 Python dict + list 模拟 KV cache 行为（理解架构不需要 CUDA） |
| **CUDA Graph** | 减少 kernel launch overhead 的优化——理解架构不需要 |
| **Speculative Decoding** | 与基础架构正交——先理解 continuous batching 和 paged attention |
| **LRU Cache Eviction** | PrefixCache 不做 eviction（简化），真实 vLLM 有容量上限和 LRU |
| **Copy-on-Write 实现** | 只做了 COW 扩展点（is_shared flag），没有真正实现写时复制 |
| **Automatic Prefix Detection** | 真实 vLLM 自动检测共享前缀长度，本项目手动控制测试 |

---

## 8. 5 分钟快速复习版

```
LLM Inference Serving ≠ Training
  
  核心问题：LLM 生成是自回归的，传统 "等一个 batch 全部做完" 浪费 GPU
  
  答案：Continuous Batching

Runtime:
  Engine Step = 1 次 schedule → 1 次 execute（事件驱动，非固定 tick）
  
Scheduler（每步 6 个 phase）:
  ┌─ 1. Finish    ── 检查谁生成了够多 token
  ├─ 2. Categorize ── 分 decode / prefill
  ├─ 3. Decode-first ── decode 先扣 token budget（为什么？延迟敏感！）
  ├─ 4. Chunked-prefill ── 继续没做完的 prefill
  ├─ 5. Admit     ── probe cache → uncached_tokens → budget check → alloc
  └─ 6. Count     ── 记录 token 统计
  
  三个为什么：
    • Decode 优先？因为 decode 延迟影响用户体验，prefill 可以等
    • Token Budget 存在？防止 OOM + 保证公平性
    • Scheduler 不执行模型？关注点分离 + 可测试性

Memory（PagedAttention + BlockTable）:
  为什么不用连续 KV Cache？碎片化 OK，不需要预留，按需分配
  Logical block 0 → Physical block 7  （逻辑连续，物理离散）
  ensure_block(write_time) = 0 浪费

Prefix Cache（两阶段）:
  Probe: 只读查缓存，不碰 ref_count
  Allocate: 共享 block，increment_ref
  
  为什么 RefCount？防止 use-after-free（A 释放时 B 还在共享）
  为什么 Scheduler 必须感知？token budget / admission / chunked prefill 都需要正确的 uncached_tokens

Metrics（面试重点）:
  TTFT  = time to first token（prefill 延迟）
  TPOT  = time per output token（decode 延迟）
  Throughput = req/s + tok/s
  Prefix cache hit rate = cached_tokens / total_prompt_tokens

与真实 vLLM 对应:
  所有核心概念一致（scheduler, budget, block, ref_count, prefix cache）
  省略的是工程优化（CUDA, distributed, speculative decoding），不是架构差异
```

---

> **一句话总结**: mini-vLLM 是真实 vLLM 的架构骨架——Continuous Batching + PagedAttention + Scheduler-aware Prefix Cache，核心概念完全一致，只是砍掉了 CUDA 和分布式等工程复杂度。
