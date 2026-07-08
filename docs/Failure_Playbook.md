# Failure Playbook — mini-vLLM Serving Runtime
# 故障排查手册 — mini-vLLM 服务运行时

> Runbook entries for the most likely production failure scenarios
> in a continuous-batching LLM serving system.

> 记录在连续批处理 LLM 服务系统中最有可能出现的生产故障场景及对应的运行手册。

---

## Scenario 1: KV Cache Block Exhaustion
## 场景 1：KV Cache 块耗尽

**What Happened:**
The server ran out of free KV cache blocks. Every new request was rejected with `BLOCK_EXHAUSTED`. Existing running requests continued decode but the system entered a "deny-all" state.

**现象：**
服务器的可用 KV cache 块耗尽。每个新请求都被拒绝，返回 `BLOCK_EXHAUSTED`。正在运行的 decode 请求不受影响继续处理，但系统进入了"全拒绝"状态。

**Why:**
The sum of `(prompt_len + max_tokens) / block_size` across all active requests exceeded the total block pool. In this implementation, the Admission Control check uses `free - needed_blocks < watermark_blocks` (default 20% watermark). The trigger was a burst of long-context requests (4k prompt + 2k generation each) that exhausted the 1024-block pool.

**根因：**
所有活跃请求的 `(prompt_len + max_tokens) / block_size` 总和超过了总块池大小。在该实现中，准入控制检查使用 `free - needed_blocks < watermark_blocks`（默认 20% 水位线）。触发条件是一波长上下文请求（每个 4k prompt + 2k 生成）耗尽了 1024 块的池。

```
block_size = 16
total_blocks = 1024
watermark = 1024 * 0.2 = 204 blocks
4 requests: each needs ceil((4096+2048)/16) = 384 blocks
384 * 4 = 1536 blocks needed, only 1024 available
→ free - needed = 1024 - 1536 = -512 < 204 → BLOCK_EXHAUSTED
```

**Impact:**
- All new requests rejected for ~45 seconds until running requests completed
- P50 latency for accepted requests increased 3x (smaller effective batch)
- Downstream clients saw HTTP 429/503 responses

**影响：**
- 所有新请求被拒绝约 45 秒，直到正在运行的请求完成
- 已接受请求的 P50 延迟增加 3 倍（有效 batch 更小）
- 下游客户端收到 HTTP 429/503 响应

**Detection:**
- Metric `block_utilization` > 90% (this system's `/metrics` endpoint)
- Error log: `admission.check() → BLOCK_EXHAUSTED`
- Grafana: KV block timeline shows flat-top at 100%

**检测：**
- 指标 `block_utilization` > 90%（本系统的 `/metrics` 端点）
- 错误日志：`admission.check() → BLOCK_EXHAUSTED`
- Grafana：KV 块时间线显示在 100% 处平顶

**Mitigation:**
- Reduce `max_num_seqs` to limit concurrent block consumers
- Increase `num_gpu_blocks` if GPU memory allows
- Set shorter `max_tokens` defaults in the serving config
- Enable request queuing with bounded queue depth

**缓解措施：**
- 降低 `max_num_seqs` 以限制并发块消费者数量
- 如果 GPU 内存允许，增加 `num_gpu_blocks`
- 在服务配置中设置更短的 `max_tokens` 默认值
- 启用有界队列深度的请求排队

**Recovery:**
- No action needed — blocks return to pool as requests finish
- For immediate relief: cancel low-priority running requests via `cancel()`
- Post-mortem: right-size block pool based on peak usage

**恢复：**
- 无需操作——请求完成时块自动返回池中
- 需要立即缓解时：通过 `cancel()` 取消低优先级正在运行的请求
- 事后分析：根据峰值使用量调整块池大小

---

## Scenario 2: Cancel Storm (Mass Abandonment)
## 场景 2：取消风暴（大规模断连）

**What Happened:**
A downstream client disconnected, triggering 50 concurrent `cancel()` calls within 100ms. The BlockManager's `free()` walked 384 block tables, decrementing ref_counts under the GIL. Engine step latency spiked.

**现象：**
下游客户端断连，在 100ms 内触发了 50 个并发的 `cancel()` 调用。BlockManager 的 `free()` 遍历了 384 个块表，在 GIL 下递减引用计数。引擎 step 延迟飙升。

**Why:**
Each `cancel_request()` calls `BlockManager.free(seq_id)` which iterates the sequence's block table and decrements `ref_count` on each block. With 50 requests owning ~8 blocks each:

**根因：**
每个 `cancel_request()` 调用 `BlockManager.free(seq_id)`，后者遍历序列的块表并递减每个块的 `ref_count`。50 个请求每个拥有约 8 个块：

```
50 cancels × 8 blocks = 400 free() operations
Each = O(block_table_length) hash lookup + ref_count decrement
Millions of Python dict ops under GIL → 200ms step latency
```

**Impact:**
- Engine step latency: 15ms → 220ms during cancel storm
- All other requests in that batch saw increased TPOT
- Metrics: `cancelled_requests` spiked, throughput dropped 40%

**影响：**
- 引擎 step 延迟：取消风暴期间从 15ms 变为 220ms
- 该 batch 中所有其他请求的 TPOT 增加
- 指标：`cancelled_requests` 飙升，吞吐量下降 40%

**Detection:**
- Metrics spike in `cancelled_requests` counter
- Step latency trace shows outlier
- `ref_count` dump in `dump_ref_counts()` shows rapid decrement pattern

**检测：**
- `cancelled_requests` 计数器出现指标尖峰
- Step 延迟跟踪显示异常值
- `dump_ref_counts()` 中的 `ref_count` 转储显示快速递减模式

**Mitigation:**
- Batch cancel operations: collect IDs, free in a single pass
- Throttle inbound cancel requests at the serving layer
- Use a cancel queue (async drain) instead of synchronous free

**缓解措施：**
- 批量取消操作：收集 ID，一次性释放
- 在服务层限流入的取消请求
- 使用取消队列（异步排空）代替同步释放

**Recovery:**
- System self-recovers: freed blocks go back to the pool
- Monitor ref_count integrity post-storm (all should be 0)
- Consider adding a cancel rate-limiter

**恢复：**
- 系统自恢复：释放的块回到池中
- 风暴后监控 ref_count 完整性（都应为 0）
- 考虑添加取消操作的速率限制器

---

## Scenario 3: Queue Overflow Cascading to All-Up Rejection
## 场景 3：队列溢出导致全量拒绝

**What Happened:**
A traffic spike filled the waiting queue to `max_queue_len=32`. Every subsequent request hit `QUEUE_OVERFLOW`. The scheduler spent 30% of its budget iterating the full waiting list each step.

**现象：**
流量尖峰将等待队列填满至 `max_queue_len=32`。之后的每个请求都命中 `QUEUE_OVERFLOW`。调度器每步花费 30% 的预算遍历完整的等待列表。

**Why:**
The Admission Control check `current_waiting() >= max_queue_len` is a hard gate — it sits BEFORE the scheduler. Once the queue is full, no request enters. But the scheduler still calls `schedule()` every step, which iterates the waiting list (still 32 entries) even though none can be admitted.

**根因：**
准入控制检查 `current_waiting() >= max_queue_len` 是一个硬性门控——它位于调度器之前。一旦队列满了，没有请求能进入。但调度器仍然每步调用 `schedule()`，它会遍历等待列表（仍有 32 个条目），即使没有一个能被准入。

```
Step N:   _check_timeouts() → 32 waiting, 0 expired
          schedule() → iterates 32 waiting, admits 0, returns empty
          → wasted 100% of schedule() work
```

**Impact:**
- All new requests rejected
- CPU burn on scheduler iteration (small in isolation, adds up)
- 32 queued requests eventually timeout if not served

**影响：**
- 所有新请求被拒绝
- 调度器迭代消耗 CPU（单独看很小，但累加起来）
- 32 个排队请求如果未被服务最终会超时

**Detection:**
- Metric `waiting_requests` == `max_queue_len`
- Error log: `admission.check() → QUEUE_OVERFLOW`
- Client receives error code `QUEUE_OVERFLOW`

**检测：**
- 指标 `waiting_requests` == `max_queue_len`
- 错误日志：`admission.check() → QUEUE_OVERFLOW`
- 客户端收到错误码 `QUEUE_OVERFLOW`

**Mitigation:**
- Right-size `max_queue_len` for expected concurrency
- Add timeout to queued requests so they don't pile up forever
- Consider load shedding at a higher layer (API gateway → admission control)

**缓解措施：**
- 根据预期并发量合理设置 `max_queue_len`
- 为队列中的请求添加超时，防止无限堆积
- 考虑在更高层做负载削减（API 网关 → 准入控制）

**Recovery:**
- Requests complete → queue drains → new traffic accepted
- For immediate relief: clear the queue or increase `max_num_seqs`

**恢复：**
- 请求完成 → 队列排空 → 新流量被接受
- 需要立即缓解时：清空队列或增加 `max_num_seqs`

---

## Scenario 4: Timeout Thundering Herd
## 场景 4：超时惊群

**What Happened:**
Requests with `max_tokens=4096` queued up during a GPU scheduling delay. 16 requests all timed out simultaneously. `_check_timeouts()` fired, cancelled all 16, triggering the same BlockManager churn as a Cancel Storm.

**现象：**
GPU 调度延迟期间，`max_tokens=4096` 的请求排队。16 个请求同时超时。`_check_timeouts()` 触发，取消了全部 16 个请求，引发了与取消风暴相同的 BlockManager 抖动。

**Why:**
All 16 requests were added with the same `arrival_time` (within 5ms). When `request_timeout_s=30.0` expired, `_check_timeouts()` found all 16 in one pass:

**根因：**
所有 16 个请求的 `arrival_time` 相同（5ms 以内）。当 `request_timeout_s=30.0` 到期时，`_check_timeouts()` 一次就找到了全部 16 个：

```
now - arrival_time > 30.0 → True for all 16
16 × ~12 blocks each = 192 free() operations in one step
```

Additionally, each timed-out request had been partially decoded (consuming GPU compute for tokens nobody received).

此外，每个超时的请求已经完成了部分解码（消耗了 GPU 计算资源生成了无人接收的 token）。

**Impact:**
- Wasted GPU compute: ~2000 generated tokens never delivered
- Step latency spike during mass free
- Clients all received errors simultaneously, potentially triggering client-side retry storms

**影响：**
- 浪费 GPU 计算：约 2000 个生成的 token 从未送达
- 批量释放期间 step 延迟尖峰
- 客户端同时收到错误，可能触发客户端重试风暴

**Detection:**
- Metric `timeout_requests` jumps by N in one step
- `num_running` drops to 0 instantly
- Log: `_check_timeouts() cancelled N requests`

**检测：**
- 指标 `timeout_requests` 在一步内跳升 N
- `num_running` 瞬间降至 0
- 日志：`_check_timeouts() cancelled N requests`

**Mitigation:**
- Stagger timeouts: add jitter to `request_timeout_s` per request
- Reduce `max_queue_len` to limit timeout blast radius
- Add backoff to client reconnect logic

**缓解措施：**
- 错开超时：为每个请求的 `request_timeout_s` 添加抖动
- 降低 `max_queue_len` 以限制超时爆炸半径
- 为客户端重连逻辑添加退避

**Recovery:**
- Blocks freed, queue cleared, system ready for new requests
- Post-mortem: investigate root cause of delay (GPU memory pressure, slow prefill, etc.)

**恢复：**
- 块已释放、队列已清空、系统可接受新请求
- 事后分析：调查延迟的根因（GPU 内存压力、prefill 慢等）

---

## Scenario 5: Rate Limiter Window Boundary Stampede
## 场景 5：速率限制窗口边界踩踏

**What Happened:**
At the start of each RPM window, traffic surged to exactly the RPM limit, then dropped to zero for the rest of the window. The pattern repeated every 60 seconds — a classic "thundering herd" hitting the reset boundary.

**现象：**
在每个 RPM 窗口的起始时刻，流量恰好激增至 RPM 限制值，然后在窗口的剩余时间内降至零。此模式每 60 秒重复一次——经典的"惊群"冲击重置边界。

**Why:**
The RateLimiter uses a sliding window counter. When the window turns over:

**根因：**
RateLimiter 使用滑动窗口计数器。当窗口翻转时：

```
Window 1 (60s): 60 requests processed → window fills
Window 2 (60s): reset to 0 → next 60 requests arrive instantly
             → 120 requests in 61 seconds → last 60 still blocked
```

The synchronised reset creates a sawtooth pattern: full capacity → zero → full capacity.

同步重置产生了锯齿模式：满容量 → 零 → 满容量。

**Impact:**
- Uneven load: server is idle for 55 seconds, saturated for 5 seconds
- Queue depth spikes during the active window
- Long-tail latency for requests that arrive just after the burst

**影响：**
- 负载不均：服务器空闲 55 秒，饱和 5 秒
- 活跃窗口期间队列深度飙升
- 在突发之后到达的请求出现长尾延迟

**Detection:**
- Metrics: `rpm_rejected` spikes every 60s
- Request latency shows periodic sawtooth pattern

**检测：**
- 指标：`rpm_rejected` 每 60 秒尖峰一次
- 请求延迟呈现周期性锯齿模式

**Mitigation:**
- Add rate limit smoothing: token bucket instead of sliding window
- Add jitter to client request timing
- Use a leaky-bucket at the client side

**缓解措施：**
- 添加速率限制平滑：使用令牌桶代替滑动窗口
- 为客户端请求时序添加抖动
- 在客户端使用漏桶算法

**Recovery:**
- No action needed — behaviour is self-sustaining but not damaging
- Reconfigure window size or switch to token bucket algorithm

**恢复：**
- 无需操作——行为自维持但不具破坏性
- 重新配置窗口大小或切换到令牌桶算法

---

## Scenario 6: Prefix Cache Stale Entry Referencing Freed Blocks
## 场景 6：前缀缓存过期条目引用已释放块

**What Happened:**
Request A completed, freeing all its blocks (`ref_count=0`). Request B with the same prefix arrived. The Prefix Cache probe found a hash match (cached_token_count > 0) but the physical block had been reused by another request. The scheduler's budget calculation under-counted prefill tokens, resulting in incorrect token counts.

**现象：**
请求 A 完成，释放了所有块（`ref_count=0`）。具有相同前缀的请求 B 到达。前缀缓存探测发现哈希匹配（cached_token_count > 0），但该物理块已被另一个请求重用。调度器的预算计算低估了 prefill token 数，导致 token 计数错误。

**Why:**
The two-phase prefix cache design (probe → allocate) handles this correctly in theory, but a subtle bug in the scheduler's budget calculation assumed `cached_token_count` means "we will reuse these blocks." In the stale-entry case, the blocks are gone (ref_count=0), so the probe correctly returns `cached_token_count=0`. But if the scheduler used a stale probe result, it would miscalculate:

**根因：**
两阶段前缀缓存设计（probe → allocate）理论上正确处理了这种情况，但调度器预算计算中的一个细微错误假设了 `cached_token_count` 意味着"我们将重用这些块"。在过期条目的情况下，块已经不存在（ref_count=0），所以探测正确地返回了 `cached_token_count=0`。但如果调度器使用了过期的探测结果，就会计算错误：

```
Step N:   probe = prefix_cache.probe("AAAA") → cached_token_count=4 (stale!)
Step N:   scheduler budget: prefill_tokens = 4 - 4 = 0  ← WRONG
          → no prefill scheduled, decode with empty KV → crash
```

**Impact:**
- Output garbage or engine crash
- Hard to debug: appears as random incorrect output under memory pressure

**影响：**
- 输出垃圾内容或引擎崩溃
- 难以调试：在内存压力下表现为随机的错误输出

**Detection:**
- Monitor `prefix_cache_hit_rate`: sudden drop indicates eviction
- `dump_ref_counts()` shows expected vs actual counts
- Regression test `test_stale_entry_not_falsely_used` catches this

**检测：**
- 监控 `prefix_cache_hit_rate`：突然下降表示驱逐发生
- `dump_ref_counts()` 显示预期与实际计数
- 回归测试 `test_stale_entry_not_falsely_used` 能捕获此问题

**Mitigation:**
- Two-phase probe design is the mitigation: probe returns `(cached_token_count, is_stale)` and the allocator checks ref_count before reusing
- Never use cached_token_count > 0 without verifying ref_count > 0

**缓解措施：**
- 两阶段探测设计本身就是缓解措施：探测返回 `(cached_token_count, is_stale)`，分配器在重用前检查 ref_count
- 在未验证 ref_count > 0 之前，永远不要使用 cached_token_count > 0

**Recovery:**
- Clear the prefix cache: `prefix_cache._cache.clear()`
- All subsequent requests will cold-start (slower but correct)
- The stale entry is naturally evicted on next allocation

**恢复：**
- 清空前缀缓存：`prefix_cache._cache.clear()`
- 后续所有请求将冷启动（更慢但正确）
- 过期条目会在下次分配时自然被淘汰

---

## Scenario 7: Stream Connection Manager Exhaustion
## 场景 7：流连接管理器耗尽

**What Happened:**
A client opened 16 streaming connections, exhausting `max_num_streams=16`. The 17th streaming request was rejected with `TOO_MANY_STREAMS`. The 16 active streams were idle — clients had disconnected without calling close, but the server-side stream tracking never released them.

**现象：**
一个客户端打开了 16 个流连接，耗尽了 `max_num_streams=16`。第 17 个流请求被拒绝，返回 `TOO_MANY_STREAMS`。16 个活跃流处于空闲状态——客户端已断开连接但未调用 close，服务器端的流跟踪从未释放它们。

**Why:**
The StreamManager uses `try_acquire(tracking_id)` which increments a counter. `release(tracking_id)` is only called when `poll_stream()` detects a finished group. If a client disconnects mid-stream, the tracking_id is never released:

**根因：**
StreamManager 使用 `try_acquire(tracking_id)` 递增加计数器。`release(tracking_id)` 仅在 `poll_stream()` 检测到完成组时被调用。如果客户端在流中间断开连接，tracking_id 永远不会被释放：

```
Client opens stream → try_acquire("sv-abc") → count=16
Client disconnects → socket closes → poll_stream never called
→ count stays at 16 → TOO_MANY_STREAMS for next 16 clients
→ leaked stream slot persists until server restart
```

**Impact:**
- All streaming requests rejected
- Non-streaming requests still accepted (different code path)
- Metrics: `active_streams` stays at 16 permanently

**影响：**
- 所有流请求被拒绝
- 非流请求仍然被接受（不同的代码路径）
- 指标：`active_streams` 永久停留在 16

**Detection:**
- Metric `active_streams` flat at `max_num_streams` even with no active clients
- `TOO_MANY_STREAMS` errors for valid streaming requests
- Compare `active_streams` with actual client connections

**检测：**
- 即使没有活跃客户端，指标 `active_streams` 也维持在 `max_num_streams`
- 有效的流请求报 `TOO_MANY_STREAMS` 错误
- 比较 `active_streams` 与实际客户端连接数

**Mitigation:**
- Add stream heartbeat/timeout: release stale streams after idle period
- Detect client disconnect via socket error → cleanup
- Separate stream tracking from engine state (use TTL-based expiry)

**缓解措施：**
- 添加流心跳/超时：在空闲期后释放过期的流
- 通过套接字错误检测客户端断连 → 清理
- 将流跟踪与引擎状态分离（使用基于 TTL 的过期）

**Recovery:**
- Manual release: `stream_manager.release(tracking_id)` for known leaked IDs
- Restart serving layer (not the engine) to reset stream counters
- Post-mortem: add proper disconnect detection

**恢复：**
- 手动释放：对已知泄漏的 ID 调用 `stream_manager.release(tracking_id)`
- 重启服务层（而非引擎）以重置流计数器
- 事后分析：添加完善的断连检测
