# Interview Q&A: mini-vLLM 架构面试问答

> 面试方向：AI Infra / LLM Serving / ML Engineering
> 本问答全部基于项目实现，不涉及泛泛的理论

---

## Runtime（20 题）

---

**Q1**: EngineCore.step() 包含哪几个阶段？顺序为什么重要？

**A**: EngineCore.step() 执行四个阶段：1) Scheduler.schedule() -> 2) Executor.prefill() -> 3) Executor.decode() -> 4) cleanup + metrics。顺序重要：先 schedule 做决策，然后分别执行。cleanup 最后做是因为 finished seq 的 KV block 要在所有 prefill/decode 完成后才能释放。

**Follow-up**: 如果 prefill 和 decode 的顺序交换会有什么问题？decode 依赖 prefill 写入的 first token，如果 decode 先跑，产出的 token 是错的。所以 prefill 必须在 decode 之前。

---

**Q2**: 什么是 Continuous Batching？在代码中如何体现？

**A**: 每一步都重新调度所有 seqs（running + waiting），不是等一个 batch 全部做完再处理下一个。Scheduler.schedule() 每 step 执行 6-phase 调度，基于当前 RequestQueue 做全新决策。每个 step 可以既有 prefill 也有 decode。

**Follow-up**: Static batch 的典型问题是什么？短请求的 TTFT 被长请求拖累。Continuous Batching 消除了 head-of-line blocking。

---

**Q3**: EngineCore.step() 为什么不固定 tick 间隔执行？

**A**: 每个 step 耗时取决于负载：纯 decode step 轻量，大 prefill step 重量。固定 tick 要么设短了（频繁空转），要么设长了（浪费 GPU）。事件驱动确保"上一步做完立即下一步"。

**Follow-up**: 如果用固定 tick（10ms），但 step 需要 15ms？下一个 step 被延迟到 20ms 才启动，累积延迟。

---

**Q4**: MetricsCollector 收集哪些指标？

**A**: 1) TTFT（prefill 速度）。2) TPOT（decode 吞吐）。3) Throughput（req/s + tok/s）。4) KV utilisation。5) Block utilisation。6) Scheduler latency。7) Prefix cache hit rate。

**Follow-up**: TTFT 和 TPOT 由哪个配置项调节？max_num_batched_tokens 越大，prefill 越多，TTFT 降低但 TPOT 上升。

---

**Q5**: LLMEngine 和 EngineCore 为什么分开？

**A**: 职责分离。LLMEngine 是用户接口（add_request, run_until_done）。EngineCore 是推理循环（schedule->prefill->decode->cleanup）。LLMEngine 还维护 RequestQueue、做打印、收集输出。

**Follow-up**: 真实 vLLM 中还有 AsyncLLMEngine 和 API Server 层。

---

**Q6**: run_until_done() 何时结束？

**A**: 循环 step() 直到 queue.num_waiting == 0 and queue.num_running == 0。返回所有 finished request 的输出。

**Follow-up**: 如果某个请求永不 finish 会怎样？无限循环。真实系统需要 timeout/abort。

---

**Q7**: ScheduleResult 包含哪些信息？谁消费它？

**A**: scheduled_prefill_groups, scheduled_decode_groups, ignored/finished/rejected_groups, 以及 token 统计、debug_reason。由 EngineCore.step() 消费。

**Follow-up**: 为什么不合并到 Scheduler 内部？契约解耦——ScheduleResult 可以被 mock、序列化、单元测试。

---

**Q8**: _print_step() 打印什么？

**A**: waiting/running 列表、scheduled prefill/decode、token budget 剩余、KV block 使用量、ignored/finished/rejected 列表。cache 命中时显示 "prefill_tokens=N (+Mcached)"。

**Follow-up**: 内存 trace 模式打印什么？打印 BlockAllocator free list、每个 seq 的 BlockTable 映射、ALLOC/FREE trace、on-demand 相比 eager allocation 节省的 block 数量。

---

**Q9**: prefill() 和 decode() 写 KV 的方式有何不同？

**A**: prefill() 写多个连续 prompt token（最多 chunk_size 个），完成后产生 first token。decode() 只写 1 个 generated token。底层都用 _write_to_kv()，但 decode 还需要先 _read_from_kv() 读已有 KV。

**Follow-up**: prefill 完成后 seq.status 怎么变？从 PREFILL 变为 RUNNING。由 executor 在 prefill 函数末尾设置。

---

**Q10**: Executor 接口为什么设计成 Protocol？

**A**: 支持 FakeModelExecutor（测试和教学）和 QwenExecutor（真实模型）的无缝替换。Scheduler 和 EngineCore 只依赖 Executor 协议——tokenize、prefill、decode、cleanup_sequence。不需要知道底层是 fake 还是真实的 HuggingFace 模型。

**Follow-up**: FakeModelExecutor 的 KV cache 如何模拟真实行为？用 Dict[block_id, List[int]] 存储，prepare_block 创建条目，release_block 删除。prefill 时写入 key+value 对，decode 时读取求和。

---

**Q11**: QwenExecutor._build_attention_mask() 解决了什么问题？

**A**: Chunked prefill 时，sequence 已有部分 KV（past_key_values），attention mask 必须覆盖已有 token 和新 token。mask = ones(1, total_len) 允许所有位置互相 attend，与因果掩码配合。

**Follow-up**: 为什么 past_kv is None 时返回 None？transformers 内部会为全量输入自动创建因果掩码。只有 chunked prefill（非 None）才需要手动构造。

---

**Q12**: Config 中的 max_num_prefill_tokens 和 max_num_batched_tokens 有什么区别？

**A**: max_num_batched_tokens 是每 step 的总 token 上限（prefill + decode）。max_num_prefill_tokens 是 prefill 的单独上限。分离后 decode 有 guaranteed budget——不会被 prefill 完全挤占。

**Follow-up**: 如果 max_num_prefill_tokens > max_num_batched_tokens 会怎样？post_init 没有 assert 这个关系，但实际中 prefill 会被 decode 先扣一并约束，不会超过 batched_tokens。

---

**Q13**: print_step_events 和 memory_trace 两种 debug 模式有什么区别？

**A**: print_step_events 打印调度摘要：谁在做 prefill/decode、token budget、KV 用量。memory_trace 打印详细内存状态：free list、BlockTable 映射、ALLOC/FREE 事件、on-demand 节省量。前者适合看调度决策，后者适合 debug 内存问题。

**Follow-up**: 生产环境应该开哪个？都不开。这些是教学和调试工具，生产应该用 metrics 和 tracing。

---

**Q14**: 当 Executor 的 get_kv_stats() 返回 kv_slot_capacity 时，这个值代表什么？

**A**: kv_slot_capacity = len(_kv_cache) * block_size。表示物理 KV cache 当前能容纳的 token 总数（不是实际写入数）。len(_kv_cache) 是已分配的 block 数量。如果所有 block 都塞满，capacity == total_tokens_processed。

**Follow-up**: slot capacity 和实际写入数的差值代表什么？还没被占满的 block 空间，即碎片化空间。

---

**Q15**: 同一个 step 内既有 prefill 又有 decode 时，token budget 如何分配？

**A**: 先扣 decode（每个 decode seq 扣 1 token）。剩余的 budget 给 prefill，但不超过 max_num_prefill_tokens。这种机制确保 decode 序列每 step 至少产 1 个 token。

**Follow-up**: 如果 decode seq 很多，prefill budget 可能为 0 吗？可能。当 decode seq 数量 >= max_num_prefill_tokens 时，prefill budget = 0，新请求不会被 admit。

---

**Q16**: Executor 的 cleanup_sequence() 做什么？

**A**: 对 FakeModelExecutor 是 no-op。对 QwenExecutor 是移除 seq 的 past_key_values。block 级别的 cleanup 由 BlockAllocator.free() 的 on_free 回调处理——EngineCore 不需要手动清理 block。

**Follow-up**: 为什么 block cleanup 不需要 EngineCore 手动触发？因为 Scheduler 在 Phase 1 (Finish) 中调用 BlockManager.free(seq_id)，后者通过 BlockAllocator.free() 释放 block、触发 on_free 回调。EngineCore 只需调用 executor.cleanup_sequence() 清理 executor 内部状态。

---

**Q17**: Sequence 的 is_prefill_finished 如何判断？

**A**: 由 prefill_cursor >= len(prompt_token_ids) 决定。prefill_cursor 初始为 0（或 cached_token_count），每次 prefill 写入完 chunk 后 cursor 前移。当 cursor 超过 prompt 末尾时 prefill 完成。

**Follow-up**: prefill_cursor 和 cached_token_count 的关系是什么？admit 时 prefill_cursor 设为 cached_token_count，表示跳过已缓存的部分。

---

**Q18**: LLMEngine 如何选择使用 fake 还是真实模型？

**A**: 通过 Config.executor_type。在 _create_worker() 中 factory：executor_type=="fake" 创建 FakeWorker，否则创建 QwenWorker。Worker 返回对应的 Executor。Scheduler 和 EngineCore 无需修改。

**Follow-up**: 为什么是 Worker 模式而不是直接创建 Executor？Worker 模式可以封装模型加载、device 管理等复杂逻辑。QwenWorker 负责加载模型到 GPU，FakeWorker 返回纯 CPU fake executor。EngineCore 只看到统一的 get_executor() 接口。

---

**Q19**: 在 test_mid_arrival_merge 中，中间添加的 Request C 会发生什么？

**A**: 在 step 3 之后添加 Request C。此时 A 和 B 正在 running（decode）。C 进入 waiting。下一个 step 的 schedule() 中，A 和 B 继续 decode（Phase 3 先扣 budget），C 在 Phase 5 被 admit（如果 budget 有剩余）。

**Follow-up**: 这个测试验证了什么？验证 Continuous Batching 的新请求可以中途加入：不需要等全部 running seqs 完成，新请求在下一个 step 就有可能被 admit。

---

**Q20**: 如果 test_ondemand_oom_during_execution 中只设 2 个 GPU block，为什么会 OOM？

**A**: 2 个 block 最多容纳 2 * block_size = 8 个 token 的 KV。两个请求的 prompt + decode 总 token 数远超 8。因为 on-demand 在写 KV 时才 allocate，错误发生在 ensure_block 时尝试分配但没有可用 block，抛出 RuntimeError("OOM")。

**Follow-up**: 为什么 OOM 不发生在 admit 时而在执行时？因为 on-demand allocation：admit 时不分配 block，block 在 executor._write_to_kv() 调用 ensure_block() 时才分配。

---


## Scheduler（20 题）

---

**Q21**: Scheduler.schedule() 的 6 个 Phase 分别做什么？

**A**: Phase 1 (Finish) 检查 running 组，标记 FINISHED 并释放 block。Phase 2 (Categorize) 分 decode 和 prefill-continue。Phase 3 (Decode-first budget) decode 先扣预算。Phase 4 (Chunked-prefill continue) 继续未完成 prefill。Phase 5 (Admit) probe cache -> uncached -> budget check -> allocate。Phase 6 汇总 token 统计。

**Follow-up**: Phase 1 必须在 Phase 5 之前？是的。先释放 finish seq 的 budget 和 slot，新请求才能被 admit。如果顺序反了，已完成的 seq 仍占 slot，新请求被错误 ignore。

---

**Q22**: Token budget 的计算过程？

**A**: 初始 = max_num_batched_tokens。decode 先扣（每 seq 1 token + 1 slot）。prefill_budget = min(剩余, max_num_prefill_tokens)。Phase 4 按顺序扣 prefill-continue。Phase 5 按顺序处理 waiting。每 seq 最多 chunk_size，防止独占。

**Follow-up**: 长 prompt 为什么不会饿死 decode？max_num_prefill_tokens 上限 + decode 在 Phase 3 先扣，prefill 无法完全占满 budget。

---

**Q23**: 为什么 decode 必须优先于 prefill？

**A**: 1) 延迟敏感（decode 影响 TPOT，prefill 只影响 TTFT）。2) 不可暂停（decode 不做用户感知卡住）。3) 轻量（decode 每 seq 1 token budget）。

**Follow-up**: 放弃 decode first 的极端情况？长 prefill 占据多步 budget，所有 decode 序列停摆，TPOT 变成 N 倍。

---

**Q24**: Chunked Prefill 的实现原理？

**A**: 长 prompt 分到多个 step，每步处理 max_prefill_chunk_size 个 token。Scheduler Phase 4 推进 cursor。解决：1) 长 prompt 不锁 GPU。2) decode 不饿死。3) Budget 不被独占。

**Follow-up**: Chunked prefill 对 TTFT 的影响？TTFT 增加，但这是公平性的代价。

---

**Q25**: Scheduler admission 条件？rejected 和 ignored 的区别？

**A**: uncached_tokens > prefill_budget 时不能 admit。如果 uncached > max_num_batched_tokens（整个 step 都放不下），标记 rejected。否则 marked ignored（等待后续 step）。rejected 意味着"请求不可能被处理"，ignored 是"现在没空间以后可能"。

**Follow-up**: rejected 请求在 RequestQueue 中进入哪个池？rejected 池。Engine 可以通知用户返回 4xx。ignored 请求下次 step 可能被 admit。

---

**Q26**: 为什么 Scheduler 不在内部执行模型推理？

**A**: 关注点分离。Scheduler 做策略，Executor 做执行。好处：可测试性（调度可单独测试）、可插拔（FakeExecutor/QwenExecutor 共用同一调度器）、性能（调度微秒级、推理毫秒级，可异步化）。

**Follow-up**: 如果 schedule 做成异步线程，BlockManager 需要什么保护？Phase 5 的 allocate_for_seq 修改共享状态（block table, ref count），需要锁保护并发访问。

---

**Q27**: schedule() 调用 allocate_for_seq() 时，seq.block_table 是空的还是已有共享 block？

**A**: 如果是 prefix cache 命中，allocate_for_seq 会把已匹配的共享 block 预先填入 block_table（add_shared_block）。此时 block_table 不为空，但只包含共享 block。未匹配的 block 在 executor.ensure_block() 时 on-demand 填充。

**Follow-up**: 纯 on-demand（无 cache）时 block_table 的长度？空（0）。所有 block 都在 ensure_block 时分配。

---

**Q28**: Phase 4 (prefill-continue) 和 Phase 5 (admit) 处理 prefill 的不同？

**A**: Phase 4 的 seq 已 admit，有 sequence 对象和 prefill_cursor。Phase 5 要新建 sequence、probe cache、allocate_for_seq。Phase 4 不做 allocation，Phase 5 做 allocate_for_seq。

**Follow-up**: 如果 Phase 4 seq 在上一步 prefill 完成，executor 将其设为 RUNNING？Phase 2 中它被分到 decode 组，不再走 Phase 4。

---

**Q29**: RequestQueue 为什么是四个池？

**A**: waiting（等待调度）、running（正在处理）、finished（已完成）、rejected（被拒绝）。Scheduler Phase 1 从 running 移到 finished，Phase 5 从 waiting 移到 running/rejected。

**Follow-up**: 四个池中 running 池为什么特殊？被 Scheduler 和 EngineCore（metrics）同时读，Executor 间接修改 seq state。单线程不需要锁。

---

**Q30**: max_num_seqs=2，waiting 有 4 个请求时如何选择？

**A**: 按 waiting 列表顺序遍历。前 2 个满足 budget/seq budget 的被 admit。后面的收到 MAX_NUM_SEQS_LIMIT ignore。这是 FCFS。

**Follow-up**: 如果要优先级调度需要改哪里？waiting 的遍历顺序和 admission 逻辑。SequenceGroup 需要 priority 字段。

---

**Q31**: ScheduleResult.debug_reason 的格式？

**A**: "prefill(4t): req-0000 | decode(1t): req-0001 | ignored: req-0002(NO_TOKEN_BUDGET) | done: req-0003"。每 step 一行。

**Follow-up**: cache 命中时 prefill 部分？"prefill(4t +8cached): req-0000"。

---

**Q32**: Scheduler 如何知道 seq 是 PREFILL 还是 RUNNING？

**A**: seq.status。Phase 2 根据 status 分类。PREFILL -> prefill_continue_groups，RUNNING -> decode_groups。

**Follow-up**: WAITING 状态的 seq 怎么处理？不会出现在 running 队列，Phase 2 不处理它。

---

**Q33**: probe_prefix_cache 在哪个 phase 调用？

**A**: Phase 5 (Admit)，在 budget check 之前。probe -> uncached = prompt_len - cached -> budget check。

**Follow-up**: Phase 4 为什么不需要 probe？prefill-continue seq 已在 admit 时 probe 过，prefill_cursor 已设好。

---

**Q34**: Scheduler 为什么需要感知 prefix cache？

**A**: 没有 cache 感知时，Scheduler 用 prompt_len 算 budget，但实际只需算 uncached tokens。导致：1) budget 浪费（为已缓存 token 留预算）。2) admission 过于保守。3) chunked prefill 长度假大。

**Follow-up**: 真实 vLLM 中 Scheduler 如何感知 cache？通过 _get_cached_prefix_len() 和 num_unmatched_tokens。设计完全一致。

---

**Q35**: chunked_prefill_enabled=False 时 admission 有何不同？

**A**: this_chunk = uncached_tokens（整个 prompt），不做切分。如果超过 budget，直接 ignore/reject。没有"分多步逐步处理"的可能。

**Follow-up**: 什么场景可能禁用 chunked prefill？KV cache 非常大且 decode 延迟不敏感时。或者使用固定 batch size 的实验设置。

---

**Q36**: test_ondemand_admits_without_allocating_blocks 的核心验证点？

**A**: Scheduler 在 admit 时不分配 block。只有 2 个 GPU block 也能 admit prompt_len=4 的请求。block_table 在 admit 后为空。

**Follow-up**: 这为什么有用？admit 时可以 overcommit，block 在写 KV 时分配。但风险是最终可能 OOM。

---

**Q37**: 如何测试 Scheduler 的 ignore/reject 决策？

**A**: test_ignored_when_budget_full：admit 一请求耗尽 budget，再加一个等待请求，验证被 ignored。test_ignored_reasons：验证 ignored_reasons dict 包含正确的原因字符串。

**Follow-up**: seq budget 和 token budget 哪个先检查？seq budget 先（Phase 5 开头 if remaining_seq_budget <= 0）。然后 check token budget。

---

**Q38**: decode groups 在 Phase 3 之前如果 seq_budget=0 会发生什么？

**A**: decode 仍然被调度。Phase 3 只扣 decode 的 token budget 和 seq budget，但不影响 decode groups 的判定。decode groups 已在 Phase 2 确定，Phase 3 只做 budget 计算。

**Follow-up**: 为什么 decode 不受 seq budget 限制？decode 已经是 running 状态，必须每步做。seq budget 只限制新 admit 的请求。

---

**Q39**: 当 decode groups 数量 > max_num_seqs 时会发生什么？

**A**: 不会发生。因为 max_num_seqs 控制的是 admit，不是 running 上限。超出 max_num_seqs 的 running seq 是之前 admit 时放入的。decode groups 包含所有 running decode seq。

**Follow-up**: 真实 vLLM 如何处理 running 超过 max_num_seqs？使用 watermark 做 preemption——超过阈值时抢占低优先级 seq。

---

**Q40**: 如果让你重新设计 Scheduler，你怎么做？

**A**: 1) 优先级调度（priority queue，基于等待时间/请求大小），不只用 FCFS。2) 预计算 cache 匹配（请求进入 waiting 时提前 probe）。3) 支持 preemption（高优请求抢占低优 prefill slot）。4) Look-ahead scheduling（预读 waiting 队列做 cache probe）。5) 分离 prefill 和 decode 的独立调度器。

**Follow-up**: Preemption 在 BlockManager 层面需要什么？保存和恢复 KV cache state，正确管理被抢占 seq 的 shared block ref_count。

---


## Memory Manager（20 题）

---

**Q41**: 为什么不用连续 KV Cache？

**A**: 连续 KV Cache 的问题：1) 预留空间难估计——预留多了浪费，少了 OOM。2) 碎片化——decode 阶段无法紧凑 packing。3) 不支持 Prefix Cache——无法让两个 seq 共享物理 block。PagedAttention 用 BlockTable 做 logical-to-physical 映射，物理上离散、逻辑上连续。

**Follow-up**: 操作系统虚拟内存和 PagedAttention 的类比？逻辑地址（logical block#）→ 物理地址（physical block#）→ BlockAllocator 分配物理页。缺页中断 = on-demand allocation。共享内存 = shared block via increment_ref。

---

**Q42**: BlockAllocator 的 allocate() 如何选择分配哪个 block？

**A**: 遍历 _free list 找第一个 True 的 slot。找到后设 _free[pid]=False, _ref_counts[pid]=1。这是最简单的 first-fit 策略。选择策略不重要（所有 block 等价），关键是 ref_count 管理。

**Follow-up**: 如果改用 best-fit 策略有用吗？没用，因为所有 block 大小相同（都是 block_size），没有 fit 的差别。BlockAllocator 就是 page allocator——页面大小固定。

---

**Q43**: RefCount 在什么场景会大于 1？

**A**: Prefix Cache 共享时。Sequence A 分配了 P0(ref=1)，Sequence B 共享 P0，increment_ref 使 ref=2。ref > 1 意味着多个 sequence 依赖这个 block 的数据，free 时只减 ref，不释放内存。

**Follow-up**: ref_count 从 2 降到 1 时，block 数据仍然有效吗？仍然有效（另一个 sequence 还在读）。只有降到 0 时 block 才被回收。

---

**Q44**: BlockTable 的 is_shared flag 有什么用？

**A**: 标识这个 logical block 是不是通过 Prefix Cache 从别的 sequence 共享的。当 is_shared=True 时，Executor._write_to_kv() 跳过写入（数据已存在）。is_shared flag 也是未来 Copy-on-Write 的扩展点——当 shared block 需要写入时，allocate 新 block 并替换。

**Follow-up**: Copy-on-Write 的触发条件是什么？当 decoder 向一个 is_shared=True 的 token position 做 KV write 时（decode 阶段）。当前的 prefill 阶段不需要 COW（prompt tokens 在所有共享者之间相同）。

---

**Q45**: On-demand Allocation 和 Eager Allocation 对比？

**A**: Eager：在 admit 时分配 (prompt_len + max_tokens) / block_size 个 block。优点：地址空间固定，不会 OOM。缺点：预留了大量未来才需要的 block。On-demand：在 _write_to_kv() 调用 ensure_block() 时才分配。优点：0 浪费。缺点：执行时可能 OOM。

**Follow-up**: 项目中 block_table 长度反映了什么？已分配 block 的总数（shared + 独占）。不是 token 数，而是 block 粒度。block_table 长度 > ceil(tokens/block_size) 的部分？正常情况下不会，但 decode 阶段 generate 的 token 会逐步增加 block_table。

---

**Q46**: BlockAllocator.free() 如何通过 ref_count 保证正确性？

**A**: free(pids) 对每个 pid: _ref_counts[pid] -= 1; if _ref_counts[pid] == 0: _free[pid] = True。当多个 sequence 共享 block 时，每个 sequence 的 free 只减 1，只有最后一个 sequence free 时才真释放。这导致同一个 block 被调用多次 free 但没有 double-free 问题——第二次 free 时 ref=0 直接跳过。

**Follow-up**: ref_count=0 的 block，_free 标记是什么？_free[pid] = True（free pool 中）。PrefixCache 中其 hash 条目仍然存在，但 probe_prefix_cache 检查 ref_count>0，所以 stale 条目不会被匹配。

---

**Q47**: ensure_block() 的分配策略——为什么先查 cache 再 allocate？

**A**: 因为 sequence 的剩余 prompt token 可能已被另一个 sequence 在后续 step 中注册到 cache。比如 seq-A 在 step 0 写入 block 0 并注册 hash，seq-B 在 step 1 才调用 ensure_block(seq-B, 4)——此时 block 2（对应 hash）可能已被 seq-A 在 step 0 的后续 allocate 注册。查 cache 可以捕获这种情况。

**Follow-up**: 这与 allocate_for_seq 时的 cache 检查有什么区别？allocate_for_seq 只检查 prefix 从 index 0 开始连续匹配的 block。ensure_block 检查的是单个 block hash——可能命中 cache 中非连续位置的 block（但实际上不连续位置不会缓存匹配）。

---

**Q48**: BlockManager 的 block_size 和 Config 的 block_size 关系？

**A**: 同一个值。BlockManager 在构造时从 Config 获得 block_size，然后传递给 BlockTable（决定逻辑地址到物理地址的映射粒度）和 PrefixCache（决定 hash 分块粒度）。block_size=4 意味着每 4 个 token 组成一个 logical block。

**Follow-up**: block_size 变大或变小的影响？小 block_size：更细粒度共享（prefix 匹配更精确），但 block table 变大。大 block_size：共享粒度粗（prefix 匹配需要连续更多 token），但 block table 更小、分配 overhead 更低。

---

**Q49**: BlockManager.stats() 返回哪些信息？

**A**: 调用 _allocator.stats() 得到 total_blocks / free_blocks / used_blocks，再加上 prefix_cache_entries = _prefix_cache.size()。这些统计信息被 EngineCore 传递给 MetricsCollector 用于计算 KV utilisation 等指标。

**Follow-up**: prefix_cache_entries 可能大于 total_blocks 吗？可能。因为 PrefixCache 是 hash->pid 映射，stale entries（pid 已被释放但 hash 还在）也计入 size。所以 prefix_cache_entries 可以超过总 block 数。

---

**Q50**: dump_tables() 返回什么？用于什么场景？

**A**: 返回 Dict[seq_id, List[{logical, physical, shared}]]，每个 seq 的 BlockTable 映射。用于 memory_trace debug 模式——打印每个 seq 的 logical->physical 映射，帮助理解 on-demand allocation 和 shared block 的状态。

**Follow-up**: memory_trace 的输出中，如何区分 shared block 和 own block？"L0->P0(shared)" vs "L0->P0"。shared 标注表示该 block 是通过 Prefix Cache 共享的。

---

**Q51**: BlockManager._block_hashes 和 _shared_prefix_blocks 的作用？

**A**: _block_hashes[seq_id] 存储 seq 的 prompt block hash 列表，用于 ensure_block 时查 cache。_shared_prefix_blocks[seq_id] 记录 seq 共享了多少个 prefix block（从 index 0 开始）。后者用于 is_block_shared 检查和 free 时 cleanup。

**Follow-up**: 为什么 _block_hashes 要在 allocate_for_seq 时立即计算？因为 compute_block_hashes 只在 prompt 阶段有意义（decode token 不需要 hash）。在 allocate_for_seq 时 prompt token 已知，计算一次后 ensure_block 可以重复使用。

---

**Q52**: BlockTable 的 add_shared_block 和 add_block 有什么区别？

**A**: add_shared_block 创建 is_shared=True 的条目（Executor 会跳过 KV write），add_block 创建 is_shared=False 的条目（Executor 会写入 KV）。物理上两者都占用一个 physical block slot，但共享 block 不需要写 KV。

**Follow-up**: add_shared_block 后 ref_count 已经在 allocate_for_seq 中 increment 过了吗？是的。BlockManager.allocate_for_seq() 中先 increment_ref(cached_pid)，然后 add_shared_block(cached_pid)。顺序必须是 increment 在前——修改 ref_count 先于记录 block table。

---

**Q53**: BlockAllocator 的 on_allocate 和 on_free 回调用于什么？

**A**: 通知 Executor 创建/释放 KV cache storage。on_allocate 调用 executor.prepare_block(pid)（创建 _kv_cache[pid] = []），on_free 调用 executor.release_block(pid)（删除 _kv_cache[pid]）。这是 BlockAllocator 与 Executor 的衔接点。

**Follow-up**: 为什么不用 BlockAllocator 直接在 allocate 时创建 KV storage？因为 BlockAllocator 不知道 Executor 的存在——它是纯物理 block 管理层。回调模式保持了 BlockAllocator 的通用性。

---

**Q54**: 如果 BlockTable 的 num_blocks() 比预期少，ensure_block 会怎样？

**A**: 在 while logical_idx >= table.num_blocks(): 循环中不断添加新 block。添加策略：先查 cache，hit 则 add_shared_block + increment_ref，miss 则 allocate + add_block。循环直到 table 足够长，覆盖目标 logical_idx。

**Follow-up**: 这个 while loop 最多循环几次？一次。因为每次循环只加一个 block（allocate(1) 或 add_shared_block），而 logical_idx 每次递增 1。最多循环到当前 logical_idx 被覆盖。

---

**Q55**: 为什么 free(seq_id) 时不需要反过来检查是不是 shared block？

**A**: free 调用 allocator.free(pids) —— 所有 block 的物理地址都被传入。allocator.free() 对每个 pid 执行 _ref_counts[pid] -= 1。shared block 的 ref 可能从 2 降到 1（不会被 true free），own block 的 ref 从 1 降到 0（被 true free）。allocator 不需要知道 block 是 shared 还是 owned，它只管理 ref count。

**Follow-up**: 如果 shared block 被 free 后(>=0)，另一个 seq 还在 probe 它？probe 检查 ref_count > 0，所以 ref=0 的 block 不会被匹配。stale cache entry 被安全处理。

---

**Q56**: BlockAllocator 如何处理 double-free？

**A**: free() 中检查 if _ref_counts[pid] == 0: continue（跳过）。不会 raise error。这在 ref_count 设计下是合理的——多次 free 同一个 pid 是安全的行为，因为只有第一次（ref 从 1 到 0）释放了 block。

**Follow-up**: 什么情况会导致 double-free？BlockManager.free(seq_id) 清理 seq 的所有 block。如果 seq 的 block_table 被修改或不一致，可能出现同一个 pid 被多次传入 free。ref_count 机制天然防护了这一点。

---

**Q57**: BlockManager.is_block_shared() 的实现细节？

**A**: 从 BlockTable 中查找 token_position 对应的 logical block，检查该 entry 的 is_shared flag。如果是 shared 返回 True，否则返回 False。如果 seq 没有 BlockTable（尚未 allocate_for_seq），返回 False。

**Follow-up**: Executor 在 _write_to_kv() 中为什么先检查 is_block_shared？如果 block 是共享的，KV 数据已被原 sequence 写入，不需要重复写入。跳过写入节省了 KV cache 内存和计算。

---

**Q58**: memory_trace 中 "blocks=N (would be M with eager), saved=M-N" 如何计算？

**A**: N = len(seq.block_table)（on-demand 分配的实际 block 数）。M = (prompt_len + max_tokens + block_size - 1) // block_size（eager allocation 需要的 block 数）。saved = M - N，显示 on-demand 相比 eager 节省了多少 block。

**Follow-up**: 如果 seq 在 prompt 阶段就 OOM 了，N 的值是多少？少于预期。但 saved 计算仍然基于最大需求，所以 saved 可能很大——但这不代表 on-demand 好，而是说明 seq 还没运行到需要更多 block 就 OOM 了。

---

**Q59**: BlockAllocator 的 num_free_blocks 如何反映系统内存压力？

**A**: 每次 allocate() 减少 num_free_blocks，每次 free()（ref 到 0 时）增加。如果 num_free_blocks == 0，allocate 返回 None -> ensure_block 抛出 RuntimeError("OOM")。当 num_free 接近 0 时，新请求的 prefill 会 OOM——但 decode 可能还正常运行（已有 block 在 ref_count 保护下）。

**Follow-up**: 如何从 free block 数量判断是否需要限制新请求 admission？当 free_blocks < avg(每个请求所需的 block 数) * 几个请求时，应该拒绝新请求。真实 vLLM 使用 watermark 机制。

---

**Q60**: dump_free_list() 和 dump_used_list() 的输出顺序有什么含义？

**A**: dump_free_list() 返回所有 _free[i]==True 的 i 列表，按 index 排序。dump_used_list() 返回所有 _free[i]==False 的 i 列表。顺序就是 block index 顺序。memory_trace 中打印这些列表可以观察碎片化程度。

**Follow-up**: 如果 free list 是 [0, 3, 5, 7]，used list 是 [1, 2, 4, 6]，这说明什么？说明 KV cache 碎片化——分配和释放导致空闲 block 不连续。但因为 PagedAttention 的 block 是等大小且通过 BlockTable 映射，这种碎片不影响分配——任何空闲 block 都能用。

---


## Prefix Cache（10 题）

---

**Q61**: PrefixCache 的核心数据结构是什么？为什么是 Dict[int, int]？

**A**: _cache: Dict[int, int] = {block_hash: physical_block_id}。key 是 token block 的 hash，value 是 physical block ID。这是最简单的 hash 索引——给定 block content 的 hash，直接返回已分配 block 的物理地址。为什么不用 List？因为 hash 值不连续，dict 提供 O(1) 查找。

**Follow-up**: 为什么不是 MultiDict（一个 hash 多 block）？因为每个 hash 对应唯一的 prompt token block 内容。相同内容在任何 seq 中 hash 相同，指向的物理 block 可以不同（但当前实现假设相同）。实际中需要 LRU 策略时可能一个 hash 多个 pid。

---

**Q62**: compute_block_hashes() 如何处理最后一个不完整的 block？

**A**: 按 block_size 步长切片 prompt_token_ids[i:i+block_size]，最后一块可能小于 block_size。Python slice 超出 index 会返回剩余部分。这个 partial block 也被 hash 并可能被 cache——如果另一个 seq 有完全相同的部分 block。

**Follow-up**: Partial block 的 hash 碰撞概率？比完整 block 高的概率不变——hash 函数对（不同长度 + 不同内容）天然区别。两个 partial block 只有 token 和长度都相同时 hash 才相同。

---

**Q63**: PrefixCache 如何插入和查找 block？

**A**: insert(hash, pid)：_cache[hash] = pid。lookup(hash)：return _cache.get(hash, None)。batch 操作 lookup_span(hashes) 对每个 hash 调用 lookup，insert_span(hashes, pids) 并发 insert 每个 pair。简单直接。

**Follow-up**: lookup_span 返回 list 中 None 代表什么？cache miss——没有 seq 注册过这个 hash 对应的 block。调用者（BlockManager）据此决定是否 allocate 新 block。

---

**Q64**: Stale cache entry 怎么产生？如何处理？

**A**: 当所有持有某个 block 的 seq 都 free 后，block 被归还 free pool（ref=0），但 PrefixCache 中的 hash->pid 映射还在。Stale entry。probe_prefix_cache 检查 ref_count > 0，忽略 stale entries。ensure_block 中的 lookup 也检查 ref_count——stale 被视为 cache miss。

**Follow-up**: Stale entry 会被覆盖或淘汰吗？当前无 LRU/eviction，stale entry 永久存在。当后续 seq 分配了同一个 pid（如果 allocator 返回相同 pid），cache 中 hash->pid 正确；否则 hash 仍然指向错误但 ref=0 的 pid，最终 hash 碰撞后 insert 会覆盖。

---

**Q65**: Prefix cache 的共享粒度是什么？为什么是这个粒度？

**A**: Block 粒度（block_size 个 token）。因为 PagedAttention 的 KV cache 以 block 为最小操作单位——allocate、free、copy-on-write 都在 block 级别。更细粒度（token 级）共享需要更复杂的数据结构且收益有限。

**Follow-up**: 如果 block_size=4，只有前 3 个 token 匹配，block-level cache 能共享吗？不能。hash 计算的是 4 个 token 的完整 block，只有全部匹配才能共享。这是 block-level caching 的固有代价——匹配粒度粗，但实现简单、overhead 低。

---

**Q66**: RefCount 的设计如何影响 Prefix Cache 的并发安全？

**A**: ref_count 保护物理 block 不被提前释放。两个 seq 共享 block P0（ref=2），A 先结束时 free(A) 将 ref 从 2 降到 1，P0 存活。B 结束后 ref 从 1 到 0 才释放。不需要锁——ref_count 的递增和递减是原子操作（当前单线程）。

**Follow-up**: 如果 A 和 B 同时 free 同一个 block（多线程）？当前没有多线程问题。真实 vLLM 中 ref_count 用 atomic operations 保护，Python 中 GIL 对此天然安全。

---

**Q67**: 解释 probe 和 allocate 为什么是两阶段的？

**A**: Phase 1 probe：Scheduler 查 cache 但不修改 ref_count——只读。目的是知道 uncached_tokens 数量，做 budget 决策。Phase 2 allocate：BlockManager 实际 attach shared block，increment_ref，add_shared_block。分离的收益：1) 多次 probe 安全（无副作用）。2) Budget 决策基于真实 uncached 量。3) 如果 budget 不足（ignored），不需要回滚 ref_count 变更。

**Follow-up**: 如果 probe 后 allocate 前 cache 状态变了（另一个 seq 释放 block）怎么办？allocate_for_seq 在 attach 时仍然检查 ref_count > 0，遇到 stale entry 就 break 停止共享。probe 是预算估计，allocate 是实际执行——allocate 自己做 final 检查。

---

**Q68**: prefix cache 命中后，Executor._write_to_kv() 如何处理 shared block？

**A**: 首先检查 is_block_shared(seq, position)，如果 True 直接 return（不写 KV，不计入总 token 数）。这意味着 executor._total_tokens_processed 不包含共享 block 的 token，反映的是"实际写入" token 数，不是"理论 token"数。

**Follow-up**: 这个跳过共享 block 的机制可能引入什么问题？如果 seq 需要在前面的 shared block 上做 attention（正常的 prefill 操作），但 KV 数据早已在另一个 seq 的 block 中。正确——因为 decoder 读 KV 时调用 _read_from_kv() 而不是 _write_to_kv()。跳过的是写，不影响读。

---

**Q69**: PrefixCache 的 size() 和 BlockAllocator 的 num_used_blocks 有什么关系？

**A**: 没有直接相等关系。size() 是 cache 中的 hash 条目数（包括 stale entries），num_used_blocks 是 ref_count > 0 的 block 数。通常 size >= num_used_blocks，因为 stale entry 占据 hash 条目但不占用 block。两者之差 = stale entry 数。

**Follow-up**: 如何清理 stale cache entries？当前的实现不清除。实际工程中需要 LRU eviction：限制 cache 容量，淘汰最久未命中的条目。或者当 ref=0 的 block 被重新分配时覆盖 hash->pid。

---

**Q70**: 为什么 Prefix Cache 不直接复制 KV？

**A**: 复制 KV = 内存翻倍 + 计算翻倍。Shared block 的核心优势是零复制——共享者不需要分配内存、不需要写 KV、不需要做注意力计算。只增加 ref_count。如果复制，第二份请求需要 allocate 新 block、写 KV（完全相同的值），完全没有节省。

**Follow-up**: 什么时候需要复制？Copy-on-Write 场景：shared block 需要被修改（decode 写入不同 token 时），此时才分配新 block、复制数据。

---


## Serving System Design（10 题）

---

**Q71**: 当前项目是否可以处理并发请求？

**A**: 单线程事件循环模型，同步处理。add_request() 是同步的，step() 也是同步的。不能同时处理两个 step。但是 Continuous Batching 允许新请求在任意 step 加入，所以不需要请求排队等待前一个完全结束——新请求在下一个 step 的 Phase 5 被 admit。

**Follow-up**: 如果要支持真正的并发请求（同时多个客户端调 add_request），需要改什么？需要锁保护 RequestQueue（多线程写 waiting），或者使用 async 模型。LLMEngine 的 add_request 需要线程安全。

---

**Q72**: 当前系统如果接收 100 个同时请求会发生什么？

**A**: add_request 全放入 waiting 队列（O(1) 操作）。每个 step 的 Phase 5 最多 admit min(max_num_seqs, budget) 个请求。如果 max_num_seqs=4，每个 step 只 admit 最多 4 个。100 个请求会逐步被 admit，最先等到的 4 个开始 prefill，后续等 budget 释放后才被 admit。会 prioritize 最先到达的（FCFS）。

**Follow-up**: 100 个中最后一个请求的等待时间是多少？取决于 max_num_seqs、平均 prompt 长度、每个 step 的 throughput。如果每个 request 需要 10 step 完成，max_num_seqs=4，100/4=25 个 batch，每个 batch 约 10 step，所以最后一个大约在 step 250 左右开始。

---

**Q73**: max_num_seqs 设为 1 但 chunked_prefill 启用时，系统还能正常服务吗？

**A**: 能，但吞吐受限。每个 step 只能处理 1 个 seq。如果这个 seq 在做 prefill，另一个 decode seq 的 slot 需要等。事实上 max_num_seqs=1 时 decode 和 prefill 不能并行——要么做 decode 要么做 prefill。此时必须设 max_num_batched_tokens 至少等于 max_prefill_chunk_size + 1。

**Follow-up**: 这个配置对 prefix cache 测试有什么用？test_identical_requests_metrics_show_cache_hits 用 max_num_seqs=1 确保每个 step 只处理 1 个 seq，方便验证 metrics 中的 cache 命中统计。

---

**Q74**: 项目如何支持不同的 executor（fake vs qwen）而不改 Scheduler？

**A**: Executor 是 Protocol（Python typing.Protocol 定义接口）。Scheduler 不持有 Executor 引用，只产出 ScheduleResult。EngineCore 将结果中的 seq 列表传给 Executor.prefill()/decode()。Executor 的替换通过 Worker 模式——FakeWorker 返回 FakeModelExecutor，QwenWorker 返回 QwenExecutor。Scheduler 完全不知道。

**Follow-up**: QwenExecutor 的 KV cache 和 FakeExecutor 的 KV cache 管理模式有何不同？QwenExecutor 额外维护 _seq_kv（transformers 的 past_key_values 格式），因为预训练模型用 transformers 的增量 cache 做推理。FakeExecutor 只有 _kv_cache dict 做模拟。

---

**Q75**: 如果新增 GPU 硬件 backend，需要改哪些文件？

**A**: 1) 新建 executor（如 triton_executor.py）实现 Executor 协议。2) 新建 worker（如 TritonWorker）返回该 executor。3) 在 Config.executor_type 添加新的 type 值。4) 在 LLMEngine._create_worker() 添加分支。Scheduler / BlockManager / PrefixCache 完全不需要修改。

**Follow-up**: GPU backend 的 KV cache 是否可以复用 BlockAllocator 的 on_allocate/on_free 回调？可以。BlockAllocator 回调通知 Executor 创建/释放 GPU KV buffer。QwenExecutor 的 prepare_block 就是填充 _kv_cache dict。

---

**Q76**: SequenceGroup 和 Sequence 的关系在 beam search 场景如何扩展？

**A**: 当前 1:1（一个 group 一个 seq）。Beam search 需要 1:N——每个 group 有 N 个 seq 做不同路径探索。每个 seq 需要独立的 block_table、prefill_cursor、output_token_ids。Scheduler 中每个 seq 独立消耗 budget。BlockManager 中每个 seq 独立管理 block table 和 free。

**Follow-up**: Beam search 时，N 个 sibling seq 共享 prefix 会如何？BlockManager 自动共享——它们有相同的 prompt token，prefix cache 命中后每个 seq 共享相同的 block。ref_count 从 2 开始（2 个 sibling 共享）。

---

**Q77**: 如何扩展当前系统支持流式输出？

**A**: 当前 run_until_done() 等待所有 token 生成完才返回。流式输出需要：1) LLMEngine.step() 在 decode 后立即返回新生成的 token。2) 应用层通过回调或队列消费这些 token。3) Scheduler 不需要改（step 的决策不变），EngineCore 的 prefill/decode 后加一步"收集新 token"。

**Follow-up**: 流式输出时，finished 状态的 seq 何时被 cleanup？engine 可以选择在最后一个 token 被消费后才 free seq 的 block。当前是在 Phase 1 finish 时调用 BlockManager.free()。

---

**Q78**: 如何扩展当前系统支持 Abort Request？

**A**: 添加 LLMEngine.abort_request(request_id) 方法，从 waiting（或 running）中移除该 SequenceGroup，调用 BlockManager.free() 释放 block。running 中的 seq 被 abort 后，EngineCore 需要跳过它的 prefill/decode。Scheduler 需要将它的 slot 提前释放给其他请求。

**Follow-up**: Abort 和 finish 在 BlockManager 的 free 调用上有区别吗？相同。都是调用 BlockManager.free(seq_id) 释放所有 block。但 abort 需要额外 cleanup executor 的 per-seq state（past_key_values）。

---

**Q79**: 如何实现请求级的超时（Time-To-Live）？

**A**: 给 SequenceGroup 添加 deadline/expiry 字段。在 Scheduler Phase 1 检查：如果 arrival_time + timeout < current_time，标记该 seq 为 TIMEOUT（类似 finish），释放 block，记录到 metrics。EngineCore 将其视为 finished 处理（cleanup + 不返回结果给用户）。

**Follow-up**: 超时发生在 prefill 中间和 decode 中间的处理有何不同？相同——都是 force free block。但 prefill 中 timeout 意味着 seq 从未输出 token，用户没看到任何内容。decode 中 timeout 意味着用户收到了部分输出。

---

**Q80**: 当前架构最大的瓶颈在哪？如何优化？

**A**: 1) 单线程事件循环——Scheduler 和 Executor 串行，GPU 在 Scheduler 执行时空闲。优化：Scheduler 异步化（Scheduler 跑在 CPU、Executor 跑在 GPU，pipeline 并行）。2) 无 LRU eviction——PrefixCache 持续增长。优化：添加 LRU 淘汰策略。3) FCFS 调度——没有优先级，无法区分高优和低优请求。优化：优先级队列 + preemption。4) 无 sliding window attention——所有历史 KV 一直占用 block。优化：实现 sliding window，释放早期 token 的 block。5) on-demand OOM——admit 时没有记忆分配检查。优化：admit 时检查 num_free_blocks，设置 watermark。

**Follow-up**: Scheduler 异步化面临的最大工程挑战？状态同步。Scheduler 在 CPU 上修改 BlockManager 和 RequestQueue，Executor 在 GPU 上写 KV cache。如果两者异步运行，Scheduler 做的 allocate_for_seq 需要保证 block 在 GPU 端已创建，否则 Executor.ensure_block 可能失败。

---
