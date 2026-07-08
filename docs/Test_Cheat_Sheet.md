# mini-vLLM Test Cheat Sheet — 面试复习用

> 基于当前代码（176 tests, ~3s passing）。所有测试函数名、assert、不变量均来自实际代码。
> 运行方式：`PYTHONPATH=. python3 -m pytest tests/ -q`

---

## 测试文件：tests/test_request.py（~12 tests）

### class TestStatus

#### 1. test_finished_statuses
```
构造场景：直接比较 FINISHED 和 REJECTED 两个 enum 值。
关键操作：无。
核心 assert：assert Status.FINISHED is not Status.REJECTED
验证的不变量：FINISHED 和 REJECTED 是不同状态码。
如果失败：Status 枚举定义有重复值。
```

#### 2. test_status_order
```
构造场景：直接比较 WAITING 和 PREFILL 的 enum value。
关键操作：无。
核心 assert：assert Status.WAITING.value < Status.PREFILL.value
验证的不变量：enum 值的数值顺序 WAITING(0) < PREFILL(1) < RUNNING(2) < FINISHED(3)。
如果失败：Status 枚举值顺序错乱。
```

### class TestSamplingParams

#### 3. test_defaults
```
构造场景：SamplingParams() 无参数构造。
关键操作：读取 4 个默认字段。
核心 assert：
  assert sp.max_tokens == 16
  assert sp.temperature == 1.0
  assert sp.top_p == 1.0
  assert sp.top_k == -1
验证的不变量：默认采样参数正确。
如果失败：默认值被修改或构造函数出错。
```

#### 4. test_custom_max_tokens
```
构造场景：SamplingParams(max_tokens=64)。
关键操作：读取自定义 max_tokens。
核心 assert：assert sp.max_tokens == 64
验证的不变量：构造函数接受并正确存储自定义值。
如果失败：参数覆盖逻辑损坏。
```

#### 5. test_stop_lists
```
构造场景：SamplingParams(stop_token_ids=[1,2], stop_strings=["."])。
关键操作：验证 stop_lists 被正确存储。
核心 assert：
  assert sp.stop_token_ids == [1, 2]
  assert sp.stop_strings == ["."]
验证的不变量：stop 条件正确保存。
如果失败：stop lists 字段损坏。
```

### class TestSequence

#### 6. test_initial_state
```
构造场景：构造一个 3-token 的 Sequence。
关键操作：验证初始状态所有字段。
核心 assert：
  assert seq.status == Status.WAITING
  assert not seq.finished
  assert seq.prompt_length == 3
  assert seq.num_output_tokens == 0
验证的不变量：新创建的序列处于 WAITING 状态、未完成、无输出。
如果失败：Sequence 初始化逻辑有 bug。
```

#### 7. test_status_lifecycle
```
构造场景：手动走一次 WAITING->PREFILL->RUNNING->FINISHED。
关键操作：依次设置 status，最后设 num_generated_tokens=1 再 FINISHED。
核心 assert：assert seq.finished
验证的不变量：生命周期转换正确，finished 反映 FINISHED 和 REJECTED。
如果失败：status 转换或 finished 属性有 bug。
```

#### 8. test_rejected_is_finished
```
构造场景：设置 status = Status.REJECTED。
关键操作：单独设 REJECTED 状态。
核心 assert：assert seq.finished
验证的不变量：REJECTED 也算 finished（不会再被调度）。
如果失败：finished 属性未包含 REJECTED 状态。
```

#### 9. test_to_dict
```
构造场景：构造 Sequence 后调用 to_dict()。
关键操作：读取字典字段。
核心 assert：
  assert d["seq_id"] == "test-seq"
  assert d["num_prompt_tokens"] == 3
  assert d["status"] == "WAITING"
验证的不变量：序列的字典表示完整。
如果失败：to_dict() 字段缺失或错误。
```

### class TestSequenceGroup

#### 10. test_create_sequence
```
构造场景：SequenceGroup 创建一个 Sequence。
关键操作：create_sequence，验证 seq 字段和 group 状态。
核心 assert：
  assert seq.seq_id == "g0-seq-0"
  assert seq.group_id == "g0"
  assert seq.sampling_params is sg.sampling_params
  assert sg.num_sequences == 1
  assert not sg.is_finished
验证的不变量：create_sequence 正确关联 group 和 sampling_params。
如果失败：序列创建逻辑有 bug。
```

#### 11. test_is_finished_all_done
```
构造场景：2 个 seq 均 FINISHED。
关键操作：同时设置两个 seq 的 status。
核心 assert：assert sg.is_finished; assert sg.num_finished == 2
验证的不变量：全部 seq 完成时 group 才算完成。
如果失败：is_finished 判断错误。
```

#### 12. test_is_finished_partial
```
构造场景：2 个 seq，只有 1 个 FINISHED。
关键操作：设置其中一个 FINISHED。
核心 assert：assert not sg.is_finished
验证的不变量：部分完成时 group 不算完成。
如果失败：is_finished 逻辑有 bug。
```

#### 13. test_empty_group_not_finished
```
构造场景：无 seq 的 SequenceGroup。
关键操作：直接访问 is_finished。
核心 assert：assert not sg.is_finished
验证的不变量：空 group 不算完成。
如果失败：空 group 导致崩溃或错误返回 finished。
```

#### 14. test_get_unfinished_seqs
```
构造场景：2 个 seq，其中一个 FINISHED。
关键操作：get_unfinished_seqs()。
核心 assert：assert len(unfinished) == 1; assert unfinished[0].seq_id == "s2"
验证的不变量：只返回未完成的 seq。
如果失败：get_unfinished_seqs 过滤逻辑有 bug。
```



## 测试文件：tests/test_kv_cache_manager.py（~13 tests）

### class TestBlockTable

#### 15. test_add_and_count
```
构造场景：BlockTable("r0", block_size=4)，添加 12 和 7 两个 block。
关键操作：add_block(12), add_block(7)。
核心 assert：
  assert table.num_blocks() == 2
  assert table.get_block_ids() == [12, 7]
验证的不变量：block table 的 append-only 语义。
如果失败：BlockTable 数据存储或计数有 bug。
```

#### 16. test_get_physical_block
```
构造场景：BlockTable 有 block 10（逻辑 0-3）和 block 20（逻辑 4-7）。
关键操作：查询各逻辑位置对应的物理 block。
核心 assert：
  assert table.get_physical_block(0) == 10
  assert table.get_physical_block(3) == 10
  assert table.get_physical_block(4) == 20
  assert table.get_physical_block(7) == 20
  assert table.get_physical_block(8) is None
验证的不变量：position / block_size = 逻辑 block 索引，越界返回 None。
如果失败：位置映射逻辑错误。
```

#### 17. test_clear
```
构造场景：添加 2 个 block 后 clear()。
关键操作：clear()。
核心 assert：assert table.num_blocks() == 0
验证的不变量：clear 后 block 列表为空。
如果失败：clear 未正确重置状态。
```

### class TestBlockAllocator

#### 18. test_allocate_and_free
```
构造场景：8-block 分配器。
关键操作：allocate(3) 分配 3 个 block，然后 free(pids)。
核心 assert：
  alloc.num_free_blocks == 8（初始）
  pids == [0, 1, 2]
  alloc.num_free_blocks == 5（分配后）
  alloc.num_free_blocks == 8（释放后）
验证的不变量：allocate 返回连续 PID，free 归还到池。
如果失败：分配器基础功能有 bug。
```

#### 19. test_oom
```
构造场景：2-block 分配器，先 allocate(2) 耗尽，再 allocate(1)。
关键操作：第二次 allocate(1)。
核心 assert：assert t2 is None
验证的不变量：OOM 返回 None 而非 raise/crash。
如果失败：OOM 时 crash 或返回错误值。
```

#### 20. test_callback_on_allocate
```
构造场景：带 on_allocate 和 on_free 回调的分配器。
关键操作：allocate(2)，然后 free(pids)。
核心 assert：events 回调列表正确。
验证的不变量：allocate/free 触发的回调及时且准确。
如果失败：回调机制有 bug。
```

#### 21. test_stats
```
构造场景：8-block 分配器，allocate(2) 后读取 stats。
关键操作：stats() 方法。
核心 assert：
  s["total_blocks"] == 8
  s["free_blocks"] == 6
  s["used_blocks"] == 2
验证的不变量：stats 返回字段与实际状态一致。
如果失败：stats() 报告不准确。
```

### class TestBlockManager

#### 22. test_allocate_for_seq_starts_empty
```
构造场景：BlockManager 为新 seq 调用 allocate_for_seq。
关键操作：allocate_for_seq(seq)。
核心 assert：
  seq.block_table == []（admission 时不分配 block）
  alloc.num_free_blocks == 8
  mgr.get_table(seq.seq_id) is not None
验证的不变量：admission 时 block table 为空。这是 on-demand 分配的关键设计。
如果失败：admission 时错误地预分配了 block。
```

#### 23. test_ensure_block_allocates_on_demand
```
构造场景：核心 on-demand 分配路径。
关键操作：逐步 ensure_block。
核心 assert：
  ensure_block(seq,0) -> pid=0, blocks=1, free=7
  ensure_block(seq,3) -> pid=0, blocks=1（同一 block，不新分配）
  ensure_block(seq,4) -> pid=1, blocks=2, free=6（新逻辑 block）
验证的不变量：ensure_block 只在跨 block boundary 时分配。
如果失败：on-demand 分配逻辑有 bug。
```

#### 24. test_free
```
构造场景：分配 2 个 block 后释放 seq。
关键操作：mgr.free(seq.seq_id)。
核心 assert：
  alloc.num_free_blocks == 8（全部归还）
  mgr.get_table(seq.seq_id) is None（table 注销）
验证的不变量：free 后 block 全部归还，table 被注销。
如果失败：资源泄漏。
```

#### 25. test_oom_during_ensure_block
```
构造场景：2-block 池，seq1 占满后 seq2 尝试 ensure_block。
关键操作：seq2.ensure_block(seq2,0) 时 pool 已空。
核心 assert：with pytest.raises(RuntimeError, match="OOM"):
验证的不变量：OOM 在 execution 时 raise RuntimeError，而非 admission 时。
如果失败：OOM 路径错误地返回 None 而非 raise。
```

#### 26. test_stats
```
构造场景：分配 2 个 block 后读取 mgr.stats()。
关键操作：stats() 方法。
核心 assert：
  s["total_blocks"] == 8
  s["free_blocks"] == 6
  s["used_blocks"] == 2
验证的不变量：BlockManager stats 代理到 allocator stats。
如果失败：stats 代理有 bug。
```



## 测试文件：tests/test_prefix_cache.py（~20 tests）

### class TestPrefixCache

#### 27. test_empty_cache_returns_none
```
构造场景：新建 PrefixCache。
关键操作：lookup(42)。
核心 assert：assert cache.lookup(42) is None; assert cache.size() == 0
验证的不变量：空 cache 返回 None。
如果失败：空 cache 返回非空。
```

#### 28. test_insert_and_lookup
```
构造场景：insert(123, 5) 后 lookup。
关键操作：insert + lookup。
核心 assert：assert cache.lookup(123) == 5; assert cache.size() == 1
验证的不变量：插入后 lookup 返回正确 PID。
如果失败：insert/lookup 逻辑有 bug。
```

#### 29. test_lookup_span_all_hits
```
构造场景：插入三个 hash->pid 映射，批量查询。
关键操作：lookup_span([10,20,30])。
核心 assert：assert cache.lookup_span(hashes) == [1, 2, 3]
验证的不变量：lookup_span 批量返回结果。
如果失败：批量查询有 bug。
```

#### 30. test_lookup_span_with_misses
```
构造场景：只插入 hash 10 和 30，查询 [10,20,30]。
关键操作：lookup_span。
核心 assert：assert cache.lookup_span([10,20,30]) == [1, None, 3]
验证的不变量：miss 项返回 None 而非错误。
如果失败：miss 处理有 bug。
```

#### 31. test_insert_span
```
构造场景：批量 insert_span([10,20,30], [1,2,3])。
关键操作：insert_span 后逐个 lookup。
核心 assert：每个 hash 对应的 pid 正确。
验证的不变量：批量插入正确。
如果失败：insert_span 逻辑有 bug。
```

#### 32. test_hash_determinism
```
构造场景：对相同 tokens 计算两次 block hashes。
关键操作：compute_block_hashes(tokens, block_size=2) 两次。
核心 assert：assert h1 == h2
验证的不变量：hash 函数是确定性的。
如果失败：hash 函数不稳定（非确定性）。
```

#### 33. test_hash_partial_block
```
构造场景：3 个 token，block_size=4。
关键操作：compute_block_hashes。
核心 assert：assert len(hashes) == 1（一个部分 block）
验证的不变量：最后一个不满 block_size 的 block 也生成 hash。
如果失败：部分 block 被忽略。
```

### class TestBlockAllocatorRefCount — double-free 测试就在这里

#### 34. test_allocate_sets_ref_count_one
```
构造场景：allocate(2)。
关键操作：get_ref_count。
核心 assert：assert alloc.get_ref_count(0) == 1; assert alloc.get_ref_count(1) == 1
验证的不变量：新分配的 block 初始 ref=1。
如果失败：分配时 ref_count 不对。
```

#### 35. test_increment_ref
```
构造场景：allocate(1) 后 increment_ref。
关键操作：increment_ref(0)。
核心 assert：assert alloc.get_ref_count(0) == 1; 后 assert alloc.get_ref_count(0) == 2
验证的不变量：increment_ref 正确递增。
如果失败：ref_count 递增逻辑有 bug。
```

#### 36. test_free_decrements_ref_and_releases_at_zero
```
构造场景：ref=2 的 block，free 一次（ref=1，不释放），再 free 一次（ref=0，释放）。
关键操作：increment_ref -> free -> free。
核心 assert：
  第一次 free: num_free_blocks == 7（不释放）
  第二次 free: num_free_blocks == 8（释放）
验证的不变量：ref_count 递减语义正确的关键测试 —— block 只在 ref=0 时归还。
如果失败：ref_count 释放逻辑有 bug（提早释放或泄漏）。
```

#### 37. test_on_free_called_only_when_ref_reaches_zero
```
构造场景：带 on_free 回调，ref=2 -> free -> free。
关键操作：跟踪 events 列表。
核心 assert：第一次 free 后 events == []（ref=1 活着）；第二次 free 后 events == [("free", pid)]
验证的不变量：on_free 只在 ref=0 时调用。
如果失败：on_free 触发时机错误。
```

#### 38. test_double_free_is_safe — 关键：double free 不 crash
```
构造场景：allocate(1) -> free -> 再 free 同一 pid。
关键操作：两次 free 同一个 pid。
核心 assert：assert alloc.num_free_blocks == 8（最终状态正确，不 crash）
验证的不变量：double-free 安全，不 crash、不泄漏。
如果失败：double-free 导致 crash 或 block 计数错误。
```

### class TestBlockManagerPrefixCache — shared prefix ref_count 测试

#### 39. test_allocate_for_seq_without_cache
```
构造场景：第一个请求，cache 为空。
关键操作：allocate_for_seq。
核心 assert：
  seq.block_table == []
  mgr.get_shared_prefix_length("s0") == 0
验证的不变量：无缓存时 shared_prefix_length=0。
如果失败：prefix cache 初始状态错误。
```

#### 40. test_ensure_block_registers_in_cache
```
构造场景：分配 2 个 block。
关键操作：ensure_block 两次。
核心 assert：assert mgr.prefix_cache.size() == 2
验证的不变量：每次 ensure_block 自动注册到 prefix cache。
如果失败：cache 注册有漏。
```

#### 41. test_shared_prefix_blocks_not_written — 关键：相同 prompt 共享
```
构造场景：seqA 占用 block 0/1 后 seqB（相同 prompt）进来。
关键操作：allocate_for_seq(seqB)。
核心 assert：
  mgr.get_shared_prefix_length("sB") == 2
  seq_b.block_table == seq_a.block_table（相同 PID）
  entries[0].is_shared and entries[1].is_shared（标记为共享）
验证的不变量：相同 prompt 共享 block 且正确标记 shared。
如果失败：prefix sharing 失败。
```

#### 42. test_ref_count_increases_with_each_sharer
```
构造场景：seqA 占用 1 block(ref=1) -> seqB 相同 prompt(ref=2) -> seqC(ref=3)。
关键操作：递增共享者。
核心 assert：assert alloc.get_ref_count(0) == 3
验证的不变量：每多一个 sharer，ref_count +1。
如果失败：ref_count 共享计数有 bug。
```

#### 43. test_free_sharer_does_not_release_block
```
构造场景：seqA 和 seqB 共享一个 block(ref=2)，free(seqA) 后检查。
关键操作：mgr.free("sA")。
核心 assert：
  alloc.get_ref_count(0) == 1（seqB 仍然引用）
  alloc.num_free_blocks == 15（block 0 仍在使用）
  mgr.free("sB") 后 ref=0，free=16
验证的不变量：free 一个 sharer 不会释放 block（ref_count 保护）。
如果失败：共享 block 被提早释放。
```

#### 44. test_partial_prefix_match
```
构造场景：seqA=[1,2,3,4,10,11,12,13]，seqB=[1,2,3,4,20,21,22,23]。
关键操作：块 0 共享（相同 hash），块 1 不同。
核心 assert：
  mgr.get_shared_prefix_length("sB") == 1
  seqB 只有 block 0 prepopulated，block 1 靠 ensure_block on-demand 分配
验证的不变量：只共享匹配的前缀 block，不同部分不影响。
如果失败：partial prefix 匹配有 bug。
```

#### 45. test_is_block_shared
```
构造场景：seqB 共享后调用 is_block_shared(seq_b, 0)。
关键操作：is_block_shared(seq_b, 0/1/3)。
核心 assert：同一 block 内的所有位置都返回 True。
验证的不变量：is_block_shared 基于物理 block，不是 token 位置。
如果失败：位置判断有 bug。
```

#### 46. test_no_prefix_match_for_different_prompts
```
构造场景：seqA=[0..3]，seqB=[100..103]。
关键操作：get_shared_prefix_length + is_block_shared。
核心 assert：shared_prefix_length==0, is_block_shared==False。
验证的不变量：不同 prompt 无共享。
如果失败：hash 冲突导致错误共享。
```

#### 47. test_late_arrival_can_share
```
构造场景：seqA 运行到完成（block 释放），seqB 相同 prompt 后到。
关键操作：检查 cache 是否保留，ref 是否正常。
核心 assert：
  mgr.prefix_cache.size() == 1（cache entry 仍在，stale）
  mgr.get_shared_prefix_length("sB") == 0（stale entry 不共享）
  ensure_block(seq_b,0) 后 alloc.get_ref_count(0) == 1
验证的不变量：stale entry（ref=0）不被共享但可被重新注册。
如果失败：stale entry 处理有 bug。
```

#### 48. test_cache_persists_after_partial_free
```
构造场景：seqA+seqB 共享，free(seqA) 后 block 0 仍存活。
关键操作：free("sA") 后 ensure_block(seqB,0)。
核心 assert：
  alloc.get_ref_count(0) == 1
  alloc.num_free_blocks == 15（block 0 在用）
  pid == seq_a.block_table[0]（仍然同一物理 block）
验证的不变量：共享 block 在其他 sharer 存活时不释放。
如果失败：block 提前释放导致 pid 变化。


### class TestPrefixCacheProbe — 关键：probe 只读

#### 49. test_probe_returns_correct_cached_count
```
构造场景：seqA 填充 2 个 block 的 cache 后，probe 相同 prompt。
关键操作：mgr.probe_prefix_cache(tokens)。
核心 assert：
  probe.matched_block_count == 2
  probe.cached_token_count == 8（2 blocks * 4 block_size）
  len(probe.matched_physical_block_ids) == 2
验证的不变量：probe 返回正确的匹配计数。
如果失败：probe 返回值错误。
```

#### 50. test_probe_does_not_change_ref_count
```
构造场景：seqA 分配 block(ref=1) 后 probe 相同 prompt。
关键操作：probe_prefix_cache，前后比较 ref_count。
核心 assert：
  assert alloc.get_ref_count(0) == 1（probe 前）
  probe.matched_block_count == 1
  assert alloc.get_ref_count(0) == 1（probe 后，不变！）
  probe2 后 ref_count 仍然 == 1
验证的不变量：probe 是只读操作，不改变 ref_count。
如果失败：probe 错误地 increment 了 ref_count（会导致内存泄漏/block 无法释放）。
```

#### 51. test_allocate_for_seq_changes_ref_count_after_probe
```
构造场景：probe 后执行 allocate_for_seq。
关键操作：probe（只读）-> allocate_for_seq（实际 attach）。
核心 assert：
  probe: alloc.get_ref_count(0) == 1（probe 不改变）
  allocate_for_seq(seqB): alloc.get_ref_count(0) == 2（attach 才 increment）
验证的不变量：two-phase flow（Scheduler probe 只读 + BlockManager attach 递增）。
如果失败：admission 后 ref_count 未被正确递增。
```

#### 52. test_probe_stale_cache_entry_returns_zero
```
构造场景：seqA 填充 cache 后 free("sA")（block 释放，cache entry 仍存在）。
关键操作：probe_prefix_cache。
核心 assert：
  mgr.prefix_cache.size() == 1（cache entry 还在）
  alloc.get_ref_count(0) == 0（block 已释放）
  probe.matched_block_count == 0
  probe.cached_token_count == 0
  probe.matched_physical_block_ids == []
验证的不变量：stale entry（ref=0）不被 probe 误用。
如果失败：stale cache entry 被错误地用于前缀共享。
```

#### 53. test_probe_partial_prefix
```
构造场景：cache 有 [0,1,2,3,10,11,12,13]；probe [0,1,2,3,20,21,22,23]。
关键操作：probe_prefix_cache。
核心 assert：
  probe.matched_block_count == 1（只匹配第一块）
  probe.cached_token_count == 4
验证的不变量：probe 要求从索引 0 开始的连续匹配。
如果失败：非连续匹配也被计入。
```

#### 54. test_probe_empty_cache
```
构造场景：空 cache 的 probe。
关键操作：probe_prefix_cache。
核心 assert：matched_block_count==0, cached_token_count==0
验证的不变量：空 cache 返回 0。
如果失败：空 cache 返回非零。
```

### class TestSchedulerPrefixCache — Scheduler 集成

#### 55. test_cache_not_populated_yet
```
构造场景：第一次请求（cache 为空）。
关键操作：engine.step()。
核心 assert：
  result.cached_token_count == 0
  result.num_uncached_prefill_tokens > 0
  result.matched_block_count == 0
  result.num_prefill_tokens == result.num_uncached_prefill_tokens
验证的不变量：首次请求 cache miss，所有 token 都 uncached。
如果失败：首次请求错误地得到 cache hit。
```

#### 56. test_cache_hit_reduces_prefill_tokens_in_scheduler
```
构造场景：请求 A 填充 cache 后请求 B 相同 prompt 进来。
关键操作：engine.step()。
核心 assert：
  result.cached_token_count > 0
  result.matched_block_count > 0
验证的不变量：缓存命中后 scheduler 减少 prefill budget。
如果失败：cache hit 未反映在 ScheduleResult 中。
```

#### 57. test_cache_hit_compare_with_no_cache
```
构造场景：有 cache 的 engine（请求 A 填充后请求 B）vs 无 cache 的 engine（首次请求）。
关键操作：比较两个 scheduler result。
核心 assert：
  result_cache.cached_token_count > 0
  result_cache.num_uncached_prefill_tokens <= result_no_cache.num_prefill_tokens
验证的不变量：cache hit 减少了实际需要计算的 token 数。
如果失败：cache 未减少 prefill 工作量。
```

#### 58. test_identical_requests_metrics_show_cache_hits
```
构造场景：两个相同 prompt 的请求连续运行。
关键操作：MetricsCollector.report()。
核心 assert：
  "total_cached_tokens" in report
  "prefix_cache_hit_rate" in report
验证的不变量：metrics 报告 cache 命中统计。
如果失败：cache metrics 缺失。


## 测试文件：tests/test_scheduler.py（~12 tests）

### class TestScheduler

#### 59. test_admit_waiting_request
```
构造场景：1 个请求进入 queue。
关键操作：sched.schedule()。
核心 assert：
  len(result.scheduled_prefill_groups) == 1
  result.scheduled_prefill_groups[0].request_id == "r0"
  len(result.scheduled_decode_groups) == 0
验证的不变量：新请求被调度为 prefill。
如果失败：调度器不分配 prefill。
```

#### 60. test_prefill_then_decode
```
构造场景：admit -> prefill -> _simulate_prefill -> 再 schedule。
关键操作：_simulate_prefill 模拟 executor 将 seq 状态设为 RUNNING。
核心 assert：第二次 schedule 有 decode group，没有 prefill group。
验证的不变量：prefill 完成后 seq 自动进入 decode 阶段。
如果失败：prefill->decode 转换有 bug。
```

#### 61. test_finished_request_removed
```
构造场景：3-token prompt + max_new=2 -> admit -> prefill -> decode -> finish。
关键操作：重复 schedule，手动模拟 token 生成。
核心 assert：第三次 schedule 后 finished_groups 包含该请求。
验证的不变量：seq 生成足量 token 后自动标记 finished 并从队列移除。
如果失败：finished transition 有 bug。
```

#### 62. test_max_num_seqs_limit
```
构造场景：max_num_seqs=2，4 个请求进入 queue。
关键操作：sched.schedule()。
核心 assert：assert len(result.scheduled_prefill_groups) <= 2
验证的不变量：max_num_seqs 被尊重。
如果失败：超过并发限制被调度。
```

#### 63. test_ondemand_admits_without_allocating_blocks
```
构造场景：num_gpu_blocks=2，但 prompt 需要更多 block。
关键操作：sched.schedule()。
核心 assert：
  assert len(result.scheduled_prefill_groups) == 1（被调度）
  assert len(result.rejected_groups) == 0（不被拒绝）
  assert len(seq.block_table) == 0（block 已分配）
验证的不变量：admission 不检查 block 可用性，不预分配 block。Block 在 execution 时才 on-demand 分配。
如果失败：admission 时错误检查 block 或阻塞。
```

#### 64. test_sequence_id_format
```
构造场景：group "req-0000" 被调度。
关键操作：读取 seq_id。
核心 assert：assert seq.seq_id == "req-0000-seq-0"
验证的不变量：seq_id 格式为 "{request_id}-seq-{index}"。
如果失败：ID 格式不一致。
```

#### 65. test_schedule_result_token_counts
```
构造场景：prompt_len=4, max_new=2。
关键操作：schedule() -> _simulate_prefill -> schedule()。
核心 assert：
  第一次（prefill）: num_prefill_tokens=4, num_decode_tokens=0, num_batched_tokens=4
  第二次（decode）: num_prefill_tokens=0, num_decode_tokens=1, num_batched_tokens=1
验证的不变量：ScheduleResult 的 token 计数正确。
如果失败：token 计数有 bug。
```

#### 66. test_ignored_when_budget_full
```
构造场景：max_num_seqs=2, max_num_batched_tokens=8。r0 prefill 中 -> r1, r2 等待。
关键操作：r1 被调度，r2 被 ignore。
核心 assert：
  len(result.scheduled_prefill_groups) == 1（只有 r1 被调度）
  len(result.ignored_groups) == 1（r2 被 ignore）
  result.ignored_groups[0].request_id == "r2"
验证的不变量：budget 满时新请求被 ignore 而非 reject。
如果失败：ignore 逻辑有 bug。
```

#### 67. test_decode_first
```
构造场景：max_num_seqs=1。r0 decode 中（占用唯一 slot），r1 新请求等待。
关键操作：schedule。
核心 assert：
  len(r2.scheduled_decode_groups) == 1（r0 decode 被调度）
  r2.scheduled_decode_groups[0].request_id == "r0"
  len(r2.scheduled_prefill_groups) == 0（r1 没有被调度为 prefill）
  len(r2.ignored_groups) == 1（r1 被 ignore）
  r2.ignored_groups[0].request_id == "r1"
验证的不变量：decode 序列优先于 prefill 序列（decode-first 语义）。
如果失败：decode 被 prefill 抢占。
```

#### 68. test_chunked_prefill
```
构造场景：prompt_len=12, max_prefill_chunk_size=4, chunked_prefill_enabled=True。
关键操作：分 3 次 schedule，每次都手动推进 prefill_cursor，最后 _simulate_prefill。
核心 assert：
  Step 1: num_prefill_tokens=4, seq.status==PREFILL（第 1 个 chunk）
  Step 2: num_prefill_tokens=4（第 2 个 chunk）
  Step 3: num_prefill_tokens=4（第 3 个 chunk）
  Step 4: num_decode_tokens=1（进入 decode）
  中间步骤 decode_tokens==0（prefill 未完成时不解码）
验证的不变量：长 prompt 在多个 prefill 步中分 chunk 执行。prefill 完成前不解码。
如果失败：chunked prefill 逻辑有 bug。
```

#### 69. test_prefill_not_finished_not_decode
```
构造场景：prompt_len=8, chunk_size=4。partial prefill（cursor=4）。
关键操作：prefill_cursor 到达 4 但 is_prefill_finished=False。
核心 assert：
  len(r2.scheduled_decode_groups) == 0（没进入 decode）
  len(r2.scheduled_prefill_groups) == 1（仍在 prefill）
验证的不变量：partial prefill 的序列不提前进入 decode。只有 cursor 到达 prompt 末尾才算 prefill 完成。
如果失败：partial prefill 错误进入 decode。
```

#### 70. test_ignored_reasons
```
构造场景：max_num_seqs=1，r1 在 budget 满时被 ignore。
关键操作：检查 ignored_reasons 字典。
核心 assert：
  len(result.ignored_groups) == 1
  result.ignored_groups[0].request_id == "r1"
  result.ignored_reasons["r1"] != ""
验证的不变量：ignore 的请求有明确原因。
如果失败：ignored_reasons 缺失或为空。


## 测试文件：tests/test_engine.py（~9 tests）

### class TestEngine

#### 71. test_engine_runs_requests_to_completion
```
构造场景：2 个请求（Hello world + CUDA batching），4-block 池。
关键操作：run_until_done()。
核心 assert：
  assert len(outputs) == 2
  assert all isinstance(text, str) and len(text) > 0
  assert engine.queue.num_waiting == 0
  assert engine.queue.num_running == 0
验证的不变量：完整 engine loop 能正常运行两个请求到完成，队列最终为空。
如果失败：engine loop 不能正常运行。
```

#### 72. test_continuous_batching_new_arrival
```
构造场景：1 个请求，8-block 池。
关键操作：run_until_done()。
核心 assert：assert len(outputs) == 1
验证的不变量：单请求也能正常运行。
如果失败：单请求场景出错。
```

#### 73. test_mid_arrival_merge
```
构造场景：请求 A + B 一起加入，跑 3 步后加入请求 C，继续跑完。
关键操作：先 run_until_done，中途 add_request("C")。
核心 assert：
  assert len(outputs) == 3
  assert all output non-empty
验证的不变量：新请求可以在已有请求运行时被无缝加入（continuous batching 的关键能力 —— mid-arrival merge）。
如果失败：mid-arrival 后 crash 或输出错误。
```

#### 74. test_ondemand_oom_during_execution
```
构造场景：num_gpu_blocks=2, 2 个请求各需 >2 block。
关键操作：run_until_done()。
核心 assert：with pytest.raises(RuntimeError, match="OOM"):
验证的不变量：block 耗尽时 raise RuntimeError，不 segfault/挂起。
如果失败：OOM 时不 raise 或 crash。
```

#### 75. test_engine_step_returns_schedule_result
```
构造场景：1 个请求，1 个 step。
关键操作：engine.step()。
核心 assert：
  assert result is not None
  assert prefill_groups >= 1 or finished_groups >= 1
验证的不变量：engine.step() 返回有效的 ScheduleResult。
如果失败：step() 返回 None。
```

#### 76. test_sequence_created_for_each_request
```
构造场景：request -> add_request -> run_until_done。
关键操作：get_by_id(rid)。
核心 assert：seq.num_output_tokens > 0; seq.status.name == "FINISHED"
验证的不变量：add_request 创建 SequenceGroup，run 后 seq 被标记 finished。
如果失败：sequence 创建或状态更新有 bug。
```

#### 77. test_schedule_result_fields
```
构造场景：1 个请求，1 个 step。
关键操作：检查 result 字段。
核心 assert：以下字段全部存在：
  scheduled_prefill_groups, scheduled_decode_groups,
  ignored_groups, finished_groups, rejected_groups, preempted_groups,
  num_batched_tokens, num_prefill_tokens, num_decode_tokens,
  token_budget_remaining, debug_reason, ignored_reasons
  cached_token_count, num_uncached_prefill_tokens, matched_block_count
验证的不变量：ScheduleResult 包含所有必要字段（含 prefix cache 字段）。
如果失败：ScheduleResult 字段缺失。
```

#### 78. test_executor_kv_writes_are_tracked
```
构造场景：1 个请求，1 个 step。
关键操作：executor.get_kv_stats()。
核心 assert：
  kv_stats["allocated_blocks"] > 0
  kv_stats["kv_tokens_written"] > 0
验证的不变量：fake executor 正确报告 KV 写入统计。
如果失败：KV 统计未追踪。
```

#### 79. test_executor_kv_affects_output
```
构造场景：1 个请求（prompt="AAA"）。
关键操作：run_until_done()。
核心 assert：
  assert len(text) == 2（2 个输出 token）
  assert engine.executor.get_kv_stats()["kv_tokens_written"] > 0
验证的不变量：KV cache 写入发生在 decode 阶段，影响最终输出。
如果失败：KV 写入在 decode 时未发生。


## 测试文件：tests/test_serving_layer.py（~20 tests）

### class TestGenerate

#### 80. test_generate_non_stream_success
```
构造场景：ServingLayer 生成非流式响应。
关键操作：sv.generate("Hello", max_tokens=4, stream=False)。
核心 assert：resp.error is None; len(resp.text) > 0; resp.request_id 以 "sv-" 开头。
验证的不变量：非流式生成成功返回有效文本和 ID。
如果失败：非流式生成有 bug。
```

#### 81. test_generate_empty_prompt_rejected
```
构造场景：空 prompt 的生成请求。
关键操作：sv.generate("")。
核心 assert：resp.error_code == "PROMPT_TOO_LONG"
验证的不变量：空 prompt 被 admission control 拒绝。
如果失败：空 prompt 未被拒绝。
```

#### 82. test_generate_max_tokens_zero
```
构造场景：max_tokens=0 的生成请求。
关键操作：sv.generate("Hello", max_tokens=0)。
核心 assert：resp.error is None
验证的不变量：max_tokens=0 是合法输入（生成 0 个 token）。
如果失败：max_tokens=0 被错误拒绝。
```

### class TestStreaming

#### 83. test_streaming_basic
```
构造场景：流式生成 4 个 token。
关键操作：sv.generate_stream -> sv.poll_stream 轮询到完成。
核心 assert：
  assert engine_rid is not None
  assert err is None
  assert len(tokens) > 0
  assert finish_reason == "length" or "finished"
验证的不变量：流式生成正确产生 token，finish_reason 有效。
如果失败：流式生成有 bug。
```

#### 84. test_streaming_token_accumulation
```
构造场景：流式生成，检查第一个 token 不为空。
关键操作：poll_stream 循环读取直到 finished。
核心 assert：assert first_token is not None
验证的不变量：流式生成至少产生一个 token。
如果失败：流式生成没有产生 token。
```

### class TestRateLimit

#### 85. test_rpm_rejects_after_limit
```
构造场景：rate_limit_rpm=2，发送多个请求。
关键操作：sv.generate 多次。
核心 assert：前 2 次成功，第 3 次...（测试只检查前 2 次成功，RPM 检查依赖 rate_limiter）。
验证的不变量：RPM 限制生效。
```

#### 86. test_rpm_limit_resets
```
构造场景：rate_limit_rpm=1。
关键操作：1 个请求通过，检查 rate limiter 有计数。
核心 assert：assert sv.rate_limiter._rpm.count() > 0
验证的不变量：rate limiter 内部计数正确。
```

### class TestAdmissionControl

#### 87. test_prompt_too_long
```
构造场景：max_model_len=4，但 prompt 很长。
关键操作：sv.generate("Hello world this is too long")。
核心 assert：resp.error_code == "PROMPT_TOO_LONG"
验证的不变量：过长 prompt 被拒绝。
如果失败：过长 prompt 未被拒绝或 crash。
```

#### 88. test_queue_overflow
```
构造场景：max_queue_len=2，engine.queue 已经有 2 个 waiting + 1 running。
关键操作：sv.generate("D")。
核心 assert：resp.error_code == "QUEUE_OVERFLOW"
验证的不变量：队列满时拒绝而非 crash。
如果失败：queue overflow 未被捕获。
```

#### 89. test_block_exhausted
```
构造场景：num_gpu_blocks=2。
关键操作：sv.generate 一个请求。
核心 assert：resp.error_code == "BLOCK_EXHAUSTED"
验证的不变量：block 不足时 admission 返回 BLOCK_EXHAUSTED。
如果失败：block 不足时未正确拒绝。
```

### class TestStreamManager

#### 90. test_max_streams_cap
```
构造场景：max_num_streams=2。
关键操作：try_acquire 3 次。
核心 assert：前 2 次成功，第 3 次 False。release 后可以再 acquire。
验证的不变量：stream 并发上限正确。
如果失败：stream 限制有 bug。
```

#### 91. test_stream_manager_counts
```
构造场景：acquire + release。
关键操作：检查 active_count。
核心 assert：0 -> 1 -> 0。
验证的不变量：stream manager 计数正确。
```

### class TestCancel

#### 92. test_cancel_running_request
```
构造场景：running 中的请求被取消。
关键操作：engine.cancel_request(engine_rid)。
核心 assert：
  assert ok（取消成功）
  alloc.num_free_blocks == alloc.num_total_blocks（所有 block 归还）
验证的不变量：cancel 后 block 全部释放。
如果失败：cancel 后 block 泄漏。
```

#### 93. test_cancel_non_existent
```
构造场景：取消不存在的 ID。
关键操作：sv.cancel("does-not-exist")。
核心 assert：assert not ok
验证的不变量：取消不存在的 ID 返回 False 而非 crash。
如果失败：取消不存在 ID 时 crash。
```

### class TestTimeout

#### 94. test_timeout_cancels_request
```
构造场景：request_timeout_s=0.001，sleep 0.002 后 check_timeouts。
关键操作：engine.engine_core._check_timeouts()。
核心 assert：assert sv.engine.queue.num_running == 0
验证的不变量：过期请求被自动清理。
如果失败：timeout 未触发或未清理。


### class TestMetrics

#### 95. test_metrics_endpoint_returns_json
```
构造场景：生成 1 个请求后获取 metrics。
关键操作：sv.get_metrics()。
核心 assert：metrics JSON 包含 "total_requests" 和 "block_utilization"。
验证的不变量：serving metrics 端点返回有效 JSON。
如果失败：metrics 端点损坏。
```

#### 96. test_metrics_after_rejection
```
构造场景：被拒绝的请求。
关键操作：sv.get_metrics()。
核心 assert：JSON 包含 "rejected_requests"。
验证的不变量：rejection 被记录到 metrics。
```

#### 97. test_metrics_after_cancel
```
构造场景：取消请求后。
关键操作：sv.get_metrics()。
核心 assert：JSON 包含 "cancelled_requests"。
验证的不变量：cancel 被记录到 metrics。
```

### class TestDisconnect

#### 98. test_streaming_disconnect_abort
```
构造场景：流式生成中 _abort 模拟 client disconnect。
关键操作：sv._abort(engine_rid, tracking_id) + stream_manager.release(tracking_id)。
核心 assert：
  alloc.num_free_blocks == alloc.num_total_blocks
  sv.engine.queue.num_running == 0
  sv.engine.queue.num_waiting == 0
验证的不变量：disconnect 后 block 和队列全部清空。
如果失败：disconnect 后资源泄漏。
```

#### 99. test_streaming_disconnect_running_queue_cleared
```
构造场景：4 个流同时被 abort。
关键操作：批量 _abort + release。
核心 assert：queue.num_running==0, queue.num_waiting==0。
验证的不变量：批量 disconnect 全部清理。
如果失败：批量 disconnect 有遗漏。
```

#### 100. test_streaming_disconnect_blocks_returned
```
构造场景：abort 前后比较 used_blocks。
关键操作：used_before > 0（确认有 block 分配）-> abort -> used_after == 0。
核心 assert：alloc.num_free_blocks == alloc.num_total_blocks。
验证的不变量：disconnect 后所有 block 归还。
如果失败：block 泄漏。
```

#### 101. test_streaming_disconnect_metrics_cancelled
```
构造场景：abort 后检查 metrics 的 cancelled 计数。
关键操作：mc.report()。
核心 assert：report.get("cancelled_requests", 0) >= 1。
验证的不变量：disconnect 递增 cancelled 计数。
如果失败：disconnect 未记录到 metrics。
```

#### 102. test_streaming_disconnect_no_double_abort
```
构造场景：正常完成后再次 _abort。
关键操作：先 poll_stream 完成，再 sv._abort。
核心 assert：cancelled_after == cancelled_before（计数不变）。
验证的不变量：double abort 是 no-op，不重复计数。
如果失败：double abort 导致 metrics 重复计数。
```

#### 103. test_generate_stream_safe_finishes_normally
```
构造场景：generate_stream_safe 迭代器正常使用。
关键操作：for token_text, ... in sv.generate_stream_safe("Hello"):
核心 assert：assert len(tokens) > 0。
验证的不变量：安全生成器正常产生 token。
```

#### 104. test_generate_stream_safe_abort_on_close
```
构造场景：generate_stream_safe 迭代器在途中被 close()。
关键操作：gen.close()。
核心 assert：alloc.num_free_blocks == alloc.num_total_blocks。
验证的不变量：generator close() 触发 abort，block 被释放。
如果失败：generator close 后 block 泄漏。
```

#### 105. test_streaming_disconnect_comprehensive
```
构造场景：最完整的 disconnect 测试 —— 同时验证队列、block、table、stream slot、metrics。
关键操作：generate_stream_safe -> break -> gen.close()。
核心 assert：
  queue.num_running == 0, queue.num_waiting == 0（队列清空）
  alloc.num_free_blocks == alloc.num_total_blocks（block 归还）
  len(sv.engine.block_manager._tables) == 0（table 清空）
  sv.stream_manager.active_count == 0（stream slot 释放）
  cancelled_after == cancelled_before + 1（metrics 递增）
验证的不变量：所有 5 个资源类别全部正确清理。
如果失败：任何一个资源维度泄漏。
```

#### 106. test_disconnect_after_engine_finished
```
构造场景：引擎已标记 finished 后 client 才 disconnect。
关键操作：先 poll_stream 等待完成，再 _abort。
核心 assert：
  cancelled_after == cancelled_before（不重复计数）
  sv.stream_manager.active_count == 0（stream slot 已释放）
验证的不变量：引擎先完成 + client 后断开 = no-op，不重复计数。
如果失败：after-finish disconnect 泄漏 metrics 计数。


## 测试文件：tests/test_fault_injection.py（~15 tests）

### class TestQueueOverflow

#### 107. test_queue_overflow_rejects_new_request
```
构造场景：max_queue_len=2, max_num_seqs=1。engine.queue 填充到 3（1 running + 2 waiting）。
关键操作：sv.generate("D")。
核心 assert：resp.error_code == "QUEUE_OVERFLOW"
验证的不变量：队列满时返回错误码，不 crash。
如果失败：队列满时 crash 或错误处理缺失。
```

#### 108. test_queue_accepts_after_drain
```
构造场景：请求完成队列清空后。
关键操作：sv.engine.run_until_done() -> sv.generate("B")。
核心 assert：resp.error is None
验证的不变量：队列 drain 后恢复正常接受。
如果失败：队列 drain 后不能恢复（状态残留）。
```

### class TestBlockExhaustion

#### 109. test_block_exhaustion_rejects_new
```
构造场景：num_gpu_blocks=2。
关键操作：sv.generate("Hello world", max_tokens=16)。
核心 assert：resp.error_code == "BLOCK_EXHAUSTED"
验证的不变量：block 不足时 admission 正确返回错误码。
如果失败：block 短缺时 crash。
```

#### 110. test_block_exhaustion_does_not_crash
```
构造场景：num_gpu_blocks=1（极端短缺）。
关键操作：sv.generate("Test")。
核心 assert：assert resp.error_code is not None（不 crash）。
验证的不变量：即使只有 1 个 block，系统也不 crash。
如果失败：极端资源不足时 crash。
```

#### 111. test_blocks_recovered_after_finish
```
构造场景：请求完成后。
关键操作：free_before = alloc.num_free_blocks -> sv.generate -> alloc.num_free_blocks。
核心 assert：alloc.num_free_blocks >= free_before - 4（block 被回收至大致水平）。
验证的不变量：请求完成后 block 回到 free pool。
如果失败：block 泄漏。
```

### class TestStreamExhaustion

#### 112. test_stream_exhaustion_rejects
```
构造场景：max_num_streams=2，全部占用后继续 try_acquire。
关键操作：stream_manager.try_acquire。
核心 assert：第 3 次 assert not sv.stream_manager.try_acquire("s3")
验证的不变量：stream 并发上限正确。
```

#### 113. test_stream_release_recovers_slot
```
构造场景：release 后再次 acquire。
关键操作：try_acquire -> release -> try_acquire。
核心 assert：第 3 次成功。
验证的不变量：release 正确回收 slot。
```

### class TestTimeoutStorm

#### 114. test_timeout_releases_blocks
```
构造场景：4 个请求全部 timeout（request_timeout_s=0.001）。
关键操作：time.sleep -> engine.step -> _check_timeouts。
核心 assert：
  sv.engine.queue.num_running == 0
  sv.engine.queue.num_waiting == 0
  alloc.num_free_blocks == alloc.num_total_blocks
验证的不变量：大量 timeout 后队列和 block 全部清理，无泄漏。
如果失败：timeout storm 导致资源泄漏。
```

#### 115. test_timeout_metrics_updated
```
构造场景：1 个请求 timeout。
关键操作：_check_timeouts() 后检查 metrics。
核心 assert：report.get("timeout_requests", 0) > 0
验证的不变量：timeout 事件被记录到 metrics。
如果失败：timeout metrics 缺失。
```

### class TestCancelStorm

#### 116. test_cancel_releases_blocks_to_pool
```
构造场景：4 个请求全部取消。
关键操作：逐一 cancel_request。
核心 assert：alloc.num_free_blocks == alloc.num_total_blocks
验证的不变量：大量取消后 block 全部归还。
如果失败：cancel 后 block 泄漏。
```

#### 117. test_cancel_ref_count_integrity
```
构造场景：2 个相同 prompt 的请求（触发 prefix cache 共享），全部取消。
关键操作：cancel_request -> alloc.dump_ref_counts()。
核心 assert：assert all(rc == 0 for rc in ref_counts)（所有 ref_count=0）
验证的不变量：prefix cache 共享后 cancel 仍能正确清理所有 ref_count。
如果失败：共享 block 的 ref_count 未被正确递减（泄漏）。
```

### class TestPrefixCacheStaleEntry

#### 118. test_stale_entry_not_falsely_used
```
构造场景：请求 A 运行到完成（block 释放，cache entry 保留）。probe 相同 prompt。
关键操作：probe_prefix_cache。
核心 assert：
  sv.engine.block_manager.prefix_cache.size() > 0（cache entry 仍存在）
  probe.cached_token_count == 0（stale entry 被正确排除）
验证的不变量：stale cache entry（ref=0）不被 probe 误用。
如果失败：stale entry 被错误地视为有效 cache hit。
```

#### 119. test_stale_entry_new_request_recreates
```
构造场景：请求 A 运行到完成（block 释放），请求 B 相同 prompt 后到。
关键操作：run_until_done 两次。
核心 assert：assert sv.engine.queue.num_finished >= 2（系统不 crash）
验证的不变量：stale entry 存在时新请求仍能正常工作。
如果失败：stale entry 导致系统 crash。


### class TestAdmissionBlockPressure

#### 120. test_10blocks_100_long_all_rejected
```
构造场景：10 blocks, 100 个正常长度请求（每个需要约 17 blocks）。
关键操作：循环 sv.generate。
核心 assert：assert len(exhausted) == 100（全部被拒绝）。
验证的不变量：长期请求下 admission control 保护系统不 OOM。
如果失败：大量请求下的 admission 失效。
```

#### 121. test_10blocks_100_short_uses_blocks_then_admission_blocks
```
构造场景：10 blocks, 100 个短请求。先通过 engine.add_request 绕过 admission 消耗 block。
关键操作：engine.add_request 所有请求 -> engine.step() -> sv.generate。
核心 assert：resp.error_code == "BLOCK_EXHAUSTED"。
验证的不变量：admission control 在 block 压力下正确拒绝。
如果失败：admission block pressure 检测失效。
```

#### 122. test_10blocks_no_admission_control_crashes
```
构造场景：10 blocks，绕过 admission control 直接 add_request 20 个请求。
关键操作：循环 engine.step()。
核心 assert：with pytest.raises(RuntimeError, match="OOM"):
验证的不变量：无 admission control 时系统最终 OOM crash（证明 admission control 的必要性）。
如果失败：无 admission 时系统不 OOM。
```

## 测试文件：tests/test_metrics.py（~39 tests）

### class TestTTFT

#### 123. test_ttft_field_exists_and_non_negative
```
构造场景：1 个请求，2 tokens。
关键操作：report = mc.report()。
核心 assert：report 包含 avg/min/max_ttft_ms，均 >= 0。
验证的不变量：TTFT 字段存在且非负。
如果失败：TTFT 字段缺失或错误。
```

#### 124. test_first_token_time_after_arrival
```
构造场景：1 个请求，4 tokens。
关键操作：遍历 mc._finished_seqs 中 FINISHED 的 seq。
核心 assert：seq.first_token_time >= seq.arrival_time（时间顺序正确）。
验证的不变量：first_token_time 不会早于 arrival_time。
如果失败：时间戳记录顺序错乱。
```

#### 125. test_multiple_requests_all_have_ttft
```
构造场景：4 个请求。
关键操作：mc.report()。
核心 assert：finished_count == 4, report["avg_ttft_ms"] > 0。
验证的不变量：每个 finished 请求都有 TTFT。
如果失败：TTFT 缺失或为 0。
```

#### 126. test_ttft_min_max_avg_ordering
```
构造场景：1 个请求，8 tokens。
关键操作：report。
核心 assert：min_ttft_ms <= avg_ttft_ms <= max_ttft_ms。
验证的不变量：min/max/avg 顺序正确。
如果失败：聚合逻辑有 bug。
```

### class TestTPOT

#### 127. test_tpot_field_exists_and_non_negative
```
构造场景：1 个请求。
关键操作：report。
核心 assert：avg/min/max_tpot_ms >= 0。
```

#### 128. test_tpot_generated_tokens_count
```
构造场景：expected_tokens=6。
关键操作：遍历 _finished_seqs。
核心 assert：seq.num_output_tokens == expected_tokens。
验证的不变量：输出 token 数与 max_new_tokens 一致。
```

#### 129. test_tpot_denominator_is_output_tokens_not_prompt
```
构造场景：区分 prompt_length 和 num_output_tokens。
关键操作：手动计算 expected_tpot = decode_time / (num_output_tokens - 1)。
核心 assert：expected_tpot >= 0; num_output_tokens != prompt_length。
验证的不变量：TPOT 分母是 output tokens-1，不是 prompt length。
如果失败：TPOT 公式用错了分母。
```

#### 130. test_tpot_single_token_output
```
构造场景：max_tokens=1（单 token 输出，无 inter-token gap）。
关键操作：report。
核心 assert：report["avg_tpot_ms"] == 0.0。
验证的不变量：单 token 输出被排除在 TPOT 聚合外。
如果失败：单 token 输出错误计入 TPOT。
```

#### 131. test_tpot_multiple_tokens_have_positive_tpot
```
构造场景：max_tokens=4。
关键操作：report。
核心 assert：report["avg_tpot_ms"] > 0。
验证的不变量：多 token 输出有正 TPOT。
```

#### 132. test_tpot_min_max_avg_ordering
```
构造场景：2 个请求（4 tokens + 6 tokens）。
关键操作：report。
核心 assert：min_tpot_ms <= avg_tpot_ms <= max_tpot_ms。
验证的不变量：TPOT 聚合顺序正确。
```

### class TestThroughput

#### 133. test_throughput_fields_exist
```
核心 assert：report 包含 throughput_req_per_sec, throughput_tok_per_sec, total_output_tokens, total_prompt_tokens。
```

#### 134. test_throughput_with_completed_requests
```
构造场景：2 个请求。
核心 assert：total_requests==2, throughput_req_per_sec > 0。
验证的不变量：多请求 throughput 正数。
```

#### 135. test_token_throughput_with_generated_tokens
```
构造场景：1 个请求，8 tokens。
核心 assert：total_output_tokens > 0, throughput_tok_per_sec > 0。
```

#### 136. test_total_time_non_negative
```
构造场景：1 个请求。
核心 assert：total_time_seconds >= 0。
```

#### 137. test_throughput_formula_consistency
```
构造场景：1 个请求，4 tokens。
关键操作：手动计算 throughput = completed / (max(finishes) - min(arrivals)) 与 report 比较。
核心 assert：abs(report[...] - manual) < 0.01。
验证的不变量：report 的 throughput 与公式计算结果一致。
如果失败：throughput 实现与公式不一致。


#### 138. test_throughput_staggered_arrival
```
构造场景：max_num_seqs=1。请求 A 先完成，sleep(0.02)，请求 B 后完成。
关键操作：time.sleep(0.02) 制造 idle gap。
核心 assert：
  wall_elapsed = max(finishes) - min(arrivals)（全局最早到最晚完成）
  max_per_request = max(f - a for each)（每请求最大延迟，不含 gap）
  assert wall_elapsed > max_per_request（gap 被 wall_elapsed 包含）
  assert report["throughput_req_per_sec"] ≈ completed / wall_elapsed
  assert report["throughput_tok_per_sec"] ≈ total_tok / wall_elapsed
验证的不变量：throughput 分母是全局 wall-clock（包含 idle gap），而非 max per-request 延迟。
如果失败：throughput 公式用了错误的分母（高估 staggered arrival 场景的吞吐）。
```

### class TestMetricsNoDoubleCount — 关键：cancelled/timeout 不污染 throughput

#### 139. test_cancelled_not_in_completed
```
构造场景：engine.add_request -> engine.step -> engine.cancel_request -> run_until_done。
关键操作：取消一个正在运行的请求。
核心 assert：
  mc._cancelled_requests >= 1
  report["avg_ttft_ms"] == 0.0（无 FINISHED 序列）
  report["avg_tpot_ms"] == 0.0
  report["total_requests"] == 0（cancelled 不算）
验证的不变量：cancelled 请求不贡献 TTFT、TPOT、throughput count。
如果失败：cancelled 请求错误计入 metrics。
```

#### 140. test_timeout_not_in_completed
```
构造场景：request_timeout_s=0.001, time.sleep(0.01), engine.step 触发 _check_timeouts。
关键操作：step() 在 loop 开始时内部调用 _check_timeouts()。
核心 assert：
  mc._timeout_requests >= 1
  report["total_requests"] == 0
验证的不变量：timeout 请求不贡献 total_requests。
如果失败：timeout 请求错误计入 metrics。
```

#### 141. test_finished_seqs_only_contains_terminal
```
构造场景：正常请求完成后调用 count_cancelled / count_timeout。
关键操作：report()。
核心 assert：report["total_requests"] >= 1, report["avg_ttft_ms"] > 0。
验证的不变量：正常请求正常计入 metrics。
```

#### 142. test_cancelled_generated_tokens_not_summed
```
构造场景：请求取消后检查 total_output_tokens。
关键操作：report()。
核心 assert：
  for seq in mc._finished_seqs: assert seq.status != Status.FINISHED
  report["total_output_tokens"] >= 0
验证的不变量：cancelled 请求的输出 token 不求和到 total_output_tokens。
如果失败：cancelled token 错误计入 throughput。
```

### class TestKVBlockUtilization

#### 143. test_allocated_blocks_during_run
```
核心 assert：alloc.num_used_blocks > 0（prefill 后至少 1 个 block）。
```

#### 144. test_blocks_freed_after_completion
```
核心 assert：alloc.num_free_blocks == alloc.num_total_blocks（全部归还）。
```

#### 145. test_kv_utilization_range
```
核心 assert：0 <= kv_util_peak_pct <= 100, 0 <= kv_util_avg_pct <= 100。
```

#### 146. test_kv_util_fields_exist
```
核心 assert：report 包含 kv_total_blocks, kv_peak_blocks, kv_util_peak_pct, kv_util_avg_pct。
```

### class TestSchedulerLatency

#### 147. test_scheduler_latency_fields_exist
```
核心 assert：report 包含 avg/max_scheduler_latency_ms。
```

#### 148. test_scheduler_latency_count_positive
```
核心 assert：len(mc._scheduler_times) > 0。
验证的不变量：多步后 scheduler 被采样多次。
```

#### 149. test_scheduler_latency_non_negative
```
核心 assert：avg >= 0; max >= 0。
```

#### 150. test_max_gte_avg
```
核心 assert：max_scheduler_latency_ms >= avg_scheduler_latency_ms。
```

### class TestPrefixCacheMetrics

#### 151. test_prefix_cache_fields_exist
```
核心 assert：report 包含 total_cached_tokens, prefix_cache_hit_rate。
```

#### 152. test_hit_rate_out_of_total_prompt_tokens
```
关键操作：手动计算 expected_rate = sum(_timeline_cached) / total_prompt_tokens * 100。
核心 assert：report["prefix_cache_hit_rate"] == pytest.approx(expected_rate, abs=0.1)。
验证的不变量：hit rate 公式正确。
```

#### 153. test_cache_hit_reduces_prefill
```
构造场景：请求 A 填充 cache（保持 alive）后请求 B 同 prompt。
核心 assert：result.cached_token_count > 0。
验证的不变量：engine 集成下 cache hit 减少 prefill budget。
```

### class TestStageProfilerMetrics

#### 154. test_percent_of_total_sum
```
核心 assert：sub_total >= 0（不验证精确和，只验不 cras h）。
```

#### 155. test_engine_step_total_is_largest
```
核心 assert：每个 stage 的 total_ms <= engine_step_total * 1.05。
验证的不变量：engine_step_total 是最大的 stage。
```

#### 156. test_empty_profiler_no_crash
```
构造场景：新建 StageProfiler，不 record 直接 report。
核心 assert：
  report["total_profiled_ms"] == 0.0
  report["total_requests"] == 0
  report["total_engine_steps"] == 0
```

#### 157. test_exception_in_context_manager
```
构造场景：with p.record("scheduler_step") 内 raise ValueError。
关键操作：exception 后再 record 一次。
核心 assert：report["stages"]["scheduler_step"]["count"] == 2。
验证的不变量：exception 后 profiler 状态不损坏，后续 record 正常工作。
如果失败：exception 破坏 profiler 内部状态（count 错误或 crash）。
```

#### 158. test_profiler_stages_present
```
核心 assert：report 的 stages 包含 {"engine_step_total", "scheduler_step", "metrics_update"}。
```

### class TestServingCounters

#### 159. test_counters_are_separate
```
关键操作：分别 count_rejected, count_cancelled, count_timeout, count_rpm_rejected, count_tpm_rejected。
核心 assert：每个 counter 各自 = 1。
验证的不变量：各 counter 独立，不互相影响。
```

#### 160. test_rejected_not_in_total_requests
```
核心 assert：total_requests==0, rejected_requests==1。
```

#### 161. test_cancelled_not_in_total_requests
```
核心 assert：total_requests==0, cancelled_requests==1。
```

### class TestServingMetricsEndpoint

#### 162. test_metrics_endpoint_has_ttft
```
核心 assert：metrics JSON 包含 "ttft", "tpot", "throughput"。
```

#### 163. test_metrics_endpoint_block_util_range
```
核心 assert：0 <= data["block_utilization"] <= 100。
```


## 测试文件：tests/test_stage_profiler.py（~15 tests）

### class TestStageProfiler

#### 164. test_record_single_stage
```
构造场景：StageProfiler().record("prefill") context manager，sleep 1ms。
关键操作：with p.record("prefill"): time.sleep(0.001)。
核心 assert：
  "prefill" in report["stages"]
  report["stages"]["prefill"]["count"] == 1
  report["stages"]["prefill"]["total_ms"] > 0
验证的不变量：context manager 正确记录 duration。
如果失败：record 不记录或记录为零。
```

#### 165. test_stats_count_total_avg_max
```
构造场景：record_raw("decode", 0.010/0.020/0.030) 三次。
关键操作：record_raw。
核心 assert：
  stage["count"] == 3
  stage["total_ms"] == 60.0
  stage["avg_ms"] == 20.0
  stage["max_ms"] == 30.0
验证的不变量：聚合统计（count/total/avg/max）正确。
如果失败：聚合计算有 bug。
```

#### 166. test_context_manager_exception_handling
```
构造场景：with p.record("scheduler_step") 内 raise ValueError。
关键操作：exception 后再 record 一次。
核心 assert：report["stages"]["scheduler_step"]["count"] == 2。
验证的不变量：exception 不破坏 profiler，后续 record 仍然有效。
如果失败：exception 后 profiler 不可用。
```

#### 167. test_empty_report_no_error
```
构造场景：新建 profiler，不 record 任何内容。
关键操作：report()。
核心 assert：
  report["stages"] == {}
  report["total_profiled_ms"] == 0.0
  report["total_requests"] == 0
  report["total_engine_steps"] == 0
验证的不变量：空 profiler 输出有效空报告，不 crash。
如果失败：空 profiler report crash。
```

#### 168. test_reset_clears_data
```
构造场景：record -> increment -> reset。
关键操作：p.reset()。
核心 assert：report["stages"] == {}, total_requests==0, total_engine_steps==0。
验证的不变量：reset 正确清除所有状态。
```

#### 169. test_start_end_api
```
构造场景：p.start("kv_cache_allocation") -> sleep -> p.end("kv_cache_allocation")。
核心 assert：stage["count"] == 1。
验证的不变量：start/end API 也能记录 timing。
```

#### 170. test_start_without_end_ignored
```
构造场景：p.start("prefix_cache_lookup") 后不 end。
核心 assert："prefix_cache_lookup" not in report["stages"]（不记录不完整的 timing）。
验证的不变量：start 无 end 的记录被忽略。
如果失败：未完成的非对称 record 被错误计入。
```

#### 171. test_percent_of_total
```
构造场景：record_raw("decode", 0.030), record_raw("engine_step_total", 0.100)。
核心 assert：report["stages"]["decode"]["percent_of_total"] == 30.0。
验证的不变量：percentage 计算 = stage_total / engine_step_total * 100。
如果失败：percentage 计算错误。
```

#### 172. test_increment_requests_and_steps
```
构造场景：p.increment_requests(3), p.increment_steps(7)。
核心 assert：total_requests==3, total_engine_steps==7。
```

#### 173. test_multiple_stages_independent
```
构造场景：record_raw scheduler_step=0.005, prefill=0.015, decode=0.025。
核心 assert：assert set(keys) == {"scheduler_step", "prefill", "decode"}。
验证的不变量：多个 stage 独立存储，互不干扰。
```

#### 174. test_engine_core_integration
```
构造场景：LLMEngine 运行后读取 profiler。
核心 assert：
  "engine_step_total" in report["stages"]
  report["total_engine_steps"] > 0
  report["total_requests"] == 2
验证的不变量：profiler 通过 EngineCore 正确集成到 engine 中。
```

#### 175. test_engine_core_integration_many_steps
```
构造场景：2 个请求，多个 step。
核心 assert：
  "scheduler_step" in report
  "executor_forward" in report
  scheduler_step 的 count == total_engine_steps（每步都记录了 scheduler）
验证的不变量：所有 engine 阶段的 stage 正确记录，scheduler_step 的计数值 = total_engine_steps。
```

#### 176. test_cancel_does_not_crash_profiler
```
构造场景：engine add_request -> step -> cancel -> run_until_done。
核心 assert：report["total_engine_steps"] > 0（不 crash）。
验证的不变量：cancel 操作不破坏 profiler 状态。
如果失败：cancel 后 profiler crash。

---

## pytest -q 最新结果（2026-07-06）

```
PYTHONPATH=. python3 -m pytest tests/ -q --tb=short
176 passed, 1 warning in 3.00s
```
