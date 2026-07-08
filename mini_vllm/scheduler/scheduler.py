from __future__ import annotations
import time
from typing import Dict, List

from ..cache.manager import BlockManager
from ..config import Config
from ..sequence.sequence import Sequence
from ..sequence.sequence_group import RequestQueue, SequenceGroup
from ..sequence.status import Status
from .schedule_result import ScheduleResult


class Scheduler:
    """Continuous-batching scheduler with chunked prefill and decode-first.
    #每次引擎迭代（engine step）调用一次 schedule()，决定当前这一步里哪些请求该干活、干多少活
    At each engine step::

        1. **Finish** — check every running group; sequences that hit
           ``max_tokens`` are marked FINISHED; groups whose every sequence
           is done move to the finished pool.
        2. **Categorize** — split remaining running groups into decode
           (status=RUNNING) and prefill-continue (status=PREFILL).
        3. **Decode-first budget** — deduct decode tokens, then compute
           remaining budget for prefill.
        4. **Chunked-prefill continue** — advance prefill cursor for
           groups that are mid-prefill.
        5. **Admit new** — admit waiting groups with chunked budget.
        6. **Token counts & debug_reason**.
    """

    def __init__(self, config: Config, block_manager: BlockManager, queue: RequestQueue) -> None:
        self._config = config
        self._block_manager = block_manager
        self._queue = queue

    def schedule(self) -> ScheduleResult:
        """Run one scheduling iteration — called once per engine step."""
        result = ScheduleResult()# 创建调度结果容器，后面逐步填入数据

        chunk_size = self._config.max_prefill_chunk_size

        # ------------------------------------------------------------------
        # Phase 1: Finish check — running groups
        # ------------------------------------------------------------------
        keep_running_groups: List[SequenceGroup] = []

        for sg in self._queue.running: #遍历所有正在运行的请求组
            for seq in sg.get_unfinished_seqs():# 拿到这个组里还没完成的所有序列
                if seq.num_generated_tokens >= seq.sampling_params.max_tokens: #  检查是否达到了最大 token 数限制。如果生成的 token 数量 ≥ 用户设置的最大值
                    # 标记完成 → 记录完成时间 → 释放它占用的 KV cache 块
                    seq.status = Status.FINISHED
                    seq.finish_time = time.time()
                    self._block_manager.free(seq.seq_id)

            if sg.is_finished:
                #如果请求组里的所有序列都完成了 → 从 running 池移到 finished 池，记录到结果里
                self._queue.mark_finished(sg)
                result.finished_groups.append(sg)
            else:
                #还没完成，下一轮继续调度你
                keep_running_groups.append(sg)

        # ------------------------------------------------------------------
        # Phase 2: Categorize running groups
        # ------------------------------------------------------------------
        decode_groups: List[SequenceGroup] = []
        prefill_continue_groups: List[SequenceGroup] = []

        for sg in keep_running_groups:
            seq = sg.get_unfinished_seqs()[0]   # 每个 running 组只看它的第一条未完成序列来决定类别（同组内所有序列状态一致）
            if seq.status == Status.PREFILL:
                #  分类逻辑：
                # - 状态是 PREFILL → prefill 继续组（还没 prefill 完，下一轮接着 prefill 剩下的 tokens）
                # - 其他状态（RUNNING）→ decode 组（prefill 已经完成，只需要逐个生成 token）
                prefill_continue_groups.append(sg)
            else:
                decode_groups.append(sg)

        # ------------------------------------------------------------------
        # Phase 3: Decode-first budget
        # ------------------------------------------------------------------
        #拿本轮的总预算：最多处理多少 token、最多处理多少条序列
        remaining_token_budget = self._config.max_num_batched_tokens
        remaining_seq_budget = self._config.max_num_seqs

        # Deduct decode first (1 token per decode sequence)
        for sg in decode_groups:
            n = len(sg.get_unfinished_seqs())
            remaining_token_budget -= n
            remaining_seq_budget -= n
        result.scheduled_decode_groups = list(decode_groups)
        # "decode 优先" — 先把 decode 组的预算扣掉（每个 decode 序列这一步生产 1 个 token）。把剩下的预算留给 prefill。
        #  为什么 decode 优先？因为用户感知到的流畅度取决于 decode 延迟（每秒输出多少个 token），不能让 prefill 把 decode 挤掉了。

        # Prefill budget = remaining tokens, capped by max_num_prefill_tokens
        prefill_budget = min(remaining_token_budget, self._config.max_num_prefill_tokens) #prefill 可用的 token 预算 = 剩下的 token (不能超过 max_num_prefill_tokens 硬上限)。
        num_prefill_tokens = 0 #  初始化计数器

        # ------------------------------------------------------------------
        # Phase 4: Chunked-prefill continue
        # ------------------------------------------------------------------
        for sg in prefill_continue_groups:#遍历上一轮没 prefill 完、这轮要继续的请求
            if remaining_seq_budget <= 0:#  序列预算用完了 — 不处理这个组，记入忽略列表，下轮再说
                result.ignored_groups.append(sg)
                result.ignored_reasons[sg.request_id] = "MAX_NUM_SEQS_LIMIT"
                continue

            seq = sg.get_unfinished_seqs()[0]
            # prefill_cursor already accounts for cached prefix (set at admit time)
            remaining_prompt = len(seq.prompt_token_ids) - seq.prefill_cursor
            this_chunk = min(remaining_prompt, chunk_size) # 计算还剩多少 prompt tokens 没处理 → 这次最多处理 chunk_size 个（防止一个长 prompt 占满所有预算）

            if this_chunk > prefill_budget: #token 预算不够 — 这个组的这一 chunk 太大，当前装不下，跳过等下一步
                result.ignored_groups.append(sg)
                result.ignored_reasons[sg.request_id] = "WAITING_FOR_NEXT_STEP"
                continue
            #"你被调度到了" → 记录到结果、扣预算、累加计数器
            result.scheduled_prefill_groups.append(sg)
            prefill_budget -= this_chunk
            remaining_token_budget -= this_chunk
            remaining_seq_budget -= 1
            num_prefill_tokens += this_chunk
            result.num_uncached_prefill_tokens += this_chunk

        # ------------------------------------------------------------------
        # Phase 5: Admit new waiting groups (with prefix cache awareness)
        # ------------------------------------------------------------------
        for sg in list(self._queue.waiting):
            #遍历所有等待中的新请求。用 list() 拷贝是为了在遍历时不会因为 mark_running 修改字典而出问题
            if remaining_seq_budget <= 0:# 序列预算用完了 → 跳过。
                result.ignored_groups.append(sg)
                result.ignored_reasons[sg.request_id] = "MAX_NUM_SEQS_LIMIT"
                continue

            prompt_len = len(sg.prompt_token_ids) # 拿到这个请求的 prompt 长度

            # Read-only probe: how many prompt tokens are already cached?
            probe = self._block_manager.probe_prefix_cache(sg.prompt_token_ids) # 前缀缓存探测 — 问 block manager：这个 prompt 的前面多少 tokens 已经在 KV cache 里了？（相同前缀复用缓存，不用重新计算）
            uncached_tokens = prompt_len - probe.cached_token_count #真正需要计算的 token 数量 = 总长度 - 已缓存的部分

            if self._config.chunked_prefill_enabled:
                this_chunk = min(uncached_tokens, chunk_size)
            else:
                this_chunk = uncached_tokens
            # 分块模式：只取一个 chunk。不分块模式：全量 prefill
            if this_chunk > prefill_budget:
                # Rejection: the uncached portion alone is too long to fit
                if uncached_tokens > self._config.max_num_batched_tokens:
                    self._queue.mark_rejected(sg)
                    result.rejected_groups.append(sg)
                else:
                    result.ignored_groups.append(sg)
                    result.ignored_reasons[sg.request_id] = "NO_TOKEN_BUDGET"
                continue
                #  装不下怎么办？ 有两种情况：
                #- 整个 prompt 本身就超出单步处理上限 → 拒绝这个请求（拒绝）
                #- 只是这步预算不够 → 先跳过，下步再说（忽略）

            seq_id = f"{sg.request_id}-seq-0" # 为该请求创建第 0 号 Sequence 对象
            seq = sg.create_sequence(seq_id)
            # allocate_for_seq does the real attach: increment_ref, add_shared_block, etc.
            self._block_manager.allocate_for_seq(seq) #为这个序列分配 KV cache 块

            seq.status = Status.PREFILL
            # Prefill cursor starts after the cached prefix (not at 0).
            # The executor's prefill loop starts from this position and only
            # processes uncached tokens.
            seq.prefill_cursor = probe.cached_token_count # prefill 游标从已缓存的位置开始 — 已缓存的部分不用再计算。没命中缓存就是 0

            self._queue.mark_running(sg) #从 waiting 池移到 running 池
            result.scheduled_prefill_groups.append(sg)
            #记录调度结果，更新各种计数
            prefill_budget -= this_chunk
            remaining_token_budget -= this_chunk
            remaining_seq_budget -= 1
            num_prefill_tokens += this_chunk
            result.cached_token_count += probe.cached_token_count
            result.num_uncached_prefill_tokens += this_chunk
            result.matched_block_count += probe.matched_block_count

        # ------------------------------------------------------------------
        # Phase 6: Token counts & debug_reason
        # ------------------------------------------------------------------
        num_decode = 0
        for sg in decode_groups:#统计这次调度了多少条 decode 序列（用于日志和监控）
            num_decode += len(sg.get_unfinished_seqs())

        result.num_prefill_tokens = num_prefill_tokens
        result.num_decode_tokens = num_decode
        result.num_batched_tokens = num_prefill_tokens + num_decode
        result.token_budget_remaining = remaining_token_budget
        result.debug_reason = self._build_debug_reason(result)
        # 填满调度结果的各种统计字段，最后生成一句 debug 字符串方便日志打印

        # 返回调度结果给引擎。引擎拿到这个 result，就知道这一步要执行哪些 prefill、哪些 decode
        return result

    # 辅助方法
    def block_manager_stats(self) -> dict:
        """Expose block manager stats for metrics reporting."""
        #  暴露 BlockManager 的统计信息给外部（metrics 系统）

        return self._block_manager.stats()

    @staticmethod
    # 构建 debug 字符串，格式如：
    #prefill(256t): req-001 | decode(32t): req-002, req-003 | done: req-000
    #一眼看出这步调度发生了什么
    #静态方法（@staticmethod）：没有默认的 self 或 cls 参数。它本质上就是一个恰好放在类里面的普通函数。它既不能访问实例属性，也不能访问类属性。
    def _build_debug_reason(result: ScheduleResult) -> str:
        parts = []
        if result.scheduled_prefill_groups:
            p = [sg.request_id for sg in result.scheduled_prefill_groups]
            if result.cached_token_count > 0:
                parts.append(
                    f"prefill({result.num_uncached_prefill_tokens}t "
                    f"+{result.cached_token_count}cached): {', '.join(p)}"
                )
            else:
                parts.append(f"prefill({result.num_prefill_tokens}t): {', '.join(p)}")
        if result.scheduled_decode_groups:
            d = [sg.request_id for sg in result.scheduled_decode_groups]
            parts.append(f"decode({result.num_decode_tokens}t): {', '.join(d)}")
        if result.ignored_groups:
            i = [sg.request_id for sg in result.ignored_groups]
            parts.append(f"ignored: {', '.join(i)}")
        if result.finished_groups:
            f = [sg.request_id for sg in result.finished_groups]
            parts.append(f"done: {', '.join(f)}")
        return " | ".join(parts) if parts else "idle"
