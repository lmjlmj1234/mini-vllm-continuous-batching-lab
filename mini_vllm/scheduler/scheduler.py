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
        self._trace_enabled: bool = False
        self._trace_records: List[dict] = []
        self._trace_step_count: int = 0

    def enable_trace(self, enabled: bool = True) -> None:
        """Enable or disable per-step scheduler trace."""
        self._trace_enabled = enabled

    def get_and_clear_trace(self) -> List[dict]:
        """Return all trace records and clear the buffer."""
        records = list(self._trace_records)
        self._trace_records.clear()
        return records

    def _record_trace(self, result: ScheduleResult) -> None:
        """Record a step trace entry (no-op if trace is disabled)."""
        if not self._trace_enabled:
            return
        self._trace_step_count += 1
        waiting_ids = [sg.request_id for sg in self._queue._waiting.values()]
        running_ids = [sg.request_id for sg in self._queue._running.values()]
        finished_ids = [sg.request_id for sg in self._queue._finished.values()]
        admitted_ids = [sg.request_id for sg in result.scheduled_prefill_groups]
        newly_finished_ids = [sg.request_id for sg in result.finished_groups]

        record = {
            "timestamp": time.time(),
            "step_id": self._trace_step_count,
            "waiting_requests": len(waiting_ids),
            "running_requests": len(running_ids),
            "finished_requests": len(finished_ids),
            "newly_admitted_requests": len(admitted_ids),
            "newly_finished_requests": len(newly_finished_ids),
            "scheduled_prefill_tokens": result.num_prefill_tokens,
            "scheduled_decode_tokens": result.num_decode_tokens,
            "effective_batch_size": len(result.scheduled_decode_groups) + len(result.scheduled_prefill_groups),
            "waiting_request_ids": waiting_ids,
            "running_request_ids": running_ids,
            "finished_request_ids": finished_ids,
            "admitted_request_ids": admitted_ids,
            "finished_request_ids_this_step": newly_finished_ids,
        }
        self._trace_records.append(record)

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
        # Static batch mode: only admit new groups when no groups are running.
        can_admit = not self._config.static_batch_mode or (
            self._config.static_batch_mode and len(decode_groups) == 0
            and len(prefill_continue_groups) == 0
        )
        if can_admit:
            for sg in list(self._queue.waiting):
                #遍历所有等待中的新请求。用 list() 拷贝是为了在遍历时不会因为 mark_running 修改字典而出问题
                if remaining_seq_budget <= 0:# 序列预算用完了 → 跳过。
                    result.ignored_groups.append(sg)
                    result.ignored_reasons[sg.request_id] = "MAX_NUM_SEQS_LIMIT"
                    continue

                prompt_len = len(sg.prompt_token_ids) # 拿到这个请求的 prompt 长度

                # Read-only probe: how many prompt tokens are already cached?
                probe = self._block_manager.probe_prefix_cache(sg.prompt_token_ids) # 前缀缓存探测
                uncached_tokens = prompt_len - probe.cached_token_count

                if self._config.chunked_prefill_enabled:
                    this_chunk = min(uncached_tokens, chunk_size)
                else:
                    this_chunk = uncached_tokens

                # Edge case: all prompt tokens are prefix-cached (this_chunk=0).
                # Process the last chunk of the prompt anyway to generate
                # first-token logits via the model forward pass.
                if this_chunk == 0 and uncached_tokens == 0:
                    this_chunk = min(chunk_size, prompt_len)

                if this_chunk > prefill_budget:
                    if uncached_tokens > self._config.max_num_batched_tokens:
                        self._queue.mark_rejected(sg)
                        result.rejected_groups.append(sg)
                    else:
                        result.ignored_groups.append(sg)
                        result.ignored_reasons[sg.request_id] = "NO_TOKEN_BUDGET"
                    continue

                seq_id = f"{sg.request_id}-seq-0"
                seq = sg.create_sequence(seq_id)
                self._block_manager.allocate_for_seq(seq)

                seq.status = Status.PREFILL
                # When all tokens are cached, start prefill from the end
                # so the executor processes fresh tokens for logits.
                if uncached_tokens == 0 and probe.cached_token_count == prompt_len:
                    seq.prefill_cursor = prompt_len - this_chunk
                else:
                    seq.prefill_cursor = probe.cached_token_count

                self._queue.mark_running(sg)
                result.scheduled_prefill_groups.append(sg)
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

        # Record trace (no-op if disabled)
        self._record_trace(result)

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
