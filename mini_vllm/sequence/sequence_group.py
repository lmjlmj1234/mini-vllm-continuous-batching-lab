from __future__ import annotations # 使类型注解变成惰性求值（字符串形式）
import time
from typing import Dict, List, Optional

from .sampling_params import SamplingParams
from .sequence import Sequence
from .status import Status


class SequenceGroup:
    """User-level request that owns one or more Sequence objects."""
    #一个用户请求，拥有零到多个 Sequence 对象" — 这是整个项目的请求级数据结构核心
    def __init__(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.prompt_token_ids: List[int] = prompt_token_ids or []
        self.arrival_time: float = arrival_time or time.time()
        self.seqs: List[Sequence] = [] # 该请求下所有 Sequence 对象的列表，初始为空。后续通过 create_sequence 添加

    def create_sequence(self, seq_id: str) -> Sequence:
        seq = Sequence(
            seq_id=seq_id,
            group_id=self.request_id,
            prompt_token_ids=list(self.prompt_token_ids),
            sampling_params=self.sampling_params,
            arrival_time=self.arrival_time,
        )
        #  如果没有 _set_group，调度器拿到一个 Sequence 时，没办法知道它属于哪个请求。它只能去遍历所有 SequenceGroup 的 seqs 列表来查找，效率很低
        seq._set_group(self) # seq._group 指向 sq_group → seq 知道自己的 group 是谁
        self.seqs.append(seq) # sq_group (SequenceGroup) │ sq_group.seqs 列表里有这个 seq → group 知道 seq 是自己的
        return seq

    @property
    #  属性 — 该请求下的 Sequence 总数
    def num_sequences(self) -> int:
        return len(self.seqs)

    @property
    #属性 — 已完成的 Sequence 数量。通过 generator 遍历 self.seqs 统计
    def num_finished(self) -> int:
        return sum(1 for s in self.seqs if s.finished)

    @property
    #  属性 — 请求是否全部完成
    def is_finished(self) -> bool:
        return self.num_sequences > 0 and self.num_finished == self.num_sequences

    def get_unfinished_seqs(self) -> List[Sequence]:
        # 返回尚未完成的 Sequence 列表
        return [s for s in self.seqs if not s.finished]

    def __repr__(self) -> str:
        return (
            f"SequenceGroup(request_id={self.request_id!r}, "
            f"seqs={self.num_sequences})"
        )


class RequestQueue:
    """Four-pool queue.  All pools store SequenceGroup objects."""

    def __init__(self) -> None:
        self._waiting: Dict[str, SequenceGroup] = {}
        self._running: Dict[str, SequenceGroup] = {}
        self._finished: Dict[str, SequenceGroup] = {}
        self._rejected: Dict[str, SequenceGroup] = {}

    def add(self, sg: SequenceGroup) -> None:
        #新请求进 waiting 池
        self._waiting[sg.request_id] = sg

    def mark_running(self, sg: SequenceGroup) -> None:
        #  从 waiting 移除放入 running。pop(..., None) 避免了 key 不存在时抛异常
        self._waiting.pop(sg.request_id, None)
        self._running[sg.request_id] = sg

    def mark_finished(self, sg: SequenceGroup) -> None:
        #  从 running 移除放入 finished
        self._running.pop(sg.request_id, None)
        self._finished[sg.request_id] = sg

    def mark_rejected(self, sg: SequenceGroup) -> None:
        # 从 waiting 移除放入 rejected
        self._waiting.pop(sg.request_id, None)
        self._rejected[sg.request_id] = sg

    @property
    def waiting(self) -> List[SequenceGroup]:
        #以列表形式暴露 waiting 池
        return list(self._waiting.values())

    @property
    def running(self) -> List[SequenceGroup]:
        # 同上，暴露 running 池
        return list(self._running.values())

    @property
    def finished(self) -> List[SequenceGroup]:
        return list(self._finished.values())

    @property
    def rejected(self) -> List[SequenceGroup]:
        return list(self._rejected.values())

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_finished(self) -> int:
        return len(self._finished)

    @property
    def num_rejected(self) -> int:
        return len(self._rejected)

    @property
    def total(self) -> int:
        return self.num_waiting + self.num_running + self.num_finished + self.num_rejected

    def get_by_id(self, request_id: str) -> SequenceGroup | None:
        # 按 id 搜索所有池子，找到就返回，找不到返回 None
        for pool in (self._waiting, self._running, self._finished, self._rejected):
            if request_id in pool:
                return pool[request_id]
        return None
