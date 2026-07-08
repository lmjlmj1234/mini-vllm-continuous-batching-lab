from __future__ import annotations #解释器不再立刻去内存里找 Sequence，而是直接把 Sequence 当作一个纯字符串 "Sequence" 存起来（这就是“惰性求值”）
from typing import TYPE_CHECKING, List, Optional
from .sampling_params import SamplingParams
from .status import Status

if TYPE_CHECKING:
    from .sequence_group import SequenceGroup


class Sequence:
    """A single generation sequence.  Analogous to vLLM's ``Sequence``."""

    def __init__(
        self,
        seq_id: str, # 当前这条序列的唯一 ID（后面会看到 group 里有多个 seq）
        group_id: str,# 所属 sequence group 的 ID
        prompt_token_ids: List[int],  # 用户输入的 prompt 被 tokenizer 切成的 token ID 列表
        sampling_params: SamplingParams,  # 这条序列的采样参数（temperature 等）
        arrival_time: float,  # 到达时间戳，用于计算 TTFT 等指标
    ) -> None:
        # 初始化
        self.seq_id = seq_id
        self.group_id = group_id
        self.prompt_token_ids: List[int] = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.sampling_params: SamplingParams = sampling_params
        self.status: Status = Status.WAITING
        self.block_table: List[int] = []
        self.arrival_time: float = arrival_time
        self.first_token_time: Optional[float] = None # 第一个 token 生成出来的时间。None 表示还没生成出第一个 token。后面用于计算 TTFT（Time to First Token）。
        self.first_scheduled_time: Optional[float] = None #调度器第一次处理这条序列的时间。不同于 arrival_time（用户请求到达），这个记录的是调度器真正开始调度它的时刻。用于分析调度延迟。
        """Wall-clock time when this sequence was first admitted by the scheduler."""
        self.finish_time: Optional[float] = None # 序列完成生成（正常结束、被取消、超时等）的时间。
        self.num_generated_tokens: int = 0 # 已生成的 token 数量计数器。注意这不一定等于 len(self.output_token_ids)——因为可能某些生成被回滚或中间有其他操作。
        self.prefill_cursor: int = 0 #prefill 阶段进度指针
        """How many prompt tokens have been written to KV cache so far."""
        self._group: Optional[SequenceGroup] = None

    @property
    def finished(self) -> bool:
        #只要状态是四种终止态之一，就认为序列已结束。用于调度器判断这条序列是否还"活着"
        return self.status in (Status.FINISHED, Status.REJECTED, Status.CANCELLED, Status.TIMEOUT)

    @property
    def prompt_length(self) -> int:
        #  用户 prompt 的 token 数量
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        # 当前已生成的 token 数量
        return len(self.output_token_ids)

    @property
    def is_prefill_finished(self) -> bool:
        # 判断 prefill 阶段是否完成
        return self.prefill_cursor >= len(self.prompt_token_ids)

    @property
    def group(self) -> Optional[SequenceGroup]:
        #对外提供只读的 group 属性，内部通过 _set_group 设置（通常在 SequenceGroup 构造时将双方互相绑定）
        return self._group

    def _set_group(self, group: SequenceGroup) -> None:
        self._group = group

    def to_dict(self) -> dict:
        #将序列的核心信息序列化为字典，通常用于日志、监控、API 返回等
        return {
            "seq_id": self.seq_id,
            "group_id": self.group_id,
            "status": self.status.name,
            "num_prompt_tokens": self.prompt_length,
            "num_output_tokens": self.num_output_tokens,
            "num_blocks": len(self.block_table),
            "prefill_cursor": self.prefill_cursor,
        }

    def __repr__(self) -> str:
        #  给 Python 解释器用的可打印表示。!r 表示用 repr() 输出（带引号），方便调试时一眼看出来是字符串。例如 print(seq_obj) 会显示 Sequence(seq_id='abc-123', status=WAITING)。
        return (
            f"Sequence(seq_id={self.seq_id!r}, status={self.status.name})"
        )
