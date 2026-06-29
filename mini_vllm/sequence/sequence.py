from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from .sampling_params import SamplingParams
from .status import Status

if TYPE_CHECKING:
    from .sequence_group import SequenceGroup


class Sequence:
    """A single generation sequence.  Analogous to vLLM's ``Sequence``."""

    def __init__(
        self,
        seq_id: str,
        group_id: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        arrival_time: float,
    ) -> None:
        self.seq_id = seq_id
        self.group_id = group_id
        self.prompt_token_ids: List[int] = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.sampling_params: SamplingParams = sampling_params
        self.status: Status = Status.WAITING
        self.block_table: List[int] = []
        self.arrival_time: float = arrival_time
        self.first_token_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        self.num_generated_tokens: int = 0
        self.prefill_cursor: int = 0
        """How many prompt tokens have been written to KV cache so far."""
        self._group: Optional[SequenceGroup] = None

    @property
    def finished(self) -> bool:
        return self.status in (Status.FINISHED, Status.REJECTED, Status.CANCELLED, Status.TIMEOUT)

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def is_prefill_finished(self) -> bool:
        return self.prefill_cursor >= len(self.prompt_token_ids)

    @property
    def group(self) -> Optional[SequenceGroup]:
        return self._group

    def _set_group(self, group: SequenceGroup) -> None:
        self._group = group

    def to_dict(self) -> dict:
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
        return (
            f"Sequence(seq_id={self.seq_id!r}, status={self.status.name})"
        )
