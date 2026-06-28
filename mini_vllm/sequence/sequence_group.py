from __future__ import annotations

import time
from typing import Dict, List, Optional

from .sampling_params import SamplingParams
from .sequence import Sequence
from .status import Status


class SequenceGroup:
    """User-level request that owns one or more Sequence objects."""

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
        self.seqs: List[Sequence] = []

    def create_sequence(self, seq_id: str) -> Sequence:
        seq = Sequence(
            seq_id=seq_id,
            group_id=self.request_id,
            prompt_token_ids=list(self.prompt_token_ids),
            sampling_params=self.sampling_params,
            arrival_time=self.arrival_time,
        )
        seq._set_group(self)
        self.seqs.append(seq)
        return seq

    @property
    def num_sequences(self) -> int:
        return len(self.seqs)

    @property
    def num_finished(self) -> int:
        return sum(1 for s in self.seqs if s.finished)

    @property
    def is_finished(self) -> bool:
        return self.num_sequences > 0 and self.num_finished == self.num_sequences

    def get_unfinished_seqs(self) -> List[Sequence]:
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
        self._waiting[sg.request_id] = sg

    def mark_running(self, sg: SequenceGroup) -> None:
        self._waiting.pop(sg.request_id, None)
        self._running[sg.request_id] = sg

    def mark_finished(self, sg: SequenceGroup) -> None:
        self._running.pop(sg.request_id, None)
        self._finished[sg.request_id] = sg

    def mark_rejected(self, sg: SequenceGroup) -> None:
        self._waiting.pop(sg.request_id, None)
        self._rejected[sg.request_id] = sg

    @property
    def waiting(self) -> List[SequenceGroup]:
        return list(self._waiting.values())

    @property
    def running(self) -> List[SequenceGroup]:
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
        for pool in (self._waiting, self._running, self._finished, self._rejected):
            if request_id in pool:
                return pool[request_id]
        return None
