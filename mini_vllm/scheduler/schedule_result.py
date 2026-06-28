from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..sequence.sequence_group import SequenceGroup


@dataclass
class ScheduleResult:
    """Rich scheduling result matching vLLM's SchedulerOutputs."""

    scheduled_prefill_groups: List[SequenceGroup] = field(default_factory=list)
    scheduled_decode_groups: List[SequenceGroup] = field(default_factory=list)
    ignored_groups: List[SequenceGroup] = field(default_factory=list)
    finished_groups: List[SequenceGroup] = field(default_factory=list)
    rejected_groups: List[SequenceGroup] = field(default_factory=list)
    preempted_groups: List[SequenceGroup] = field(default_factory=list)
    num_batched_tokens: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    token_budget_remaining: int = 0
    # Prefix cache fields (aggregate across all scheduled prefill groups)
    cached_token_count: int = 0
    """Number of prefill tokens that were already in Prefix Cache (not computed)."""
    num_uncached_prefill_tokens: int = 0
    """Number of prefill tokens that actually need computation (uncached portion)."""
    matched_block_count: int = 0
    """Number of logical blocks shared via Prefix Cache (aggregate)."""
    debug_reason: str = ""
    ignored_reasons: Dict[str, str] = field(default_factory=dict)
    """Maps request_id of ignored groups to their reason string."""
