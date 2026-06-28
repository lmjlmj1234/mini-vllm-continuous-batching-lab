from dataclasses import dataclass, field
from typing import List


@dataclass
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop_token_ids: List[int] = field(default_factory=list)
    stop_strings: List[str] = field(default_factory=list)
