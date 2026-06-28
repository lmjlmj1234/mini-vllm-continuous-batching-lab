from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    """Global configuration for the mini-vLLM engine."""

    # --- Scheduler ---
    max_num_seqs: int = 4
    """Maximum number of sequences processed in one step (running + prefill)."""

    max_num_batched_tokens: int = 16
    """Maximum total tokens across all sequences in a single step."""

    max_num_prefill_tokens: int = 16
    """Maximum prefill tokens per step.  Separate from max_num_batched_tokens
    so that decode gets guaranteed budget even under heavy prefill."""

    chunked_prefill_enabled: bool = True
    """If True, prompts longer than max_prefill_chunk_size are split across
    multiple steps instead of being ignored/rejected."""

    max_prefill_chunk_size: int = 4
    """Number of prompt tokens processed in a single prefill step."""

    decode_first: bool = True
    """If True, running decode sequences are scheduled before any prefill."""

    # --- KV Cache ---
    block_size: int = 4
    """Number of tokens per physical block."""

    num_gpu_blocks: int = 8
    """Total physical blocks in the KV cache pool."""

    # --- Fake Model Runner ---
    vocab_size: int = 256
    """Fake vocabulary size used for tokenisation and sampling."""

    # --- Executor ---
    executor_type: str = "fake"
    """Which executor to use: ``"fake"`` (FakeModelExecutor) or ``"qwen"``
    (QwenExecutor via HuggingFace Transformers)."""

    # --- Engine ---
    print_step_events: bool = True
    """Whether to print schedule events at each engine step."""

    memory_trace: bool = False
    """If True, print detailed BlockAllocator free list and per-sequence
    BlockTable at each step.  Used for educational debugging."""

    DTYPE: str = "float16"
    """Data type used for the KV cache (documentation)."""

    def __post_init__(self) -> None:
        assert self.max_num_seqs > 0
        assert self.max_num_batched_tokens > 0
        assert self.max_num_prefill_tokens > 0
        assert self.block_size > 0
        assert self.num_gpu_blocks > 0
        assert self.vocab_size > 0
        assert self.max_prefill_chunk_size > 0
        assert self.executor_type in ("fake", "qwen"), (
            f"executor_type must be 'fake' or 'qwen', got {self.executor_type!r}"
        )
