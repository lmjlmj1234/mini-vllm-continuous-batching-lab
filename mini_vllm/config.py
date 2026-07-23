from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    """Global configuration for the mini-vLLM engine."""

    # --- Scheduler ---
    max_num_seqs: int = 4
    """Maximum number of sequences processed in one step (running + prefill)."""

    max_num_batched_tokens: int = 16
    """Maximum total tokens across all sequences in a single step."""

    max_num_prefill_tokens: int = 16
    """Maximum prefill tokens per step."""

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
    # "每个物理块存几个 token"。KV cache 按块管理，一块存 n 个 token 的 key/value。块越小，内存浪费越少但管理开销越大；块越大则相反

    num_gpu_blocks: int = 8
    """Total physical blocks in the KV cache pool."""

    gpu_memory_utilization: float = 0.90
    """Fraction of available GPU memory to use for the KV cache (0, 1]."""

    peak_runtime_estimate: int = 0
    """Estimated peak runtime memory in bytes (activations + intermediates).
    Measured by B7 profile workload (128 prefill + 16 decode) on the target
    GPU.  When 0, the budget computation ignores runtime activation memory —
    safe only for small models or large safety margins."""

    # --- Serving ---
    max_model_len: int = 2048
    """Maximum prompt length the model can accept."""

    max_queue_len: int = 32
    """Maximum number of requests in the waiting queue."""

    max_num_streams: int = 16
    """Maximum concurrent active SSE streams."""

    rate_limit_rpm: int = 60
    """Maximum requests per minute."""

    rate_limit_tpm: int = 100000
    """Maximum tokens per minute (prompt + generation)."""

    request_timeout_s: float = 60.0
    """Maximum wall-clock time per request before auto-cancel."""

    # --- Fake Model Runner ---
    vocab_size: int = 256
    """Fake vocabulary size used for tokenisation and sampling."""

    # --- Executor ---
    executor_type: str = "fake"
    """Which executor to use: ``"fake"``, ``"qwen"``, or ``"paged"``."""

    attention_backend: str = "reference"
    """Attention backend: ``"triton"`` (GPU kernel) or ``"reference"`` (test-only PyTorch SDPA)."""

    model_path: str = ""
    """Local path to the model directory (used by QwenWorker/Executor).
    If empty, the executor falls back to its default HuggingFace model name."""

    # --- Device ---
    device: str = "cuda"
    """Target device for model and KV cache tensors.
    Must be parseable by ``torch.device()``, e.g. ``"cpu"``, ``"cuda"``,
    ``"cuda:0"``.  When set to ``"cuda"``, raises ``RuntimeError`` if CUDA
    is unavailable.  Use ``"cpu"`` for testing on systems without GPU."""

    # --- Engine ---
    print_step_events: bool = True
    """Whether to print schedule events at each engine step."""

    memory_trace: bool = False
    """If True, print detailed BlockAllocator free list at each step."""

    trace_enabled: bool = False
    """If True, record per-step scheduler trace (default off)."""

    enable_prefix_caching: bool = True
    """If True, BlockManager uses prefix cache to share blocks between
    sequences with identical prompt prefixes.  When False, every sequence
    gets fresh blocks — avoids duplicate slot mappings required by Triton's
    ``triton_cache_write`` assertion."""

    static_batch_mode: bool = False
    """If True, scheduler does NOT admit new waiting groups.
    Running groups must all finish before any waiting groups are considered.
    Default off — used only for static batching baselines."""

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
        assert self.executor_type in ("fake", "qwen", "paged"), (
            f"executor_type must be 'fake', 'qwen', or 'paged', "
            f"got {self.executor_type!r}"
        )
        assert self.max_queue_len > 0
        assert self.max_num_streams > 0
        assert self.rate_limit_rpm > 0
        assert self.rate_limit_tpm > 0
        assert self.request_timeout_s > 0
        assert 0 < self.gpu_memory_utilization <= 1.0, (
            f"gpu_memory_utilization must be in (0, 1], "
            f"got {self.gpu_memory_utilization}"
        )

        # Validate device
        try:
            dev = torch.device(self.device)
        except RuntimeError as e:
            raise ValueError(f"Invalid device string: {self.device!r} — {e}")

        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Config.device is '{self.device}' but CUDA is not available. "
                f"Set device='cpu' for CPU-only operation."
            )
        if dev.type not in ("cpu", "cuda"):
            raise ValueError(
                f"Unsupported device type '{dev.type}'. "
                f"Supported: 'cpu', 'cuda', 'cuda:N'."
            )
