# ------------------------------------------------------------------
# Sequence layer
# ------------------------------------------------------------------
from .sequence.status import Status
from .sequence.sampling_params import SamplingParams
from .sequence.sequence import Sequence
from .sequence.sequence_group import SequenceGroup, RequestQueue

# ------------------------------------------------------------------
# Cache layer
# ------------------------------------------------------------------
from .cache.block import Block
from .cache.block_table import BlockTable, BlockTableEntry
from .cache.allocator import BlockAllocator
from .cache.manager import BlockManager
from .cache.prefix_cache import PrefixCache, PrefixCacheProbeResult
from .cache.pool import KVCachePool, compute_num_gpu_blocks
from .cache.cache_write import write_to_paged_cache

# ------------------------------------------------------------------
# Scheduler layer
# ------------------------------------------------------------------
from .scheduler.schedule_result import ScheduleResult
from .scheduler.scheduler import Scheduler

# ------------------------------------------------------------------
# Executor layer
# ------------------------------------------------------------------
from .executor.base import Executor
from .executor.executor import FakeModelExecutor

# ------------------------------------------------------------------
# Model layer
# ------------------------------------------------------------------
from .model.fake_model import FakeModel

# --- Model layer (Phase 2) ---
from .model.rms_norm import RMSNorm
from .model.rotary import RotaryEmbedding
from .model.qkv_proj import QKVProjection
from .model.mlp import SwiGLUMLP
from .model.transformer_layer import QwenDecoderLayer
from .model.qwen_model import QwenModel

# ------------------------------------------------------------------
# Worker layer
# ------------------------------------------------------------------
from .worker.fake_worker import FakeWorker

# --- Optional imports (torch / transformers not required for fake executor) ---
try:
    from .executor.qwen_executor import QwenExecutor
except ImportError:
    QwenExecutor = None  # type: ignore[assignment]

try:
    from .executor.paged_executor import PagedExecutor
except ImportError:
    PagedExecutor = None

try:
    from .worker.qwen_worker import QwenWorker
except ImportError:
    QwenWorker = None  # type: ignore[assignment]

try:
    from .worker.paged_worker import PagedWorker
except ImportError:
    PagedWorker = None

# ------------------------------------------------------------------
# Engine layer
# ------------------------------------------------------------------
from .engine.engine_core import EngineCore
from .engine.engine import LLMEngine
from .engine.metrics import MetricsCollector
from .engine.stage_profiler import StageProfiler

# ------------------------------------------------------------------
# Model Runner layer (NEW — Phase 1)
# ------------------------------------------------------------------
from .model_runner.base import (
    ModelRunner,
    ModelInput,
    AttentionMetadata,
    AttentionGroup,
    ModelConfig,
    ModelRunnerOutput,
    SequenceExecutionInfo,
)
from .model_runner.config_adapter import ConfigAdapter

# ------------------------------------------------------------------
# Attention layer (NEW — Phase 1)
# ------------------------------------------------------------------
from .attention.backend import AttentionBackend
from .attention.paged_attention_ref import AttentionBackendRef

# ------------------------------------------------------------------
# Input Builder (NEW — Phase 1)
# ------------------------------------------------------------------
from .engine.input_builder import ModelInputBuilder

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
from .config import Config

__all__ = [
    # config
    "Config",
    # sequence
    "Status",
    "SamplingParams",
    "Sequence",
    "SequenceGroup",
    "RequestQueue",
    # cache
    "Block",
    "BlockTable",
    "BlockTableEntry",
    "BlockAllocator",
    "BlockManager",
    "PrefixCache",
    "PrefixCacheProbeResult",
    "KVCachePool",
    "compute_num_gpu_blocks",
    "write_to_paged_cache",
    # scheduler
    "Scheduler",
    "ScheduleResult",
    # executor
    "Executor",
    "FakeModelExecutor",
    "QwenExecutor",
    "PagedExecutor",
    # model
    "FakeModel",
    "RMSNorm",
    "RotaryEmbedding",
    "QKVProjection",
    "SwiGLUMLP",
    "QwenDecoderLayer",
    "QwenModel",
    # worker
    "FakeWorker",
    "QwenWorker",
    "PagedWorker",
    # engine
    "EngineCore",
    "LLMEngine",
    "MetricsCollector",
    "StageProfiler",
    # model_runner (Phase 1)
    "ModelRunner",
    "ModelInput",
    "AttentionMetadata",
    "AttentionGroup",
    "ModelConfig",
    "ModelRunnerOutput",
    "SequenceExecutionInfo",
    "ConfigAdapter",
    # attention
    "AttentionBackend",
    "AttentionBackendRef",
    # input builder (Phase 1)
    "ModelInputBuilder",
]
