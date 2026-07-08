from __future__ import annotations
# 惰性类型注解：让类名可以在字符串形式下前向引用，避免循环导入

from typing import Dict, List, Protocol, TYPE_CHECKING, runtime_checkable
# Protocol: Python 的"接口"机制，定义类必须实现的方法
# runtime_checkable: 允许运行时用 isinstance() 检查是否实现了这个 Protocol
# TYPE_CHECKING: 只在类型检查时导入，避免运行时循环导入

from ..config import Config
# 全局配置对象（调度参数、模型参数、服务参数等）

from ..sequence.sequence import Sequence
# 序列对象——一条生成序列的数据结构

if TYPE_CHECKING:
    # 只在类型检查时才导入，运行时不会执行
    # 因为 BlockManager 导入了 Sequence，Sequence 又可能引用 Executor，容易循环依赖
    from ..cache.manager import BlockManager


@runtime_checkable
# 装饰器：允许 isinstance(obj, Executor) 在运行时检查
class Executor(Protocol):
    """Abstract protocol for model executors.
    # 模型执行器的抽象接口协议
    # 定义了引擎核心（EngineCore）和任何模型执行器之间的契约
    # 无论是 fake（教学用）还是 real（Qwen/HuggingFace）都必须实现这些方法

    Defines the contract between the engine core and any model runner,
    whether fake (educational) or real (Qwen/HuggingFace).

    Responsibilities:
    - Tokenize / detokenize text using the model's tokenizer
    - Run prefill: process prompt tokens, write KV cache, produce first token
    - Run decode: read KV cache, produce one new token, write new KV
    - Maintain KV cache storage in sync with BlockAllocator lifecycle
    - Provide KV cache statistics for monitoring
    """

    def __init__(self, config: Config, block_manager: BlockManager | None = None) -> None: ...
    # 构造器：接收配置和可选的 BlockManager 引用
    # ... 表示"不实现，子类必须提供"

    # ------------------------------------------------------------------
    # Block allocator callbacks
    # BlockAllocator 回调函数
    # 当 BlockAllocator 分配或释放物理块时，executor 需要同步创建/销毁 KV 存储
    # ------------------------------------------------------------------

    def prepare_block(self, block_id: int) -> None:
        """Called when BlockAllocator allocates a new physical block.
        # 当 BlockAllocator 分配了一个新物理块时调用
        # 子类应该为这个块创建 KV 数据存储（比如在 _kv_cache 字典里创建条目）

        Subclasses should create KV data storage for this block.
        """
        ...

    def release_block(self, block_id: int) -> None:
        """Called when BlockAllocator frees a physical block.
        # 当 BlockAllocator 释放了一个物理块时调用
        # 子类应该释放这个块的 KV 数据存储

        Subclasses should release KV data storage for this block.
        """
        ...

    def make_block_allocator_callbacks(self) -> dict:
        """Return callbacks dict for BlockAllocator.set_callbacks()."""
        # 返回一个字典，包含 on_allocate 和 on_free 两个回调
        # 这个字典会传给 BlockAllocator.set_callbacks()，建立起"分配→创建KV"的联动
        ...

    # ------------------------------------------------------------------
    # Tokenization
    # tokenizer/detokenizer 接口：文本 ↔ token ID 列表
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> List[int]: ...
    # 将字符串 prompt 转成 token ID 列表

    def detokenize(self, token_ids: List[int]) -> str: ...
    # 将 token ID 列表转回字符串

    # ------------------------------------------------------------------
    # Execution
    # 执行接口：prefill（处理 prompt）/ decode（逐个生成 token）
    # ------------------------------------------------------------------

    def prefill(self, sequences: List[Sequence]) -> None:
        """Process prompt tokens and write KV cache.
        # 处理 prompt token 并写入 KV cache
        #
        # 后置条件：prefill 完成的序列会设置好第一个输出 token，状态变为 RUNNING

        Post-condition: sequences with finished prefill have their first
        output token set and status changed to RUNNING.
        """
        ...

    def decode(self, sequences: List[Sequence]) -> None:
        """Produce one new output token per sequence.
        # 每条序列生成一个新的输出 token
        #
        # 读取已有 KV cache，产生 logits，采样一个 token，
        # 并将新 token 的 KV 数据写回 cache

        Reads existing KV cache, produces logits, samples a token, and
        writes the new token's KV data back to the cache.
        """
        ...

    def cleanup_sequence(self, seq_id: str) -> None:
        """Remove per-sequence state (KV cache, etc.) after finish.
        # 序列完成后清理其专有状态（KV cache 等）
        #
        # EngineCore 在请求组完成时调用，让 executor 释放序列级别的资源

        Called by EngineCore when a group finishes, so the executor
        can release any sequence-specific resources.
        """
        ...

    # ------------------------------------------------------------------
    # Stats
    # 统计信息
    # ------------------------------------------------------------------

    def get_kv_stats(self) -> Dict[str, int]:
        """Return KV cache usage statistics."""
        # 返回 KV cache 使用统计（已写 token 数、容量、已分配块数等）
        ...

    @property
    def total_tokens_processed(self) -> int:
        """Total KV write operations performed so far."""
        # 属性：截至目前总共处理的 token 数量
        ...
