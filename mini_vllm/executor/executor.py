from __future__ import annotations
# 惰性类型注解

from typing import Dict, List, TYPE_CHECKING
# Dict/List：类型注解用
# TYPE_CHECKING：条件导入，只在类型检查时执行

from ..config import Config
# 全局配置

from ..model.fake_model import FakeModel
# 假模型——用数学函数模拟 key/value/logits，不依赖 torch

from ..sequence.sequence import Sequence
# 序列对象

from ..sequence.status import Status
# 状态枚举：WAITING / PREFILL / RUNNING / FINISHED / REJECTED / CANCELLED / TIMEOUT

from .base import Executor
# Executor 协议（接口规范），FakeModelExecutor 遵循这个协议

if TYPE_CHECKING:
    # 只在类型检查时导入，避免循环依赖
    from ..cache.manager import BlockManager


class FakeModelExecutor:
    """Simulated executor: fake KV cache + fake model.
    # 模拟执行器：假 KV cache + 假模型
    #
    # 职责：
    # - 维护一个内存中的 KV cache（Dict[block_id, List[int]]）
    # - 监听 BlockAllocator 回调，同步创建/释放 KV 存储
    # - 运行 prefill：将 prompt token 写入 KV，计算第一个输出 token
    # - 运行 decode：读取 KV，生成下一个 token，写入生成 token 的 KV
    # - 假的 tokenizer/detokenizer，用于引擎集成测试

    Responsibilities:
    - Maintains an in-memory KV cache (Dict[block_id, List[int]])
    - Listens to BlockAllocator callbacks to create/release KV storage
    - Runs prefill: writes prompt tokens to KV, computes first output token
    - Runs decode: reads KV, produces next token, writes generated token KV
    - Fake tokenizer / detokenizer for engine integration

    Implements the ``Executor`` protocol.
    """

    def __init__(self, config: Config, block_manager: BlockManager | None = None) -> None:
        # 构造器
        # config: 全局配置（block_size, vocab_size 等）
        # block_manager: 可选的块管理器引用（没有时也能工作，降级为直接读 seq.block_table）
        self._config = config
        # 保存配置引用

        self._block_manager = block_manager
        # 保存 BlockManager 引用（可能为 None）

        self._model = FakeModel(vocab_size=config.vocab_size)
        # 创建一个假模型实例，词表大小由配置决定

        self._kv_cache: Dict[int, List[int]] = {}
        # KV cache 模拟：物理块 ID → [key1, value1, key2, value2, ...]
        # 每个块里按顺序存 key/value 对

        self._total_tokens_processed: int = 0
        # 总处理 token 计数器（用于统计）

    # ------------------------------------------------------------------
    # Block allocator callbacks
    # BlockAllocator 回调——当 allocator 分配/释放物理块时同步更新 KV 存储
    # ------------------------------------------------------------------

    def prepare_block(self, block_id: int) -> None:
        # 当 BlockAllocator 分配了一个新物理块时调用
        # 在 _kv_cache 里为该块创建一个空列表，准备存 key/value 数据
        self._kv_cache[block_id] = []

    def release_block(self, block_id: int) -> None:
        # 当 BlockAllocator 释放了一个物理块时调用
        # 从 _kv_cache 中移除该块的数据
        # 用 pop(..., None) 防止重复删除时抛异常
        self._kv_cache.pop(block_id, None)

    def make_block_allocator_callbacks(self) -> dict:
        # 构造回调字典，传给 BlockAllocator.set_callbacks()
        # 这样 allocator 分配块时自动调用 prepare_block，释放时自动调用 release_block
        return {
            "on_allocate": self.prepare_block,
            "on_free": self.release_block,
        }

    # ------------------------------------------------------------------
    # KV cache simulation
    # KV cache 模拟——读写假数据到 _kv_cache 字典
    # ------------------------------------------------------------------

    def _write_to_kv(self, seq: Sequence, token_position: int, token_id: int) -> None:
        """Write a token's KV data, allocating blocks on-demand.
        # 将某个 token 的 KV 数据写入 cache，必要时按需分配物理块
        #
        # 对于通过前缀缓存共享的块，KV 数据已经存在，跳过写入和计数

        For blocks shared via Prefix Cache (is_block_shared), the KV
        data already exists — we skip the write and the token count.
        """
        if self._block_manager is not None:
            # 如果有 BlockManager，用它来分配/查询块
            if self._block_manager.is_block_shared(seq, token_position):
                # 这个位置的块是通过前缀缓存共享的——数据已经由原始序列写好了
                # 不需要重复写入，也不需要增加计数
                # 但块已经在 allocate_for_seq 时预填入了 block_table
                return

            # 确保这个 token 位置对应的物理块已存在
            # 如果还没有，会触发 allocate（从 allocator 拿一个空闲块）
            pid = self._block_manager.ensure_block(seq, token_position)
        else:
            # 没有 BlockManager 时的降级方案：直接从 seq.block_table 拿物理块 ID
            pid = seq.block_table[token_position // self._config.block_size]

        # 用 FakeModel 生成假的 key 值和 value 值
        key = self._model._fake_key(token_id)
        value = self._model._fake_value(token_id)

        # 把 key+value 追加到该物理块对应的 KV 列表里
        self._kv_cache[pid].extend([key, value])

        # 总处理 token 数 +1
        self._total_tokens_processed += 1

    def _read_from_kv(self, seq: Sequence, token_position: int) -> int:
        """Read KV bias for a token position."""
        # 读取某个 token 位置的 KV "偏置值"
        # 返回一个整数，供 decode 时影响下一个 token 的生成

        if self._block_manager is not None:
            # 通过 BlockManager 确保块已存在，拿到物理块 ID
            pid = self._block_manager.ensure_block(seq, token_position)
        else:
            # 降级：直接按位置算逻辑块编号，从 seq.block_table 取物理块 ID
            pid = seq.block_table[token_position // self._config.block_size]

        # 从 KV cache 中取出该块的数据（可能为空）
        kv_data = self._kv_cache.get(pid, [])

        # 如果 KV 数据存在，将所有值求和后对词表大小取模，返回一个"伪 logit 偏置"
        # 如果数据不存在，返回 0（不影响采样）
        return sum(kv_data) % self._model.vocab_size if kv_data else 0

    # ------------------------------------------------------------------
    # Prefill / Decode
    # prefill（初次处理 prompt）/ decode（逐个生成 token）
    # ------------------------------------------------------------------

    def prefill(self, sequences: List[Sequence]) -> None:
        """Chunk-aware prefill: write from cursor, only complete when done.
        # 分块感知的 prefill：从光标位置开始写，只有写完了才算完成
        # 支持分块 prefill——一次可能只处理 prompt 的一部分
        """
        # chunk_size = 每一步最多处理多少个 prompt token（来自配置）
        chunk_size = self._config.max_prefill_chunk_size

        for seq in sequences:
            # 从 prefill 光标位置开始（上次停在哪，这次从哪继续）
            start = seq.prefill_cursor
            # 结束位置 = min(prompt 末尾, start + chunk_size)
            end = min(len(seq.prompt_token_ids), start + chunk_size)

            # 逐个 token 写入 KV cache
            for pos in range(start, end):
                # 写入位置 pos 处的 prompt token
                self._write_to_kv(seq, pos, seq.prompt_token_ids[pos])

            # 更新 prefill 光标位置
            seq.prefill_cursor = end

            if seq.is_prefill_finished:
                # prefill 完成了（所有 prompt token 都写入了 KV cache）
                # 用假模型生成第一个输出 token（基于 prompt 最后一个 token）
                first_token = self._model.prefill_token(seq.prompt_token_ids[-1])
                seq.output_token_ids = [first_token]
                # 已生成 token 数设为 1
                seq.num_generated_tokens = 1
                # 状态从 PREFILL 变为 RUNNING（进入 decode 阶段）
                seq.status = Status.RUNNING

    def decode(self, sequences: List[Sequence]) -> None:
        """Read KV, produce next token, write generated token KV."""
        # 读取已有 KV，生成下一个 token，将生成 token 的 KV 写回去

        for seq in sequences:
            # 当前需要读取的 KV 位置 = prompt 长度 + 已输出长度 - 1
            # 也就是最后一个已生成 token 的位置（用于 attention 计算）
            position = len(seq.prompt_token_ids) + len(seq.output_token_ids) - 1

            # 从 KV cache 中读取"偏置值"
            kv_bias = self._read_from_kv(seq, position)

            # 取最后一个已生成的 token
            prev = seq.output_token_ids[-1]

            # 用假模型基于上一个 token + KV 偏置生成下一个 token
            next_token = self._model.decode_token(prev, kv_bias)

            # 新 token 的位置（prompt 长度 + 已输出长度）
            new_pos = len(seq.prompt_token_ids) + len(seq.output_token_ids)
            # 将这个新 token 的 KV 数据写入 cache
            # 真实 LLM 也会做同样的事——生成的 token 也需要有 KV cache
            self._write_to_kv(seq, new_pos, next_token)

            # 将新 token 追加到输出列表
            seq.output_token_ids.append(next_token)
            # 已生成 token 数 +1
            seq.num_generated_tokens += 1

    # ------------------------------------------------------------------
    # Fake tokenizer / detokenizer
    # 假的 tokenizer/detokenizer：用 ASCII 码简单映射，不依赖真实 tokenizer
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> List[int]:
        """Convert a string to a list of fake token IDs."""
        # 将字符串转成假的 token ID 列表
        vocab = self._model.vocab_size
        # 简单映射：每个字符的 ASCII 码对词表大小取模
        return [ord(c) % vocab for c in prompt]

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to a fake string."""
        # 将 token ID 列表转回假的字符串
        chars = []
        for t in token_ids:
            # 映射回可打印 ASCII 范围（32~126）
            c = chr(t % 95 + 32)
            chars.append(c)
        return "".join(chars)

    # ------------------------------------------------------------------
    # Sequence cleanup
    # 序列清理
    # ------------------------------------------------------------------

    def cleanup_sequence(self, seq_id: str) -> None:
        """No-op: FakeModelExecutor doesn't maintain per-sequence state."""
        # 空操作：FakeModelExecutor 不维护序列级别的状态
        # QwenExecutor 才需要清理 per-sequence 的 past_key_values
        pass

    # ------------------------------------------------------------------
    # Stats
    # 统计信息
    # ------------------------------------------------------------------

    def get_kv_stats(self) -> dict:
        # 返回 KV cache 使用统计
        return {
            "kv_tokens_written": self._total_tokens_processed,
            # 已写入的 token 总数

            "kv_slot_capacity": len(self._kv_cache) * self._config.block_size,
            # KV cache 理论容量 = 已分配块数 × 每块 token 数

            "allocated_blocks": len(self._kv_cache),
            # 已分配的物理块数量
        }

    @property
    def total_tokens_processed(self) -> int:
        # 属性：总共处理了多少 token
        return self._total_tokens_processed
