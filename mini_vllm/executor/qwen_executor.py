from __future__ import annotations
# 惰性类型注解

import time
# 用于记录 prefill 完成时间（first_token_time）

from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple
# 类型注解工具

import torch
# PyTorch——用于模型推理、张量操作

from ..config import Config
# 全局配置

from ..sequence.sequence import Sequence
# 序列对象

from ..sequence.status import Status
# 状态枚举

if TYPE_CHECKING:
    # 只在类型检查时导入，避免循环依赖
    from ..cache.manager import BlockManager


_HF_CACHE: Optional[str] = None
# 可选的 HF_HOME 覆盖路径
# 可以在导入本模块之前设置，指定 HuggingFace 模型下载目录


def _get_model_and_tokenizer() -> Tuple[Any, Any]:
    """Lazy-load Qwen2-0.5B from HuggingFace.
    # 从 HuggingFace 懒加载 Qwen2-0.5B 模型和 tokenizer
    #
    # 引入放在函数内部，这样即使没有 torch/transformers 也能导入 mini_vllm
    # （只有走 QwenExecutor 路径才需要它们）

    Import is inside the function so ``mini_vllm`` can be imported without
    torch/transformers installed (only the QwenExecutor path needs them).
    """
    import os
    # os.environ 用于设置 HF_HOME

    from transformers import AutoModelForCausalLM, AutoTokenizer
    # HuggingFace 的模型和 tokenizer 加载工具
    # 放在函数内部 = 懒导入，只有真正调这个函数时才 import

    if _HF_CACHE is not None:
        # 如果设置了 HF 缓存目录，就覆盖环境变量
        os.environ["HF_HOME"] = _HF_CACHE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 优先用 GPU，没有 GPU 就回退到 CPU

    dtype = torch.float16 if device == "cuda" else torch.float32
    # GPU 用 float16 省显存，CPU 用 float32（CPU 上 float16 通常没加速）

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        # 加载 Qwen2 0.5B 参数量的模型
        torch_dtype=dtype,
        device_map=device,
        # 自动分配到设备（GPU 或 CPU）
        use_cache=True,
        # 启用 KV cache（transformers 原生）
    )
    model.eval()
    # 切到评估模式（关闭 dropout/BN 等训练专用层）

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    # 加载对应的 tokenizer

    if tokenizer.pad_token_id is None:
        # 如果 tokenizer 没有 pad_token，用 eos_token 代替
        # 这是常见处理，因为 Qwen 没有专门定义 pad_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
    # 返回 (model, tokenizer) 元组


class QwenExecutor:
    """Real model executor using Qwen2-0.5B via HuggingFace Transformers.
    # 真正的模型执行器——通过 HuggingFace Transformers 使用 Qwen2-0.5B
    #
    # 职责：
    # - 懒加载 Qwen2-0.5B 模型 + tokenizer（首次实例化时才加载）
    # - 运行真正的 prefill（prompt → KV cache → 第一个 token）
    # - 运行真正的 decode（KV cache → 下一个 token）
    # - 管理 per-sequence 的 past_key_values，支持连续批处理
    # - 追踪 per-block KV 存储以响应 BlockAllocator 回调
    # - cleanup_sequence() 时清理 per-sequence 状态

    Responsibilities:
    - Loads Qwen2-0.5B model + tokenizer (lazy, on first instantiation)
    - Runs real prefill (prompt → KV cache → first token)
    - Runs real decode (KV cache → next token)
    - Manages per-sequence ``past_key_values`` for continuous batching
    - Tracks per-block KV storage for BlockAllocator callbacks
    - Cleans up per-sequence state when ``cleanup_sequence()`` is called

    Implements the ``Executor`` protocol.
    """

    def __init__(self, config: Config, block_manager: BlockManager | None = None) -> None:
        # 构造器
        self._config = config
        # 保存配置

        self._block_manager = block_manager
        # 保存 BlockManager 引用

        self._model, self._tokenizer = _get_model_and_tokenizer()
        # 加载 Qwen2-0.5B 模型和 tokenizer

        self._model_config = self._model.config
        # 保存模型配置（hidden_size, num_layers 等）

        # Per-block KV tracking: {block_id: list_of_token_positions_in_block}
        # 按块追踪 KV：block_id → 该块中所有 token 位置的列表
        # 实际的张量数据存在 _seq_kv 里，按 seq_id 索引
        self._kv_cache: Dict[int, List[int]] = {}

        # Per-sequence past_key_values (transformers KV cache format)
        # per-sequence 的 past_key_values（transformers 的原生 KV cache 格式）
        # 字典：seq_id → tuple of (key_states, value_states) per layer
        # transformers 的标准格式是 tuple of tuples
        self._seq_kv: Dict[str, Any] = {}

        self._total_tokens_processed: int = 0
        # 总处理 token 计数器

    # ------------------------------------------------------------------
    # Block allocator callbacks
    # BlockAllocator 回调——分配块时创建 KV 槽位，释放时清理
    # ------------------------------------------------------------------

    def prepare_block(self, block_id: int) -> None:
        # 分配新块时：在 _kv_cache 中创建空列表，准备记录 token 位置
        self._kv_cache[block_id] = []

    def release_block(self, block_id: int) -> None:
        # 释放块时：从 _kv_cache 中移除该块的记录
        self._kv_cache.pop(block_id, None)

    def make_block_allocator_callbacks(self) -> dict:
        # 构造回调字典，供 BlockAllocator.set_callbacks() 使用
        return {
            "on_allocate": self.prepare_block,
            "on_free": self.release_block,
        }

    # ------------------------------------------------------------------
    # Tokenization
    # 使用 Qwen 真实 tokenizer
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> List[int]:
        # 用 Qwen 的 tokenizer 将文本编码成 token ID 列表
        return self._tokenizer.encode(prompt, add_special_tokens=True)

    def detokenize(self, token_ids: List[int]) -> str:
        # 用 Qwen 的 tokenizer 将 token ID 列表解码回文本
        # skip_special_tokens=True 去掉特殊 token（如 <|endoftext|>）
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Execution
    # 执行：prefill（处理 prompt）/ decode（逐个生成 token）
    # 用 @torch.no_grad() 关闭梯度计算，节省显存和计算
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prefill(self, sequences: List[Sequence]) -> None:
        # prefill：处理 prompt tokens，写入 KV cache，生成第一个输出 token
        # 支持分块 prefill（chunked prefill），一次可能只处理 prompt 的一部分

        chunk_size = self._config.max_prefill_chunk_size
        # 每步最多处理多少个 prompt token

        device = self._model.device
        # 模型所在的设备（cuda:0 或 cpu）

        for seq in sequences:
            # 从 prefill 光标位置开始
            start = seq.prefill_cursor
            # 本次处理的结束位置 = min(prompt 末尾, start + chunk_size)
            end = min(len(seq.prompt_token_ids), start + chunk_size)
            # 取出这部分 token ID
            token_ids = seq.prompt_token_ids[start:end]

            # 转为 PyTorch 张量，送到模型所在设备
            # shape = [1, chunk_size]（batch_size=1, seq_len=chunk_size）
            input_ids = torch.tensor([token_ids], device=device)

            # 获取该序列已有的 past_key_values（如果是第一块 prefill，则为 None）
            past_kv = self._seq_kv.get(seq.seq_id)

            # 调用 transformers 模型做 forward pass
            outputs = self._model(
                input_ids=input_ids,
                past_key_values=past_kv,
                # 传入已有的 KV cache，模型会自动追加新 token 的 KV
                use_cache=True,
                # 返回更新后的 past_key_values
                attention_mask=self._build_attention_mask(
                    seq, past_kv, len(token_ids), device
                ),
                # 构建注意力掩码，覆盖已有 KV 和新 token
            )

            # 保存更新后的 past_key_values
            # transformes 会自动将新 token 的 KV 追加到 past_key_values 末尾
            self._seq_kv[seq.seq_id] = outputs.past_key_values

            # 为每个写入的 token 更新块追踪
            for pos in range(start, end):
                if self._block_manager is not None:
                    # 确保该位置对应的物理块已存在（按需分配）
                    block_id = self._block_manager.ensure_block(seq, pos)
                    # 记录这个 token 属于哪个块
                    self._kv_cache.setdefault(block_id, []).append(pos)

            # 更新总处理计数
            self._total_tokens_processed += end - start
            # 更新 prefill 光标
            seq.prefill_cursor = end

            if seq.is_prefill_finished:
                # prefill 完成（整个 prompt 处理完了）
                # 取最后一个位置的 logits（shape = [1, seq_len, vocab_size]）
                logits = outputs.logits[0, -1, :]
                # 采样下一个 token（贪心：argmax）
                next_token = self._sample_token(logits)

                # 设置第一个输出 token
                seq.output_token_ids = [next_token]
                seq.num_generated_tokens = 1
                # 状态转为 RUNNING，进入 decode 阶段
                seq.status = Status.RUNNING
                # 记录首次生成 token 的时间（用于计算 TTFT）
                seq.first_token_time = time.time()

    @torch.no_grad()
    def decode(self, sequences: List[Sequence]) -> None:
        # decode：读已有 KV cache + 上一个 token，生成下一个 token
        # 每条序列这一步只生成一个 token

        device = self._model.device

        for seq in sequences:
            # 解码时输入的是上一个已生成的 token
            prev_token = seq.output_token_ids[-1]
            # shape = [1, 1]（batch_size=1, 一个 token）
            input_ids = torch.tensor([[prev_token]], device=device)

            # 获取该序列的过去 KV cache
            past_kv = self._seq_kv.get(seq.seq_id)

            # 调用模型 forward：输入 1 个 token + 全部历史 KV cache
            # 模型只需做一次 attention（新 token query 对所有已有 KV 做 attention）
            # 然后自动把新 token 的 KV 追加到 past_key_values 末尾
            outputs = self._model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
                # 不需要传 attention_mask——transformers 会基于 past_kv 的
                # 长度自动构建 causal mask
            )

            # 保存更新后的 past_key_values
            self._seq_kv[seq.seq_id] = outputs.past_key_values

            # 从 logits 采样下一个 token
            logits = outputs.logits[0, -1, :]
            next_token = self._sample_token(logits)

            # 在 BlockManager 中追踪新 token 的 KV 写入
            # 新 token 的位置 = prompt 长度 + 已经输出的 token 数
            new_pos = len(seq.prompt_token_ids) + len(seq.output_token_ids)
            if self._block_manager is not None:
                # 确保该位置有物理块（按需分配）
                block_id = self._block_manager.ensure_block(seq, new_pos)
                # 记录 token 位置到块追踪字典
                self._kv_cache.setdefault(block_id, []).append(new_pos)

            # 更新计数
            self._total_tokens_processed += 1

            # 将新 token 追加到输出列表
            seq.output_token_ids.append(next_token)
            seq.num_generated_tokens += 1

    # ------------------------------------------------------------------
    # Sequence cleanup
    # 序列清理
    # ------------------------------------------------------------------

    def cleanup_sequence(self, seq_id: str) -> None:
        """Remove per-sequence KV cache for a finished sequence.
        # 移除已完成序列的 per-sequence KV cache
        #
        # 块级别的 KV 追踪（_kv_cache）由 release_block() 回调清理
        # ——当 BlockAllocator 释放块时会自动触发
        # 这里只清理 per-sequence 的 past_key_values（transformers 格式）

        Block-level KV tracking (``_kv_cache``) is cleaned up by
        ``release_block()`` callbacks when the BlockAllocator frees
        blocks.  Here we only clean the per-sequence past_key_values.
        """
        self._seq_kv.pop(seq_id, None)

    # ------------------------------------------------------------------
    # Stats
    # 统计信息
    # ------------------------------------------------------------------

    def get_kv_stats(self) -> Dict[str, int]:
        # 返回 KV cache 使用统计
        return {
            "kv_tokens_written": self._total_tokens_processed,
            # 已写入的 token 总数
            "kv_slot_capacity": len(self._kv_cache) * self._config.block_size,
            # KV cache 容量 = 已分配块数 × 每块 token 数
            "allocated_blocks": len(self._kv_cache),
            # 已分配的物理块数
        }

    @property
    def total_tokens_processed(self) -> int:
        # 属性：总共处理了多少 token
        return self._total_tokens_processed

    # ------------------------------------------------------------------
    # Internal helpers
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_token(logits: torch.Tensor) -> int:
        """Greedy sampling (argmax)."""
        # 贪心采样：取 logits 中值最大的那个 token ID
        # argmax → 概率最高的 token
        # .item() 将单元素张量转成 Python int
        return torch.argmax(logits).item()

    @staticmethod
    def _build_attention_mask(
        seq: Sequence,
        past_kv: Any,
        new_tokens: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Build attention mask for prefill with existing past_key_values.
        # 为已有 past_key_values 的 prefill 构建注意力掩码
        #
        # 对于分块 prefill，序列的一部分 token 已经在 KV cache 中了
        # 注意力掩码必须覆盖已有 token 和新 token 两部分

        For chunked prefill, the sequence already has some tokens in KV.
        The attention mask must cover both existing and new tokens.
        """
        if past_kv is None:
            # 没有已有 KV cache（第一块 prefill）
            # 返回 None——transformers 会自动创建 causal mask
            return None

        # 总长度 = 已有 token 数（prefill_cursor）+ 新 token 数
        total_len = seq.prefill_cursor + new_tokens
        # 创建一个全 1 的 mask，表示所有位置都可见
        # shape = [1, total_len]
        mask = torch.ones(1, total_len, dtype=torch.long, device=device)
        return mask
