from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BlockTableEntry:
    """A single entry in the logical-to-physical block mapping.
     "逻辑块→物理块映射表中的一条记录"。dataclass 自动生成构造器、repr、eq 等方法
     真实 PagedAttention 需要的 CUDA kernel 是在 executor 层调用，在 独立的 .cu / .py 文件（使用 torch.utils.cpp_extension 或 triton）里实现。这个 mini 项目只做到了管理层面，计算层面用的
  fake，所以看起来 "kernel 不存在"。
    ``is_shared`` is ``True`` when this block was obtained via Prefix
    Cache (i.e., the block belongs to another sequence and we are
    sharing it).  In the future, this flag will trigger Copy-on-Write:
    when a sequence needs to write to a shared block, the executor
    allocates a new block, copies data, and replaces this entry.
    """
    physical_block_id: int # 这个逻辑块对应的是哪块物理块（整数编号）
    is_shared: bool = False#  这个块是否是和其他序列共享的（默认 False） 前缀是共享的（不能直接改），但后续生成的新Token要追加写入。为了防篡改，只能先复制一份新的，再往新块里写。

    def to_tuple(self) -> Tuple[int, bool]:
        return (self.physical_block_id, self.is_shared) #把记录转成元组 (物理块ID, 是否共享)，方便序列化或解包


class BlockTable:
    """Logical-to-physical block mapping (PagedAttention).
    "PagedAttention 的逻辑→物理块映射" — 这就是 PagedAttention 的核心思想：让连续的逻辑块可以映射到不连续的物理块
     Each entry maps a logical block → physical block.  Entries may be
    shared (multiple BlockTables pointing to the same physical block
    via Prefix Cache).  The ``is_shared`` flag on each entry enables
    future Copy-on-Write: when a shared block needs to diverge, the
    writer allocates a new physical block and replaces the entry.
    """

    def __init__(self, request_id: str, block_size: int) -> None:
        self._request_id = request_id
        self._block_size = block_size
        self._entries: List[BlockTableEntry] = [] #是这个表的核心：按顺序排列的逻辑块列表，entries[0] 是第 0 逻辑块对应的物理块

    def add_block(self, physical_block_id: int) -> None:
        """Add a non-shared block (owned exclusively by this sequence)."""
        #添加一个独占块 — 新增一个逻辑块→物理块的映射，标记为非共享（is_shared=False）。这是常规情况：新分配的物理块只属于自己
        self._entries.append(BlockTableEntry(physical_block_id, is_shared=False))

    def add_shared_block(self, physical_block_id: int) -> None:
        """Add a block shared via Prefix Cache.

        The physical block is owned by another sequence; this entry
        shares it via reference counting in the BlockAllocator.
        """
        # 添加一个共享块 — 逻辑块映射到一个已有的物理块（来自前缀缓存），标记为共享。这个物理块同时属于多个序列的 BlockTable，底层通过引用计数管理生命周期
        self._entries.append(BlockTableEntry(physical_block_id, is_shared=True))

    def clear(self) -> None:
        self._entries.clear()
    #清空所有条目。BlockManager.free() 在释放序列时调用这个
    def num_blocks(self) -> int:
        #当前该序列占了多少个逻辑块（注意不是物理块，因为多块可能指向同一物理块）
        return len(self._entries)

    def get_block_ids(self) -> List[int]:
        #把整张表展开成物理块 ID 列表，按逻辑顺序排列。这个列表会被写回 Sequence.block_table（Sequence 对象上有个同名字段存这个）
        return [e.physical_block_id for e in self._entries]

    def get_shared_flags(self) -> List[bool]:
        """Return whether each block is shared (for COW detection)."""
        #返回每个逻辑块是否共享的标记列表。执行器用这个来判断哪些块需要走 Copy-on-Write
        return [e.is_shared for e in self._entries]

    def get_entries(self) -> List[BlockTableEntry]:
        """Return the full entries list (for COW mutation)."""
        # 返回完整的条目列表（拷贝），供外部遍历检查。返回拷贝是为了防止外部修改内部状态
        return list(self._entries)

    def get_physical_block(self, token_position: int) -> int | None:
        """ "给 token 位置，返回物理块 ID" — 这是最常用的查询方法。执行器在推理时调用：
            - token_position=5，block_size=4 → logical_idx=1 → 返回 entries[1].physical_block_id
            - 如果这个 logical_idx 还没分配（超出 entries 长度），返回 None             """
        logical_idx = token_position // self._block_size
        if logical_idx < len(self._entries):
            return self._entries[logical_idx].physical_block_id
        return None

    def is_shared_at(self, token_position: int) -> bool:
        """Check whether the block at this token position is shared.
         "这个 token 位置对应的块是共享的吗？" 执行器在 decode 写 KV cache 前调用。如果返回 True，说明是共享块，不能直接覆盖，需要走 COW
        Used by the executor during decode to detect shared blocks
        that need Copy-on-Write before writing.
        """
        logical_idx = token_position // self._block_size
        if logical_idx < len(self._entries):
            return self._entries[logical_idx].is_shared
        return False

    def export_block_table_tensor(self, max_blocks: int) -> "torch.Tensor":
        """Export the block ID list as a padded ``torch.Tensor``.

        Shape: ``[max_blocks]`` with ``-1`` sentinel padding for unused
        trailing entries.  This is the format consumed by the GPU
        PagedAttention kernel.
        """
        import torch
        ids = self.get_block_ids()
        padded = ids + [-1] * (max_blocks - len(ids))
        return torch.tensor(padded[:max_blocks], dtype=torch.long)

    def dump_mapping(self) -> List[dict]:
        """ 导出完整的映射表，格式是 [{"logical": 0, "physical": 3, "shared": False}, ...]。调试/测试用 调试打印：BlockTable(request=req-001, blocks=[3, 7, 2])"""
        return [
            {"logical": i, "physical": e.physical_block_id, "shared": e.is_shared}
            for i, e in enumerate(self._entries)
        ]

    def __repr__(self) -> str:
        ids = [e.physical_block_id for e in self._entries]
        return f"BlockTable(request={self._request_id}, blocks={ids})"
