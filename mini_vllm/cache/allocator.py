from __future__ import annotations # 惰性类型注解
from typing import Callable, List, Optional


class BlockAllocator:
    """Low-level physical block pool with reference-count tracking.

    Each physical block carries a ``ref_count``.  When a block is shared
    via Prefix Cache, ``increment_ref()`` bumps the count.  ``free()``
    decrements the count and only truly releases the block when the
    reference count reaches zero.

    This design naturally supports Copy-on-Write: when a shared block
    needs to diverge (two sequences write different tokens to the same
    logical position), the writer allocates a *new* physical block,
    copies data, and decrements the original's ref_count.
    """
    # 解释引用计数机制：
    # - 每个物理块有个引用计数器ref_count
    # - 多个序列可以共享同一个物理块（通过前缀缓存）
    # - increment_ref() = 加引用（共享时用）
    # - free() = 减引用，只有减到0才真正释放

    def __init__(
        self,
        num_blocks: int,# num_blocks — 总共有多少个物理块（对应 Config 里的 num_gpu_blocks）
        on_allocate: Optional[Callable[[int], None]] = None, #预留一个监控钩子；on_allocate / on_free — 可选回调，分配/释放块时的钩子，可用于日志或监控
        on_free: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._num_blocks = num_blocks #  保存总块数

        # Free-list: True = free, False = in use
        self._free: List[bool] = [True] * num_blocks #空闲列表：True = 空闲，False = 在用。一开始全是 True

        # Reference count per block.  0 = free; >0 = number of references.
        self._ref_counts: List[int] = [0] * num_blocks

        self._on_allocate = on_allocate # 保存回调函数引用（如果有的话）
        self._on_free = on_free

    def set_callbacks(
        self,
        on_allocate: Optional[Callable[[int], None]] = None,
        on_free: Optional[Callable[[int], None]] = None,
    ) -> None:
        if on_allocate is not None:
            self._on_allocate = on_allocate
        if on_free is not None:
            self._on_free = on_free

    # ------------------------------------------------------------------
    # Allocation / Free with ref-count semantics
    # ------------------------------------------------------------------

    def allocate(self, num_blocks: int) -> Optional[List[int]]:
        """Allocate *num_blocks* physical blocks.

        Sets ``ref_count = 1`` for each newly allocated block.
        Returns ``None`` if insufficient free blocks.
        """
        if num_blocks > self.num_free_blocks:# "够用才分配" — 空闲不够就直接返回 None，表示分配失败。上层调用者（BlockManager）需要处理这种情况
            return None

        indices: List[int] = []
        for i, free in enumerate(self._free):# 从前往后扫描空闲列表，找到第一个空闲块就加入结果，凑够需要的数量就停。这是最简单的"首次适应"策略
            if free:
                indices.append(i)
                if len(indices) == num_blocks:
                    break

        for pid in indices:# 标记占用、设置初始引用计数为 1，触发分配回调（如果有的话）
            self._free[pid] = False
            self._ref_counts[pid] = 1  # first reference
            if self._on_allocate:
                self._on_allocate(pid)

        return indices # 返回分配到的物理块 ID 列表

    def free(self, physical_block_ids: List[int]) -> None:
        """Release one reference per block.
        传入一批块 ID，每个减一次引用
        The block is only returned to the free pool when its ref_count
        reaches zero.  Calling ``free`` on a block that this caller does
        *not* own is safe — it merely decrements the shared count.
        """
        for pid in physical_block_ids:
            if self._ref_counts[pid] == 0:
                continue  # already free; guard against double-free
            self._ref_counts[pid] -= 1
            if self._ref_counts[pid] == 0:
                self._free[pid] = True
                if self._on_free:
                    self._on_free(pid)

    def increment_ref(self, pid: int) -> None:
        """Bump reference count for a shared block (Prefix Cache)."""
        assert not self._free[pid], f"Cannot increment ref on free block {pid}"
        self._ref_counts[pid] += 1

    def get_ref_count(self, pid: int) -> int:
        return self._ref_counts[pid]

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return sum(self._free)

    @property
    def num_total_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_used_blocks(self) -> int:
        return self._num_blocks - self.num_free_blocks

    def dump_free_list(self) -> List[int]:
        return [i for i, free in enumerate(self._free) if free]

    def dump_used_list(self) -> List[int]:
        return [i for i, free in enumerate(self._free) if not free]

    def dump_ref_counts(self) -> List[int]:
        """Expose ref counts (for debugging / tests)."""
        return list(self._ref_counts)

    def stats(self) -> dict:
        return {
            "total_blocks": self._num_blocks,
            "free_blocks": self.num_free_blocks,
            "used_blocks": self.num_used_blocks,
        }
