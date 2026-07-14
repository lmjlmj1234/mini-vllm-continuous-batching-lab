"""Minimal worker wrapper for PagedExecutor.

Follows the same pattern as ``FakeWorker`` / ``QwenWorker`` but uses
the PagedExecutor directly.
"""

from __future__ import annotations

from ..config import Config
from ..executor.paged_executor import PagedExecutor


class PagedWorker:
    """Worker wrapping a PagedExecutor."""

    def __init__(self, config: Config) -> None:
        self._executor = PagedExecutor(config)

    def get_executor(self) -> PagedExecutor:
        return self._executor
