from __future__ import annotations

from ..config import Config
from ..executor.executor import FakeModelExecutor


class FakeWorker:
    """Worker that owns a FakeModelExecutor (simulates GPU worker)."""

    def __init__(self, config: Config) -> None:
        self.executor = FakeModelExecutor(config)
        self.config = config

    def get_executor(self) -> FakeModelExecutor:
        return self.executor
