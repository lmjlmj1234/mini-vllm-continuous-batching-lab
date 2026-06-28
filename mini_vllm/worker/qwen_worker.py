from __future__ import annotations

from ..config import Config
from ..executor.qwen_executor import QwenExecutor


class QwenWorker:
    """Worker that owns a QwenExecutor (real HuggingFace model on GPU/CPU).

    Responsibilities:
    - Loads Qwen2-0.5B via QwenExecutor
    - Returns the executor to EngineCore via ``get_executor()``

    Follows the same pattern as ``FakeWorker``, ensuring the engine
    can swap workers without changing scheduling or memory management.
    """

    def __init__(self, config: Config) -> None:
        self.executor = QwenExecutor(config)
        self.config = config

    def get_executor(self) -> QwenExecutor:
        return self.executor
