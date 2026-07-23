from .base import Executor
# 导出 Executor 协议（接口规范）

from .executor import FakeModelExecutor
# 导出 FakeModelExecutor（假模型执行器，用于测试和原型验证）

__all__ = ["Executor", "FakeModelExecutor"]
# 公开 API：from mini_vllm.executor import * 时只导出这两个名字
# QwenExecutor 不在此处导出，需要时由用户手动 import
