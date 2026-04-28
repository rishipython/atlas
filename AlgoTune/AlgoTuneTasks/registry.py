"""Compatibility alias for the minimal local task registry."""

from .base import TASK_REGISTRY, Task, register_task

__all__ = ["TASK_REGISTRY", "Task", "register_task"]

