from abc import ABC, abstractmethod
from typing import Any, Tuple

from torch import nn


class Task(ABC):
    """A self-contained unit of computation inside a pipeline."""

    def __init__(self, name):
        self.name = name
        self.metadata = {}

    @abstractmethod
    def prepare(self, data):
        """Optional: create masks, split edges, etc."""
        return data

    @abstractmethod
    def train(self, model, data):
        """Train a task-specific head."""
        ...

    @abstractmethod
    def evaluate(self, model, data):
        """Return evaluation results + updated metadata."""
        ...



class Pretrain:
    __slots__ = ("objective", "name", "metadata")

    def __init__(self, objective: "Task", name: str | None = None):
        self.objective: "Task" = objective
        self.name: str = name or getattr(objective, "name", "pretrain")
        self.metadata: dict = {}

    def run(self, model: nn.Module, data: Any) -> Tuple[nn.Module, dict]:
        data = self.objective.prepare(data)
        model = self.objective.train(model, data)
        self.metadata = getattr(self.objective, "metadata", {})
        return model, self.metadata
