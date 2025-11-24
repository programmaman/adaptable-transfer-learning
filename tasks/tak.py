from abc import ABC, abstractmethod


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
