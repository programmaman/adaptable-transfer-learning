import os
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from sklearn.decomposition import TruncatedSVD
from torch import nn
import torch.nn.functional as functional
from utils import logger


class Task(ABC):
    """A self-contained unit of computation inside a pipeline."""

    def __init__(self, name: str, epochs: int = 30):
        self.name = name
        self.epochs = epochs  # default number of training epochs
        self.metadata = {}
        logger.info(f"Task initialized: {self.name} | epochs={self.epochs}")

    @abstractmethod
    def prepare(self, data):
        """Optional: create masks, split edges, etc."""
        logger.info(f"[{self.name}] Preparing data...")
        return data

    @abstractmethod
    def train(self, model, data):
        """Train a task-specific head."""
        logger.info(f"[{self.name}] Starting training...")
        return model

    @abstractmethod
    def evaluate(self, model, data):
        """Return evaluation results + updated metadata."""
        logger.info(f"[{self.name}] Starting evaluation...")
        ...



class Pretrain:
    __slots__ = ("objective", "name", "metadata")

    def __init__(self, objective: "Task", name: str | None = None):
        self.objective: "Task" = objective
        self.name: str = name or getattr(objective, "name", "pretrain")
        self.metadata: dict = {}
        logger.info(f"Pretrain wrapper initialized. Objective: {self.objective.name}")

    def run(self, model: nn.Module, data: Any) -> Tuple[nn.Module, dict]:
        logger.info(f"\n----- Starting Pretrain Run: {self.name} -----")

        # 1. Prepare data using the objective's prepare method
        data = self.objective.prepare(data)

        # 2. Train the model using the objective's train method
        model = self.objective.train(model, data)

        # 3. Retrieve metadata
        self.metadata = getattr(self.objective, "metadata", {})

        logger.info(f"----- Pretrain Run Complete: {self.name} -----")
        return model, self.metadata



class GraphLoRAPretrain(Task):
    def __init__(self, base_model_path: str, seed: int = 42, epochs: int = 10,
                 name: str = "graph_lora_pretrain"):
        super().__init__(name, epochs=epochs)
        self.base_model_path = base_model_path
        self.seed = seed
        self.metadata = {}

    def prepare(self, data: Any):
        return data

    def train(self, model: nn.Module, data: Any, lr=0.01, weight_decay=5e-4,
              feat_reduce_dim=256, safety_factor=0.7) -> nn.Module:
        logger.info(f"[{self.name}] Starting GraphLoRA pretraining...")

        device = next(model.parameters()).device
        data = data.to(device)

        # Memory-aware feature reduction
        n_nodes, n_features = data.x.size()
        required_bytes = n_nodes * n_features * 4 * 2
        use_reduction = False
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0]
            if required_bytes > free_mem * safety_factor:
                use_reduction = True

        if use_reduction:
            logger.info(f"[{self.name}] Reducing features from {n_features} → {feat_reduce_dim}")
            x_cpu = data.x.cpu().numpy()
            svd = TruncatedSVD(n_components=feat_reduce_dim, random_state=self.seed)
            x_reduced = torch.tensor(svd.fit_transform(x_cpu), dtype=torch.float32)
            data = data.__class__(x=x_reduced.to(device), edge_index=data.edge_index)
            if hasattr(model, "reset_with_input_dim"):
                model.reset_with_input_dim(data.x.size(1))

        # Decoder + optimizer
        decoder = nn.Linear(model.gnn_frozen.conv[-1].out_channels, data.x.size(1)).to(device)
        optimizer = torch.optim.Adam(
            list(model.gnn_frozen.parameters()) + list(decoder.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        # Training loop uses self.epochs
        for epoch in range(self.epochs):
            model.gnn_frozen.train()
            decoder.train()
            optimizer.zero_grad()
            emb = model.gnn_frozen(data.x, data.edge_index)
            recon = decoder(emb)
            loss = functional.mse_loss(recon, data.x)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"[{self.name}] Epoch {epoch+1}: Loss {loss.item():.4f}")

        # Save pretrained weights
        os.makedirs(os.path.dirname(self.base_model_path), exist_ok=True)
        torch.save(model.gnn_frozen.state_dict(), self.base_model_path)
        logger.info(f"[{self.name}] Saved pretrained weights to {self.base_model_path}")

        return model

    def evaluate(self, model: nn.Module, data: Any):
        class Result:
            accuracy = None
            f1 = None
            metadata = {}
        return Result()


