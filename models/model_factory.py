import torch
import torch.nn as nn
from typing import Any, Tuple, List, Callable

from models.gat import GAT
from models.graphsage import GraphSAGE


class ModelFactory:
    """
    Factory class to initialize specific GNN models based on input data
    dimensions (num_features, num_classes).

    The methods return only the PyTorch model (nn.Module).
    Task and pretraining definitions are handled externally.

    Individual initialization methods are placeholders, raising NotImplementedError,
    until the concrete model logic is added.
    """

    @staticmethod
    def initialize_gcn(data: Any, labels: Any) -> nn.Module:
        """Initializes a generic GNN model (e.g.,  GCN)."""
        raise NotImplementedError("GNN model initialization is not implemented.")

    @staticmethod
    #TODO Replace with multi-layer gcn argument
    def initialize_deep_gcn(data: Any, labels: Any) -> nn.Module:
        """Initializes a Deep GCN model."""
        raise NotImplementedError("Deep GCN model initialization is not implemented.")

    @staticmethod
    def initialize_gpt_gnn(data: Any, labels: Any) -> nn.Module:
        """Initializes a GPT-GNN model."""
        raise NotImplementedError("GPT-GNN model initialization is not implemented.")

    @staticmethod
    def initialize_structg(data: Any, labels: Any) -> nn.Module:
        """Initializes a Struct-G model."""
        # This method is used by two separate pipeline configurations
        raise NotImplementedError("Struct-G model initialization is not implemented.")

    @staticmethod
    def initialize_graphsage(data: Any, labels: Any) -> nn.Module:
        """Initializes a GraphSAGE model."""
        # Assuming GraphSAGE is available in the current context

        if hasattr(data, 'num_features'):
            in_dim = data.num_features
        elif hasattr(data, 'x') and hasattr(data.x, 'shape'):
            in_dim = data.x.shape[-1]
        else:
            raise ValueError("Cannot determine input dimension (in_dim) from data.")

        if hasattr(labels, 'num_classes'):
            num_classes = labels.num_classes
        elif isinstance(labels, torch.Tensor):
            num_classes = len(torch.unique(labels))
        else:
            raise ValueError("Cannot determine number of classes (num_classes) from labels.")

        # According to the GNNFactory contract, we return a single nn.Module.
        # We initialize the model with the dimension needed for the final downstream task (classification).
        base_model = GraphSAGE(in_channels=in_dim, out_channels=num_classes)
        return base_model

    @staticmethod
    def initialize_gat(data: Any, labels: Any) -> nn.Module:
        """Initializes a GAT model."""

        if hasattr(data, 'num_features'):
            in_dim = data.num_features
        elif hasattr(data, 'x') and hasattr(data.x, 'shape'):
            in_dim = data.x.shape[-1]
        else:
            raise ValueError("Cannot determine input dimension (in_dim) from data.")

        if hasattr(labels, 'num_classes'):
            num_classes = labels.num_classes
        elif isinstance(labels, torch.Tensor):
            num_classes = len(torch.unique(labels))
        else:
            raise ValueError("Cannot determine number of classes (num_classes) from labels.")
        heads = 8
        base_model = GAT(in_channels=in_dim, out_channels=num_classes, heads=heads)
        return base_model

    # --- Initialization of All Models ---

    @staticmethod
    def initialize_all() -> List[Tuple[str, Callable[[Any, Any], nn.Module]]]:
        """
        Returns a list of tuples, where each tuple contains:
        1. The user-friendly name of the pipeline/model.
        2. The static function to initialize that specific model.

        Note: The two Struct-G pipelines are separate logically, so we duplicate
        the setup function call with their distinct pipeline names.
        """
        return [
            # I use initialize_gnn here to match the pipeline name "GNN"
            ("GNN", ModelFactory.initialize_gcn),
            ("Deep GCN", ModelFactory.initialize_deep_gcn),
            ("GPT-GNN", ModelFactory.initialize_gpt_gnn),
            ("Struct-G Structural Only Pretrain", ModelFactory.initialize_structg),
            ("GraphSAGE", ModelFactory.initialize_graphsage),
            ("GAT", ModelFactory.initialize_gat),
        ]