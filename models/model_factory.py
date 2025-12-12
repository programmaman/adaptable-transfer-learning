import torch.nn as nn
from typing import Any, Tuple, List, Callable, Dict, Union

# The return type is now just the PyTorch model
ModelSetup = nn.Module


class GNNFactory:
    """
    Factory class to initialize specific GNN models based on input data
    dimensions (num_features, num_classes).

    The methods return only the PyTorch model (nn.Module).
    Task and pretraining definitions are handled externally.

    Individual initialization methods are placeholders, raising NotImplementedError,
    until the concrete model logic is added.
    """

    # --- Individual Model Initialization Methods ---

    @staticmethod
    def initialize_gcn(data: Any, labels: Any) -> ModelSetup:
        """Initializes a generic GNN model (e.g.,  GCN)."""
        # Note: I renamed this from initialize_gcn to initialize_gnn
        # to match the name used in your initialize_all return list ("GNN").
        raise NotImplementedError("GNN model initialization is not implemented.")

    @staticmethod
    #TODO Replace with multi-layer gcn argument
    def initialize_deep_gcn(data: Any, labels: Any) -> ModelSetup:
        """Initializes a Deep GCN model."""
        raise NotImplementedError("Deep GCN model initialization is not implemented.")

    @staticmethod
    def initialize_gpt_gnn(data: Any, labels: Any) -> ModelSetup:
        """Initializes a GPT-GNN model."""
        raise NotImplementedError("GPT-GNN model initialization is not implemented.")

    @staticmethod
    def initialize_structg(data: Any, labels: Any) -> ModelSetup:
        """Initializes a Struct-G model."""
        # This method is used by two separate pipeline configurations
        raise NotImplementedError("Struct-G model initialization is not implemented.")

    @staticmethod
    def initialize_graphsage(data: Any, labels: Any) -> ModelSetup:
        """Initializes a GraphSAGE model."""
        raise NotImplementedError("GraphSAGE model initialization is not implemented.")

    @staticmethod
    def initialize_gat(data: Any, labels: Any) -> ModelSetup:
        """Initializes a GAT model."""
        raise NotImplementedError("GAT model initialization is not implemented.")

    # --- Initialization of All Models ---

    @staticmethod
    def initialize_all() -> List[Tuple[str, Callable[[Any, Any], ModelSetup]]]:
        """
        Returns a list of tuples, where each tuple contains:
        1. The user-friendly name of the pipeline/model.
        2. The static function to initialize that specific model.

        Note: The two Struct-G pipelines are separate logically, so we duplicate
        the setup function call with their distinct pipeline names.
        """
        return [
            # I use initialize_gnn here to match the pipeline name "GNN"
            ("GNN", GNNFactory.initialize_gcn),
            ("Deep GCN", GNNFactory.initialize_deep_gcn),
            ("GPT-GNN", GNNFactory.initialize_gpt_gnn),
            ("Struct-G Structural Only Pretrain", GNNFactory.initialize_structg),
            ("GraphSAGE", GNNFactory.initialize_graphsage),
            ("GAT", GNNFactory.initialize_gat),
        ]



class ModelConfig:
    """
    Configuration class to hold all common and structural-specific
    hyperparameters for GNN models.
    """

    def __init__(
            self,
            # Structural Dependencies
            structural_encoder: Optional[StructuralInformationEncoder] = None,
            gate_integrator: Optional[StructuralSignalIntegrator] = None,

            # Dimension Parameters
            input_dim: int = 0,  # Should be set based on data.num_features
            num_classes: Optional[int] = None,  # Should be set based on labels.max().item() + 1
            hidden_dim: int = 64,
            output_dim: int = 32,

            # Architecture Parameters
            num_layers: int = 2,
            use_gat: bool = False,  # Set to True for GAT models

            # Task/Loss Parameters
            feat_reconstruction: bool = False  # Used for pretraining objectives
    ):
        """
        Initializes the model configuration.
        """
        self.structural_encoder = structural_encoder
        self.gate_integrator = gate_integrator

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.num_layers = num_layers
        self.use_gat = use_gat

        self.feat_reconstruction = feat_reconstruction

    @classmethod
    def from_data(cls, data: Any, labels: Any, **kwargs):
        """
        Helper method to automatically set input_dim and num_classes from data.
        """
        input_dim = data.num_features
        num_classes = labels.max().item() + 1 if labels is not None else None

        return cls(input_dim=input_dim, num_classes=num_classes, **kwargs)