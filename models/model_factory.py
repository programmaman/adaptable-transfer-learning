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
        """Initializes a GCN model."""
        from models.gcn import GCN

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

        base_model = GCN(in_channels=in_dim, out_channels=num_classes)
        return base_model

    @staticmethod
    def initialize_deep_gcn(data: Any, labels: Any) -> nn.Module:
        """Initializes a Deep GCN model."""
        from models.deep_gcn import ThreeLayerGCN

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

        hidden_channels = max(64, in_dim * 2)
        mid_channels = max(32, in_dim)

        class DeepGCNWithHead(nn.Module):
            def __init__(self, in_channels, hidden_channels, mid_channels, num_classes):
                super().__init__()
                self.gcn = ThreeLayerGCN(in_channels=in_channels, hidden_channels=hidden_channels,
                                         mid_channels=mid_channels)
                self.head = nn.Linear(1, num_classes)

            def forward(self, x, edge_index):
                x = self.gcn(x, edge_index)
                x = x.unsqueeze(-1)  # Reshape [num_nodes] to [num_nodes, 1]
                x = self.head(x)
                return x

        base_model = DeepGCNWithHead(in_channels=in_dim, hidden_channels=hidden_channels, mid_channels=mid_channels,
                                     num_classes=num_classes)
        return base_model

    @staticmethod
    def initialize_gpt_gnn(data: Any, labels: Any) -> nn.Module:
        """Initializes a GPT-GNN model."""
        from models.gpt_gnn import GPT_GNN, GNN

        if hasattr(data, "num_features"):
            in_dim = data.num_features
        elif hasattr(data, "x") and hasattr(data.x, "shape"):
            in_dim = data.x.shape[-1]
        else:
            raise ValueError("Cannot determine input dimension (in_dim) from data.")

        if hasattr(labels, "num_classes"):
            num_classes = labels.num_classes
        elif isinstance(labels, torch.Tensor):
            num_classes = len(torch.unique(labels))
        else:
            raise ValueError("Cannot determine number of classes (num_classes) from labels.")

        n_hid = 64
        num_types = 1
        num_relations = 1
        n_heads = 4
        n_layers = 2
        device = "cuda" if torch.cuda.is_available() else "cpu"

        gnn = GNN(
            in_dim=in_dim,
            n_hid=n_hid,
            num_types=num_types,
            num_relations=num_relations,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=0.2,
            conv_name="hgt",
        )

        gpt_gnn = GPT_GNN(
            gnn=gnn,
            rem_edge_list={},
            attr_decoder=None,
            types=list(range(num_types)),
            neg_samp_num=5,
            device=device,
        )

        class GPT_GNN_Adapter(nn.Module):
            def __init__(self, gpt_gnn_model: nn.Module, num_classes: int):
                super().__init__()
                self.gpt_gnn = gpt_gnn_model
                self.classifier = nn.Linear(n_hid, num_classes)

            def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
                dev = x.device
                edge_time = torch.zeros(edge_index.shape[1], dtype=torch.long, device=dev)
                edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=dev)
                node_type = torch.zeros(x.shape[0], dtype=torch.long, device=dev)

                # Correct argument order: (node_feature, node_type, edge_time, edge_index, edge_type)
                gnn_out = self.gpt_gnn.gnn(x, node_type, edge_time, edge_index, edge_type)
                return self.classifier(gnn_out)

        return GPT_GNN_Adapter(gpt_gnn, num_classes)

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
            ("GNN", ModelFactory.initialize_gcn),
            ("Deep GCN", ModelFactory.initialize_deep_gcn),
            ("GPT-GNN", ModelFactory.initialize_gpt_gnn),
            ("Struct-G Structural Only Pretrain", ModelFactory.initialize_structg),
            ("GraphSAGE", ModelFactory.initialize_graphsage),
            ("GAT", ModelFactory.initialize_gat),
        ]