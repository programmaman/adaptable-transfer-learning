from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import nn



class FusionType(Enum):
    """
    Possible simple methods for combining features and structure
    before the gate calculation.
    """
    CONCAT = 'concat'   # Concatenation (Original SimpleFeatureGate)
    ADD = 'add'         # Element-wise addition
    MUL = 'mul'         # Element-wise multiplication
    SUB = 'sub'         # Element-wise subtraction (Feature - Structure)


class EmbeddingFusion(nn.Module, ABC):
    """
    Abstract base class for combining feature and structural embeddings.
    """

    def __init__(self, feature_dimension: int, structural_dimension: int, hidden_dimension: int):
        super().__init__()
        self.feature_dimension = feature_dimension
        self.structural_dimension = structural_dimension
        self.hidden_dimension = hidden_dimension

    @abstractmethod
    def get_input_dimension(self) -> int:
        """Returns the dimension of the tensor after fusion."""
        pass

    @abstractmethod
    def forward(self, feature_subset: torch.Tensor, structure_subset: torch.Tensor) -> torch.Tensor:
        """Perform the fusion operation."""
        pass


class ConcatFusion(EmbeddingFusion):
    """Concatenates features and structural embeddings."""

    def __init__(self, feature_dimension: int, structural_dimension: int, hidden_dimension: int):
        super().__init__(feature_dimension, structural_dimension, hidden_dimension)

    def get_input_dimension(self) -> int:
        return self.feature_dimension + self.structural_dimension

    def forward(self, feature_subset: torch.Tensor, structure_subset: torch.Tensor) -> torch.Tensor:
        return torch.cat([feature_subset, structure_subset], dim=-1)


class ElementwiseFusion(EmbeddingFusion):
    """
    Handles element-wise operations (ADD, MUL, SUB).
    Requires feature_dimension == structural_dimension.
    """

    def __init__(self, feature_dimension: int, structural_dimension: int, hidden_dimension: int, op: FusionType):
        super().__init__(feature_dimension, structural_dimension, hidden_dimension)

        if feature_dimension != structural_dimension:
            raise ValueError(
                f"Element-wise fusion ({op.value}) requires feature_dimension "
                f"({feature_dimension}) to equal structural_dimension "
                f"({structural_dimension})."
            )

        self.op = op

    def get_input_dimension(self) -> int:
        return self.feature_dimension  # Result is same dimension

    def forward(self, feature_subset: torch.Tensor, structure_subset: torch.Tensor) -> torch.Tensor:
        if self.op == FusionType.ADD:
            return feature_subset + structure_subset
        elif self.op == FusionType.MUL:
            return feature_subset * structure_subset
        elif self.op == FusionType.SUB:
            return feature_subset - structure_subset
        else:
            raise NotImplementedError(f"Operation {self.op} not implemented.")



class StructuralSignalIntegrator(nn.Module):
    def integrate(self, node_features, structural_encodings,
                  edge_indices, node_indices=None):
        raise NotImplementedError


class Gate(StructuralSignalIntegrator):
    def __init__(self, feature_dimension, structural_dimension, hidden_dimension):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.feature_projection = nn.Linear(feature_dimension, hidden_dimension)
        self.structural_projection = nn.Linear(structural_dimension, hidden_dimension)

    def integrate(self, node_features, structural_encodings, edge_indices, node_indices=None):
        raise NotImplementedError


# --- Updated SimpleFeatureGate (Now Abstract/General) ---

class SimpleFeatureGate(Gate):
    def __init__(self, feature_dimension, structural_dimension, hidden_dimension,
                 fusion_type: FusionType = FusionType.CONCAT):
        super().__init__(feature_dimension, structural_dimension, hidden_dimension)

        # 1. Instantiate the chosen Fusion Module
        self.fusion_module = self._create_fusion_module(
            fusion_type,
            feature_dimension,
            structural_dimension,
            hidden_dimension
        )

        # 2. Use the dynamically determined input dimension for the gate
        gate_input_dim = self.fusion_module.get_input_dimension()

        self.fusion_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dimension),
            nn.Sigmoid()
        )

    @staticmethod
    def _create_fusion_module(fusion_type: FusionType, feature_dim: int, structural_dim: int,
                              hidden_dim: int) -> EmbeddingFusion:
        """Factory method to get the correct fusion implementation."""
        if fusion_type == FusionType.CONCAT:
            return ConcatFusion(feature_dim, structural_dim, hidden_dim)
        elif fusion_type in [FusionType.ADD, FusionType.MUL, FusionType.SUB]:
            return ElementwiseFusion(feature_dim, structural_dim, hidden_dim, fusion_type)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def integrate(self, node_features, structural_encodings, edge_indices, node_indices=None):

        if node_indices is None:
            node_indices = torch.arange(
                node_features.size(0),
                device=node_features.device
            )

        feature_subset = node_features[node_indices]
        structure_subset = structural_encodings[node_indices]

        # Use the configured fusion module
        combined_input = self.fusion_module(feature_subset, structure_subset)

        # Gate calculation remains the same
        gate_values = self.fusion_gate(combined_input)

        projected_features = self.feature_projection(feature_subset)
        projected_structure = self.structural_projection(structure_subset)

        # Final gated combination remains the same
        fused_representation = (
                gate_values * projected_structure
                + (1 - gate_values) * projected_features
        )

        updated_node_features = node_features.clone()
        updated_node_features[node_indices] = fused_representation

        return updated_node_features, None


class SelfSupervisedGate(Gate):
    def __init__(self, feature_dimension, structural_dimension, hidden_dimension):
        super().__init__(feature_dimension, structural_dimension, hidden_dimension)

        self.structural_prediction_head = nn.Linear(hidden_dimension, hidden_dimension)
        self.feature_prediction_head = nn.Linear(hidden_dimension, hidden_dimension)

    def integrate(self, node_features, structural_encodings, edge_indices, node_indices=None):

        if node_indices is None:
            node_indices = torch.arange(
                node_features.size(0),
                device=node_features.device
            )

        feature_subset = node_features[node_indices]
        structure_subset = structural_encodings[node_indices]

        feature_latent = self.feature_projection(feature_subset)
        structural_latent = self.structural_projection(structure_subset)

        # Build Adjacency Matrix
        source_indices, destination_indices = edge_indices
        number_of_nodes = node_features.size(0)
        device = node_features.device

        adjacency_matrix = torch.zeros(number_of_nodes, number_of_nodes, device=device)
        adjacency_matrix[source_indices, destination_indices] = 1

        node_degree = adjacency_matrix.sum(dim=1, keepdim=True) + 1e-6
        normalized_adjacency = adjacency_matrix / node_degree

        # Neighborhood Feature Target (Stop-Gradient)
        with torch.no_grad():
            neighbor_feature_targets = normalized_adjacency @ node_features
            neighbor_feature_targets = neighbor_feature_targets[node_indices]

        # Predict targets
        predicted_from_structure = self.structural_prediction_head(structural_latent)
        predicted_from_features = self.feature_prediction_head(feature_latent)

        # Target projection into hidden space
        target_latent = self.feature_projection(neighbor_feature_targets)

        # MSE losses -> Gating signal
        structural_prediction_loss = (predicted_from_structure - target_latent).pow(2).mean(dim=1)
        feature_prediction_loss = (predicted_from_features - target_latent).pow(2).mean(dim=1)

        loss_based_gating_logits = torch.stack(
            [-structural_prediction_loss, -feature_prediction_loss],
            dim=-1
        )

        gating_weights = torch.softmax(loss_based_gating_logits, dim=-1)

        structural_gate = gating_weights[:, 0].unsqueeze(1)
        feature_gate = gating_weights[:, 1].unsqueeze(1)

        fused_representation = (
            structural_gate * structural_latent
            + feature_gate * feature_latent
        )

        updated_node_features = node_features.clone()
        updated_node_features[node_indices] = fused_representation

        return updated_node_features, None