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
    def get_output_dimension(self) -> int:
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

    def get_output_dimension(self) -> int:
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

    def get_output_dimension(self) -> int:
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
    def __init__(self, feature_dimension, structural_dimension, calculation_dimension):
        super().__init__()
        self.hidden_dimension = calculation_dimension
        self.feature_projection = nn.Linear(feature_dimension, calculation_dimension)
        self.structural_projection = nn.Linear(structural_dimension, calculation_dimension)

    def integrate(self, node_features, structural_encodings, edge_indices, node_indices=None):
        raise NotImplementedError


class SimpleFeatureGate(Gate):
    def __init__(self, feature_dimension, structural_dimension, calculation_dimension,
                 fusion_type: FusionType = FusionType.CONCAT):
        super().__init__(feature_dimension, structural_dimension, calculation_dimension)

        # 1. Instantiate the chosen Fusion Module
        self.fusion_module = self._create_fusion_module(
            fusion_type,
            feature_dimension,
            structural_dimension,
            calculation_dimension
        )

        # 2. Use the dynamically determined input dimension for the gate
        gate_input_dimension = self.fusion_module.get_output_dimension()

        self.fusion_gate = nn.Sequential(
            nn.Linear(gate_input_dimension, calculation_dimension),
            nn.Sigmoid()
        )

        # 3. Conditional output projection
        if calculation_dimension != feature_dimension:
            self.output_projection = nn.Linear(calculation_dimension, feature_dimension)
        else:
            self.output_projection = nn.Identity()

    @staticmethod
    def _create_fusion_module(fusion_type: FusionType, feature_dim: int, structural_dim: int,
                              hidden_dim: int) -> EmbeddingFusion:
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

        combined_input = self.fusion_module(feature_subset, structure_subset)

        gate_values = self.fusion_gate(combined_input)

        projected_features = self.feature_projection(feature_subset)
        projected_structure = self.structural_projection(structure_subset)

        fused_representation = (
                gate_values * projected_structure + (1 - gate_values) * projected_features
        )

        # Apply conditional projection here
        fused_representation = self.output_projection(fused_representation)

        updated_node_features = node_features.clone()
        updated_node_features[node_indices] = fused_representation

        return updated_node_features, None


class SelfSupervisedGate(Gate):
    def __init__(self, feature_dimension, structural_dimension, calculation_dimension):
        super().__init__(feature_dimension, structural_dimension, calculation_dimension)

        self.structural_prediction_head = nn.Linear(calculation_dimension, calculation_dimension)
        self.feature_prediction_head = nn.Linear(calculation_dimension, calculation_dimension)

        if calculation_dimension != feature_dimension:
            self.output_projection = nn.Linear(calculation_dimension, feature_dimension)
        else:
            self.output_projection = nn.Identity()

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

        # Sparse-to-dense adjacency logic
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

        # [FIX 1 Applied] Project back to feature dimension before assignment
        fused_representation = self.output_projection(fused_representation)

        updated_node_features = node_features.clone()
        # Now shapes match: [Batch, 16] into [Batch, 16]
        updated_node_features[node_indices] = fused_representation

        # [FIX 2] Return the actual auxiliary loss so the model learns
        total_aux_loss = (structural_prediction_loss + feature_prediction_loss).mean()

        return updated_node_features, total_aux_loss


class AdaptiveGate(Gate):
    """
    Implements the Adaptive Integration Mechanism from AG-GNN (Section 4.3).

    This gate fuses three sources of information:
    1. Structural Information (H_GNN)
    2. Feature Information (H_MLP)
    3. Initial Feature Information (H_MLP^0)

    It computes two gating coefficients:
    - Alpha: Balances GNN vs. MLP pathways.
    - Beta: Controls the contribution of the initial features (skip-connection).
    """

    def __init__(self, feature_dimension: int, structural_dimension: int, calculation_dimension: int,
                 initial_feature_dimension: int = None):
        super().__init__(feature_dimension, structural_dimension, calculation_dimension)

        # If initial dimension is not specified, assume it matches the current feature dimension
        if initial_feature_dimension is None:
            initial_feature_dimension = feature_dimension

        # --- Projection for the Initial Features Pathway ---
        # Projects H^(0) to the calculation dimension to make it compatible for fusion
        self.initial_projection = nn.Linear(initial_feature_dimension, calculation_dimension)

        # --- Alpha Gate Components (Eq. 11) ---
        # Determines balance between GNN and MLP representations
        # Alpha = Sigmoid( v2 * ReLU( LayerNorm( W1 * [H_GNN || H_MLP] + b1 ) ) + b2 )

        # W1: Projects concatenated [H_GNN || H_MLP] -> Hidden Dim
        self.alpha_W1 = nn.Linear(calculation_dimension * 2, calculation_dimension)

        # LayerNorm applied after W1 projection
        self.alpha_layernorm = nn.LayerNorm(calculation_dimension)

        # v2: Projects to a scalar coefficient
        self.alpha_v2 = nn.Linear(calculation_dimension, 1)

        # --- Beta Gate Components (Eq. 12) ---
        # Determines contribution of initial features
        # Beta = Sigmoid( v_res * H_MLP^(0) )
        self.beta_vres = nn.Linear(calculation_dimension, 1)

        # --- Output Projection ---
        # Maps back to feature_dimension if necessary
        if calculation_dimension != feature_dimension:
            self.output_projection = nn.Linear(calculation_dimension, feature_dimension)
        else:
            self.output_projection = nn.Identity()

        # Initialize weights to ensure balanced starting state (Section 4.3.2)
        self._reset_adaptive_parameters()

    def _reset_adaptive_parameters(self):
        """
        Initializes gating coefficients with small random values as described in Section 4.3.2.
        This ensures the model starts with approximately equal importance for all pathways.
        """
        # Initialize the final scalar projections with small values
        # so that sigmoid outputs are close to 0.5 initially.
        nn.init.uniform_(self.alpha_v2.weight, -0.01, 0.01)
        nn.init.constant_(self.alpha_v2.bias, 0.0)

        nn.init.uniform_(self.beta_vres.weight, -0.01, 0.01)
        nn.init.constant_(self.beta_vres.bias, 0.0)

    def integrate(self, node_features, structural_encodings, edge_indices, node_indices=None, initial_features=None):
        """
        Args:
            node_features: H_MLP^(l) - Features processed by the MLP pathway.
            structural_encodings: H_GNN^(l) - Features processed by the GNN pathway.
            edge_indices: Graph connectivity (unused in this specific gate logic, but part of signature).
            node_indices: Indices of nodes to process.
            initial_features: H_MLP^(0) - The transformed initial features (Required for Beta gate).
        """

        if node_indices is None:
            node_indices = torch.arange(node_features.size(0), device=node_features.device)

        # 1. Prepare Subsets
        feature_subset = node_features[node_indices]  # H_MLP
        structure_subset = structural_encodings[node_indices]  # H_GNN

        # Handle initial features (H^0). If not provided, fallback to zeros or current features
        # (though algorithmic logic dictates H^0 should be present).
        if initial_features is not None:
            initial_subset = initial_features[node_indices]
        else:
            # Fallback warning or behavior if initial features are missing in the pipeline
            initial_subset = torch.zeros_like(feature_subset)

            # 2. Project all inputs to the calculation latent space
        # H_MLP^(l)
        feature_latent = self.feature_projection(feature_subset)
        # H_GNN^(l)
        structural_latent = self.structural_projection(structure_subset)
        # H_MLP^(0)
        initial_latent = self.initial_projection(initial_subset)

        # 3. Compute Alpha Gate (Eq. 11)
        # Concatenate [H_GNN || H_MLP]
        concat_features = torch.cat([structural_latent, feature_latent], dim=-1)

        # W1 -> LayerNorm -> ReLU
        alpha_hidden = self.alpha_W1(concat_features)
        alpha_hidden = self.alpha_layernorm(alpha_hidden)
        alpha_hidden = torch.relu(alpha_hidden)

        # v2 -> Sigmoid
        alpha_score = self.alpha_v2(alpha_hidden)
        alpha = torch.sigmoid(alpha_score)  # Shape: [Batch, 1]

        # 4. Compute Beta Gate (Eq. 12)
        # v_res * H_MLP^(0) -> Sigmoid
        beta_score = self.beta_vres(initial_latent)
        beta = torch.sigmoid(beta_score)  # Shape: [Batch, 1]

        # 5. Final Integration (Eq. 13)
        # H = alpha * H_GNN + (1 - alpha) * H_MLP + beta * H_MLP^0
        fused_representation = (
                alpha * structural_latent +
                (1 - alpha) * feature_latent +
                beta * initial_latent
        )

        # 6. Output Projection and Assignment
        fused_representation = self.output_projection(fused_representation)

        updated_node_features = node_features.clone()
        updated_node_features[node_indices] = fused_representation

        # Return updated features and None for aux_loss (as this gate has no aux loss)
        return updated_node_features, None


import torch
import torch.nn as nn
import torch.nn.functional as functional

class CombinedAdaptiveSelfSupervisedGate(Gate):
    """
    Combines the AdaptiveGate (learned alpha & beta, LayerNorm + MLP gating)
    with the SelfSupervisedGate (neighborhood-prediction losses used as signals).

    Behavior:
      - Projects structural, feature and initial-feature inputs to a shared latent space.
      - Computes self-supervised neighbor-feature prediction losses for structure and feature pathways.
      - Embeds the two per-node losses and concatenates them with the latents to form the gating input.
      - Computes learned alpha (balance between GNN and MLP) via LN -> Linear -> ReLU -> Linear -> sigmoid.
      - Computes beta (contribution of initial features) from the initial-feature latent via a small linear -> sigmoid.
      - Fuses as: H = alpha * H_GNN + (1 - alpha) * H_MLP + beta * H_init.
      - Returns updated features and auxiliary self-supervised loss (mean of prediction losses).
    """

    def __init__(
        self,
        feature_dimension: int,
        structural_dimension: int,
        calculation_dimension: int,
        initial_feature_dimension: int = None,
    ):
        super().__init__(feature_dimension, structural_dimension, calculation_dimension)

        if initial_feature_dimension is None:
            initial_feature_dimension = feature_dimension

        # Prediction heads (self-supervised)
        self.structural_prediction_head = nn.Linear(calculation_dimension, calculation_dimension)
        self.feature_prediction_head = nn.Linear(calculation_dimension, calculation_dimension)

        # Projection for initial features (H^(0)) into calculation latent
        self.initial_projection = nn.Linear(initial_feature_dimension, calculation_dimension)

        # Loss embedding: turn two scalar losses -> vector in calculation_dimension
        self.loss_projection = nn.Linear(2, calculation_dimension)

        # Alpha gate: input will be [structural_latent || feature_latent || loss_embedding]
        # so input dimension is calculation_dimension * 3
        self.alpha_W1 = nn.Linear(calculation_dimension * 3, calculation_dimension)
        self.alpha_layernorm = nn.LayerNorm(calculation_dimension)
        self.alpha_v2 = nn.Linear(calculation_dimension, 1)  # projects to scalar per node

        # Beta gate: projects initial_latent -> scalar
        self.beta_vres = nn.Linear(calculation_dimension, 1)

        # Output projection back to original feature dimension if necessary
        if calculation_dimension != feature_dimension:
            self.output_projection = nn.Linear(calculation_dimension, feature_dimension)
        else:
            self.output_projection = nn.Identity()

        # Initialize gating projection weights small so sigmoid ~0.5 initially
        self._reset_adaptive_parameters()

    def _reset_adaptive_parameters(self):
        nn.init.uniform_(self.alpha_v2.weight, -0.01, 0.01)
        nn.init.constant_(self.alpha_v2.bias, 0.0)
        nn.init.uniform_(self.beta_vres.weight, -0.01, 0.01)
        nn.init.constant_(self.beta_vres.bias, 0.0)

        # Also initialize loss_projection to small values so loss embedding starts near zero
        nn.init.uniform_(self.loss_projection.weight, -0.01, 0.01)
        nn.init.constant_(self.loss_projection.bias, 0.0)

    @staticmethod
    def _aggregate_neighbor_mean(node_features: torch.Tensor, edge_index: tuple):
        """
        Efficient neighbor aggregation that computes, for each source node i,
        the mean of features of its destination neighbors:
            aggregated[i] = mean_{(i -> j) in edges} node_features[j]
        This mirrors adjacency[src, dst] = 1 then normalized_adjacency @ node_features.
        """
        src, dst = edge_index  # expected LongTensor vectors of same length E
        num_nodes = node_features.size(0)
        feat_dim = node_features.size(1)
        device = node_features.device

        # Sum features of destinations into the corresponding source rows:
        aggregated = node_features.new_zeros((num_nodes, feat_dim))
        # index_add_: for each edge e, add node_features[dst[e]] to aggregated[src[e]]
        aggregated.index_add_(0, src, node_features[dst])

        # degree = number of outgoing edges per source
        degree = torch.bincount(src, minlength=num_nodes).float().unsqueeze(1).to(device)
        degree = degree.clamp(min=1e-6)  # avoid divide-by-zero

        neighbor_mean = aggregated / degree
        return neighbor_mean

    def integrate(self, node_features, structural_encodings, edge_indices, node_indices=None, initial_features=None):
        """
        Args:
            node_features: tensor [N, feat_dim]  (H_MLP^(l))
            structural_encodings: tensor [N, struct_dim]  (H_GNN^(l))
            edge_indices: tuple (src, dst) where src,dst are LongTensors of length E
            node_indices: optional selection of node indices to update
            initial_features: tensor [N, feat_dim0]  (H_MLP^(0)), required for beta gate (if None, zeros used)
        Returns:
            updated_node_features: tensor [N, feat_dim]
            aux_loss: scalar tensor (mean of self-supervised losses)
        """
        if node_indices is None:
            node_indices = torch.arange(node_features.size(0), device=node_features.device)

        device = node_features.device

        # 1) Subset the node-specific latents
        feature_subset = node_features[node_indices]                    # [B, feat_dim]
        structure_subset = structural_encodings[node_indices]          # [B, struct_dim]

        # 2) Project to calculation latent space (assume base Gate defines these)
        feature_latent = self.feature_projection(feature_subset)       # [B, calc_dim]
        structural_latent = self.structural_projection(structure_subset)  # [B, calc_dim]

        # 3) Initial features latent
        if initial_features is not None:
            initial_subset = initial_features[node_indices]
        else:
            # fallback to zeros -- consistent with earlier design choice
            initial_subset = torch.zeros_like(feature_subset, device=device)

        initial_latent = self.initial_projection(initial_subset)       # [B, calc_dim]

        # 4) Build neighborhood feature targets (over all nodes) WITHOUT grad, then select node_indices
        neighbor_feature_targets_full = None
        with torch.no_grad():
            # Use efficient index_add aggregation (mirrors normalized_adjacency @ node_features)
            neighbor_feature_targets_full = self._aggregate_neighbor_mean(node_features, edge_indices)  # [N, feat_dim]
        neighbor_feature_targets = neighbor_feature_targets_full[node_indices]  # [B, feat_dim]

        # 5) Predict targets from each pathway (in latent space)
        # map structural & feature latents -> predicted neighbor latent
        predicted_from_structure = self.structural_prediction_head(structural_latent)  # [B, calc_dim]
        predicted_from_features = self.feature_prediction_head(feature_latent)         # [B, calc_dim]

        # Project neighbor targets into calculation latent (same projection used for features)
        target_latent = self.feature_projection(neighbor_feature_targets)             # [B, calc_dim]

        # 6) Self-supervised per-node MSE losses (mean across latent dims -> scalar per node)
        structural_prediction_loss = functional.mse_loss(predicted_from_structure, target_latent, reduction='none').mean(dim=1)  # [B]
        feature_prediction_loss = functional.mse_loss(predicted_from_features, target_latent, reduction='none').mean(dim=1)     # [B]

        # 7) Embed the two losses -> vector and create gating input
        # Stack losses -> [B, 2], then project to [B, calc_dim]
        loss_pair = torch.stack([structural_prediction_loss, feature_prediction_loss], dim=1)  # [B, 2]
        loss_emb = self.loss_projection(loss_pair)                                             # [B, calc_dim]

        # Gating input: [structural_latent || feature_latent || loss_emb]
        gate_input = torch.cat([structural_latent, feature_latent, loss_emb], dim=-1)          # [B, calc_dim*3]

        # 8) Compute alpha (learned gating coefficient) per node
        alpha_hidden = self.alpha_W1(gate_input)
        alpha_hidden = self.alpha_layernorm(alpha_hidden)
        alpha_hidden = functional.relu(alpha_hidden)
        alpha_score = self.alpha_v2(alpha_hidden)          # [B, 1]
        alpha = torch.sigmoid(alpha_score)                 # [B, 1]

        # 9) Compute beta (learned importance of initial features)
        beta_score = self.beta_vres(initial_latent)        # [B, 1]
        beta = torch.sigmoid(beta_score)                   # [B, 1]

        # 10) Final fusion following AG-GNN eq (alpha, 1-alpha, beta)
        fused_representation = (
            alpha * structural_latent
            + (1.0 - alpha) * feature_latent
            + beta * initial_latent
        )  # [B, calc_dim]

        # 11) Project back to feature space, assign to updated features
        fused_representation = self.output_projection(fused_representation)  # [B, feat_dim]
        updated_node_features = node_features.clone()
        updated_node_features[node_indices] = fused_representation

        # 12) Auxiliary loss: average of both prediction losses (mean over batch)
        total_aux_loss = (structural_prediction_loss + feature_prediction_loss).mean()

        return updated_node_features, total_aux_loss
