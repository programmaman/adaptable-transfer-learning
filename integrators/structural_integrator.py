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

    """
    Implements the Adaptive Gated Integration layer of AG-GNN
    (Begga et al., Information Sciences 2026, Section 4.3, Eq. 11–13).

    This method corresponds exactly to the paper's adaptive fusion:

        H^(l) = α ⊙ H_GNN^(l) + (1 - α) ⊙ H_MLP^(l) + β ⊙ H^(0)

    Where:
      - H_GNN^(l) comes from `structural_encodings`
      - H_MLP^(l) comes from `node_features`
      - H^(0) comes from `initial_features`

    α (alpha) is computed from concatenated [H_GNN || H_MLP] using:
        α = sigmoid( v2 * ReLU( LayerNorm( W1 * [H_GNN || H_MLP] ) ) )

    β (beta) is computed from the transformed initial features:
        β = sigmoid( v_res * H^(0) )

    This is the core AG-GNN mechanism that adaptively balances:
      - structure-driven propagation (GCN / SAGE path),
      - feature-driven MLP processing,
      - and direct access to original features to avoid over-smoothing,
    as described in Section 4.2–4.3 of the paper.
    """

    def integrate(self, node_features, structural_encodings, edge_indices, node_indices=None, initial_features=None):
        """
        Args:
            node_features: H_MLP^(l)
            structural_encodings: H_GNN^(l)
            edge_indices: unused (kept for interface consistency)
            node_indices: optional indices of nodes to update
            initial_features: H_MLP^(0) (recommended; if None, zeros are used)
        """
        if node_indices is None:
            node_indices = torch.arange(node_features.size(0), device=node_features.device)

        # 1) Subsets
        feature_subset = node_features[node_indices]  # H_MLP
        structure_subset = structural_encodings[node_indices]  # H_GNN

        if initial_features is not None:
            initial_subset = initial_features[node_indices]  # H^0
        else:
            initial_subset = torch.zeros_like(feature_subset)

        # 2) Project to calculation latent space
        feature_latent = self.feature_projection(feature_subset)  # [B, calc_dim]
        structural_latent = self.structural_projection(structure_subset)  # [B, calc_dim]
        initial_latent = self.initial_projection(initial_subset)  # [B, calc_dim]

        # 3) Alpha gate (AG-GNN-style)
        concat_features = torch.cat([structural_latent, feature_latent], dim=-1)
        alpha_hidden = self.alpha_W1(concat_features)
        alpha_hidden = self.alpha_layernorm(alpha_hidden)
        alpha_hidden = torch.relu(alpha_hidden)
        alpha = torch.sigmoid(self.alpha_v2(alpha_hidden))  # [B, 1]

        # 4) Beta gate (skip from H^0)
        beta = torch.sigmoid(self.beta_vres(initial_latent))  # [B, 1]

        # 5) Fuse
        fused_representation = (
                alpha * structural_latent
                + (1.0 - alpha) * feature_latent
                + beta * initial_latent
        )

        fused_representation = self.output_projection(fused_representation)

        updated_node_features = node_features.clone()
        updated_node_features[node_indices] = fused_representation
        return updated_node_features, None


class AdaptiveGateWithSparsity(Gate):
    """
    Adaptive gate with sparsity regularization on alpha to encourage
    ignoring structure unless it is truly useful.
    """

    def __init__(
        self,
        feature_dimension: int,
        structural_dimension: int,
        calculation_dimension: int,
        initial_feature_dimension: int = None,
        lambda_sparse: float = 1e-3,   # <<< NEW
    ):
        super().__init__(feature_dimension, structural_dimension, calculation_dimension)

        self.lambda_sparse = lambda_sparse

        if initial_feature_dimension is None:
            initial_feature_dimension = feature_dimension

        self.initial_projection = nn.Linear(initial_feature_dimension, calculation_dimension)

        self.alpha_W1 = nn.Linear(calculation_dimension * 2, calculation_dimension)
        self.alpha_layernorm = nn.LayerNorm(calculation_dimension)
        self.alpha_v2 = nn.Linear(calculation_dimension, 1)

        self.beta_vres = nn.Linear(calculation_dimension, 1)

        if calculation_dimension != feature_dimension:
            self.output_projection = nn.Linear(calculation_dimension, feature_dimension)
        else:
            self.output_projection = nn.Identity()

        self._reset_adaptive_parameters()

    def _reset_adaptive_parameters(self):
        nn.init.uniform_(self.alpha_v2.weight, -0.01, 0.01)
        nn.init.constant_(self.alpha_v2.bias, 0.0)
        nn.init.uniform_(self.beta_vres.weight, -0.01, 0.01)
        nn.init.constant_(self.beta_vres.bias, 0.0)

    def integrate(self, node_features, structural_encodings, edge_indices, node_indices=None, initial_features=None):

        if node_indices is None:
            node_indices = torch.arange(node_features.size(0), device=node_features.device)

        feature_subset = node_features[node_indices]
        structure_subset = structural_encodings[node_indices]

        if initial_features is not None:
            initial_subset = initial_features[node_indices]
        else:
            initial_subset = torch.zeros_like(feature_subset)

        # Project to latent
        feature_latent = self.feature_projection(feature_subset)
        structural_latent = self.structural_projection(structure_subset)
        initial_latent = self.initial_projection(initial_subset)

        # Alpha gate
        concat_features = torch.cat([structural_latent, feature_latent], dim=-1)
        alpha_hidden = self.alpha_layernorm(self.alpha_W1(concat_features))
        alpha_hidden = torch.relu(alpha_hidden)
        alpha = torch.sigmoid(self.alpha_v2(alpha_hidden))  # [B, 1]

        # Beta gate
        beta = torch.sigmoid(self.beta_vres(initial_latent))  # [B, 1]

        # Fusion
        fused_latent = (
            alpha * structural_latent
            + (1.0 - alpha) * feature_latent
            + beta * initial_latent
        )

        fused_feat = self.output_projection(fused_latent)

        updated_node_features = node_features.clone()
        updated_node_features[node_indices] = fused_feat

        # ----------- NEW: sparsity regularization -----------
        # Penalize using structure unless necessary
        sparsity_loss = self.lambda_sparse * alpha.mean()

        return updated_node_features, sparsity_loss

class DisagreementAwareAdaptiveGate(Gate):
    """
    Disagreement- and Reliability-Aware AG-GNN Gate.

    This extends AG-GNN by:
      1) Measuring agreement between structural and feature representations
      2) Measuring local consistency (reliability) of each pathway
      3) Using all of these signals to decide how much to trust structure vs features

    It keeps the same fusion equation:

        H = alpha * H_GNN + (1 - alpha) * H_MLP + beta * H_0

    But alpha is now computed from:
      - learned AG-GNN alpha
      - representation agreement
      - structural consistency
      - feature consistency
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

        # Projection for initial features
        self.initial_projection = nn.Linear(initial_feature_dimension, calculation_dimension)

        # --- Standard AG-GNN alpha MLP ---
        self.alpha_W1 = nn.Linear(calculation_dimension * 2, calculation_dimension)
        self.alpha_layernorm = nn.LayerNorm(calculation_dimension)
        self.alpha_v2 = nn.Linear(calculation_dimension, 1)

        # --- Combine signals into final alpha ---
        # Inputs: [logit(alpha_learned), agreement, struct_consistency, feat_consistency]
        self.alpha_fusion = nn.Linear(4, 1)

        # --- Beta gate ---
        self.beta_vres = nn.Linear(calculation_dimension, 1)

        # --- Output projection ---
        if calculation_dimension != feature_dimension:
            self.output_projection = nn.Linear(calculation_dimension, feature_dimension)
        else:
            self.output_projection = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.alpha_v2.weight, -0.01, 0.01)
        nn.init.constant_(self.alpha_v2.bias, 0.0)

        nn.init.uniform_(self.beta_vres.weight, -0.01, 0.01)
        nn.init.constant_(self.beta_vres.bias, 0.0)

        # Initialize fusion to be conservative initially
        nn.init.zeros_(self.alpha_fusion.weight)
        nn.init.zeros_(self.alpha_fusion.bias)

    @staticmethod
    def _neighbor_mean(x: torch.Tensor, edge_index: tuple):
        """
        Compute mean of neighbor features for each node:
            mean[i] = mean_{(i -> j) in edges} x[j]
        """
        src, dst = edge_index
        num_nodes = x.size(0)
        dim = x.size(1)
        device = x.device

        out = x.new_zeros((num_nodes, dim))
        out.index_add_(0, src, x[dst])

        deg = torch.bincount(src, minlength=num_nodes).float().unsqueeze(1).to(device)
        deg = deg.clamp(min=1.0)

        return out / deg

    def integrate(self, node_features, structural_encodings, edge_indices, node_indices=None, initial_features=None):

        if node_indices is None:
            node_indices = torch.arange(node_features.size(0), device=node_features.device)

        # Subsets
        feature_subset = node_features[node_indices]
        structure_subset = structural_encodings[node_indices]

        if initial_features is not None:
            initial_subset = initial_features[node_indices]
        else:
            initial_subset = torch.zeros_like(feature_subset)

        # Project to latent space
        feature_latent = self.feature_projection(feature_subset)
        structural_latent = self.structural_projection(structure_subset)
        initial_latent = self.initial_projection(initial_subset)

        # -------------------------------------------------
        # 1) Standard AG-GNN learned alpha
        # -------------------------------------------------
        concat_features = torch.cat([structural_latent, feature_latent], dim=-1)
        alpha_hidden = self.alpha_W1(concat_features)
        alpha_hidden = self.alpha_layernorm(alpha_hidden)
        alpha_hidden = torch.relu(alpha_hidden)
        alpha_learned = torch.sigmoid(self.alpha_v2(alpha_hidden))  # [B, 1]

        # Convert to logit for fusion
        eps = 1e-6
        alpha_learned_logit = torch.log(alpha_learned.clamp(eps, 1 - eps) / (1 - alpha_learned.clamp(eps, 1 - eps)))

        # -------------------------------------------------
        # 2) Agreement signal: cosine(H_gnn, H_mlp)
        # -------------------------------------------------
        f_norm = torch.nn.functional.normalize(feature_latent, dim=1)
        s_norm = torch.nn.functional.normalize(structural_latent, dim=1)
        agreement = (f_norm * s_norm).sum(dim=1, keepdim=True)  # [-1, 1]

        # -------------------------------------------------
        # 3) Consistency / reliability signals
        # -------------------------------------------------
        with torch.no_grad():
            feat_nb = self._neighbor_mean(node_features, edge_indices)
            struct_nb = self._neighbor_mean(structural_encodings, edge_indices)

        feat_nb_subset = feat_nb[node_indices]
        struct_nb_subset = struct_nb[node_indices]

        feat_nb_latent = self.feature_projection(feat_nb_subset)
        struct_nb_latent = self.structural_projection(struct_nb_subset)

        # Cosine consistency
        feat_consistency = torch.nn.functional.cosine_similarity(feature_latent, feat_nb_latent, dim=1).unsqueeze(1)
        struct_consistency = torch.nn.functional.cosine_similarity(structural_latent, struct_nb_latent, dim=1).unsqueeze(1)

        # -------------------------------------------------
        # 4) Fuse signals into final alpha
        # -------------------------------------------------
        alpha_input = torch.cat(
            [
                alpha_learned_logit,
                agreement,
                struct_consistency,
                feat_consistency,
            ],
            dim=1,
        )  # [B, 4]

        alpha = torch.sigmoid(self.alpha_fusion(alpha_input))  # [B, 1]

        # -------------------------------------------------
        # 5) Beta gate
        # -------------------------------------------------
        beta = torch.sigmoid(self.beta_vres(initial_latent))  # [B, 1]

        # -------------------------------------------------
        # 6) Final fusion
        # -------------------------------------------------
        fused_latent = (
            alpha * structural_latent
            + (1.0 - alpha) * feature_latent
            + beta * initial_latent
        )

        fused_feat = self.output_projection(fused_latent)

        updated_node_features = node_features.clone()
        updated_node_features[node_indices] = fused_feat

        return updated_node_features, None

class JumpingKnowledgeGate(Gate):
    """
    Jumping Knowledge gate for 1-layer GCN.

    This implements a learned skip connection between:
        h0 = input features (pre-GCN)
        h1 = output of 1-layer GCN

    Form:
        gamma = sigmoid( MLP([h1 || h0]) )
        h = gamma * h1 + (1 - gamma) * h0

    This is the correct JK-style depth selection in a 1-layer network.
    """

    def __init__(
        self,
        feature_dimension: int,
        structural_dimension: int,      # unused, kept for interface compatibility
        calculation_dimension: int,
    ):
        super().__init__(feature_dimension, structural_dimension, calculation_dimension)

        # Project both to latent
        self.h1_projection = nn.Linear(feature_dimension, calculation_dimension)
        self.h0_projection = nn.Linear(feature_dimension, calculation_dimension)

        # Gate MLP: decides between h1 and h0
        self.gamma_W1 = nn.Linear(calculation_dimension * 2, calculation_dimension)
        self.gamma_ln = nn.LayerNorm(calculation_dimension)
        self.gamma_v = nn.Linear(calculation_dimension, 1)

        # Output projection back to feature dim if needed
        if calculation_dimension != feature_dimension:
            self.output_projection = nn.Linear(calculation_dimension, feature_dimension)
        else:
            self.output_projection = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize gate to ~0.5 so it starts as simple averaging
        nn.init.uniform_(self.gamma_v.weight, -0.01, 0.01)
        nn.init.constant_(self.gamma_v.bias, 0.0)

    def integrate(self, node_features, structural_encodings, edge_indices, node_indices=None, initial_features=None):
        """
        Args:
            node_features: h1 (GCN output)           [N, D]
            initial_features: h0 (input features)    [N, D]   (REQUIRED)
        """

        if initial_features is None:
            raise ValueError("JumpingKnowledgeGate requires initial_features (h0).")

        if node_indices is None:
            node_indices = torch.arange(node_features.size(0), device=node_features.device)

        # Subsets
        h1 = node_features[node_indices]
        h0 = initial_features[node_indices]

        # Project to latent
        h1_latent = self.h1_projection(h1)
        h0_latent = self.h0_projection(h0)

        # Gate
        concat = torch.cat([h1_latent, h0_latent], dim=-1)
        hidden = self.gamma_W1(concat)
        hidden = self.gamma_ln(hidden)
        hidden = torch.relu(hidden)
        gamma = torch.sigmoid(self.gamma_v(hidden))  # [B, 1]

        # Fuse
        fused_latent = gamma * h1_latent + (1.0 - gamma) * h0_latent
        fused_feat = self.output_projection(fused_latent)

        # Write back
        updated_node_features = node_features.clone()
        updated_node_features[node_indices] = fused_feat

        return updated_node_features, None

