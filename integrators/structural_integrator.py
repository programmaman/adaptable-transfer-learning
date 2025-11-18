import torch
from torch import nn


class StructuralSignalIntegrator(nn.Module):
    """
    Abstract interface that defines how structural signals
    (e.g., node2vec embeddings, positional encodings, geometry descriptors)
    are incorporated into a GNN pipeline.
    """

    def integrate(self, node_features, structural_encodings,
                  edge_indices, node_indices=None):
        """
        Main hook for altering node features using structural signals.
        """
        raise NotImplementedError



# ============================================================
# 1. Gated Fusion Integrator (Simple Feature/Structure Gate)
# ============================================================

class GatedStructureFeatureIntegrator(StructuralSignalIntegrator):
    """
    Applies a gating mechanism to fuse standard node features
    with structural encodings into a hidden embedding.
    """

    def __init__(self, feature_dim, structural_dim, hidden_dim):
        super().__init__()

        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.structural_projection = nn.Linear(structural_dim, hidden_dim)

        # Gate decides weighting between (projected features) and (projected structure)
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim + structural_dim, hidden_dim),
            nn.Sigmoid()
        )

    def integrate(self, node_features, structural_encodings, edge_indices, node_indices=None):

        if node_indices is None:
            node_indices = torch.arange(
                node_features.size(0),
                device=node_features.device
            )

        # Subsets
        features_subset = node_features[node_indices]
        structure_subset = structural_encodings[node_indices]

        combined_input = torch.cat([features_subset, structure_subset], dim=-1)

        gate_values = self.fusion_gate(combined_input)

        projected_features = self.feature_projection(features_subset)
        projected_structure = self.structural_projection(structure_subset)

        fused_representation = (
            gate_values * projected_structure
            + (1 - gate_values) * projected_features
        )

        updated_node_features = node_features.clone()
        updated_node_features[node_indices] = fused_representation

        return updated_node_features, None



# ============================================================
# 2. Self-Supervised Gating Integrator
# ============================================================

class SelfSupervisedStructureFeatureIntegrator(StructuralSignalIntegrator):
    """
    Computes latent predictions from structure and features, then
    uses their prediction error as a self-supervised gating signal.
    """

    def __init__(self, feature_dim, structural_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.structural_projection = nn.Linear(structural_dim, hidden_dim)

        # Heads predicting neighborhood-based targets
        self.structural_prediction_head = nn.Linear(hidden_dim, hidden_dim)
        self.feature_prediction_head = nn.Linear(hidden_dim, hidden_dim)

    def integrate(self, node_features, structural_encodings, edge_indices, node_indices=None):

        if node_indices is None:
            node_indices = torch.arange(
                node_features.size(0),
                device=node_features.device
            )

        # Subsets
        features_subset = node_features[node_indices]
        structure_subset = structural_encodings[node_indices]

        # Latent encodings
        feature_latent = self.feature_projection(features_subset)
        structural_latent = self.structural_projection(structure_subset)

        # Build adjacency
        src, dst = edge_indices
        num_nodes = node_features.size(0)
        device = node_features.device

        adjacency = torch.zeros(num_nodes, num_nodes, device=device)
        adjacency[src, dst] = 1

        degree = adjacency.sum(dim=1, keepdim=True) + 1e-6
        normalized_adj = adjacency / degree

        # Neighborhood feature target (stop-gradient)
        with torch.no_grad():
            neighbor_feature_targets = normalized_adj @ node_features
            neighbor_feature_targets = neighbor_feature_targets[node_indices]

        # Predict targets from structure and from features
        predicted_from_structure = self.structural_prediction_head(structural_latent)
        predicted_from_features = self.feature_prediction_head(feature_latent)

        # Target projection into hidden space
        target_latent = self.feature_projection(neighbor_feature_targets)

        # MSE losses → gating signal
        structural_loss = (predicted_from_structure - target_latent).pow(2).mean(dim=1)
        feature_loss = (predicted_from_features - target_latent).pow(2).mean(dim=1)

        loss_based_gating_logits = torch.stack(
            [-structural_loss, -feature_loss],
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
