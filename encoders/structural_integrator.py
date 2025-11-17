import torch
from torch import nn
import torch.nn.functional as function



class StructuralIntegrator(nn.Module):
    """
    Abstract interface for integrating structural information
    (e.g., Node2Vec, geometric descriptors) into a GNN model.

    The integrator defines *how* structure is used — whether via
    gating, edge-aware modulation, attention biasing, etc.
    """

    def integrate(self, x, struct, edge_index, node_indices=None):
        """
        Main hook for modifying GNN state using structure.

        Args:
            x            : [N, D] tensor of node features
            struct       : [N, S] tensor of structural encodings
            edge_index   : [2, E] edge list
            node_indices : Optional subset of nodes to apply to

        Returns:
            Tuple of:
                - x_mod: [N, D'] updated node embeddings
                - edge_weights (optional): [E] edge modulation weights
        """
        raise NotImplementedError


class GatingIntegrator(StructuralIntegrator):
    def __init__(self, feat_dim, struct_dim, hidden_dim):
        super().__init__()
        self.fusion_proj = nn.Linear(feat_dim + struct_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(feat_dim + struct_dim, hidden_dim),
            nn.Sigmoid()
        )

    def integrate(self, x, struct, edge_index, node_indices=None):
        if node_indices is None:
            node_indices = torch.arange(x.size(0), device=x.device)

        combined = torch.cat([x[node_indices], struct[node_indices]], dim=-1)
        gate = self.gate(combined)
        fused = self.fusion_proj(combined)

        raw_proj = self.fusion_proj(torch.cat([
            x[node_indices],
            torch.zeros_like(struct[node_indices])
        ], dim=-1))

        x_mod = torch.clone(x)
        x_mod[node_indices] = gate * fused + (1 - gate) * raw_proj

        return x_mod, None


class Node2VecEdgeIntegrator(StructuralIntegrator):
    def integrate(self, x, struct, edge_index, node_indices=None):
        src, dst = edge_index
        sim = function.cosine_similarity(struct[src], struct[dst], dim=-1)
        return x, sim


class GeometryAwareIntegrator(StructuralIntegrator):
    def integrate(self, x, shape_descriptors, edge_index, node_indices=None):
        raise NotImplementedError
