# encoders/base.py
import torch.nn as nn

class StructuralEncoder(nn.Module):
    """
    Abstract interface for any structural or geometrical encoder.

    Implementations produce fixed-size embeddings for given node indices,
    derived from either:
        - random-walk structure (Node2Vec),
        - geometric descriptors (from Java),
        - spectral embeddings, etc.
    """

    def forward(self, node_indices):
        """
        node_indices: LongTensor of node IDs.
        Returns:
            Tensor [num_nodes, embedding_dim] representing node embeddings.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @property
    def embedding_dim(self):
        """Return the dimensionality of produced embeddings."""
        raise NotImplementedError("Subclasses must define embedding_dim")
