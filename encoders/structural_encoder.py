# encoders/base.py
import torch.nn as nn

class StructuralInformationEncoder(nn.Module):
    """
    Abstract interface for any structural or geometrical encoder.
    """

    def forward(self, node_index_tensor):
        """
        Arguments:
            node_index_tensor: LongTensor containing integer node identifiers.
        Returns:
            Tensor of shape [NumberOfNodes, EmbeddingDimension]
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    @property
    def embedding_dimension(self):
        """
        Returns:
            Integer representing the dimensionality of the embeddings.
        """
        raise NotImplementedError("Subclasses must define the embedding_dimension property.")