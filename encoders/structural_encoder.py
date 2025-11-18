# encoders/base.py

import torch.nn as neural_network


class StructuralInformationEncoder(neural_network.Module):
    """
    Abstract interface for any structural or geometrical encoder.

    Implementations produce fixed-size embedding vectors for the
    provided node index values. These embeddings may be derived from:

        - random-walk structural information (for example, Node To Vector),
        - geometric descriptors,
        - spectral embedding methods,
        - or other structural information sources.
    """

    def forward(self, node_index_tensor):
        """
        Arguments:
            node_index_tensor:
                LongTensor containing the integer node identifiers.

        Returns:
            Tensor of shape [NumberOfNodes, EmbeddingDimension]
            representing structural embedding vectors for each node.
        """
        raise NotImplementedError(
            "Subclasses must implement the forward method."
        )

    @property
    def embedding_dimension(self):
        """
        Returns:
            Integer representing the dimensionality of the structural
            embedding vectors produced by this encoder.
        """
        raise NotImplementedError(
            "Subclasses must define the embedding_dimension property."
        )
