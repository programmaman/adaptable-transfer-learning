# encoders/base.py

from torch import nn
import scipy.sparse as sp
import numpy as np
import torch
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import to_scipy_sparse_matrix

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



class RandomStructuralEncoder(nn.Module):
    def __init__(self, num_nodes, dim):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, dim)
        nn.init.normal_(self.emb.weight, std=1.0)
        self.emb.weight.requires_grad = False

        self.embedding_dimension = dim

    def forward(self, nodes):
        return self.emb(nodes)

class DegreeStructuralEncoder(nn.Module):
    def __init__(self, edge_index, num_nodes):
        super().__init__()

        deg = torch.zeros(num_nodes, dtype=torch.float)
        for i in range(edge_index.size(1)):
            deg[edge_index[0, i]] += 1

        deg = deg.unsqueeze(1)  # shape [N, 1]

        self.register_buffer("deg", deg)
        self.embedding_dimension = 1

    def forward(self, nodes):
        return self.deg[nodes]


class LaplacianStructuralEncoder(nn.Module):
    def __init__(self, edge_index, num_nodes, dim=8):
        super().__init__()

        # Build scipy sparse adjacency
        A = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

        # Degree
        deg = np.array(A.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0

        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
        I = sp.eye(num_nodes)
        L = I - D_inv_sqrt @ A @ D_inv_sqrt

        # Try to compute eigenvectors robustly
        k = dim + 1
        success = False

        while not success and k >= 2:
            try:
                eigvals, eigvecs = sp.linalg.eigsh(L, k=k, which="SM", tol=1e-2)
                success = True
            except Exception as e:
                print(f"[LaplacianPE] eigsh failed for k={k}, retrying with k={k-1}")
                k = k - 1

        if not success:
            raise RuntimeError("Laplacian eigendecomposition failed completely.")

        # Take non-trivial ones (skip first eigenvector)
        usable_dim = min(dim, eigvecs.shape[1] - 1)
        pe = torch.from_numpy(eigvecs[:, 1 : 1 + usable_dim]).float()

        self.register_buffer("pe", pe)
        self.embedding_dimension = pe.size(1)

    def forward(self, nodes):
        return self.pe[nodes]



class Node2VecEncoder(StructuralInformationEncoder):
    """
    Node2Vec-based structural encoder.
    Wraps PyTorch Geometric's Node2Vec implementation.
    """

    def __init__(self, num_nodes, edge_index, embedding_dim=128,
                 walk_length=10, context_size=5, walks_per_node=10,
                 num_negative_samples=1, p=1.0, q=1.0, sparse=True):
        super().__init__()

        self.node2vec = pyg_nn.Node2Vec(
            edge_index=edge_index,
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p, q=q,
            sparse=sparse
        )

        self._embedding_dimension = embedding_dim

    @property
    def embedding_dimension(self):
        # [FIX] Renamed from embedding_dim to match base class property
        return self._embedding_dimension

    def forward(self, node_indices=None):
        """
        Compute Node2Vec embeddings for given node indices.
        """
        return self.node2vec(node_indices)

    def train_encoder(self, epochs=1, batch_size=128, lr=0.01, verbose=True):
        """
        Pretrain Node2Vec.
        Note: This adapts automatically to the device the model currently resides on.
        """
        # [NOTE] SparseAdam is efficient for embeddings but check CUDA support if
        # using very old PyTorch versions. Modern PyTorch handles this fine.
        optimizer = torch.optim.SparseAdam(self.node2vec.parameters(), lr=lr)
        loader = self.node2vec.loader(batch_size=batch_size, shuffle=True)

        # [FIX] Dynamically get the device from the model parameters
        device = next(self.node2vec.parameters()).device

        for epoch in range(epochs):
            self.node2vec.train()
            total_loss = 0.0

            for pos_rw, neg_rw in loader:
                # Move sampled walks to the same device as the model
                pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)

                optimizer.zero_grad()
                loss = self.node2vec.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose:
                avg = total_loss / len(loader)
                print(f"[Node2Vec Epoch {epoch + 1}/{epochs}] loss={avg:.4f} | device={device}")