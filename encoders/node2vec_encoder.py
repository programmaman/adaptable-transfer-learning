# encoders/node2vec_encoder.py
import torch
import torch_geometric.nn as pyg_nn

from encoders.structural_encoder import StructuralEncoder


class Node2VecEncoder(StructuralEncoder):
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
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def forward(self, node_indices):
        """
        Compute Node2Vec embeddings for given node indices.
        If node_indices is None, return all embeddings.
        """
        if node_indices is None:
            node_indices = torch.arange(self.node2vec.num_nodes,
                                        device=self.node2vec.embedding.weight.device)
        return self.node2vec(node_indices)

    def train_encoder(self, epochs=1, batch_size=128, lr=0.01, verbose=True):
        """
        Pretrain Node2Vec before using it in a StructuralGNN.
        """
        optimizer = torch.optim.SparseAdam(self.node2vec.parameters(), lr=lr)
        loader = self.node2vec.loader(batch_size=batch_size, shuffle=True)
        device = next(self.node2vec.parameters()).device

        for epoch in range(epochs):
            self.node2vec.train()
            total_loss = 0.0
            for pos_rw, neg_rw in loader:
                pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)
                optimizer.zero_grad()
                loss = self.node2vec.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose:
                avg = total_loss / len(loader)
                print(f"[Node2Vec Epoch {epoch+1}/{epochs}] loss={avg:.4f}")
