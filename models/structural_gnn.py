import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class StructLayer(nn.Module):
    def __init__(self, num_nodes, edge_index, embedding_dim=128, walk_length=10, context_size=5,
                 walks_per_node=10, num_negative_samples=1, p=1.0, q=1.0, sparse=True):
        super(StructLayer, self).__init__()
        self.node2vec = pyg_nn.Node2Vec(
            edge_index=edge_index,
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=sparse
        )

    def forward(self, node_indices):
        return self.node2vec(node_indices)

class StructuralGNN(nn.Module):
    def __init__(self, num_nodes, edge_index, input_dim, hidden_dim=64, output_dim=32, num_layers=2, embedding_dim=128):
        super(StructuralGNN, self).__init__()
        self.node2vec = StructLayer(num_nodes, edge_index, embedding_dim=embedding_dim)
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(pyg_nn.GCNConv(input_dim + embedding_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.gnn_layers.append(pyg_nn.SAGEConv(hidden_dim, hidden_dim))
        self.gat_layer = pyg_nn.GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, node_indices):
        node2vec_embeddings = self.node2vec(node_indices)
        x = torch.cat([x, node2vec_embeddings], dim=1)
        for gnn in self.gnn_layers:
            x = gnn(x, edge_index)
            x = torch.relu(x)
        x = self.gat_layer(x, edge_index)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x
