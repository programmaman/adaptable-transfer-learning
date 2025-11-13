import torch.nn as nn
from torch_geometric.nn import GCNConv, TransformerConv

from models.graphlora.graphlora import GATConvLoRA


class GNNLoRA(nn.Module):
    def __init__(self, input_dim, out_dim, activation, gnn, gnn_type='GAT', gnn_layer_num=2, r=32):
        super().__init__()
        self.gnn = gnn
        self.activation = activation
        self.gnn_layer_num = gnn_layer_num
        GraphConv = {"GCN": GCNConv, "GAT": GATConvLoRA, "TransformerConv": TransformerConv}[gnn_type]

        layers = []
        if gnn_layer_num == 1:
            layers.append(GraphConv(input_dim, out_dim, r=r))
        else:
            layers.append(GraphConv(input_dim, 2 * out_dim, r=r))
            for _ in range(gnn_layer_num - 2):
                layers.append(GraphConv(2 * out_dim, 2 * out_dim, r=r))
            layers.append(GraphConv(2 * out_dim, out_dim, r=r))
        self.conv = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for i in range(self.gnn_layer_num - 1):
            x = self.gnn.conv[i](x, edge_index) + self.conv[i](x, edge_index)
        node_emb1 = self.gnn.conv[-1](x, edge_index)
        node_emb2 = self.conv[-1](x, edge_index)
        return node_emb1 + node_emb2, node_emb1, node_emb2