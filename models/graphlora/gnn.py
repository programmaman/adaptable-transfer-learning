from torch import nn
from torch_geometric.nn import GATConv, GCNConv, TransformerConv


class GNN(nn.Module):
    def __init__(self, input_dim, out_dim, activation, gnn_type='TransformerConv', gnn_layer_num=2):
        super().__init__()
        self.gnn_layer_num = gnn_layer_num
        self.activation = activation
        self.gnn_type = gnn_type

        GraphConv = {"GCN": GCNConv, "GAT": GATConv, "TransformerConv": TransformerConv}[gnn_type]
        layers = []
        if gnn_layer_num == 1:
            layers.append(GraphConv(input_dim, out_dim))
        else:
            layers.append(GraphConv(input_dim, 2 * out_dim))
            for _ in range(gnn_layer_num - 2):
                layers.append(GraphConv(2 * out_dim, 2 * out_dim))
            layers.append(GraphConv(2 * out_dim, out_dim))
        self.conv = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for conv in self.conv[:-1]:
            x = self.activation(conv(x, edge_index))
        return self.conv[-1](x, edge_index)