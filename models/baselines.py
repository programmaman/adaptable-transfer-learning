import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class SimpleGNN(nn.Module):
    """
    Graph Neural Network with one layer for node classification.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(SimpleGNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        return x


class SimpleGAT(nn.Module):
    """
    Graph Attention Network (GAT) with one layer for node classification.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1):
        super(SimpleGAT, self).__init__()
        self.conv1 = pyg_nn.GATConv(in_channels, out_channels, heads=heads)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        return x


class SimpleGraphSAGE(nn.Module):
    """
    GraphSAGE with one layer for node classification.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(SimpleGraphSAGE, self).__init__()
        self.conv1 = pyg_nn.SAGEConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        return x