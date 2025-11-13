import torch
from torch import nn
import torch_geometric.nn as pyg_nn

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) with one layer for node classification.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1):
        super(GAT, self).__init__()
        self.conv1 = pyg_nn.GATConv(in_channels, out_channels, heads=heads)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        return x
