import torch
from torch import nn
import torch_geometric.nn as pyg_nn

class GCN(nn.Module):
    """
    Graph Neural Network with one layer for node classification.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        return x

