import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as f

class SimpleGNN(nn.Module):
    """
    Graph Neural Network with two layers for node classification.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(SimpleGNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = f.relu(x)
        x = self.conv2(x, edge_index)
        return x
