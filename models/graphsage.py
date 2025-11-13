import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn



class GraphSAGE(nn.Module):
    """
    GraphSAGE with one layer for node classification.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(GraphSAGE, self).__init__()
        self.conv1 = pyg_nn.SAGEConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        return x
