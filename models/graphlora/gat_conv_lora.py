import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn.dense.linear import Linear
from typing import Tuple, Union

class GATConvLoRA(GATConv):
    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, r: int = 32, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.r = r
        if isinstance(in_channels, int):
            self.lin_src_a = Linear(in_channels, r, bias=False)
            self.lin_src_b = Linear(r, self.heads * out_channels, bias=False)
            self.lin_src = nn.Sequential(self.lin_src_a, self.lin_src_b)
            self.lin_dst = self.lin_src
        else:
            self.lin_src_a = Linear(in_channels[0], r, bias=False)
            self.lin_src_b = Linear(r, self.heads * out_channels, bias=False)
            self.lin_src = nn.Sequential(self.lin_src_a, self.lin_src_b)
            self.lin_dst_a = Linear(in_channels[1], r, bias=False)
            self.lin_dst_b = Linear(r, self.heads * out_channels, bias=False)
            self.lin_dst = nn.Sequential(self.lin_dst_a, self.lin_dst_b)
        self.reset_parameters_lora()

    def reset_parameters_lora(self):
        torch.nn.init.kaiming_normal_(self.lin_src[0].weight)
        torch.nn.init.zeros_(self.lin_src[1].weight)
        torch.nn.init.kaiming_normal_(self.lin_dst[0].weight)
        torch.nn.init.zeros_(self.lin_dst[1].weight)
