# graphlora.py
"""
GraphLoRA: Low-Rank Adaptation for Graph Neural Networks
---------------------------------------------------------
This module contains model definitions for:
  - GNN: generic GCN/GAT/Transformer-based encoder
  - GATConvLoRA: Low-Rank adaptation of GATConv
  - GNNLoRA: combines frozen base GNN with trainable LoRA adapter
  - GraphLoRAWrapped: high-level classification wrapper around GNNLoRA
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.nn.dense.linear import Linear
from typing import Tuple, Union


# -------------------------------------------------------------------------
# Utility activation selector (purely for model internals)
# -------------------------------------------------------------------------
def get_activation(act_type: str = "relu"):
    if act_type == "leakyrelu":
        return F.leaky_relu
    elif act_type == "tanh":
        return torch.tanh
    elif act_type == "relu":
        return F.relu
    elif act_type == "prelu":
        return nn.PReLU()
    elif act_type == "sigmoid":
        return F.sigmoid
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


# -------------------------------------------------------------------------
# Generic GNN backbone (GCN, GAT, Transformer)
# -------------------------------------------------------------------------
class GNN(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, activation, gnn_type: str = "TransformerConv", gnn_layer_num: int = 2):
        super().__init__()
        self.gnn_layer_num = gnn_layer_num
        self.activation = activation

        if gnn_type == "GCN":
            GraphConv = GCNConv
        elif gnn_type == "GAT":
            GraphConv = GATConv
        elif gnn_type == "TransformerConv":
            GraphConv = TransformerConv
        else:
            raise KeyError("gnn_type must be one of ['GCN', 'GAT', 'TransformerConv']")

        self.gnn_type = gnn_type

        if gnn_layer_num < 1:
            raise ValueError(f"GNN layer_num should >= 1 but got {gnn_layer_num}")

        layers = []
        if gnn_layer_num == 1:
            layers.append(GraphConv(input_dim, out_dim))
        elif gnn_layer_num == 2:
            layers.extend([
                GraphConv(input_dim, 2 * out_dim),
                GraphConv(2 * out_dim, out_dim)
            ])
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


# -------------------------------------------------------------------------
# LoRA-augmented GATConv
# -------------------------------------------------------------------------
class GATConvLoRA(GATConv):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: int | None = None,
        fill_value: Union[float, torch.Tensor, str] = "mean",
        bias: bool = True,
        r: int = 32,
        **kwargs,
    ):
        super().__init__(
            in_channels, out_channels, heads, concat,
            negative_slope, dropout, add_self_loops, edge_dim,
            fill_value, bias, **kwargs
        )
        self.r = r

        # Build low-rank adapters
        if isinstance(in_channels, int):
            self.lin_src = nn.Sequential(
                Linear(in_channels, self.r, bias=False, weight_initializer="glorot"),
                Linear(self.r, heads * out_channels, bias=False, weight_initializer="glorot")
            )
            self.lin_dst = self.lin_src
        else:
            self.lin_src = nn.Sequential(
                Linear(in_channels[0], self.r, bias=False, weight_initializer="glorot"),
                Linear(self.r, heads * out_channels, bias=False, weight_initializer="glorot")
            )
            self.lin_dst = nn.Sequential(
                Linear(in_channels[1], self.r, bias=False, weight_initializer="glorot"),
                Linear(self.r, heads * out_channels, bias=False, weight_initializer="glorot")
            )

        self.reset_parameters_lora()

    def reset_parameters_lora(self):
        torch.nn.init.kaiming_normal_(self.lin_src[0].weight)
        torch.nn.init.zeros_(self.lin_src[1].weight)
        torch.nn.init.kaiming_normal_(self.lin_dst[0].weight)
        torch.nn.init.zeros_(self.lin_dst[1].weight)


# -------------------------------------------------------------------------
# GNNLoRA: combines frozen base GNN + LoRA adapters
# -------------------------------------------------------------------------
class GNNLoRA(nn.Module):
    def __init__(self, input_dim, out_dim, activation, gnn, gnn_type="GAT", gnn_layer_num=2, r=32):
        super().__init__()
        self.gnn = gnn  # frozen base GNN
        self.gnn_layer_num = gnn_layer_num
        self.activation = activation

        if gnn_type == "GCN":
            GraphConv = GCNConv
        elif gnn_type == "GAT":
            GraphConv = GATConvLoRA
        elif gnn_type == "TransformerConv":
            GraphConv = TransformerConv
        else:
            raise KeyError("gnn_type must be one of ['GCN', 'GAT', 'TransformerConv']")

        self.gnn_type = gnn_type

        # Construct LoRA adapters
        layers = []
        if gnn_layer_num == 1:
            layers.append(GraphConv(input_dim, out_dim, r=r))
        elif gnn_layer_num == 2:
            layers.extend([
                GraphConv(input_dim, 2 * out_dim, r=r),
                GraphConv(2 * out_dim, out_dim, r=r)
            ])
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


# -------------------------------------------------------------------------
# GraphLoRAWrapped: complete model for downstream tasks
# -------------------------------------------------------------------------
class GraphLoRAWrapped(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_classes,
        base_model_path,
        gnn_type="GCN",
        num_layers=2,
        r=8,
        activation="relu",
    ):
        super().__init__()
        self.base_model_path = base_model_path
        self.activation = get_activation(activation)

        # Load or initialize frozen base GNN
        self.gnn_frozen = GNN(in_dim, out_dim, self.activation, gnn_type, num_layers)
        if os.path.exists(base_model_path):
            self.gnn_frozen.load_state_dict(torch.load(base_model_path, map_location="cpu"))
            for p in self.gnn_frozen.parameters():
                p.requires_grad = False
            self.gnn_frozen.eval()
        else:
            print(f"[GraphLoRAWrapped] No checkpoint at {base_model_path}. Will train from scratch.")

        # Attach LoRA adapters
        self.gnn_lora = GNNLoRA(in_dim, out_dim, self.activation, self.gnn_frozen,
                                gnn_type=gnn_type, gnn_layer_num=num_layers, r=r)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x, edge_index):
        emb, _, _ = self.gnn_lora(x, edge_index)
        logits = self.classifier(emb)
        return F.normalize(logits, p=2, dim=-1)

    def get_embeddings(self, x, edge_index):
        emb, _, _ = self.gnn_lora(x, edge_index)
        return emb

    def reset_with_input_dim(self, new_in_dim: int):
        """
        Rebuilds modules to accept a new input feature size
        while keeping same output dimensions and configuration.
        """
        out_dim = self.classifier.out_features
        num_classes = self.classifier.out_features
        gnn_type = self.gnn_frozen.gnn_type
        num_layers = self.gnn_frozen.gnn_layer_num
        activation = self.gnn_frozen.activation
        r = self.gnn_lora.conv[0].lin_src[0].out_channels if hasattr(self.gnn_lora.conv[0], "lin_src") else 8

        self.gnn_frozen = GNN(new_in_dim, out_dim, activation, gnn_type, num_layers)
        self.gnn_lora = GNNLoRA(new_in_dim, out_dim, activation, self.gnn_frozen,
                                gnn_type=gnn_type, gnn_layer_num=num_layers, r=r)
        self.classifier = nn.Linear(out_dim, num_classes).to(self.classifier.weight.device)
