# gnn_pipeline/helper.py
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.utils import to_dense_adj

def get_ppr_matrix(dataset, alpha=0.05):
    A_tilde = to_dense_adj(dataset.edge_index)[0]
    D_tilde = torch.diag(1 / torch.sqrt(A_tilde.sum(dim=1)))
    H = D_tilde @ A_tilde @ D_tilde
    num_nodes = A_tilde.shape[0]
    return alpha * torch.linalg.inv(torch.eye(num_nodes, device=A_tilde.device) - (1 - alpha) * H)

def get_ppr_weight(dataset):
    P = get_ppr_matrix(dataset)
    P[P == 0] = P[P != 0].min()
    P = torch.log(1 + 1 / P)
    P = P / P.sum(1, keepdim=True) * P.shape[0]
    return P

def few_shot_masks(data, shots=5, dataname="Cora", device="cpu"):
    np.random.seed(0)
    y = data.y.cpu()
    num_classes = int(y.max()) + 1
    selected = []
    train_mask = torch.zeros(len(y)).bool().to(device)
    for i in range(num_classes):
        candidates = (torch.arange(len(y))[y == i])
        selected.append(np.random.choice(candidates, shots))
    train_mask[np.concatenate(selected)] = True
    remaining = np.setdiff1d(np.arange(len(y)), np.concatenate(selected))
    val_mask = torch.zeros(len(y)).bool().to(device)
    test_mask = torch.zeros(len(y)).bool().to(device)
    np.random.shuffle(remaining)
    val_mask[remaining[:int(0.2 * len(remaining))]] = True
    test_mask[remaining[int(0.2 * len(remaining)):]] = True
    return train_mask, val_mask, test_mask
