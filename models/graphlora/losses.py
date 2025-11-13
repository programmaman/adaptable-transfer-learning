# gnn_pipeline/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


class SMMDLoss(nn.Module):
    """Structure-aware MMD loss with optional PPR weighting."""
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        L2_dist = torch.cdist(total, total, p=2) ** 2
        bandwidth = self.fix_sigma or torch.sum(L2_dist) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        return sum(torch.exp(-L2_dist / bw) for bw in bandwidth_list)

    def forward(self, source, target, ppr=None):
        batch = source.size(0)
        kernels = self.gaussian_kernel(source, target)
        XX = (kernels[:batch, :batch] * (ppr if ppr is not None else 1)).mean()
        YY = kernels[batch:, batch:].mean()
        XY = kernels[:batch, batch:].mean()
        YX = kernels[batch:, :batch].mean()
        return XX + YY - XY - YX


def batched_contrastive_loss(z1, z2, labels, tau=0.5, batch_size=1000):
    """Memory-safe label-based contrastive loss."""
    N = z1.size(0)
    indices = torch.arange(N, device=z1.device)
    losses = []

    def sim(a, b):
        a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
        return torch.mm(a, b.t())

    for i in range(0, N, batch_size):
        idx = indices[i:i + batch_size]
        refl = torch.exp(sim(z1[idx], z1) / tau)
        between = torch.exp(sim(z1[idx], z2) / tau)
        same = (labels[idx].unsqueeze(1) == labels.unsqueeze(0)).float()
        numer = (same * between).sum(1) + 1e-12
        denom = refl.sum(1) + between.sum(1) - torch.diagonal(refl[:, i:i + len(idx)])
        losses.append(-torch.log(numer / denom))
    return torch.cat(losses).mean()


def edge_reconstruction_loss(logits, edge_index, num_nodes):
    """Predicts positive and negative edges using softmaxed node scores."""
    pos_edge = edge_index
    neg_edge = negative_sampling(pos_edge, num_nodes=num_nodes, num_neg_samples=pos_edge.size(1))
    log_probs = torch.softmax(logits, dim=1)
    logits_pos = (log_probs[pos_edge[0]] * log_probs[pos_edge[1]]).sum(dim=1)
    logits_neg = (log_probs[neg_edge[0]] * log_probs[neg_edge[1]]).sum(dim=1)
    target = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)])
    pred = torch.cat([logits_pos, logits_neg])
    return F.binary_cross_entropy_with_logits(pred, target)
