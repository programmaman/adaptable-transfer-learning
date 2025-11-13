import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaGCN(nn.Module):
    """
    A simple GCN implementation for use in AdaGCNModel.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Simplified forward pass for demonstration purposes
        return self.linear(x)

class AdaGCNModel(nn.Module):
    """
    AdaGCN: ensemble of GCNs with boosting weights (alphas).
    """

    def __init__(self, in_channels, out_channels, num_learners=5):
        super().__init__()
        self.learners = nn.ModuleList(
            [
                AdaGCN(in_channels, out_channels)
                for _ in range(num_learners)
            ]
        )
        self.alphas = nn.Parameter(
            torch.zeros(num_learners)
        )  # trainable alpha weights

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        logits_list = [learner(x, edge_index) for learner in self.learners]
        return logits_list

    def classify_nodes(self, logits_list):
        """
        Combines weak learners using softmax-weighted ensemble voting.
        """
        weighted_logits = 0
        alpha_softmax = F.softmax(self.alphas, dim=0)
        for alpha, logits in zip(alpha_softmax, logits_list):
            weighted_logits += alpha * F.log_softmax(logits, dim=1)
        return weighted_logits

    def node_classification_loss(self, logits_list, labels, mask=None):
        """
        Weighted sum of negative log-likelihoods for all weak learners.
        """
        alpha_softmax = F.softmax(self.alphas, dim=0)
        losses = []
        for logits in logits_list:
            logits = logits[mask] if mask is not None else logits
            target = labels[mask] if mask is not None else labels
            loss = F.nll_loss(F.log_softmax(logits, dim=1), target)
            losses.append(loss)
        return sum(alpha * loss for alpha, loss in zip(alpha_softmax, losses))
