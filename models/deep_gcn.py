import logging

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch_geometric.nn as pyg_nn

from experiments.experiment_utils import sample_negative_edges

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StructuralGcn(nn.Module):
    """
    Graph Neural Network backbone for predicting structural features.
    This version outputs a single continuous value per node (for regression).
    """

    def __init__(self, in_channels: int, hidden_channels: int, mid_channels: int):
        super(StructuralGcn, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, mid_channels)
        # Final layer outputs one continuous value per node.
        self.conv3 = pyg_nn.GCNConv(mid_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = f.relu(x)
        x = self.conv2(x, edge_index)
        x = f.relu(x)
        x = self.conv3(x, edge_index)
        return x.squeeze()  # Squeeze to shape [num_nodes]


class GnnClassifierHead(nn.Module):
    """
    Classification head that fine-tunes a StructuralGnn backbone.
    It maps the backbone's 1D output to class logits.
    """

    def __init__(self, pretrained_model: StructuralGcn, out_channels: int):
        super(GnnClassifierHead, self).__init__()
        self.backbone = pretrained_model
        self.fc = nn.Linear(1, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Use the backbone directly, which is more idiomatic.
        features = self.backbone(x, edge_index).unsqueeze(-1)  # Shape: [num_nodes, 1]
        out = self.fc(features)
        return out


def train_structural_feature_predictor(model, data, epochs=100, lr=0.01, weight_decay=0.01, device=torch.device('cpu')):
    """
    Pretrain the GNN backbone by regressing the clustering coefficient for each node.
    """
    model.to(device)
    data = data.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    logger.info("Starting structural feature prediction pretraining...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.structural_targets)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return model


def fine_tune_model(model, data, task_labels, epochs=50, lr=0.01, weight_decay=0.01, device=torch.device('cpu')):
    """
    Fine-tune the classification model (backbone + classification head) on a downstream task.
    """
    model.to(device)
    data = data.to(device)
    task_labels = task_labels.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    logger.info("Starting fine-tuning...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, task_labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Fine-tune Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return model


def fine_tune_link_prediction(model, data, rem_edge_list, epochs=50, lr=0.01, weight_decay=0.01, device=torch.device('cpu')):
    """
    Fine-tune the model for link prediction using only retained edges.
    """
    model.to(device)
    data = data.to(device)

    # Retain 80% of edges for training (data.edge_index has already been updated)
    pos_edge_index = data.edge_index
    pos_edges_eval = rem_edge_list[0][0]  # 20% held out

    n = data.num_nodes
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    logger.info("Starting fair fine-tuning for link prediction...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        emb = model(data.x, pos_edge_index)
        if emb.dim() == 1:
            emb = emb.unsqueeze(-1)

        # Sample negative edges
        neg_edges = sample_negative_edges(pos_edges_eval, n).to(emb.device)

        pos_scores = (emb[pos_edges_eval[:, 0]] * emb[pos_edges_eval[:, 1]]).sum(dim=-1)
        neg_scores = (emb[neg_edges[:, 0]] * emb[neg_edges[:, 1]]).sum(dim=-1)

        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ])

        loss = loss_fn(torch.cat([pos_scores, neg_scores]), labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch:03d} | Fair LP Fine-tune Loss: {loss.item():.4f}")

    return model

