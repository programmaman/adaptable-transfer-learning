import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
import torch.nn.functional as f

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuralGnn(nn.Module):
    """
    Graph Neural Network backbone for predicting structural features.
    This version outputs a single continuous value per node (for regression).
    """
    def __init__(self, in_channels: int, hidden_channels: int, mid_channels: int):
        super(StructuralGnn, self).__init__()
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
    def __init__(self, pretrained_model: StructuralGnn, out_channels: int):
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

def fine_tune_link_prediction(model, data, epochs=50, lr=0.01, weight_decay=0.01, device=torch.device('cpu')):
    """
    Fine-tune the model for link prediction (predicting missing edges).
    """
    model.to(device)
    data = data.to(device)

    # Positive edges = existing edges
    pos_edge_index = data.edge_index

    # Generate negative edges (random node pairs that do NOT have edges)
    neg_edge_index = torch.randint(0, data.num_nodes, (2, pos_edge_index.shape[1]), dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    logger.info("Starting fine-tuning for link prediction...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Compute node embeddings (Ensure shape is [num_nodes, embedding_dim])
        node_embeddings = model(data.x, data.edge_index)

        if node_embeddings.dim() == 1:
            node_embeddings = node_embeddings.unsqueeze(-1)  # Ensure it is [num_nodes, 1]

        # Score positive and negative edges using dot product
        pos_scores = (node_embeddings[pos_edge_index[0]] * node_embeddings[pos_edge_index[1]]).sum(dim=-1)
        neg_scores = (node_embeddings[neg_edge_index[0]] * node_embeddings[neg_edge_index[1]]).sum(dim=-1)

        # Labels: 1 for positive edges, 0 for negative edges
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

        # Compute loss
        loss = loss_fn(torch.cat([pos_scores, neg_scores]), labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Fine-tune Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, data, labels, device=torch.device('cpu')):
    """
    Evaluate the classification model on the provided labels.
    """
    model.eval()
    data = data.to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, labels)
        predictions = out.argmax(dim=1)
        accuracy = (predictions == labels).to(torch.float).mean().item()

    return loss.item(), accuracy