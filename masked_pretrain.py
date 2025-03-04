import logging

import torch
import torch.nn as nn
import torch.nn.functional as functional_neural_network
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import networkx as nx

def get_device():
    """
    Returns the best available device: GPU if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("GPU not available, using CPU")
    return device

def compute_clustering_coefficients(data):
    # Convert PyG Data object to a NetworkX graph.
    graph = nx.Graph()
    edges = data.edge_index.t().tolist()  # Convert tensor to list of edges.
    graph.add_edges_from(edges)
    # Ensure all nodes are present, including isolated ones.
    graph.add_nodes_from(range(data.num_nodes))

    # Compute clustering coefficient for each node.
    clustering = nx.clustering(graph)

    # Create a tensor with clustering coefficients in the node order.
    clustering_tensor = torch.tensor([clustering[i] for i in range(data.num_nodes)], dtype=torch.float)
    return clustering_tensor


class GNNModel(nn.Module):
    """
    Graph Neural Network backbone for predicting structural features.
    This version outputs a single continuous value per node (for regression).
    """

    def __init__(self, in_channels: int, hidden_channels: int, mid_channels: int):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, mid_channels)
        # Final layer outputs one continuous value per node.
        self.conv3 = GCNConv(mid_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = functional_neural_network.relu(x)
        x = self.conv2(x, edge_index)
        x = functional_neural_network.relu(x)
        x = self.conv3(x, edge_index)
        return x.squeeze()  # Squeeze to shape [num_nodes]


def generate_synthetic_graph(num_nodes=200, num_edges=500, feature_dim=16):
    """
    Generate a synthetic graph with random node features and edges.
    Then compute and attach the clustering coefficient as the structural target.
    """
    x = torch.randn((num_nodes, feature_dim))
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    data.structural_targets = compute_clustering_coefficients(data)
    return data


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


class FineTuneGNN(nn.Module):
    """
    Fine-tuning model that attaches a classification head to the pretrained backbone.
    """
    def __init__(self, pretrained_model: GNNModel, out_channels: int):
        super(FineTuneGNN, self).__init__()
        self.backbone = pretrained_model
        # New classification head. It takes the backbone's 1D output and maps it to class logits.
        self.fc = nn.Linear(1, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward(x, edge_index).unsqueeze(-1)  # Shape: [num_nodes, 1]
        out = self.fc(features)
        return out


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


def experiment_node_classification(data, task_labels, device=torch.device('cpu')):
    # Step 2: Pretraining using structural feature prediction.
    backbone = GNNModel(in_channels=16, hidden_channels=32, mid_channels=64)
    pretrained_backbone = train_structural_feature_predictor(backbone, data, epochs=100, lr=0.01, device=device)

    ### NODE CLASSIFICATION FINE-TUNING ###
    pretrained_node_model = FineTuneGNN(pretrained_backbone, out_channels=2)
    pretrained_node_model = fine_tune_model(pretrained_node_model, data, task_labels, epochs=50, lr=0.01, device=device)

    # Compare performance with training from scratch for node classification
    scratch_node_backbone = GNNModel(in_channels=16, hidden_channels=32, mid_channels=64)
    scratch_node_model = FineTuneGNN(scratch_node_backbone, out_channels=2)
    scratch_node_model = fine_tune_model(scratch_node_model, data, task_labels, epochs=50, lr=0.01, device=device)

    pretrained_node_loss, pretrained_node_acc = evaluate_model(pretrained_node_model, data, task_labels, device)
    scratch_node_loss, scratch_node_acc = evaluate_model(scratch_node_model, data, task_labels, device)

    logger.info(f"Pretrained Node Model - Loss: {pretrained_node_loss:.4f}, Accuracy: {pretrained_node_acc:.4f}")
    logger.info(f"Scratch Node Model - Loss: {scratch_node_loss:.4f}, Accuracy: {scratch_node_acc:.4f}")


def experiment_link_prediction(data, device=torch.device('cpu')):
    # Step 2: Pretraining using structural feature prediction.
    backbone = GNNModel(in_channels=16, hidden_channels=32, mid_channels=64)
    pretrained_backbone = train_structural_feature_predictor(backbone, data, epochs=100, lr=0.01, device=device)

    ### LINK PREDICTION FINE-TUNING ###
    fine_tune_link_prediction(pretrained_backbone, data, epochs=50, lr=0.01, device=device)

    # Compare performance with training from scratch for link prediction
    scratch_link_backbone = GNNModel(in_channels=16, hidden_channels=32, mid_channels=64)
    fine_tune_link_prediction(scratch_link_backbone, data, epochs=50, lr=0.01, device=device)

    logger.info("Comparison of pretraining vs training from scratch completed.")


def main():
    # Get device
    device = get_device()

    # Step 1: Generate synthetic graph data.
    data = generate_synthetic_graph(num_nodes=1000, num_edges=1500, feature_dim=16)

    # Generate random task labels for the downstream node classification task.
    task_labels = torch.randint(0, 2, (data.num_nodes,))

    # Run experiments
    experiment_node_classification(data, task_labels, device)
    experiment_link_prediction(data, device)

from ogb.nodeproppred import PygNodePropPredDataset

class PygOgbnArxiv(PygNodePropPredDataset):
    def __init__(self):
        root, name, transform = '/kaggle/input', 'ogbn-arxiv', T.ToSparseTensor()
        master = pd.read_csv(osp.join(root, name, 'ogbn-master.csv'), index_col = 0)
        meta_dict = master[name]
        meta_dict['dir_path'] = osp.join(root, name)
        super().__init__(name = name, root = root, transform = transform, meta_dict = meta_dict)
    def get_idx_split(self):
        split_type = self.meta_info['split']
        path = osp.join(self.root, 'split', split_type)
        train_idx = dt.fread(osp.join(path, 'train.csv'), header = False).to_numpy().T[0]
        train_idx = torch.from_numpy(train_idx).to(torch.long)
        valid_idx = dt.fread(osp.join(path, 'valid.csv'), header = False).to_numpy().T[0]
        valid_idx = torch.from_numpy(valid_idx).to(torch.long)
        test_idx = dt.fread(osp.join(path, 'test.csv'), header = False).to_numpy().T[0]
        test_idx = torch.from_numpy(test_idx).to(torch.long)
        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

# Main init
if __name__ == '__main__':
    main()

