import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import logging
import networkx as nx

# Import both models
from models.gpt_gnn import GPT_GNN, GNN
from masked_pretrain import GNNModel, FineTuneGNN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate synthetic graph (Same graph for both models)
def generate_synthetic_graph(num_nodes=1000, num_edges=1500, feature_dim=16):
    x = torch.randn((num_nodes, feature_dim))  # Random node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)  # Random edges
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

    # Generate structural targets (for GPT-GNN's auxiliary tasks)
    graph = nx.Graph()
    edges = data.edge_index.t().tolist()
    graph.add_edges_from(edges)
    graph.add_nodes_from(range(data.num_nodes))
    clustering = nx.clustering(graph)
    data.structural_targets = torch.tensor([clustering[i] for i in range(data.num_nodes)], dtype=torch.float)

    return data


# Define task labels for node classification (Random binary labels)
def generate_task_labels(data):
    return torch.randint(0, 2, (data.num_nodes,))  # Binary classification


# Train GPT-GNN on Link Prediction
def train_gpt_gnn(data, task_labels):
    num_types = 1  # Assume one node type
    num_relations = 1  # Assume one relation type
    gnn = GNN(in_dim=16, n_hid=32, num_types=num_types, num_relations=num_relations, n_heads=2, n_layers=2)
    attr_decoder = nn.Linear(32, 1)  # Simple attribute decoder

    model = GPT_GNN(gnn, rem_edge_list={}, attr_decoder=attr_decoder, types=num_types, neg_samp_num=5, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    task_labels = task_labels.to(device)

    logger.info("Training GPT-GNN for Node Classification...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, torch.zeros_like(data.x[:, 0]), None, data.edge_index, None)
        loss = loss_fn(output, task_labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return model


# Train GCN-based model (from your script)
def train_gcn(data, task_labels):
    backbone = GNNModel(in_channels=16, hidden_channels=32, mid_channels=64)
    model = FineTuneGNN(backbone, out_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    data = data.to(device)
    task_labels = task_labels.to(device)

    logger.info("Training GCN-based model for Node Classification...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = loss_fn(output, task_labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return model


# Evaluate models
def evaluate_model(model, data, labels):
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

def evaluate_gpt_gnn(model, data, labels):
    """
    Evaluate GPT-GNN model with correct argument structure.
    """
    model.eval()
    data = data.to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        out = model(data.x, torch.zeros_like(data.x[:, 0]), None, data.edge_index, None)  # Ensure correct arguments
        loss = loss_fn(out, labels)
        predictions = out.argmax(dim=1)
        accuracy = (predictions == labels).to(torch.float).mean().item()

    return loss.item(), accuracy


# Experiment
def run_experiment():
    data = generate_synthetic_graph()
    task_labels = generate_task_labels(data)

    # Train models
    gpt_model = train_gpt_gnn(data, task_labels)
    gcn_model = train_gcn(data, task_labels)

    # Evaluate models
    gpt_loss, gpt_acc = evaluate_gpt_gnn(gpt_model, data, task_labels)
    gcn_loss, gcn_acc = evaluate_model(gcn_model, data, task_labels)

    logger.info(f"GPT-GNN Model - Loss: {gpt_loss:.4f}, Accuracy: {gpt_acc:.4f}")
    logger.info(f"GCN Model - Loss: {gcn_loss:.4f}, Accuracy: {gcn_acc:.4f}")


if __name__ == "__main__":
    run_experiment()
