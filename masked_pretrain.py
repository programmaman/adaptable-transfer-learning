import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj, degree
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNModel(nn.Module):
    """
    Graph Neural Network with two layers for node classification.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def generate_synthetic_graph(num_nodes=200, num_edges=500, feature_dim=16, mask_fraction=0.2):
    """
    Generate a synthetic graph with random node features, edges, and masked nodes.
    """
    x = torch.randn((num_nodes, feature_dim))
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

    # Compute node degrees
    deg = degree(edge_index[0], num_nodes=num_nodes)

    # Determine dead nodes based on a threshold
    threshold = 5
    labels = (deg < threshold).long()

    # Select nodes to mask (remove their features and edges)
    num_masked = int(mask_fraction * num_nodes)
    masked_nodes = random.sample(range(num_nodes), num_masked)

    # Mask out node features
    x[masked_nodes] = 0  # Set feature vectors to zero

    return Data(x=x, edge_index=edge_index, labels=labels, masked_nodes=torch.tensor(masked_nodes))


def train_model(model, data, epochs=100, lr=0.01, device='cpu'):
    """
    Train the model to predict dead nodes while handling masked nodes.
    """
    model.to(device)
    data = data.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return model


def main():
    data = generate_synthetic_graph(num_nodes=200, num_edges=500, feature_dim=16, mask_fraction=0.2)
    model = GNNModel(in_channels=16, hidden_channels=32, out_channels=2)
    trained_model = train_model(model, data, epochs=100, lr=0.01, device='cpu')
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
