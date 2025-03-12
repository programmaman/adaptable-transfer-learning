import torch
import logging
import networkx as nx
import random
from torch_geometric.data import Data
from torch_geometric.utils import degree


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def generate_masked_synthetic_graph(num_nodes=200, num_edges=500, feature_dim=16, mask_fraction=0.2):
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


__all__ = ["get_device", "generate_masked_synthetic_graph", "generate_synthetic_graph"]
