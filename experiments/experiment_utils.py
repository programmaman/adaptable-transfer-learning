import torch
import networkx as nx
from torch_geometric.data import Data

def generate_synthetic_graph(num_nodes=1000, num_edges=1500, feature_dim=16):
    # Generate random node features
    x = torch.randn((num_nodes, feature_dim))

    # Generate random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

    # Build PyG data object
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

    # Generate structural targets using clustering coefficient (for auxiliary tasks, not labels)
    graph = nx.Graph()
    edges = edge_index.t().tolist()
    graph.add_edges_from(edges)
    graph.add_nodes_from(range(num_nodes))
    clustering = nx.clustering(graph)
    data.structural_targets = torch.tensor([clustering[i] for i in range(num_nodes)], dtype=torch.float)

    return data


def generate_task_labels(data, num_classes=5):
    # Generate semi-structured labels based on node features
    # Use k-means clustering to assign labels based on feature similarity
    from sklearn.cluster import KMeans
    import numpy as np

    x_np = data.x.cpu().numpy()
    kmeans = KMeans(n_clusters=num_classes, random_state=42).fit(x_np)
    labels = torch.tensor(kmeans.labels_, dtype=torch.long)
    return labels



