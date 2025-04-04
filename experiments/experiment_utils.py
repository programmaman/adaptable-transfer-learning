import json

import networkx as nx
from sklearn.preprocessing import LabelEncoder


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
    
    # generate labels
    labels = generate_task_labels(data)

    print(f"Loaded graph with {data.num_nodes} nodes, {data.num_edges} edges, {x.size(1)} features")
    print(f"Label coverage: {(labels >= 0).sum().item()} / {len(labels)} nodes labeled")
    
    return data, labels


def generate_task_labels(data, num_classes=5):
    # Generate semi-structured labels based on node features
    # Use k-means clustering to assign labels based on feature similarity
    from sklearn.cluster import KMeans

    x_np = data.x.cpu().numpy()
    kmeans = KMeans(n_clusters=num_classes, random_state=42).fit(x_np)
    labels = torch.tensor(kmeans.labels_, dtype=torch.long)
    return labels



def load_musae_facebook_dataset(edge_path, features_path, target_path):
    # Load edges
    edges_df = pd.read_csv(edge_path)
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)  # shape [2, num_edges]

    # Load features
    with open(features_path, 'r') as f:
        features_dict = json.load(f)

    # Build a consistent node ID mapping (important!)
    node_ids = sorted(set(int(k) for k in features_dict.keys()))
    node_id_map = {node_id: i for i, node_id in enumerate(node_ids)}  # external → internal ID

    # Build feature matrix
    num_nodes = len(node_ids)
    num_features = max(f for feats in features_dict.values() for f in feats) + 1
    x = torch.zeros((num_nodes, num_features))
    for raw_id, feats in features_dict.items():
        mapped_id = node_id_map[int(raw_id)]
        x[mapped_id, feats] = 1.0

    # Load labels
    target_df = pd.read_csv(target_path)
    target_df = target_df[target_df['id'].isin(node_ids)]  # filter only nodes with features
    target_df['mapped_id'] = target_df['id'].map(node_id_map)
    label_encoder = LabelEncoder()
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    encoded_labels = label_encoder.fit_transform(target_df['page_type'])
    labels[target_df['mapped_id']] = torch.tensor(encoded_labels, dtype=torch.long)

    edge_list = edge_index.t().tolist()

    # Keep only edges where both endpoints exist in node_id_map
    filtered_edges = [
        [node_id_map[src], node_id_map[dst]]
        for src, dst in edge_list
        if src in node_id_map and dst in node_id_map
    ]

    # Convert back to tensor
    edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()

    # Build PyG Data object
    data = Data(x=x, edge_index=edge_index)

    print(f"Loaded graph with {data.num_nodes} nodes, {data.num_edges} edges, {x.size(1)} features")
    print(f"Label coverage: {(labels >= 0).sum().item()} / {len(labels)} nodes labeled")

    return data, labels, label_encoder

import torch
from torch_geometric.data import Data
import pandas as pd

def load_email_eu_core_dataset(edge_path: str, label_path: str):
    # === Load edge list ===
    edge_df = pd.read_csv(edge_path, sep=" ", header=None, names=["src", "dst"])
    edge_index = torch.tensor(edge_df.values.T, dtype=torch.long)

    # === Load labels ===
    label_df = pd.read_csv(label_path, sep=" ", header=None, names=["node_id", "label"])
    max_node = max(edge_index.max().item(), label_df["node_id"].max())
    num_nodes = max_node + 1

    labels = torch.full((num_nodes,), -1, dtype=torch.long)  # default -1 (unlabeled)
    labels[label_df["node_id"].values] = torch.tensor(label_df["label"].values, dtype=torch.long)

    # === Dummy node features (e.g. identity or one-hot) ===
    x = torch.eye(num_nodes)  # identity features

    # === Create PyG Data object ===
    data = Data(x=x, edge_index=edge_index)

    print(f"Loaded graph with {data.num_nodes} nodes, {data.num_edges} edges, {x.size(1)} features")
    print(f"Label coverage: {(labels >= 0).sum().item()} / {len(labels)} nodes labeled")

    return data, labels
