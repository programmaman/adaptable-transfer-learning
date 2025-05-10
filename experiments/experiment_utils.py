import random

import networkx as nx
import numpy as np


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_synthetic_graph(num_nodes=10000, num_edges=15000, feature_dim=16):
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

def load_deezer_europe_dataset(edge_path, features_path, target_path):
    # --- Load edges ---
    edges_df = pd.read_csv(edge_path)
    edge_index = torch.tensor(edges_df[['node_1', 'node_2']].values.T, dtype=torch.long)

    # --- Load features ---
    with open(features_path, 'r') as f:
        features_dict = json.load(f)

    node_ids = sorted(set(int(k) for k in features_dict.keys()))
    node_id_map = {node_id: i for i, node_id in enumerate(node_ids)}
    num_nodes = len(node_ids)
    num_features = max(f for feats in features_dict.values() for f in feats) + 1

    x = torch.zeros((num_nodes, num_features))
    for raw_id, feats in features_dict.items():
        mapped_id = node_id_map[int(raw_id)]
        x[mapped_id, feats] = 1.0

    # --- Load labels ---
    target_df = pd.read_csv(target_path)
    target_df = target_df[target_df['id'].isin(node_ids)]
    target_df['mapped_id'] = target_df['id'].map(node_id_map)

    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    labels[target_df['mapped_id']] = torch.tensor(target_df['target'].values, dtype=torch.long)

    # --- Filter valid edges ---
    edge_list = edge_index.t().tolist()
    filtered_edges = [
        [node_id_map[src], node_id_map[dst]]
        for src, dst in edge_list
        if src in node_id_map and dst in node_id_map
    ]
    edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)

    print(f"Loaded Deezer Europe: {data.num_nodes} nodes, {data.num_edges} edges, {x.size(1)} features")
    print(f"Label coverage: {(labels >= 0).sum().item()} / {len(labels)}")

    return data, labels


def load_twitch_gamers_dataset(edge_path: str, target_path: str, use_metadata_as_features=True):
    # Load metadata
    meta_df = pd.read_csv(target_path)
    node_ids = sorted(meta_df["numeric_id"].unique())
    node_id_map = {raw_id: i for i, raw_id in enumerate(node_ids)}  # raw → internal

    num_nodes = len(node_ids)

    # --- Initialize features ---
    if use_metadata_as_features:
        feature_cols = ["views", "life_time", "affiliate"]
        x = torch.zeros((num_nodes, len(feature_cols)))
        for _, row in meta_df.iterrows():
            idx = node_id_map[row["numeric_id"]]
            x[idx] = torch.tensor([row[col] for col in feature_cols], dtype=torch.float)
    else:
        x = torch.eye(num_nodes)

    # --- Initialize labels ---
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    for _, row in meta_df.iterrows():
        idx = node_id_map[row["numeric_id"]]
        labels[idx] = int(row["mature"])

    # --- Load and remap edges ---
    edge_df = pd.read_csv(edge_path)
    raw_edges = edge_df[["numeric_id_1", "numeric_id_2"]].values.tolist()

    filtered_edges = [
        [node_id_map[u], node_id_map[v]]
        for u, v in raw_edges
        if u in node_id_map and v in node_id_map
    ]
    edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()

    # --- Create PyG Data object ---
    data = Data(x=x, edge_index=edge_index)

    print(f"Loaded Twitch Gamers graph with {data.num_nodes} nodes, {data.num_edges} edges, {x.size(1)} features")
    print(f"Label coverage: {(labels >= 0).sum().item()} / {len(labels)} nodes labeled")

    return data, labels


def load_musae_facebook_dataset(edge_path, features_path, target_path):
    # Load edges
    edges_df = pd.read_csv(edge_path)
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)  # shape [2, num_edges]

    # Load features
    with open(features_path, 'r') as f:
        features_dict = json.load(f)

    # Build a consistent node ID mapping
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


import json
from sklearn.preprocessing import LabelEncoder


def load_musae_github_dataset(edge_path, features_path, target_path):
    # Load edges
    edges_df = pd.read_csv(edge_path)
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)  # shape [2, num_edges]

    # Load features
    with open(features_path, 'r') as f:
        features_dict = json.load(f)

    # Build a consistent node ID mapping
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
    target_df = target_df[target_df['id'].isin(node_ids)]
    target_df['mapped_id'] = target_df['id'].map(node_id_map)

    label_encoder = LabelEncoder()
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    encoded_labels = label_encoder.fit_transform(target_df['ml_target'])
    labels[target_df['mapped_id']] = torch.tensor(encoded_labels, dtype=torch.long)

    # Filter valid edges
    edge_list = edge_index.t().tolist()
    filtered_edges = [
        [node_id_map[src], node_id_map[dst]]
        for src, dst in edge_list
        if src in node_id_map and dst in node_id_map
    ]
    edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()

    # Build PyG Data object
    data = Data(x=x, edge_index=edge_index)

    print(f"Loaded GitHub graph with {data.num_nodes} nodes, {data.num_edges} edges, {x.size(1)} features")
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


def sample_negative_edges(pos_edges: torch.Tensor, num_nodes: int, num_samples: int = None,
                          existing_edge_set: set = None) -> torch.Tensor:
    """
    Sample negative edges (non-existent links) for link prediction evaluation.

    Args:
        pos_edges (Tensor): Shape [num_pos_edges, 2] — existing positive edges.
        num_nodes (int): Number of nodes in the graph.
        num_samples (int, optional): How many negative edges to sample. Defaults to len(pos_edges).
        existing_edge_set (set, optional): Optional set of (u, v) edges to avoid. If not provided, uses pos_edges.

    Returns:
        Tensor: Negative edges of shape [num_samples, 2].
    """
    if num_samples is None:
        num_samples = pos_edges.size(0)

    # Initialize set of positive or existing edges
    if existing_edge_set is None:
        existing_edge_set = set((u.item(), v.item()) for u, v in pos_edges)

    neg_edges = set()
    while len(neg_edges) < num_samples:
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()
        if u != v and (u, v) not in existing_edge_set and (v, u) not in existing_edge_set:
            neg_edges.add((u, v))

    # Convert to tensor
    neg_edges_tensor = torch.tensor(list(neg_edges), dtype=torch.long)
    return neg_edges_tensor


def split_edges_for_link_prediction(edge_index: torch.Tensor, removal_ratio: float = 0.1):
    """
    Randomly removes a subset of edges for link prediction.

    Returns:
        remaining_edges: Tensor [2, num_remaining]
        removed_edges_dict: Dict format matching rem_edge_list
    """
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    split_idx = int((1 - removal_ratio) * num_edges)
    remaining_edges = edge_index[:, perm[:split_idx]]
    removed_edges = edge_index[:, perm[split_idx:]]

    # Match expected format: {0: [Tensor of shape [num_removed, 2]]}
    rem_edge_list = {0: [removed_edges.t()]}  # [num_edges, 2]
    return remaining_edges, rem_edge_list


from dataclasses import dataclass
from typing import Optional, Any


from dataclasses import dataclass, field
from typing import Any, Optional, Dict


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: Optional[float] = None
    ap: Optional[float] = None
    preds: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"Accuracy: {self.accuracy:.4f}",
            f"Precision: {self.precision:.4f}",
            f"Recall: {self.recall:.4f}",
            f"F1: {self.f1:.4f}",
        ]
        if self.auc is not None:
            parts.append(f"AUC: {self.auc:.4f}")
        if self.ap is not None:
            parts.append(f"AP: {self.ap:.4f}")
        return " | ".join(parts)

    def as_dict(self):
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "ap": self.ap,
            **self.metadata
        }

