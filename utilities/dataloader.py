# gnn_pipeline/data.py
"""
Unified Dataset Loader for Graph Neural Networks
-------------------------------------------------
Handles all data sources:
 - Built-in PyG datasets (Cora, CiteSeer, PubMed, Amazon)
 - MUSAE datasets (Facebook, GitHub)
 - Deezer Europe
 - Twitch Gamers
 - Email-EU Core
 - Synthetic graph generation

All loaders return:
    data: torch_geometric.data.Data
    labels: torch.Tensor
    metadata: dict (optional)
"""

import json
import random
from typing import Tuple, Dict, Any

import torch
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops
import torch_geometric.transforms as transforms
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.transforms import SVDFeatureReduction


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_self_loops(dataset):
    dataset.edge_index = add_remaining_self_loops(dataset.edge_index)[0]
    return dataset


def apply_feature_reduction(dataset, out_channels=100):
    reducer = SVDFeatureReduction(out_channels=out_channels)
    return reducer(dataset)

def _ensure_masks(data, train_ratio=0.6, val_ratio=0.2):
    if all(hasattr(data, attr) for attr in ("train_mask", "val_mask", "test_mask")):
        return data
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[perm[:train_end]] = True
    data.val_mask[perm[train_end:val_end]] = True
    data.test_mask[perm[val_end:]] = True
    return data

# ----------------------------------------------------------------------
# Built-in datasets
# ----------------------------------------------------------------------
def load_pyg_dataset(name: str, root: str = "./datasets"):
    name = name.lower()
    transform = transforms.NormalizeFeatures()
    if name in ["computers", "photo"]:
        dataset = Amazon(root, name.capitalize(), transform)
    elif name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root=root, name=name.capitalize(), transform=transform)
    else:
        raise ValueError(f"Unknown built-in dataset: {name}")
    data = _ensure_masks(dataset[0])
    return data, data.y, {"source": "pyg_builtin"}

# ----------------------------------------------------------------------
# Synthetic dataset
# ----------------------------------------------------------------------


def generate_synthetic_graph(
        num_nodes: int = 1000,
        num_edges: int = 1500,
        feature_dim: int = 16,
        num_classes: int = 5
) -> Tuple[Data, torch.Tensor, Dict[str, Any]]:
    # Generate random node features and edges
    x = torch.randn((num_nodes, feature_dim))
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

    # Compute clustering coefficients
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edge_index.t().tolist())
    clustering = nx.clustering(g)
    data.structural_targets = torch.tensor([clustering[i] for i in range(num_nodes)], dtype=torch.float)

    # Generate labels using k-means clustering
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init='auto')
    kmeans.fit(x.numpy())
    labels = torch.tensor(kmeans.labels_, dtype=torch.long)
    data.y = labels

    data = _ensure_masks(data)
    print(f"[Synthetic] {num_nodes} nodes | {num_edges} edges | {feature_dim} features | {num_classes} classes")
    return data, labels, {"source": "synthetic"}



# ----------------------------------------------------------------------
# Deezer Europe
# ----------------------------------------------------------------------
def load_deezer_europe(edge_path, features_path, target_path):
    edges_df = pd.read_csv(edge_path)
    edge_index = torch.tensor(edges_df[["node_1", "node_2"]].values.T, dtype=torch.long)

    with open(features_path, "r") as f:
        features_dict = json.load(f)

    node_ids = sorted(map(int, features_dict.keys()))
    node_id_map = {nid: i for i, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)
    num_features = max(f for feats in features_dict.values() for f in feats) + 1

    x = torch.zeros((num_nodes, num_features))
    for raw, feats in features_dict.items():
        x[node_id_map[int(raw)], feats] = 1.0

    target_df = pd.read_csv(target_path)
    target_df = target_df[target_df["id"].isin(node_ids)]
    target_df["mapped_id"] = target_df["id"].map(node_id_map)
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    labels[target_df["mapped_id"]] = torch.tensor(target_df["target"].values)

    data = Data(x=x, edge_index=edge_index, y=labels)
    data = _ensure_masks(data)
    return data, labels, {"source": "deezer_europe"}




# ----------------------------------------------------------------------
# Twitch Gamers
# ----------------------------------------------------------------------
def load_twitch_gamers(edge_path, meta_path, use_metadata_as_features=True):
    meta_df = pd.read_csv(meta_path)
    node_ids = sorted(meta_df["numeric_id"].unique())
    num_nodes = len(node_ids)

    if use_metadata_as_features:
        feature_cols = ["views", "life_time", "affiliate"]
        x = torch.tensor(meta_df[feature_cols].values, dtype=torch.float)
    else:
        x = torch.eye(num_nodes)

    labels = torch.tensor(meta_df["mature"].astype(int).values, dtype=torch.long)

    edge_df = pd.read_csv(edge_path)
    edge_index = torch.tensor(edge_df[["numeric_id_1", "numeric_id_2"]].values.T, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=labels)
    data = _ensure_masks(data)
    return data, labels, {"source": "twitch_gamers"}



# ----------------------------------------------------------------------
# MUSAE (Facebook & GitHub)
# ----------------------------------------------------------------------
def _load_musae(edge_path, features_path, target_path, label_col):
    edges_df = pd.read_csv(edge_path)
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)

    with open(features_path, "r") as f:
        features_dict = json.load(f)

    node_ids = sorted(map(int, features_dict.keys()))
    node_id_map = {nid: i for i, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)
    num_features = max(f for feats in features_dict.values() for f in feats) + 1

    x = torch.zeros((num_nodes, num_features))
    for raw_id, feats in features_dict.items():
        x[node_id_map[int(raw_id)], feats] = 1.0

    target_df = pd.read_csv(target_path)
    target_df = target_df[target_df["id"].isin(node_ids)]
    target_df["mapped_id"] = target_df["id"].map(node_id_map)

    le = LabelEncoder()
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    encoded = le.fit_transform(target_df[label_col])
    labels[target_df["mapped_id"]] = torch.tensor(encoded)

    filtered_edges = [
        [node_id_map[src], node_id_map[dst]]
        for src, dst in edge_index.t().tolist()
        if src in node_id_map and dst in node_id_map
    ]
    edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=labels)
    data = _ensure_masks(data)
    return data, labels, {"encoder": le, "source": f"musae_{label_col}"}



def load_musae_facebook(edge, features, target):
    return _load_musae(edge, features, target, "page_type")


def load_musae_github(edge, features, target):
    return _load_musae(edge, features, target, "ml_target")


# ----------------------------------------------------------------------
# Email-EU Core
# ----------------------------------------------------------------------
def load_email_eu_core(edge_path, label_path):
    edge_df = pd.read_csv(edge_path, sep=" ", header=None, names=["src", "dst"])
    edge_index = torch.tensor(edge_df.values.T, dtype=torch.long)
    label_df = pd.read_csv(label_path, sep=" ", header=None, names=["node_id", "label"])
    num_nodes = max(edge_index.max().item(), label_df["node_id"].max()) + 1
    x = torch.eye(num_nodes)
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    labels[label_df["node_id"]] = torch.tensor(label_df["label"].values)
    data = Data(x=x, edge_index=edge_index, y=labels)
    data = _ensure_masks(data)
    return data, labels, {"source": "email_eu_core"}



# ----------------------------------------------------------------------
# Dataset Dispatcher
# ----------------------------------------------------------------------
def load_dataset(name: str, root: str = "./datasets", **kwargs):
    """
    Unified dataset loader by name.
    Examples:
        load_dataset("Cora")
        load_dataset("MUSAE-Facebook", edge_path=..., features_path=..., target_path=...)
    """
    name = name.lower()
    if name in ["cora", "citeseer", "pubmed", "computers", "photo"]:
        return load_pyg_dataset(name, root)
    elif name == "deezer-europe":
        # log arguments for debugging
        print(f"Loading Deezer Europe with arguments: {kwargs}")
        return load_deezer_europe(**kwargs)
    elif name == "twitch-gamers":
        return load_twitch_gamers(**kwargs)
    elif name == "musae-facebook":
        return load_musae_facebook(**kwargs)
    elif name == "musae-github":
        return load_musae_github(**kwargs)
    elif name == "email-eu-core":
        return load_email_eu_core(**kwargs)
    elif name == "synthetic":
        return generate_synthetic_graph(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

def get_all_dataset_names():
    return [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Computers",
        "Photo",
        "Deezer-Europe",
        "Twitch-Gamers",
        "MUSAE-Facebook",
        "MUSAE-GitHub",
        "Email-EU-Core",
        "Synthetic"
    ]
