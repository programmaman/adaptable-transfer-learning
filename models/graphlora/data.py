# gnn_pipeline/data.py
import os
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.utils import add_remaining_self_loops


def load_dataset(name: str, path: str = "./datasets"):
    """Load a known PyG dataset (Planetoid or Amazon)."""
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo']
    transform = T.NormalizeFeatures()
    if name in ['Computers', 'Photo']:
        return Amazon(path, name, transform)
    return Planetoid(path, name, transform)


def apply_feature_reduction(dataset, out_channels=100):
    """Optionally apply SVD-based feature reduction."""
    reducer = SVDFeatureReduction(out_channels=out_channels)
    return reducer(dataset)


def ensure_self_loops(dataset):
    """Guarantees self-loops are present in edge_index."""
    dataset.edge_index = add_remaining_self_loops(dataset.edge_index)[0]
    return dataset
