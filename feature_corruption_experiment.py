import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import numpy as np
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from utilities.dataloader import load_dataset

# ----------------------------
# Device & Repro
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Encoders (Node2Vec ONLY, as in Exp 1)
# ----------------------------
from encoders.structural_encoder import Node2VecEncoder

# ----------------------------
# Gates
# ----------------------------
from integrators.structural_integrator import (
    SimpleFeatureGate,
    AdaptiveGateWithSparsity,
    ResidualAdaptiveGate,
)

# ----------------------------
# Feature corruption
# ----------------------------
def corrupt_features_mask(x: torch.Tensor, p: float):
    """
    Randomly zero out each feature entry with probability p.
    """
    if p <= 0.0:
        return x
    mask = torch.rand_like(x) > p
    return x * mask.float()

# --------------------------------------------------------------------------
# Model (IDENTICAL to Experiment 1)
# --------------------------------------------------------------------------
class StructuralGCN(nn.Module):
    def __init__(self, structural_encoder, feature_dim, hidden_dim, num_classes, gate=None):
        super().__init__()
        self.structural_encoder = structural_encoder
        self.gate = gate

        self.conv1 = GCNConv(feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        aux_loss = None

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)

        if self.gate is not None and self.structural_encoder is not None:
            all_nodes = torch.arange(x.size(0), device=x.device)
            struct_emb = self.structural_encoder(all_nodes)

            if isinstance(self.gate, (AdaptiveGateWithSparsity, ResidualAdaptiveGate)):
                h, aux_loss = self.gate.integrate(h, struct_emb, edge_index, initial_features=h)
            else:
                h, aux_loss = self.gate.integrate(h, struct_emb, edge_index)

        logits = self.classifier(h)
        return logits, aux_loss

# --------------------------------------------------------------------------
# Builders (IDENTICAL to Experiment 1)
# --------------------------------------------------------------------------
def build_gate(fusion_type, hidden_dim, structural_dim):
    if fusion_type == "Standard":
        return None
    if fusion_type == "Simple":
        return SimpleFeatureGate(hidden_dim, structural_dim, hidden_dim)
    if fusion_type == "Adaptive":
        return AdaptiveGateWithSparsity(hidden_dim, structural_dim, hidden_dim, hidden_dim)
    if fusion_type == "AdaptiveResidual":
        return ResidualAdaptiveGate(hidden_dim, structural_dim, hidden_dim, hidden_dim, residual_scale_init=0.0)
    raise ValueError(f"Unknown fusion type: {fusion_type}")

def build_model(data, num_classes, fusion_type, node2vec_encoder):
    in_dim = data.x.size(1)
    hidden_dim = 64

    if fusion_type == "Standard":
        gate = None
    else:
        gate = build_gate(fusion_type, hidden_dim, node2vec_encoder.embedding_dimension)
        gate = gate.to(DEVICE)

    model = StructuralGCN(node2vec_encoder, in_dim, hidden_dim, num_classes, gate)
    return model.to(DEVICE)

# --------------------------------------------------------------------------
# Train / Eval (IDENTICAL)
# --------------------------------------------------------------------------
def train_and_eval(model, data, epochs=200, lr=0.01):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits, aux_loss = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

        if aux_loss is not None:
            loss = loss + aux_loss.mean()

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)
        acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    return acc

# --------------------------------------------------------------------------
# Main Experiment (ONLY ADD corruption loop)
# --------------------------------------------------------------------------
def main():
    datasets = ["Cora", "CiteSeer"]
    corrupt_levels = [0.0, 0.3, 0.5, 0.7]
    fusions = ["Standard", "Simple", "Adaptive", "AdaptiveResidual"]
    seeds = [0, 1, 2, 3, 4]

    records = []

    for dataset_name in datasets:
        print(f"\n================ Dataset: {dataset_name} ================")

        for seed in seeds:
            set_seed(seed)

            data, labels, _ = load_dataset(dataset_name)
            data = data.to(DEVICE)
            num_classes = int(labels.max().item() + 1)

            # ----------------------------
            # Train Node2Vec ONCE (same as Exp 1)
            # ----------------------------
            print("Training Node2Vec prior...")
            node2vec = Node2VecEncoder(data.num_nodes, data.edge_index, embedding_dim=64)
            node2vec = node2vec.to(DEVICE)
            node2vec.train_encoder(epochs=10, verbose=False)

            # Freeze it
            for p in node2vec.parameters():
                p.requires_grad = False
            node2vec.eval()

            for p_corrupt in corrupt_levels:
                print(f"\n--- Feature mask p = {p_corrupt} ---")

                # Corrupt features (ONLY CHANGE)
                data_corrupt = data.clone()
                data_corrupt.x = corrupt_features_mask(data_corrupt.x, p_corrupt)

                for fusion_type in fusions:
                    print(f"Seed={seed} Fusion={fusion_type}")

                    model = build_model(data_corrupt, num_classes, fusion_type, node2vec)
                    acc = train_and_eval(model, data_corrupt)

                    records.append({
                        "dataset": dataset_name,
                        "mask_p": p_corrupt,
                        "seed": seed,
                        "fusion": fusion_type,
                        "accuracy": acc,
                    })

    df = pd.DataFrame(records)
    df.to_csv("results/feature_corruption_raw.csv", index=False)

    print("\n================ RAW RESULTS ================")
    print(df)

    summary = (
        df.groupby(["dataset", "mask_p", "fusion"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv("results/feature_corruption_summary.csv", index=False)

    print("\n================ SUMMARY ================")
    print(summary)

if __name__ == "__main__":
    main()
