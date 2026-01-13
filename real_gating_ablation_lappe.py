import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import logging
import random
import numpy as np
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import AddLaplacianEigenvectorPE

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
# Gates
# ----------------------------
from integrators.structural_integrator import (
    SimpleFeatureGate,
    SelfSupervisedGate,
    AdaptiveGateWithSparsity,
    ResidualAdaptiveGate,
    CombinedAdaptiveSelfSupervisedGate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("RealGatingAblation_LapPE")

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class StructuralGCN(nn.Module):
    def __init__(self, structural_features, feature_dim, hidden_dim, num_classes, gate=None):
        super().__init__()

        # Register as buffer so it moves with .to(device)
        self.register_buffer("structural_features", structural_features)

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

        if self.gate is not None:
            # Ensure same device as h
            struct_emb = self.structural_features.to(h.device)

            if isinstance(self.gate, (AdaptiveGateWithSparsity, ResidualAdaptiveGate, CombinedAdaptiveSelfSupervisedGate)):
                h, aux_loss = self.gate.integrate(h, struct_emb, edge_index, initial_features=h)
            else:
                h, aux_loss = self.gate.integrate(h, struct_emb, edge_index)

        logits = self.classifier(h)
        return logits, aux_loss


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def build_gate(fusion_type, hidden_dim, structural_dim):
    if fusion_type == "Standard":
        return None

    if fusion_type == "Simple":
        return SimpleFeatureGate(hidden_dim, structural_dim, hidden_dim)

    if fusion_type == "SSL":
        return SelfSupervisedGate(hidden_dim, structural_dim, hidden_dim)

    if fusion_type == "Adaptive":
        return AdaptiveGateWithSparsity(hidden_dim, structural_dim, hidden_dim, hidden_dim)

    if fusion_type == "AdaptiveResidual":
        return ResidualAdaptiveGate(hidden_dim, structural_dim, hidden_dim, hidden_dim, residual_scale_init=0.0)

    if fusion_type == "Combined":
        return CombinedAdaptiveSelfSupervisedGate(hidden_dim, structural_dim, hidden_dim, hidden_dim)

    raise ValueError(f"Unknown fusion type: {fusion_type}")

def build_model(data, num_classes, fusion_type):
    in_dim = data.x.size(1)
    hidden_dim = 64

    # Structural prior = Laplacian PE (already computed & frozen)
    struct_feat = data.lap_pe
    structural_dim = struct_feat.size(1)

    if fusion_type == "Standard":
        gate = None
    else:
        gate = build_gate(fusion_type, hidden_dim, structural_dim)
        gate = gate.to(DEVICE)

    model = StructuralGCN(struct_feat, in_dim, hidden_dim, num_classes, gate)
    return model.to(DEVICE)

# --------------------------------------------------------------------------
# Train / Eval
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
# Main Experiment
# --------------------------------------------------------------------------

def main():
    datasets = ["Cora", "CiteSeer", "PubMed", "Computers", "Photo"]
    fusions = ["Standard", "Simple", "SSL", "Adaptive", "AdaptiveResidual"]
    seeds = [0, 1, 2, 3, 4]

    records = []

    for dataset_name in datasets:
        logger.info(f"\n================ Dataset: {dataset_name} ================")

        for seed in seeds:
            set_seed(seed)

            data, labels, _ = load_dataset(dataset_name)
            data = data.to(DEVICE)
            num_classes = int(labels.max().item() + 1)

            # ----------------------------
            # Build Laplacian PE prior (ONCE per seed & dataset)
            # ----------------------------
            logger.info("Computing Laplacian positional encodings...")
            pe_dim = 8
            transform = AddLaplacianEigenvectorPE(k=pe_dim, attr_name="lap_pe")
            data = transform(data)

            # Freeze it (treat as prior)
            data.lap_pe = data.lap_pe.detach()

            for fusion_type in fusions:
                logger.info(f"Seed={seed} | Fusion={fusion_type}")

                model = build_model(data, num_classes, fusion_type)
                acc = train_and_eval(model, data)

                records.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "fusion": fusion_type,
                    "accuracy": acc,
                })

    df = pd.DataFrame(records)
    df.to_csv("real_gating_lappe_raw_results.csv", index=False)

    print("\n================ RAW RESULTS ================")
    print(df)

    summary = df.groupby(["dataset", "fusion"])["accuracy"].agg(["mean", "std"]).reset_index()
    summary.to_csv("results/real_gating_lappe_summary.csv", index=False)

    print("\n================ SUMMARY ================")
    print(summary)

if __name__ == "__main__":
    main()
