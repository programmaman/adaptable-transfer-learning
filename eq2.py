import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import logging
import random
import numpy as np
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from utilities.dataloader import load_dataset
from encoders.structural_encoder import (
    Node2VecEncoder,
    RandomStructuralEncoder,
    LaplacianStructuralEncoder,
    DegreeStructuralEncoder,
)
from integrators.structural_integrator import (
    SimpleFeatureGate,
    SelfSupervisedGate,
    AdaptiveGateWithSparsity,
    CombinedAdaptiveSelfSupervisedGate,
    ResidualAdaptiveGate,
)

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
# Model
# ----------------------------

class StructuralGCN(nn.Module):
    def __init__(self, structural_encoder, feature_dim, hidden_dim, num_classes, gate=None):
        super().__init__()
        self.structural_encoder = structural_encoder
        self.gate = gate

        self.conv1 = GCNConv(feature_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.cached_struct_emb = None

    def forward(self, x, edge_index):
        aux_loss = None
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h0 = h.clone()

        if self.gate is not None and self.structural_encoder is not None:
            if self.cached_struct_emb is None:
                all_nodes = torch.arange(x.size(0), device=x.device)
                with torch.no_grad():
                    self.cached_struct_emb = self.structural_encoder(all_nodes).detach()
            struct_emb = self.cached_struct_emb

            if isinstance(self.gate, (AdaptiveGateWithSparsity, ResidualAdaptiveGate, CombinedAdaptiveSelfSupervisedGate)):
                h, aux_loss = self.gate.integrate(h, struct_emb, edge_index, initial_features=h0)
            else:
                h, aux_loss = self.gate.integrate(h, struct_emb, edge_index)

        logits = self.classifier(h)
        return logits, aux_loss

# ----------------------------
# Builders
# ----------------------------

def build_encoder(encoder_type, data):
    if encoder_type == "none":
        return None
    if encoder_type == "random":
        enc = RandomStructuralEncoder(data.num_nodes, 64)
    elif encoder_type == "laplacian":
        enc = LaplacianStructuralEncoder(data.edge_index, data.num_nodes, dim=16)
    elif encoder_type == "degree":
        enc = DegreeStructuralEncoder(data.edge_index, data.num_nodes)
    elif encoder_type == "node2vec":
        enc = Node2VecEncoder(data.num_nodes, data.edge_index, embedding_dim=64)
        enc = enc.to(DEVICE)
        enc.train_encoder(epochs=10, verbose=False)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    enc = enc.to(DEVICE)
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()
    return enc

def infer_structural_dim(encoder, num_nodes):
    with torch.no_grad():
        nodes = torch.arange(num_nodes, device=DEVICE)
        z = encoder(nodes)
    return z.size(1)

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
    if fusion_type == "AdaptiveGatingWithSparsity":
        return AdaptiveGateWithSparsity(hidden_dim, structural_dim, hidden_dim, hidden_dim)
    raise ValueError(f"Unknown fusion type: {fusion_type}")

def build_model(data, num_classes, encoder_type, fusion_type):
    raw_dim = data.x.size(1)
    hidden_dim = 64
    encoder = build_encoder(encoder_type, data)
    if fusion_type == "Standard" or encoder is None:
        gate = None
    else:
        structural_dim = infer_structural_dim(encoder, data.num_nodes)
        gate = build_gate(fusion_type, hidden_dim, structural_dim)
        gate = gate.to(DEVICE)
    model = StructuralGCN(encoder, raw_dim, hidden_dim, num_classes, gate)
    return model.to(DEVICE)

# ----------------------------
# Train / Eval
# ----------------------------

def train_and_eval(model, data, epochs=200, lr=0.01):
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, aux_loss = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        if aux_loss is not None:
            loss = loss + 0.1 * aux_loss.mean()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)
        acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return acc

# ----------------------------
# Dataset loader
# ----------------------------

def load_experiment_dataset(name):
    if name == "musae-facebook":
        data, labels, _ = load_dataset(
            "musae-facebook",
            edge="./datasets/facebook_large/musae_facebook_edges.csv",
            features="./datasets/facebook_large/musae_facebook_features.json",
            target="./datasets/facebook_large/musae_facebook_target.csv",
        )
    else:
        data, labels, _ = load_dataset(name)
    data.y = labels
    return data, labels

# ----------------------------
# Experiment Runner (Real Graphs)
# ----------------------------

def run_experiment_eq2():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger("GatingExp-EQ2")

    logger.info("Running EQ2: Gating performance on real datasets")

    real_datasets = ["cora", "computers", "musae-facebook"]
    encoder_types = ["none", "degree", "laplacian"]
    fusion_types = ["Standard", "Simple", "Adaptive", "SSL"]
    rows = []

    for graph_name in real_datasets:
        for run in range(3):
            seed = 100 + run
            set_seed(seed)

            logger.info(f"\n[REAL] Dataset={graph_name} | Run={run+1} | Seed={seed}")

            data, labels = load_experiment_dataset(graph_name)
            data = data.to(DEVICE)

            valid = labels >= 0
            num_classes = labels[valid].max().item() + 1

            for encoder_type in encoder_types:
                for fusion_type in fusion_types:

                    if fusion_type != "Standard" and encoder_type == "none":
                        continue
                    if fusion_type == "Standard" and encoder_type != "none":
                        continue

                    logger.info(f"  [REAL] Encoder={encoder_type} | Fusion={fusion_type}")

                    model = build_model(data, num_classes, encoder_type, fusion_type)
                    acc = train_and_eval(model, data, epochs=200)

                    rows.append({
                        "domain": "real",
                        "graph": graph_name,
                        "run": run + 1,
                        "encoder": encoder_type,
                        "fusion": fusion_type,
                        "accuracy": acc,
                    })

    df = pd.DataFrame(rows)
    df.to_csv("results/gating_eq2_real.csv", index=False)

    print("\n" + "=" * 80)
    print("RAW RESULTS (EQ2)")
    print(df)
    print("=" * 80)

    summary = df.groupby(["domain", "graph", "encoder", "fusion"])["accuracy"].agg(["mean", "std"]).reset_index()
    summary.to_csv("results/gating_eq2_real_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("SUMMARY (mean ± std, EQ2)")
    print(summary)
    print("=" * 80)

if __name__ == "__main__":
    run_experiment_eq2()
