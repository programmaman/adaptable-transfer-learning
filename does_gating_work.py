import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import logging
import random
import numpy as np
from torch.optim import Adam
from torch_geometric.nn import GCNConv


"""
This script runs a controlled empirical study on how to integrate global structural information
with local message-passing in graph neural networks. The central goal is to understand:

    When should a GNN rely on global structural priors versus learned local features,
    and how should these two information sources be combined?

To answer this, we decompose the representation into two streams:
    (1) A local semantic stream learned by a shallow GCN.
    (2) A frozen global structural stream provided by a precomputed structural encoder
        (e.g., degree, Laplacian, or Node2Vec embeddings).

We then study different fusion strategies, with particular focus on adaptive gating mechanisms
that allow the model to arbitrarily interpolate between structure and features on a per-node basis.

The experiments are organized into four conceptual blocks:

------------------------------------------------------------
Experiment 1 — Do frozen structural bases help at all?
------------------------------------------------------------
We compare a plain GCN against models augmented with different frozen structural encoders:
    - None (no structure, baseline)
    - Degree (local structural signal)
    - Laplacian (spectral / diffusion-based global signal)
    - Node2Vec (random-walk-based global signal)
    - Random (control)

This experiment isolates whether decoupled, frozen global structure provides any benefit
over pure message passing.

------------------------------------------------------------
Experiment 2 — Is adaptive gating better than naive fusion?
------------------------------------------------------------
For each structural encoder, we compare multiple fusion mechanisms:
    - No fusion (baseline)
    - Simple fusion (e.g., concatenation / linear mixing)
    - Adaptive gating (our main mechanism)

This experiment tests whether *how* structure is injected matters, and whether allowing the model
to adaptively control reliance on structure vs features is superior to fixed fusion.

------------------------------------------------------------
Experiment 3 — When does structure matter? (Synthetic graphs)
------------------------------------------------------------
Using controlled synthetic graph families (random, SBM, role-based graphs), we evaluate:
    - Which graph regimes benefit from global structural information
    - Which regimes prefer local message passing
    - How different fusion strategies behave under varying structural conditions

This experiment is designed to reveal the causal role of global structure and to study
structure–feature arbitration under controlled conditions.

------------------------------------------------------------
Experiment 4 — Full ablation over structure × fusion choices
------------------------------------------------------------
We run the full grid over:
    - Structural encoders × fusion mechanisms × graph types × random seeds

and report mean ± std performance. This serves to:
    - Isolate the contribution of the structural stream
    - Isolate the contribution of the fusion mechanism
    - Verify that gains are not due to a specific encoder or dataset artifact

------------------------------------------------------------
Datasets
------------------------------------------------------------
We evaluate on:
    - Synthetic graphs: random, SBM, and role-based graphs (for controlled analysis)
    - Real benchmarks: Cora, Amazon-Computers, and MUSAE-Facebook

------------------------------------------------------------
Key Scientific Questions Addressed
------------------------------------------------------------
1. Does decoupled, frozen global structure help GNNs beyond local message passing?
2. Is adaptive fusion strictly better than naive structural feature injection?
3. On what types of graphs does the model rely more on structure vs features?
4. Is Node2Vec special, or do other structural bases work similarly?

Overall, this script implements a principled study of structural–semantic decoupling and
adaptive arbitration in graph neural networks, rather than a benchmark-driven model comparison.
"""


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
# Encoders
# ----------------------------
from encoders.structural_encoder import (
    Node2VecEncoder,
    RandomStructuralEncoder,
    LaplacianStructuralEncoder,
    DegreeStructuralEncoder,
)

# ----------------------------
# Gates
# ----------------------------
from integrators.structural_integrator import (
    SimpleFeatureGate,
    SelfSupervisedGate,
    AdaptiveGateWithSparsity,
)

# ----------------------------
# Synthetic graphs
# ----------------------------
from utilities.experiment_utils import (
    generate_synthetic_graph,
    generate_synthetic_graph_with_structure,
    generate_role_graph,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("GatingExp")

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class StructuralGCN(nn.Module):
    def __init__(self, structural_encoder, feature_dim, hidden_dim, num_classes, gate=None):
        super().__init__()
        self.structural_encoder = structural_encoder
        self.gate = gate

        # Single GCN layer
        self.conv1 = GCNConv(feature_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.cached_struct_emb = None

    def forward(self, x, edge_index):
        aux_loss = None

        # ----------------
        # 1-layer GNN
        # ----------------
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        # Save pre-gate features for residual / initial-feature gates
        h0 = h.clone()

        # ----------------
        # Structural fusion
        # ----------------
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

        # ----------------
        # Classifier
        # ----------------
        logits = self.classifier(h)
        return logits, aux_loss


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

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

    # --------- CRITICAL: freeze encoder ---------
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
        return AdaptiveGateWithSparsity(
            feature_dimension=hidden_dim,
            structural_dimension=structural_dim,
            calculation_dimension=hidden_dim,
            initial_feature_dimension=hidden_dim,
        )

    if fusion_type == "AdaptiveResidual":
        return ResidualAdaptiveGate(
            feature_dimension=hidden_dim,
            structural_dimension=structural_dim,
            calculation_dimension=hidden_dim,
            initial_feature_dimension=hidden_dim,
            residual_scale_init=0.0,
        )

    if fusion_type == "Combined":
        return CombinedAdaptiveSelfSupervisedGate(
            feature_dimension=hidden_dim,
            structural_dimension=structural_dim,
            calculation_dimension=hidden_dim,
            initial_feature_dimension=hidden_dim,
        )

    if fusion_type == "AdaptiveGatingWithSparsity":
        return AdaptiveGateWithSparsity(
            feature_dimension=hidden_dim,
            structural_dimension=structural_dim,
            calculation_dimension=hidden_dim,
            initial_feature_dimension=hidden_dim,
        )

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

# --------------------------------------------------------------------------
# Train / Eval
# --------------------------------------------------------------------------

def train_and_eval(model, data, epochs=150, lr=0.01):
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

# --------------------------------------------------------------------------
# Dataset factory
# --------------------------------------------------------------------------

def load_experiment_dataset(name):
    name = name.lower()

    # -----------------------
    # Synthetic datasets
    # -----------------------
    if name == "random":
        data, labels, _ = load_dataset("synthetic")
    elif name == "sbm":
        data, labels = generate_synthetic_graph_with_structure()
    elif name == "role":
        data, labels = generate_role_graph()

    # -----------------------
    # Real datasets
    # -----------------------
    elif name in ["cora", "computers", "musae-facebook"]:
        if name == "musae-facebook":
            data, labels, _ = load_dataset(
                "musae-facebook",
                edge="./datasets/facebook_large/musae_facebook_edges.csv",
                features="./datasets/facebook_large/musae_facebook_features.json",
                target="./datasets/facebook_large/musae_facebook_target.csv",
            )
        else:
            data, labels, _ = load_dataset(name)

    else:
        raise ValueError(f"Unknown dataset: {name}")


    data.y = labels

    # For synthetic graphs: create random split
    if name in ["random", "sbm", "role"]:
        N = data.num_nodes
        perm = torch.randperm(N)
        split = int(0.8 * N)
        data.train_mask = torch.zeros(N, dtype=torch.bool)
        data.test_mask = torch.zeros(N, dtype=torch.bool)
        data.train_mask[perm[:split]] = True
        data.test_mask[perm[split:]] = True

    # For real datasets: masks already exist, do nothing
    return data, labels


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def run_experiment():
    logger.info("Running FULL gating experiment (synthetic + real)")

    rows = []

    # ============================================================
    # Part 1: Synthetic experiments (full grid, as before)
    # ============================================================
    graph_types = ["random", "sbm", "role"]
    encoder_types = ["none", "random", "laplacian", "degree", "node2vec"]
    fusion_types = ["Standard", "Simple", "SSL", "Adaptive", "AdaptiveResidual", "Combined", "AdaptiveGatingWithSparsity"]

    for graph_name in graph_types:
        for run in range(3):
            seed = 42 + run
            set_seed(seed)

            logger.info(f"\n[SYNTH] Graph={graph_name} | Run={run+1} | Seed={seed}")

            data, labels = load_experiment_dataset(graph_name)
            data = data.to(DEVICE)
            num_classes = labels.max().item() + 1

            for encoder_type in encoder_types:
                for fusion_type in fusion_types:

                    # Skip meaningless combos
                    if fusion_type != "Standard" and encoder_type == "none":
                        continue
                    if fusion_type == "Standard" and encoder_type != "none":
                        continue

                    logger.info(f"  [SYNTH] Encoder={encoder_type} | Fusion={fusion_type}")

                    model = build_model(data, num_classes, encoder_type, fusion_type)
                    acc = train_and_eval(model, data, epochs=150)

                    rows.append({
                        "domain": "synthetic",
                        "graph": graph_name,
                        "run": run + 1,
                        "encoder": encoder_type,
                        "fusion": fusion_type,
                        "accuracy": acc,
                    })

    # ============================================================
    # Part 2: Real experiments (small, focused grid)
    # ============================================================
    real_datasets = ["cora", "computers", "musae-facebook"]
    real_encoders = ["none", "degree", "laplacian"]
    real_fusions = ["Standard", "Simple", "Adaptive", "SSL"]

    for graph_name in real_datasets:
        for run in range(3):
            seed = 100 + run
            set_seed(seed)

            logger.info(f"\n[REAL] Dataset={graph_name} | Run={run+1} | Seed={seed}")

            data, labels = load_experiment_dataset(graph_name)
            data = data.to(DEVICE)

            # Some datasets may have -1 labels (unlabeled nodes)
            valid = labels >= 0
            num_classes = labels[valid].max().item() + 1

            for encoder_type in real_encoders:
                for fusion_type in real_fusions:

                    # Skip meaningless combos
                    if fusion_type != "Standard" and encoder_type == "none":
                        continue
                    if fusion_type == "Standard" and encoder_type != "none":
                        continue

                    logger.info(f"  [REAL] Encoder={encoder_type} | Fusion={fusion_type}")

                    model = build_model(data, num_classes, encoder_type, fusion_type)
                    acc = train_and_eval(model, data, epochs=200)  # a bit longer for real data

                    rows.append({
                        "domain": "real",
                        "graph": graph_name,
                        "run": run + 1,
                        "encoder": encoder_type,
                        "fusion": fusion_type,
                        "accuracy": acc,
                    })

    # ============================================================
    # Save results
    # ============================================================
    df = pd.DataFrame(rows)
    df.to_csv("results/gating_results_all.csv", index=False)

    print("\n" + "=" * 80)
    print("RAW RESULTS")
    print(df)
    print("=" * 80)

    summary = df.groupby(["domain", "graph", "encoder", "fusion"])["accuracy"].agg(["mean", "std"]).reset_index()
    summary.to_csv("results/gating_results_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("SUMMARY (mean ± std)")
    print(summary)
    print("=" * 80)



if __name__ == "__main__":
    run_experiment()
