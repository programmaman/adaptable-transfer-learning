import os
import logging
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from utilities.dataloader import load_dataset

# Encoders
from encoders.structural_encoder import Node2VecEncoder

# Gates
from integrators.structural_integrator import ResidualAdaptiveGate

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
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("CrossGraphTransfer")

# ----------------------------
# Feature hashing projection
# ----------------------------
def hashed_feature_projection(x: torch.Tensor, out_dim: int, hash_seed: int) -> torch.Tensor:
    """
    Project features of arbitrary input dimension -> fixed out_dim using a deterministic hashing trick.
    This avoids Cora (1433) vs CiteSeer (3703) dimension mismatch while keeping transfer meaningful.

    Implementation:
        y = (R^T @ x^T)^T where R^T is sparse [out_dim, in_dim] with exactly one +/-1 per column.
    """
    assert x.dim() == 2, "x must be [N, F]"
    n, in_dim = x.shape
    device = x.device

    g = torch.Generator(device="cpu")
    g.manual_seed(hash_seed)

    buckets = torch.randint(low=0, high=out_dim, size=(in_dim,), generator=g, dtype=torch.long)  # [F]
    signs = torch.randint(low=0, high=2, size=(in_dim,), generator=g, dtype=torch.long) * 2 - 1  # {-1,+1}

    # Build sparse R^T of shape [out_dim, in_dim]
    # indices: [2, nnz] where rows=buckets, cols=feature_index
    rows = buckets
    cols = torch.arange(in_dim, dtype=torch.long)
    indices = torch.stack([rows, cols], dim=0)
    values = signs.to(torch.float32)

    Rt = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(out_dim, in_dim),
        dtype=torch.float32,
        device=device,
    ).coalesce()

    # y^T = Rt @ x^T  => y = (Rt @ x^T)^T
    y_t = torch.sparse.mm(Rt, x.t())  # [out_dim, N]
    y = y_t.t().contiguous()          # [N, out_dim]
    return y

# ----------------------------
# Model
# ----------------------------
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

            # ResidualAdaptiveGate expects initial_features
            h, aux_loss = self.gate.integrate(h, struct_emb, edge_index, initial_features=h)

        logits = self.classifier(h)
        return logits, aux_loss

# ----------------------------
# Builders
# ----------------------------
def build_standard_model(input_dim: int, num_classes: int, hidden_dim: int = 64) -> nn.Module:
    return StructuralGCN(
        structural_encoder=None,
        feature_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        gate=None,
    ).to(DEVICE)

def build_adaptive_residual_model(input_dim: int, num_classes: int, structural_dim: int, hidden_dim: int = 64) -> nn.Module:
    gate = ResidualAdaptiveGate(
        feature_dimension=hidden_dim,
        structural_dimension=structural_dim,
        calculation_dimension=hidden_dim,
        initial_feature_dimension=hidden_dim,
        residual_scale_init=0.0,
    ).to(DEVICE)

    # structural_encoder is injected/updated per-graph (source vs target)
    return StructuralGCN(
        structural_encoder=None,
        feature_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        gate=gate,
    ).to(DEVICE)

# ----------------------------
# Train / Eval
# ----------------------------
def train_on_source(model: nn.Module, data, epochs: int = 200, lr: float = 0.01) -> nn.Module:
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

    return model

@torch.no_grad()
def eval_on_target(model: nn.Module, data) -> float:
    model.eval()
    logits, _ = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1)
    acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return acc

def train_node2vec_prior(data, seed: int, embedding_dim: int = 64, epochs: int = 10) -> Node2VecEncoder:
    """
    Train Node2Vec unsupervised on a graph, then freeze it.
    """
    node2vec = Node2VecEncoder(data.num_nodes, data.edge_index, embedding_dim=embedding_dim)
    node2vec = node2vec.to(DEVICE)
    node2vec.train_encoder(epochs=epochs, verbose=False)

    for p in node2vec.parameters():
        p.requires_grad = False
    node2vec.eval()
    return node2vec

# ----------------------------
# Experiment 3
# ----------------------------
def main():
    os.makedirs("results", exist_ok=True)

    source_name = "Cora"
    target_name = "CiteSeer"

    # Transfer-safe shared feature dimension
    PROJ_DIM = 256

    # Keep projection fixed across all seeds (so transfer is consistent)
    PROJ_HASH_SEED = 1337

    seeds = [0, 1, 2, 3, 4]
    methods = ["Standard", "AdaptiveResidual"]

    records = []

    for seed in seeds:
        set_seed(seed)
        logger.info(f"\n================ Seed: {seed} | {source_name} → {target_name} ================")

        # ---- Load source
        src_data, _, _ = load_dataset(source_name)
        src_data = src_data.to("cpu")  # projection on CPU (safe), then move to GPU
        src_data.x = hashed_feature_projection(src_data.x, out_dim=PROJ_DIM, hash_seed=PROJ_HASH_SEED)
        src_data = src_data.to(DEVICE)

        # ---- Load target
        tgt_data, _, _ = load_dataset(target_name)
        tgt_data = tgt_data.to("cpu")
        tgt_data.x = hashed_feature_projection(tgt_data.x, out_dim=PROJ_DIM, hash_seed=PROJ_HASH_SEED)
        tgt_data = tgt_data.to(DEVICE)

        # ---- Class count: use source classes for training head
        # (Cora=7, CiteSeer=6). For transfer, we must pick a consistent output space.
        # The cleanest thing is: run transfer on datasets with same label-space OR restrict to shared tasks.
        #
        # However your stated goal is "Cora → CiteSeer" to show feature overfitting hurts.
        # To keep the experiment runnable, we evaluate using the *target labels* by mapping the classifier to target classes.
        #
        # Practical approach: train a backbone on source, then re-init the classifier for target (no training),
        # which is NOT meaningful. So instead: do a representation-quality transfer metric:
        # - train on source
        # - evaluate target with a linear probe trained ONLY on target train_mask (fast)
        #
        # That matches typical transfer protocols and avoids label-space mismatch.
        src_num_classes = int(src_data.y.max().item() + 1)
        tgt_num_classes = int(tgt_data.y.max().item() + 1)

        logger.info(f"Source classes={src_num_classes} | Target classes={tgt_num_classes} | proj_dim={PROJ_DIM}")

        # ---- Train Node2Vec priors per-graph (unsupervised)
        logger.info("Training Node2Vec prior on SOURCE graph...")
        src_node2vec = train_node2vec_prior(src_data, seed=seed, embedding_dim=64, epochs=10)

        logger.info("Training Node2Vec prior on TARGET graph...")
        tgt_node2vec = train_node2vec_prior(tgt_data, seed=seed, embedding_dim=64, epochs=10)

        # ---- Train + Evaluate (with linear probe on target)
        for method in methods:
            logger.info(f"[Seed {seed}] Method={method}")

            if method == "Standard":
                # Train backbone+classifier on source label space
                model = build_standard_model(input_dim=PROJ_DIM, num_classes=src_num_classes)
                model = train_on_source(model, src_data, epochs=200, lr=0.01)

                # Transfer: use backbone weights, train linear probe on TARGET train_mask
                backbone = nn.Sequential(model.conv1, model.conv2).to(DEVICE)
                probe = nn.Linear(64, tgt_num_classes).to(DEVICE)

                opt = Adam(probe.parameters(), lr=0.01, weight_decay=5e-4)
                for _ in range(200):
                    backbone.train()
                    probe.train()
                    opt.zero_grad()

                    h = backbone[0](tgt_data.x, tgt_data.edge_index)
                    h = F.relu(h)
                    h = F.dropout(h, p=0.5, training=True)
                    h = backbone[1](h, tgt_data.edge_index)

                    logits = probe(h)
                    loss = F.cross_entropy(logits[tgt_data.train_mask], tgt_data.y[tgt_data.train_mask])
                    loss.backward()
                    opt.step()

                probe.eval()
                with torch.no_grad():
                    h = backbone[0](tgt_data.x, tgt_data.edge_index)
                    h = F.relu(h)
                    h = backbone[1](h, tgt_data.edge_index)
                    logits = probe(h)
                    preds = logits.argmax(dim=1)
                    acc = (preds[tgt_data.test_mask] == tgt_data.y[tgt_data.test_mask]).float().mean().item()

            elif method == "AdaptiveResidual":
                # Train on source with SOURCE prior
                model = build_adaptive_residual_model(
                    input_dim=PROJ_DIM,
                    num_classes=src_num_classes,
                    structural_dim=src_node2vec.embedding_dimension,
                )
                model.structural_encoder = src_node2vec
                model = train_on_source(model, src_data, epochs=200, lr=0.01)

                # Swap in TARGET prior for evaluation/probing
                model.structural_encoder = tgt_node2vec

                # Same transfer protocol: linear probe on target embeddings
                # We reuse the trained conv weights + gate behavior; probe is trained on target train_mask.
                backbone_model = model  # contains convs + gate
                probe = nn.Linear(64, tgt_num_classes).to(DEVICE)

                opt = Adam(probe.parameters(), lr=0.01, weight_decay=5e-4)
                for _ in range(200):
                    backbone_model.train()
                    probe.train()
                    opt.zero_grad()

                    # get embeddings from convs (+ gate) but bypass classifier
                    h1 = backbone_model.conv1(tgt_data.x, tgt_data.edge_index)
                    h1 = F.relu(h1)
                    h1 = F.dropout(h1, p=0.5, training=True)
                    h = backbone_model.conv2(h1, tgt_data.edge_index)

                    # gate integrate
                    all_nodes = torch.arange(tgt_data.x.size(0), device=DEVICE)
                    struct_emb = backbone_model.structural_encoder(all_nodes)
                    h, aux_loss = backbone_model.gate.integrate(h, struct_emb, tgt_data.edge_index, initial_features=h)

                    logits = probe(h)
                    loss = F.cross_entropy(logits[tgt_data.train_mask], tgt_data.y[tgt_data.train_mask])
                    if aux_loss is not None:
                        loss = loss + aux_loss.mean()

                    loss.backward()
                    opt.step()

                probe.eval()
                backbone_model.eval()
                with torch.no_grad():
                    h1 = backbone_model.conv1(tgt_data.x, tgt_data.edge_index)
                    h1 = F.relu(h1)
                    h = backbone_model.conv2(h1, tgt_data.edge_index)

                    all_nodes = torch.arange(tgt_data.x.size(0), device=DEVICE)
                    struct_emb = backbone_model.structural_encoder(all_nodes)
                    h, _ = backbone_model.gate.integrate(h, struct_emb, tgt_data.edge_index, initial_features=h)

                    logits = probe(h)
                    preds = logits.argmax(dim=1)
                    acc = (preds[tgt_data.test_mask] == tgt_data.y[tgt_data.test_mask]).float().mean().item()

            else:
                raise ValueError(method)

            records.append({
                "source": source_name,
                "target": target_name,
                "seed": seed,
                "method": method,
                "accuracy": acc,
            })

            logger.info(f"[Seed {seed}] {method} transfer accuracy: {acc:.4f}")

    df = pd.DataFrame(records)
    raw_path = "results/cross_graph_transfer_raw.csv"
    df.to_csv(raw_path, index=False)

    summary = df.groupby(["source", "target", "method"])["accuracy"].agg(["mean", "std"]).reset_index()
    summary_path = "results/cross_graph_transfer_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n================ RAW RESULTS ================")
    print(df)
    print("\n================ SUMMARY (mean ± std) ================")
    print(summary)
    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")

if __name__ == "__main__":
    main()
