import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import numpy as np
from torch.optim import Adam
from torch_geometric.nn import GCNConv
import logging

from utilities.dataloader import load_dataset
from encoders.structural_encoder import (
    Node2VecEncoder, RandomStructuralEncoder,
    LaplacianStructuralEncoder, DegreeStructuralEncoder
)
from integrators.structural_integrator import (
    SimpleFeatureGate,
    SelfSupervisedGate,
    AdaptiveGate,
    AdaptiveGateWithSparsity,
    DisagreementAwareAdaptiveGate,
    JumpingKnowledgeGate,          # <<< NEW (literature baseline)
)

from utilities.experiment_utils import (
    generate_synthetic_graph,
    generate_synthetic_graph_with_structure,
    generate_role_graph
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("Experiment1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Gate Registry (SINGLE SOURCE OF TRUTH)  <<< THIS DEFINES EQ.1
# ============================================================

class GateType:
    BASELINE_NO_GATE = "BASELINE_NO_GATE"

    # Literature baselines
    JUMPING_KNOWLEDGE_GATE = "JUMPING_KNOWLEDGE_GATE"
    SIMPLE_FEATURE_GATE = "SIMPLE_FEATURE_GATE"
    SELF_SUPERVISED_GATE = "SELF_SUPERVISED_GATE"

    # AG-GNN family
    AGGNN_BASE_GATE = "AGGNN_BASE_GATE"
    AGGNN_SPARSE_GATE = "AGGNN_SPARSE_GATE"

    # Your Eq.1 contribution
    AGGNN_DISAGREE_GATE = "AGGNN_DISAGREE_GATE"


ALL_GATE_TYPES = [
    GateType.BASELINE_NO_GATE,

    # Literature
    GateType.JUMPING_KNOWLEDGE_GATE,
    GateType.SIMPLE_FEATURE_GATE,
    GateType.SELF_SUPERVISED_GATE,

    # AG-GNN
    GateType.AGGNN_BASE_GATE,
    GateType.AGGNN_SPARSE_GATE,

    # Your method
    GateType.AGGNN_DISAGREE_GATE,
]


# ============================================================
# Model
# ============================================================

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

        # h0 = input to GCN (for JK)
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        # h0 = pre-gate hidden representation (for JK and AGGNN)
        h0 = h.clone()

        if self.gate is not None:

            # Gates that do NOT use structure (JK, etc.)
            if isinstance(self.gate, JumpingKnowledgeGate):
                h, aux_loss = self.gate.integrate(h, None, edge_index, initial_features=h0)

            else:
                # Structure-dependent gates
                if self.structural_encoder is None:
                    raise RuntimeError("Structure-dependent gate used without structural encoder.")

                if self.cached_struct_emb is None:
                    all_nodes = torch.arange(x.size(0), device=x.device)
                    with torch.no_grad():
                        self.cached_struct_emb = self.structural_encoder(all_nodes).detach()

                struct_emb = self.cached_struct_emb

                if isinstance(
                    self.gate,
                    (
                        AdaptiveGateWithSparsity,
                        DisagreementAwareAdaptiveGate,
                        AdaptiveGate,
                    ),
                ):
                    h, aux_loss = self.gate.integrate(h, struct_emb, edge_index, initial_features=h0)
                else:
                    h, aux_loss = self.gate.integrate(h, struct_emb, edge_index)

        logits = self.classifier(h)
        return logits, aux_loss


# ============================================================
# Builders
# ============================================================

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
        enc = Node2VecEncoder(data.num_nodes, data.edge_index, embedding_dim=64).to(device)
        enc.train_encoder(epochs=10, verbose=False)
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")

    enc = enc.to(device)
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()
    return enc


def infer_structural_dim(encoder, num_nodes):
    with torch.no_grad():
        z = encoder(torch.arange(num_nodes, device=device))
    return z.size(1)


def build_gate(gate_type, hidden_dim, structural_dim):
    if gate_type == GateType.BASELINE_NO_GATE:
        return None

    # -------- Literature --------
    if gate_type == GateType.JUMPING_KNOWLEDGE_GATE:
        return JumpingKnowledgeGate(hidden_dim, structural_dim, hidden_dim)

    if gate_type == GateType.SIMPLE_FEATURE_GATE:
        return SimpleFeatureGate(hidden_dim, structural_dim, hidden_dim)

    if gate_type == GateType.SELF_SUPERVISED_GATE:
        return SelfSupervisedGate(hidden_dim, structural_dim, hidden_dim)

    # -------- AG-GNN family --------
    if gate_type == GateType.AGGNN_BASE_GATE:
        return AdaptiveGate(hidden_dim, structural_dim, hidden_dim, hidden_dim)

    if gate_type == GateType.AGGNN_SPARSE_GATE:
        return AdaptiveGateWithSparsity(hidden_dim, structural_dim, hidden_dim, hidden_dim)

    if gate_type == GateType.AGGNN_DISAGREE_GATE:
        return DisagreementAwareAdaptiveGate(hidden_dim, structural_dim, hidden_dim, hidden_dim)

    raise ValueError(f"Unknown gate type: {gate_type}")


def build_model(data, num_classes, encoder_type, gate_type):
    raw_dim = data.x.size(1)
    hidden_dim = 64

    encoder = build_encoder(encoder_type, data)

    # JK does NOT need structural encoder
    if gate_type == GateType.JUMPING_KNOWLEDGE_GATE:
        gate = build_gate(gate_type, hidden_dim, 0).to(device)
        return StructuralGCN(None, raw_dim, hidden_dim, num_classes, gate).to(device)

    gate = None
    if gate_type != GateType.BASELINE_NO_GATE and encoder is not None:
        struct_dim = infer_structural_dim(encoder, data.num_nodes)
        gate = build_gate(gate_type, hidden_dim, struct_dim).to(device)

    return StructuralGCN(encoder, raw_dim, hidden_dim, num_classes, gate).to(device)


# ============================================================
# Training
# ============================================================

def train_and_eval(model, data, epochs=150):
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=5e-4)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits, aux_loss = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

        if aux_loss is not None:
            loss = loss + 0.1 * aux_loss

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)
        acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    return acc


# ============================================================
# Experiment
# ============================================================

def run_experiment1():
    graph_types = ["random", "sbm", "role"]
    encoder_types = ["none", "random", "laplacian", "degree", "node2vec"]
    gate_types = ALL_GATE_TYPES

    rows = []

    for graph_name in graph_types:
        for run in range(3):
            seed = 42 + run
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            if graph_name == "random":
                data, labels, _ = load_dataset("synthetic")
            elif graph_name == "sbm":
                data, labels = generate_synthetic_graph_with_structure()
            elif graph_name == "role":
                data, labels = generate_role_graph()
            else:
                continue

            data.y = labels
            N = data.num_nodes
            perm = torch.randperm(N)
            split = int(0.8 * N)

            data.train_mask = torch.zeros(N, dtype=torch.bool)
            data.test_mask = torch.zeros(N, dtype=torch.bool)
            data.train_mask[perm[:split]] = True
            data.test_mask[perm[split:]] = True

            data = data.to(device)
            num_classes = labels.max().item() + 1

            for encoder_type in encoder_types:
                for gate_type in gate_types:

                    # Valid combinations
                    if gate_type == GateType.JUMPING_KNOWLEDGE_GATE:
                        if encoder_type != "none":
                            continue
                    else:
                        if gate_type != GateType.BASELINE_NO_GATE and encoder_type == "none":
                            continue
                        if gate_type == GateType.BASELINE_NO_GATE and encoder_type != "none":
                            continue

                    logger.info(
                        f"[SYNTH] Graph={graph_name}, Run={run+1}, "
                        f"Encoder={encoder_type}, Gate={gate_type}"
                    )

                    model = build_model(data, num_classes, encoder_type, gate_type)
                    acc = train_and_eval(model, data)

                    rows.append({
                        "domain": "synthetic",
                        "graph": graph_name,
                        "run": run + 1,
                        "encoder": encoder_type,
                        "gate": gate_type,
                        "accuracy": acc,
                    })

    df = pd.DataFrame(rows)
    df.to_csv("results/experiment1_synthetic.csv", index=False)
    print("Saved Experiment 1 results to results/experiment1_synthetic.csv")


if __name__ == "__main__":
    run_experiment1()
