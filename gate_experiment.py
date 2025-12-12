import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import logging
from torch.optim import Adam
from torch_geometric.nn import GCNConv

# Assumed Imports
from encoders.node2vec_encoder import Node2VecEncoder
from experiments.pipeline import TaskPipeline
# Ensure AdaptiveGate is imported or defined above
from integrators.structural_integrator import SimpleFeatureGate, SelfSupervisedGate, AdaptiveGate, \
    CombinedAdaptiveSelfSupervisedGate
from tasks.task import Task
from experiments.experiment_utils import generate_synthetic_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("GatingExp")


class StructuralGCN(nn.Module):
    """
    Modified architecture:
        1. GCN produces h_gcn
        2. Gate fuses h_gcn with structural embeddings
        3. Classifier predicts labels
    """
    def __init__(self, structural_encoder, feature_dim, hidden_dim, num_classes, gate=None):
        super().__init__()
        self.structural_encoder = structural_encoder
        self.gate = gate

        # GCN trunk
        self.conv1 = GCNConv(feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        aux_loss = None

        # --- GCN trunk ---
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)

        # --- Optional post-GCN gating ---
        if self.gate is not None:
            device = x.device
            all_nodes = torch.arange(x.size(0), device=device)
            struct_emb = self.structural_encoder(all_nodes)

            if isinstance(self.gate, AdaptiveGate):
                gated, aux_loss = self.gate.integrate(
                    h, struct_emb, edge_index, initial_features=h
                )
            else:
                gated, aux_loss = self.gate.integrate(h, struct_emb, edge_index)
            h = gated

        return h, aux_loss

    def classify_nodes(self, embeddings):
        return self.classifier(embeddings)


# --------------------------------------------------------------------------
# Updated build_model uses the new architecture unchanged
# --------------------------------------------------------------------------

def build_model(data, num_classes, mode):
    # Raw feature dimension (e.g., 16)
    raw_dim = data.x.size(1)
    # Internal hidden dimension (e.g., 64)
    hidden_dim = 64

    encoder = Node2VecEncoder(data.num_nodes, data.edge_index, embedding_dim=64)
    # (Optional: reduce epochs for debugging speed)
    encoder.train_encoder(epochs=1, verbose=False)

    gate = None

    # [FIX] For all gates, 'feature_dimension' must match the GCN output (hidden_dim),
    # NOT the raw input dimension (raw_dim), because the gate comes AFTER the GCN.

    if mode == "Simple":
        # Changed raw_dim -> hidden_dim
        gate = SimpleFeatureGate(hidden_dim, encoder.embedding_dimension, hidden_dim)

    elif mode == "SSL":
        # Changed raw_dim -> hidden_dim
        gate = SelfSupervisedGate(hidden_dim, encoder.embedding_dimension, hidden_dim)

    elif mode == "Adaptive":
        gate = AdaptiveGate(
            feature_dimension=hidden_dim,
            structural_dimension=encoder.embedding_dimension,
            calculation_dimension=hidden_dim,
            initial_feature_dimension=hidden_dim
        )

    elif mode == "Combined":
        gate = CombinedAdaptiveSelfSupervisedGate(
            feature_dimension=hidden_dim,
            structural_dimension=encoder.embedding_dimension,
            calculation_dimension=hidden_dim,
            initial_feature_dimension=hidden_dim
        )

    return StructuralGCN(encoder, raw_dim, hidden_dim, num_classes, gate)


# --------------------------------------------------------------------------
# 2. Task Definition (Unchanged)
# --------------------------------------------------------------------------
class StructuralGatingTask(Task):
    def __init__(self, name, epochs=50, lr=0.01):
        super().__init__(name, epochs=epochs)
        self.lr = lr

    def prepare(self, data):
        if not hasattr(data, 'train_mask'):
            N = data.num_nodes
            perm = torch.randperm(N)
            split = int(0.8 * N)
            data.train_mask = torch.zeros(N, dtype=torch.bool)
            data.test_mask = torch.zeros(N, dtype=torch.bool)
            data.train_mask[perm[:split]] = True
            data.test_mask[perm[split:]] = True
        return data

    def train(self, model, data):
        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)
        model.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out, aux_loss = model(data.x, data.edge_index)
            logits = model.classify_nodes(out)

            cls_loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

            if aux_loss is not None and aux_loss.dim() > 0:
                total_loss = cls_loss + aux_loss.mean()
            else:
                total_loss = cls_loss

            total_loss.backward()
            optimizer.step()
        return model

    def evaluate(self, model, data):
        model.eval()
        with torch.no_grad():
            out, _ = model(data.x, data.edge_index)
            logits = model.classify_nodes(out)
            preds = logits.argmax(dim=1)
            acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        class Result:
            accuracy = acc
            metadata = {}

        return Result()



def run_experiment():
    logger.info("Starting Experiment: Standard vs Simple vs SSL vs Adaptive")
    results = []

    for i in range(5):
        logger.info(f"\n--- Run {i + 1} ---")
        data, labels = generate_synthetic_graph()
        data.y = labels
        num_classes = labels.max().item() + 1
        pipeline = TaskPipeline(seed=42 + i, device='cpu')

        # 1. Standard
        model_std = build_model(data, num_classes, "Standard")
        _, res_std = pipeline.run(model_std, data, [StructuralGatingTask("Standard", epochs=3)])

        # 2. Simple
        model_simple = build_model(data, num_classes, "Simple")
        _, res_simple = pipeline.run(model_simple, data, [StructuralGatingTask("Simple", epochs=3)])

        # 3. SSL
        model_ssl = build_model(data, num_classes, "SSL")
        _, res_ssl = pipeline.run(model_ssl, data, [StructuralGatingTask("SSL", epochs=3)])

        # 4. Adaptive (New)
        model_adaptive = build_model(data, num_classes, "Adaptive")
        _, res_adaptive = pipeline.run(model_adaptive, data, [StructuralGatingTask("Adaptive", epochs=3)])

        # 5. Combined
        model_combined = build_model(data, num_classes, "Combined")
        _, res_combined = pipeline.run(model_combined, data, [StructuralGatingTask("Combined", epochs=3)])

        logger.info(
            f"Accuracies -> Std: {res_std['Standard'].accuracy:.4f} | "
            f"Simple: {res_simple['Simple'].accuracy:.4f} | "
            f"SSL: {res_ssl['SSL'].accuracy:.4f} | "
            f"Adaptive: {res_adaptive['Adaptive'].accuracy:.4f} | "
            f"Combined: {res_combined['Combined'].accuracy:.4f}"
        )

        results.append({
            "Run": i + 1,
            "Standard_GCN": res_std['Standard'].accuracy,
            "Simple_Gate": res_simple['Simple'].accuracy,
            "SSL_Gate": res_ssl['SSL'].accuracy,
            "Adaptive_Gate": res_adaptive['Adaptive'].accuracy,
            "Combined_Gate": res_combined['Combined'].accuracy,
        })

    df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("SUMMARY")
    print(df)
    print("-" * 50)
    print(f"Avg Standard: {df['Standard_GCN'].mean():.4f}")
    print(f"Avg Simple:   {df['Simple_Gate'].mean():.4f}")
    print(f"Avg SSL:      {df['SSL_Gate'].mean():.4f}")
    print(f"Avg Adaptive: {df['Adaptive_Gate'].mean():.4f}")
    print(f"Avg Combined: {df['Combined_Gate'].mean():.4f}")
    print("=" * 50)


if __name__ == "__main__":
    run_experiment()