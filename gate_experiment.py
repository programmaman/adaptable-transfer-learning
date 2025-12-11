import torch
print("hello world1 ")
import torch.nn as nn
print("hello world2")
import pandas as pd
import logging
from torch.optim import Adam
print("hello world3")

print(torch.__version__)
print(torch.version.cuda)

from encoders.node2vec_encoder import Node2VecEncoder
print("hello world4")
from experiments.pipeline import TaskPipeline
from integrators.structural_integrator import SimpleFeatureGate, SelfSupervisedGate
from models.struct_g import StructuralGNN
from tasks.task import Task
from experiments.experiment_utils import generate_synthetic_graph


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("GatingExp")


class StructuralGatingTask(Task):
    """
    A specific task that handles the (output, aux_loss) tuple signature
    of the StructuralGNN.
    """

    def __init__(self, name, epochs=50, lr=0.01, num_classes=None):
        super().__init__(name, epochs=epochs)
        self.lr = lr
        self.num_classes = num_classes

    def prepare(self, data):
        # Quick synthetic split
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

            # --- TUPLE HANDLING HERE ---
            out, aux_loss = model(data.x, data.edge_index)

            # Main Loss
            logits = model.classify_nodes(out)
            cls_loss = nn.functional.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

            # Total Loss
            total_loss = cls_loss + aux_loss

            total_loss.backward()
            optimizer.step()

        return model

    def evaluate(self, model, data):
        model.eval()
        with torch.no_grad():
            out, _ = model(data.x, data.edge_index)  # Ignore aux_loss
            logits = model.classify_nodes(out)
            preds = logits.argmax(dim=1)

            acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        class Result:
            accuracy = acc
            metadata = {}

        return Result()


def build_model(data, num_classes, gate_type):
    # Shared Encoder (trained on fly)
    encoder = Node2VecEncoder(data.num_nodes, data.edge_index, embedding_dim=64,device='cuda' if torch.cuda.is_available() else 'cpu')
    encoder.train_encoder(epochs=5, verbose=False)

    dim = data.x.size(1)

    if gate_type == "Simple":
        gate = SimpleFeatureGate(dim, encoder.embedding_dim, 64)
    else:
        gate = SelfSupervisedGate(dim, encoder.embedding_dim, 64)

    return StructuralGNN(
        structural_encoder=encoder,
        input_dim=dim,
        hidden_dim=64,
        output_dim=64,
        num_classes=num_classes,
        gate_integrator=gate
    )


def run_experiment():
    logger.info("Starting Structural Gating Experiment")
    results = []

    for i in range(5):
        logger.info(f"--- Run {i + 1} ---")
        data, labels = generate_synthetic_graph()
        data.y = labels
        num_classes = labels.max().item() + 1

        pipeline = TaskPipeline(seed=42 + i)

        # 1. Simple Gate
        model_simple = build_model(data, num_classes, "Simple")
        task_simple = StructuralGatingTask("Simple", num_classes=num_classes)
        _, res_simple = pipeline.run(model_simple, data, [task_simple])

        # 2. Self-Supervised Gate
        model_ssl = build_model(data, num_classes, "SSL")
        task_ssl = StructuralGatingTask("SSL", num_classes=num_classes)
        _, res_ssl = pipeline.run(model_ssl, data, [task_ssl])

        logger.info(f"Result: Simple={res_simple['Simple'].accuracy:.4f} | SSL={res_ssl['SSL'].accuracy:.4f}")
        results.append({"Run": i + 1, "Simple": res_simple['Simple'].accuracy, "SSL": res_ssl['SSL'].accuracy})

    df = pd.DataFrame(results)
    print("\nSummary:\n", df)
    print(f"\nAvg Simple: {df['Simple'].mean():.4f}")
    print(f"Avg SSL:    {df['SSL'].mean():.4f}")


if __name__ == "__main__":
    run_experiment()