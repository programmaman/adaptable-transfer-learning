from abc import ABC
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score
)

from utilities.experiment_utils import (
    EvaluationResult,
    split_edges_for_link_prediction,
    sample_negative_edges
)
from tasks.task import Task


class LinkPredictionTask(Task):
    """
    Link prediction task using dot-product decoding.
    """

    def __init__(self, name="link_prediction", epochs=30, learning_rate=0.01,
                 weight_decay=5e-4, negative_sample_size=None, log_every=10):
        super().__init__(name, epochs=epochs)  # epochs handled by base class
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.log_every = log_every
        self.negative_sample_size = negative_sample_size  # may be unused by dot-product LP

    # ----------------------------
    # 1. Data Preparation
    # ----------------------------
    def prepare(self, data):
        """
        Split edges into:
            train_edges
            val_edges
            test_edges

        Using the existing helper function.
        """
        data_lp = data.clone()
        edge_index_split, remaining_edges_list = split_edges_for_link_prediction(data_lp.edge_index)
        data_lp.edge_index = edge_index_split
        data_lp.remaining_edges_list = remaining_edges_list
        return data_lp

    # ----------------------------
    # 2. Training Loop
    # ----------------------------
    def train(self, model, data):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        loss_function = torch.nn.BCEWithLogitsLoss()

        # Extract raw edges (E, 2)
        positive_edges_raw = data.remaining_edges_list[0][0]

        # -------------------------------------------------------
        # Debug prints
        # -------------------------------------------------------
        print("\n=== LinkPredictionTask.train Debug ===")
        print("data keys:", list(data.keys()))
        print("remaining_edges_list keys:", data.remaining_edges_list.keys())
        print("remaining_edges_list[0] type:", type(data.remaining_edges_list[0]))
        print("remaining_edges_list[0][0] type:", type(positive_edges_raw))
        print("positive_edges_raw shape:", positive_edges_raw.shape)
        print("positive_edges_raw device:", positive_edges_raw.device)
        print("positive_edges_raw dtype:", positive_edges_raw.dtype)
        print("positive_edges_raw sample (first 5):")
        print(positive_edges_raw[:5])

        # (2, E) for scoring
        positive_edges = positive_edges_raw.t().contiguous().to(data.x.device)

        print("positive_edges (transposed) shape:", positive_edges.shape)
        print("positive_edges device:", positive_edges.device)
        print("num_nodes:", data.num_nodes)
        print("======================================\n")

        number_of_nodes = data.num_nodes

        # -------------------------------------------------------
        # Training Loop
        # -------------------------------------------------------
        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()

            node_embeddings = model(data.x, data.edge_index)

            negative_edges_raw = sample_negative_edges(
                positive_edges_raw, num_nodes=number_of_nodes
            ).to(node_embeddings.device)

            negative_edges = negative_edges_raw.t().contiguous()

            positive_scores = (node_embeddings[positive_edges[0]] *
                               node_embeddings[positive_edges[1]]).sum(dim=1)

            negative_scores = (node_embeddings[negative_edges[0]] *
                               node_embeddings[negative_edges[1]]).sum(dim=1)

            scores = torch.cat([positive_scores, negative_scores])
            labels = torch.cat([
                torch.ones_like(positive_scores),
                torch.zeros_like(negative_scores)
            ])

            loss = loss_function(scores, labels)
            loss.backward()
            optimizer.step()

            if epoch % self.log_every == 0:
                self.metadata[f"epoch_{epoch}"] = {"loss": loss.item()}

        return model

    # ----------------------------
    # 3. Evaluation
    # ----------------------------
    def evaluate(self, model, data):
        model.eval()

        with torch.no_grad():
            node_embeddings = model(data.x, data.edge_index)

        positive_edges_raw = data.remaining_edges_list[0][0]

        positive_edges = torch.tensor(
            positive_edges_raw,
            dtype=torch.long,
            device=data.x.device
        ).t().contiguous()

        number_of_nodes = data.num_nodes

        negative_edges = sample_negative_edges(
            positive_edges_raw, num_nodes=number_of_nodes
        ).to(node_embeddings.device)

        positive_scores = (node_embeddings[positive_edges[0]] *
                           node_embeddings[positive_edges[1]]).sum(dim=1)

        negative_scores = (node_embeddings[negative_edges[0]] *
                           node_embeddings[negative_edges[1]]).sum(dim=1)

        scores = torch.cat([positive_scores, negative_scores]).cpu()
        labels = torch.cat([
            torch.ones_like(positive_scores),
            torch.zeros_like(negative_scores)
        ]).cpu()

        probabilities = torch.sigmoid(scores)
        predictions = (probabilities > 0.5).float()

        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, zero_division=0)
        auc = roc_auc_score(labels, probabilities)
        average_precision = average_precision_score(labels, probabilities)

        return EvaluationResult(
            accuracy=accuracy,
            precision=0,
            recall=0,
            f1=f1,
            auc=auc,
            ap=average_precision,
            preds=predictions
        )
