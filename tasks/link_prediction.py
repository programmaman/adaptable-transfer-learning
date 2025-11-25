from abc import ABC
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score
)

from experiments.experiment_utils import (
    EvaluationResult,
    split_edges_for_link_prediction,
    sample_negative_edges
)
from task import Task


class LinkPredictionTask(Task, ABC):
    """
    Link prediction task using dot-product decoding.
    """

    def __init__(self, name="link_prediction", epochs=30, learning_rate=0.01,
                 weight_decay=5e-4, negative_sample_size=None, log_every=10):
        super().__init__(name)
        self.epochs = epochs
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

        positive_edges = data.remaining_edges_list[0].to(data.x.device)
        number_of_nodes = data.num_nodes

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()

            node_embeddings = model(data.x, data.edge_index)

            negative_edges = sample_negative_edges(
                positive_edges, num_nodes=number_of_nodes
            ).to(node_embeddings.device)

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

        positive_edges = data.remaining_edges_list[2].to(data.x.device)
        number_of_nodes = data.num_nodes

        negative_edges = sample_negative_edges(
            positive_edges, num_nodes=number_of_nodes
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
