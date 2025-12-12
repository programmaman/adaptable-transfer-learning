# tasks/structg_tasks.py
import logging
from utilities.experiment_utils import EvaluationResult, sample_negative_edges
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from tasks.task import Task

logger = logging.getLogger(__name__)

from typing import Any
import torch
from torch import nn


# --- Structural GNN Pretraining Task ---
class StructuralPretrainTask(Task):
    """
    Combines the StructuralGNN internal multi-task pretraining
    (node class, link pred, feat rec, n2v align)
    """

    def __init__(self, name="StructuralGNN_Pretrain", epochs=100, lr=0.01, neg_sample_size=5):
        super().__init__(name, epochs)
        self.lr = lr
        self.neg_sample_size = neg_sample_size

    def prepare(self, data):
        # Data preparation is minimal for pretraining
        return data

    def train(self, model: nn.Module, data: Any):
        # Extract the relevant pretraining logic from StructGPipeline.pretrain
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()

            # This call encapsulates the multi-task loss calculation
            _, total_loss = model.forward_and_loss(
                data,
                neg_sample_size=self.neg_sample_size,
                do_node_class=True,
                do_linkpred=True,
                do_featrec=True,
                do_n2v_align=True,
                train_mask=None,
            )
            total_loss.backward()
            optimizer.step()
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info(f"[{self.name} Epoch {epoch:03d}] Total Loss: {total_loss.item():.4f}")

        return model

    def evaluate(self, model, data):
        return


# --- Optional: Node2Vec Initialization Task ---
class Node2VecPretrainTask(Task):
    """
    Handles the initial Node2Vec embedding pretraining.
    This can be executed first in the pretrain_tasks list.
    """

    def __init__(self, name="Node2Vec_Pretrain", epochs=100, batch_size=128, lr=0.01):
        super().__init__(name, epochs)
        self.batch_size = batch_size
        self.lr = lr

    def prepare(self, data):
        return data

    def train(self, model: nn.Module, data: Any):
        logger.info(f"[{self.name}] Starting Node2Vec pretraining...")
        # Assumes model has train_node2vec method
        model.train_node2vec(
            num_epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            verbose=True,
        )
        return model

    def evaluate(self, model, data):
        return EvaluationResult(metadata=self.metadata)


class NodeClassificationTask(Task):
    """
    Fine-tuning and evaluation for Node Classification.
    """

    def __init__(self, labels, name="Node_Classification", epochs=30, lr=0.01, weight_decay=5e-4):
        super().__init__(name, epochs)
        self.labels = labels
        self.lr = lr
        self.weight_decay = weight_decay

    def prepare(self, data):
        # Attach labels for internal use by the model/task
        data.y = self.labels
        return data

    def train(self, model: nn.Module, data: Any):
        logger.info("Fine-tuning StructuralGNN for Node Classification")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        node_indices = torch.arange(data.num_nodes, device=data.y.device)

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()

            embeddings = model(data.x, data.edge_index, node_indices)
            loss = model.node_classification_loss(embeddings, data.y)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == self.epochs:  # Log every 10 epochs
                val_result = self.evaluate(model, data, mask=data.val_mask, verbose=False)
                logger.info(
                    f"[{self.name} Epoch {epoch:03d}] Loss {loss.item():.4f} | Val Acc {val_result.accuracy:.4f}")

        return model

    def evaluate(self, model: nn.Module, data: Any, mask=None, verbose=True) -> EvaluationResult:
        logger.info(f"Evaluating StructuralGNN for Node Classification on {self.name} task")
        if mask is None:
            mask = data.test_mask

        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)
            logits = model.classify_nodes(embeddings)
            preds = logits[mask].argmax(dim=1).cpu()
            true = data.y[mask].cpu()

        acc = accuracy_score(true, preds)
        precision = precision_score(true, preds, average="macro", zero_division=0)
        recall = recall_score(true, preds, average="macro", zero_division=0)
        f1 = f1_score(true, preds, average="macro", zero_division=0)
        try:
            # Need probabilities/scores for AUC, but using argmax preds for simplicity here
            # In a full implementation, you'd use logits/softmax output for AUC
            auc = roc_auc_score(true, preds, multi_class="ovr", average="macro")
        except ValueError:
            auc = None

        self.metadata['accuracy'] = acc

        if verbose:
            logger.info(
                f"[{self.name} Eval] Acc {acc:.4f} | Prec {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f} | AUC {auc}")

        return EvaluationResult(acc, precision, recall, f1, auc, preds, metadata=self.metadata)


class LinkPredictionTask(Task):
    """
    Fine-tuning and evaluation for Link Prediction.
    """

    def __init__(self, rem_edge_list, name="Link_Prediction", epochs=30, lr=0.01, neg_sample_size=5):
        super().__init__(name, epochs)
        self.rem_edge_list = rem_edge_list
        self.lr = lr
        self.neg_sample_size = neg_sample_size

    def prepare(self, data):
        # Link prediction often requires a split of edges (train/val/test edges)
        # We assume rem_edge_list is the split edges used for evaluation
        # The training logic relies on data.edge_index being the training graph
        return data

    def train(self, model: nn.Module, data: Any):
        # Logic from finetune_link_prediction
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)
        node_indices = torch.arange(data.num_nodes, device=data.edge_index.device)

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()

            # The edge list for training loss is assumed to be contained in the model logic
            # Here we explicitly pass the training edges (rem_edge_list[0][0])
            embeddings = model(data.x, data.edge_index, node_indices)
            loss = model.link_prediction_loss(
                embeddings,
                self.rem_edge_list[0][0].transforms,  # Assuming this is the set of positive training edges
                neg_sample_size=self.neg_sample_size
            )
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == self.epochs:
                logger.info(f"[{self.name} Epoch {epoch:03d}] Loss {loss.item():.4f}")
        return model

    def evaluate(self, model: nn.Module, data: Any, verbose=True) -> EvaluationResult:

        model.eval()
        node_indices = torch.arange(data.num_nodes, device=data.edge_index.device)

        with torch.no_grad():
            gnn_emb = model(data.x, data.edge_index, node_indices)
            n2v_emb = model.node2vec_layer(node_indices)  # Accesses the precomputed/learned N2V embeddings

        # We assume the evaluation uses rem_edge_list[1] or similar
        pos_edges = self.rem_edge_list[1][0].transforms.contiguous()  # Using a different split for eval

        # Ensure pos_edges are on the correct device
        pos_edges = pos_edges.to(data.edge_index.device)
        neg_edges = sample_negative_edges(pos_edges, data.num_nodes).to(data.edge_index.device)

        def score(u, v):
            # Assumes model._pairwise_score handles the combination of GNN and N2V embeddings
            return model._pairwise_score(gnn_emb[u], gnn_emb[v], n2v_emb[u], n2v_emb[v]).squeeze()

        pos_scores = score(pos_edges[:, 0], pos_edges[:, 1])
        neg_scores = score(neg_edges[:, 0], neg_edges[:, 1])
        scores = torch.cat([pos_scores, neg_scores]).detach().cpu()
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu()

        preds = (torch.sigmoid(scores) > 0.5).float()

        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

        self.metadata['accuracy'] = acc

        if verbose:
            logger.info(f"[{self.name} Eval] Acc {acc:.4f} | AUC {auc:.4f} | AP {ap:.4f}")

        return EvaluationResult(acc, precision, recall, f1, auc, ap, preds, metadata=self.metadata)