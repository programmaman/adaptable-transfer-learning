# tasks/structg_tasks.py
import os
import logging
from typing import Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.task import Task
from experiments.experiment_utils import EvaluationResult, sample_negative_edges
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


# -------------------------
# 1) Node2Vec pretraining (task)
# -------------------------
class Node2VecPretrainTask(Task):
    def __init__(self, name: str = "node2vec_pretrain", epochs: int = 10, batch_size: int = 128, lr: float = 0.01, verbose: bool = False):
        super().__init__(name, epochs=epochs)
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose

    def prepare(self, data: Any):
        # Node2Vec uses edge_index and num_nodes; no extra preparation required here.
        return data

    def train(self, model: nn.Module, data: Any):
        """Train Node2Vec module attached to the model (if present)."""
        if not hasattr(model, "train_node2vec"):
            raise RuntimeError("Model does not implement train_node2vec() required by Node2VecPretrainTask")

        logger.info(f"[{self.name}] Starting Node2Vec pretraining for {self.epochs} epochs")
        model.train_node2vec(num_epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, verbose=self.verbose)
        return model

    def evaluate(self, model: nn.Module, data: Any):
        # Node2Vec pretraining normally doesn't produce a concrete EvaluationResult.
        return EvaluationResult(accuracy=None, precision=None, recall=None, f1=None, auc=None, preds=None)


# -------------------------
# 2) Structural multi-loss pretraining (task)
#    Runs combined losses: classification, linkpred, featrec, n2v_align (if available)
# -------------------------
class StructuralMultiLossPretrainTask(Task):
    def __init__(
        self,
        base_model_path: str,
        name: str = "structg_multiloss_pretrain",
        epochs: int = 10,
        lr: float = 1e-2,
        weight_decay: float = 5e-4,
        neg_sample_size: int = 5,
        feat_reduce_dim: Optional[int] = None,
        safety_factor: float = 0.7,
    ):
        super().__init__(name, epochs=epochs)
        self.base_model_path = base_model_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_sample_size = neg_sample_size
        self.feat_reduce_dim = feat_reduce_dim
        self.safety_factor = safety_factor

    def prepare(self, data: Any):
        # No modification by default; tasks could override for special preprocessing
        return data

    def _maybe_reduce_features(self, model: nn.Module, data: Any):
        """Optional SVD reduction if memory is constrained."""
        if self.feat_reduce_dim is None:
            return data
        n_nodes, n_features = data.x.size()
        required_bytes = n_nodes * n_features * 4 * 2
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0]
            if required_bytes > free_mem * self.safety_factor:
                logger.info(f"[{self.name}] Reducing features {n_features} -> {self.feat_reduce_dim} due to memory limits")
                x_cpu = data.x.cpu().numpy()
                svd = TruncatedSVD(n_components=self.feat_reduce_dim, random_state=0)
                x_reduced = torch.tensor(svd.fit_transform(x_cpu), dtype=torch.float32)
                data = data.__class__(x=x_reduced.to(data.x.device), edge_index=data.edge_index)
                if hasattr(model, "reset_with_input_dim"):
                    model.reset_with_input_dim(data.x.size(1))
        return data

    def train(self, model: nn.Module, data: Any):
        """
        Train the model with multiple losses combined.
        The model is expected to expose:
          - forward(x, edge_index) -> embeddings
          - node_classification_loss(embeddings, y, mask)
          - link_prediction_loss(embeddings, edge_index, neg_sample_size)
          - feature_reconstruction_loss(embeddings, x)  (optional)
          - node2vec_alignment_loss(embeddings, node2vec_raw) (optional)
        """
        device = next(model.parameters()).device
        data = data.to(device)
        data = self._maybe_reduce_features(model, data)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        logger.info(f"[{self.name}] Multi-loss pretraining for {self.epochs} epochs")

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()

            embeddings = model(data.x, data.edge_index)

            total_loss = 0.0
            # classification loss (if available & labels present)
            if hasattr(model, "node_classification_loss") and getattr(data, "y", None) is not None:
                # here we pass train_mask if present; else None (task can interpret)
                mask = getattr(data, "train_mask", None)
                total_loss = total_loss + model.node_classification_loss(embeddings, data.y, mask=mask)

            # linkpred
            if hasattr(model, "link_prediction_loss"):
                total_loss = total_loss + model.link_prediction_loss(embeddings, data.edge_index, neg_sample_size=self.neg_sample_size)

            # feature reconstruction
            if hasattr(model, "feature_reconstruction_loss") and getattr(model, "feat_reconstruction", False):
                total_loss = total_loss + model.feature_reconstruction_loss(embeddings, data.x)

            # n2v alignment
            if hasattr(model, "node2vec_alignment_loss") and getattr(data, "node2vec_raw", None) is not None:
                total_loss = total_loss + model.node2vec_alignment_loss(embeddings, data.node2vec_raw)

            total_loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == self.epochs:
                logger.info(f"[{self.name}] Epoch {epoch:03d} | Loss {total_loss.item():.6f}")

        # save backbone weights if requested
        os.makedirs(os.path.dirname(self.base_model_path) or ".", exist_ok=True)
        if hasattr(model, "gnn_frozen"):
            try:
                torch.save(model.gnn_frozen.state_dict(), self.base_model_path)
                logger.info(f"[{self.name}] Saved backbone to {self.base_model_path}")
            except Exception:
                # best-effort save; ignore failures for unit tests
                logger.exception("Failed to save backbone weights (non-fatal).")

        return model

    def evaluate(self, model: nn.Module, data: Any):
        # Pretraining task: typically no evaluation result; return empty EvaluationResult-like object
        return EvaluationResult(accuracy=None, precision=None, recall=None, f1=None, auc=None, preds=None)


# -------------------------
# 3) Node classification finetune task (wrap pipeline finetune logic)
# -------------------------
class NodeClassificationFineTuneTask(Task):
    def __init__(self, name: str = "node_classification_finetune", epochs: int = 30, lr: float = 0.01, weight_decay: float = 5e-4, log_every: int = 10):
        super().__init__(name, epochs=epochs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_every = log_every

    def prepare(self, data: Any):
        # Ensure labels exist; attach to data if necessary outside
        assert hasattr(data, "y"), "NodeClassificationFineTuneTask.prepare: data must have data.y"
        return data

    def train(self, model: nn.Module, data: Any):
        device = next(model.parameters()).device
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        logger.info(f"[{self.name}] Fine-tuning for {self.epochs} epochs")

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()
            node_indices = torch.arange(data.num_nodes, device=device)
            embeddings = model(data.x.to(device), data.edge_index.to(device), node_indices)
            # call model-provided classification loss (keeps model heads encapsulated)
            if not hasattr(model, "node_classification_loss"):
                raise RuntimeError("Model must implement node_classification_loss for NodeClassificationFineTuneTask")
            loss = model.node_classification_loss(embeddings, data.y, mask=getattr(data, "train_mask", None))
            loss.backward()
            optimizer.step()

            if epoch % self.log_every == 0 or epoch == self.epochs:
                # evaluate on val_mask if available
                from experiments.experiment_utils import EvaluationResult as _ER  # local import
                val_mask = getattr(data, "val_mask", None)
                val_res = self.evaluate(model, data) if val_mask is None else self.evaluate(model, data, mask=val_mask)
                logger.info(f"[{self.name}] Epoch {epoch:03d} | Loss {loss.item():.6f} | Val Acc {getattr(val_res, 'accuracy', None)}")

        return model

    def evaluate(self, model: nn.Module, data: Any, mask: Optional[torch.Tensor] = None) -> EvaluationResult:
        if mask is None:
            mask = getattr(data, "test_mask", None)
        model.eval()
        device = next(model.parameters()).device
        data = data.to(device)
        with torch.no_grad():
            embeddings = model(data.x.to(device), data.edge_index.to(device))
            if not hasattr(model, "classify_nodes"):
                raise RuntimeError("Model must implement classify_nodes for evaluation")
            logits = model.classify_nodes(embeddings)
            assert mask is not None, "NodeClassificationFineTuneTask.evaluate: no mask provided"
            preds = logits[mask].argmax(dim=1).cpu()
            true = data.y[mask].cpu()

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        acc = accuracy_score(true, preds)
        prec = precision_score(true, preds, average="macro", zero_division=0)
        rec = recall_score(true, preds, average="macro", zero_division=0)
        f1 = f1_score(true, preds, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(true, preds, multi_class="ovr", average="macro")
        except Exception:
            auc = None

        return EvaluationResult(accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc, preds=preds)


# -------------------------
# 4) Link prediction finetune task
# -------------------------
class LinkPredictionFineTuneTask(Task):
    def __init__(self, name: str = "link_prediction_finetune", epochs: int = 30, lr: float = 0.01, weight_decay: float = 5e-4, neg_sample_size: int = 5, log_every: int = 10):
        super().__init__(name, epochs=epochs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_sample_size = neg_sample_size
        self.log_every = log_every

    def prepare(self, data: Any):
        # expect rem_edge_list or training split already present
        assert hasattr(data, "remaining_edges_list"), "LinkPredictionFineTuneTask.prepare: data must have remaining_edges_list"
        return data

    def train(self, model: nn.Module, data: Any):
        device = next(model.parameters()).device
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        logger.info(f"[{self.name}] Fine-tuning link prediction for {self.epochs} epochs")

        # positive edges for training are the 'remaining_edges_list' returned by your helper
        pos_edges = data.remaining_edges_list[0].to(device)

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()
            node_indices = torch.arange(data.num_nodes, device=device)
            embeddings = model(data.x.to(device), data.edge_index.to(device), node_indices)

            # If model provides a link_prediction_loss use it, else compute here
            if hasattr(model, "link_prediction_loss"):
                loss = model.link_prediction_loss(embeddings, pos_edges.T, neg_sample_size=self.neg_sample_size)
            else:
                # fallback: sample negatives and compute BCE with dot-product
                neg_edges = sample_negative_edges(pos_edges, data.num_nodes, num_samples=pos_edges.size(0)).to(device)
                pos_scores = (embeddings[pos_edges[:, 0]] * embeddings[pos_edges[:, 1]]).sum(dim=1)
                neg_scores = (embeddings[neg_edges[:, 0]] * embeddings[neg_edges[:, 1]]).sum(dim=1)
                logits = torch.cat([pos_scores, neg_scores], dim=0)
                labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0).to(device)
                loss = F.binary_cross_entropy_with_logits(logits, labels)

            loss.backward()
            optimizer.step()

            if epoch % self.log_every == 0 or epoch == self.epochs:
                logger.info(f"[{self.name}] Epoch {epoch:03d} | Loss {loss.item():.6f}")

        return model

    def evaluate(self, model: nn.Module, data: Any, verbose: bool = True) -> EvaluationResult:
        model.eval()
        device = next(model.parameters()).device
        data = data.to(device)
        node_indices = torch.arange(data.num_nodes, device=device)
        with torch.no_grad():
            gnn_emb = model(data.x.to(device), data.edge_index.to(device), node_indices)
            # try to get node2vec embeddings (if model exposes it)
            n2v_emb = None
            if hasattr(model, "node2vec_layer"):
                try:
                    with torch.no_grad():
                        n2v_emb = model.node2vec_layer(torch.arange(data.num_nodes, device=device))
                except Exception:
                    n2v_emb = None

        pos_edges = data.remaining_edges_list[0].to(device)
        neg_edges = sample_negative_edges(pos_edges, data.num_nodes, num_samples=pos_edges.size(0)).to(device)

        # scoring: use model._pairwise_score if present else dot product
        if hasattr(model, "_pairwise_score"):
            def score(u, v):
                if n2v_emb is not None:
                    return model._pairwise_score(gnn_emb[u], gnn_emb[v], n2v_emb[u], n2v_emb[v]).squeeze()
                else:
                    return model._pairwise_score(gnn_emb[u], gnn_emb[v], torch.zeros_like(gnn_emb[u]), torch.zeros_like(gnn_emb[v])).squeeze()
        else:
            def score(u, v):
                return (gnn_emb[u] * gnn_emb[v]).sum(dim=1)

        pos_scores = score(pos_edges[:, 0], pos_edges[:, 1])
        neg_scores = score(neg_edges[:, 0], neg_edges[:, 1])

        scores = torch.cat([pos_scores, neg_scores]).detach().cpu()
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu()
        preds = (torch.sigmoid(scores) > 0.5).float()

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

        if verbose:
            logger.info(f"[{self.name}] LP Eval | Acc {acc:.4f} | Prec {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f} | AUC {auc:.4f} | AP {ap:.4f}")

        return EvaluationResult(accuracy=acc, precision=precision, recall=recall, f1=f1, auc=auc, ap=ap, preds=preds)
