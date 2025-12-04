from typing import Any

import torch
from torch import nn

from experiments.experiment_utils import EvaluationResult, sample_negative_edges

from tasks.task import Task


class PromptSAGEClassificationTask(Task):
    """
    Task wrapper so Prompt-SAGE can be used inside the same pipeline
    as PyG models.

    This handles:
    - PyG → DGL conversion
    - Prompt-SAGE node classification training
    - Evaluation returning EvaluationResult
    """

    def __init__(self, model, device):
        super().__init__(model, device)
        self.dgl_graph = None
        self.labels = None
        self.train_nids = None
        self.val_nids = None
        self.test_nids = None

    # -------------------------
    # Prepare
    # -------------------------
    def prepare(self, data, labels=None, **kwargs):
        import dgl
        import torch
        import torch_geometric

        # Use the same device as the data or default to CUDA if available
        device = data.x.device if hasattr(data, 'x') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Store labels on local device
        self.labels = labels.to(device)

        # PyG → networkx → DGL
        nx_graph = torch_geometric.utils.to_networkx(
            data, to_undirected=True, remove_self_loops=True
        )
        self.dgl_graph = dgl.from_networkx(nx_graph).to(device)

        # Attach node features
        self.dgl_graph.ndata["feat"] = data.x.to(device)
        self.dgl_graph.ndata["label"] = labels.to(device)

        # masks → indices
        self.train_nids = torch.where(data.train_mask)[0].to(device)
        self.val_nids = torch.where(data.val_mask)[0].to(device)
        self.test_nids = torch.where(data.test_mask)[0].to(device)

    # -------------------------
    # Train
    # -------------------------
    def train(self, model: nn.Module, data: Any, epochs=50, lr=0.01, weight_decay=5e-4, log_every=10, **kwargs):
        import torch
        import logging
        logger = logging.getLogger(__name__)

        # Determine device from data or model
        device = data.x.device if hasattr(data, 'x') else next(model.parameters()).device
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_nids = self.train_nids.to(device)
        labels = self.labels.to(device)

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            logits = model(self.dgl_graph.to(device), self.dgl_graph.ndata["feat"].to(device))
            loss = torch.nn.functional.cross_entropy(
                logits[train_nids],
                labels[train_nids]
            )

            loss.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == epochs:
                val_result = self.evaluate(model, data, split="val", verbose=False)
                logger.info(
                    f"[PromptSAGE Class Epoch {epoch:03d}] Loss {loss.item():.4f} | Val Acc {val_result.accuracy:.4f}"
                )

        return model

    # -------------------------
    # Evaluate
    # -------------------------
    def evaluate(self, model: nn.Module, split="test", verbose=True, **kwargs) -> EvaluationResult:
        import torch
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        import logging
        logger = logging.getLogger(__name__)

        # Choose node indices based on split
        if split == "test":
            nids = self.test_nids
        elif split == "val":
            nids = self.val_nids
        else:
            nids = self.train_nids

        # Determine device
        device = next(model.parameters()).device
        model = model.to(device)

        nids = nids.to(device)
        labels = self.labels.to(device)

        model.eval()
        with torch.no_grad():
            logits = model(self.dgl_graph.to(device), self.dgl_graph.ndata["feat"].to(device))
            preds = logits[nids].argmax(dim=1).cpu()
            true = labels[nids].cpu()

        acc = accuracy_score(true, preds)
        precision = precision_score(true, preds, average="macro", zero_division=0)
        recall = recall_score(true, preds, average="macro", zero_division=0)
        f1 = f1_score(true, preds, average="macro", zero_division=0)

        try:
            auc = roc_auc_score(true, preds, multi_class="ovr", average="macro")
        except ValueError:
            auc = None

        if verbose:
            logger.info(
                f"PromptSAGE Eval | Split {split} | "
                f"Acc {acc:.4f} | Prec {precision:.4f} | Recall {recall:.4f} | "
                f"F1 {f1:.4f} | AUC {auc}"
            )

        return EvaluationResult(acc, precision, recall, f1, auc, preds)


class PromptSAGELinkPredictionTask(Task):
    """
    Task wrapper for Prompt-SAGE Link Prediction.
    Converts PyG → DGL, fine-tunes link prediction, and evaluates with standard metrics.
    """

    def __init__(self, rem_edge_list, name="PromptSAGE_LinkPred", epochs=20, lr=2e-3, weight_decay=5e-4):
        super().__init__(name, epochs)
        self.rem_edge_list = rem_edge_list
        self.lr = lr
        self.weight_decay = weight_decay
        self.dgl_graph = None
        self.pos_edges = None
        self.device = None

    # -------------------------
    # Prepare
    # -------------------------
    def prepare(self, data, **kwargs):
        import dgl
        # Determine device
        self.device = data.x.device if hasattr(data, 'x') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # PyG → DGL
        self.dgl_graph = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
        self.dgl_graph = dgl.to_simple(self.dgl_graph).to(self.device)
        self.dgl_graph.ndata['feat'] = data.x.to(self.device)

        # Positive edges for training
        self.pos_edges = self.rem_edge_list[0][0].T.to(self.device)  # shape (E, 2)

    # -------------------------
    # Train / Fine-tune
    # -------------------------
    def train(self, model: nn.Module, data: Any):
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()

            # Forward embeddings
            emb = model.forward_smc(self.dgl_graph, self.dgl_graph.ndata['feat'])

            # Sample negatives
            neg_edges = sample_negative_edges(self.pos_edges, self.dgl_graph.num_nodes()).to(self.device)

            # Compute dot product scores
            pos_scores = (emb[self.pos_edges[:, 0]] * emb[self.pos_edges[:, 1]]).sum(dim=-1)
            neg_scores = (emb[neg_edges[:, 0]] * emb[neg_edges[:, 1]]).sum(dim=-1)

            # Binary cross-entropy loss
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            scores = torch.cat([pos_scores, neg_scores])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)

            loss.backward()
            optimizer.step()

        return model

    # -------------------------
    # Evaluate
    # -------------------------
    def evaluate(self, model: nn.Module, verbose=True) -> EvaluationResult:
        model.eval()
        with torch.no_grad():
            emb = model.forward_smc(self.dgl_graph, self.dgl_graph.ndata['feat'])

            # Positive edges for evaluation
            pos_edges = self.rem_edge_list[1][0].T.to(self.device)
            neg_edges = sample_negative_edges(pos_edges, self.dgl_graph.num_nodes()).to(self.device)

            pos_scores = (emb[pos_edges[:, 0]] * emb[pos_edges[:, 1]]).sum(dim=-1)
            neg_scores = (emb[neg_edges[:, 0]] * emb[neg_edges[:, 1]]).sum(dim=-1)

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

            preds = (torch.sigmoid(scores) > 0.5).float()

            acc = accuracy_score(labels.cpu(), preds.cpu())
            prec = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
            rec = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
            f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)
            auc = roc_auc_score(labels.cpu(), scores.cpu())
            ap = average_precision_score(labels.cpu(), scores.cpu())

        if verbose:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[{self.name} Eval] Acc {acc:.4f} | AUC {auc:.4f} | AP {ap:.4f}")

        return EvaluationResult(acc, prec, rec, f1, auc, ap, preds)