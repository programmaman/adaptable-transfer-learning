import logging
from abc import ABC
from experiments.experiment_utils import  EvaluationResult

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def _initialize_device(device):
    return device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _initialize_seed(seed):
    if seed is not None:
        return seed
    generated_seed = int(time.time() * 1e6) % (2 ** 32)
    logger.warning("No seed provided. Using time-based seed for reproducibility.")
    return generated_seed


class Pipeline(ABC):
    def __init__(self, seed=None, device=None, train_ratio=0.6, val_ratio=0.2):
        self.seed = _initialize_seed(seed)
        self.device = _initialize_device(device)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.metadata = {"seed": self.seed}

        logger.info(f"Pipeline initialized with seed {self.seed} on device {self.device}")
        if self.device != "cuda":
            logger.warning("CUDA not available, using CPU. This may be slow!")

    def run(self, data, labels, model, pretrain_epochs=100, finetune_epochs=30):
        """
        Runs the pipeline on the provided model.
        """
        if model is None:
            raise ValueError("You must pass a model instance to run().")

        logger.info("Starting pipeline run...")
        logger.info(f"Data has {data.num_nodes} nodes and {data.num_edges} edges.")
        logger.info(f"Labels have {len(labels.unique())} unique classes.")
        logger.info(f"Pretraining for {pretrain_epochs} epochs, Finetuning for {finetune_epochs} epochs.")

        self._set_seed()
        start = time.time()

        data = self.prepare_data(data)

        # Phase 1: Pretraining
        model = self.pretrain(model, data, pretrain_epochs)

        # Phase 2: Classification
        classifier_train_start = time.time()
        model = self.finetune_classification(model, data, labels, finetune_epochs)
        self.metadata["classifier_time"] = time.time() - classifier_train_start

        classification_results = self.evaluate_classification(model, data, labels)

        # Phase 3: Link Prediction
        data.edge_index, rem_edge_list = split_edges_for_link_prediction(data.edge_index)
        lp_train_start = time.time()
        model = self.finetune_link_prediction(model, data, rem_edge_list, finetune_epochs)
        self.metadata["link_pred_time"] = time.time() - lp_train_start

        link_pred_results = self.evaluate_link_prediction(model, data, rem_edge_list)

        self.metadata["total_time"] = time.time() - start
        classification_results.metadata.update(self.metadata)
        link_pred_results.metadata.update(self.metadata)

        return model, classification_results, link_pred_results

    # ---- Default Train/Val/Test Split ----
    def prepare_data(self, data):
        """Creates train/val/test masks using the pipeline's seed for reproducibility."""
        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)  # uses the seeded RNG
        train_cut = int(self.train_ratio * num_nodes)
        val_cut = int((self.train_ratio + self.val_ratio) * num_nodes)

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[perm[:train_cut]] = True
        data.val_mask[perm[train_cut:val_cut]] = True
        data.test_mask[perm[val_cut:]] = True
        return data

    # ---- Standardized Seeding ----
    def _set_seed(self):
        # Use the unified seed to control torch, numpy, and Python RNGs
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    # ---- Default Methods (Overridable) ----

    def pretrain(self, model, data, epochs):
        """
        Default: skip pretraining and return model unchanged.
        Override if your model has a pretraining phase.
        """
        logger.info("No pretraining implemented; skipping.")
        return model

    def finetune_classification(self, model, data, labels, epochs):
        """
        Default: raise NotImplementedError.
        Subclasses should override with training loop.
        """
        raise NotImplementedError("finetune_classification() must be implemented by a subclass.")

    def evaluate_classification(self, model, data, labels) -> EvaluationResult:
        """
        Default: raise NotImplementedError.
        Subclasses should override with evaluation logic.
        """
        raise NotImplementedError("evaluate_classification() must be implemented by a subclass.")

    def finetune_link_prediction(self, model, data, rem_edge_list, epochs):
        """
        Default: no link prediction fine-tuning (identity).
        Override if you have a LP-specific training loop.
        """
        logger.info("No link prediction fine-tuning implemented; skipping.")
        return model

    def evaluate_link_prediction(self, model, data, rem_edge_list) -> EvaluationResult:
        """
        Default: raise NotImplementedError.
        Subclasses should override with LP evaluation.
        """
        raise NotImplementedError("evaluate_link_prediction() must be implemented by a subclass.")



from experiments.experiment_utils import EvaluationResult, sample_negative_edges


class DefaultPipeline(Pipeline):
    """
    A default pipeline implementation of the abstract Pipeline.
    Provides standard classification and link prediction logic
    that can be used with GNN-style models (e.g., SimpleGNN, SimpleGAT).
    """

    # ------------------------------
    # Fine-tune for Node Classification
    # ------------------------------
    def finetune_classification(self, model, data, labels, epochs=30,
                                lr=0.01, weight_decay=5e-4, log_every=10):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        logger.info("Fine-tuning on Node Classification")
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], labels[data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == epochs:
                val_result = self.evaluate_classification(model, data, labels, mask=data.val_mask)
                logger.info(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val Acc {val_result.accuracy:.4f}")
        return model

    # ------------------------------
    # Evaluate Classification
    # ------------------------------
    def evaluate_classification(self, model, data, labels, mask=None, verbose=False) -> EvaluationResult:
        if mask is None:
            mask = data.test_mask

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1)[mask].cpu()
            trues = labels[mask].cpu()

        acc = accuracy_score(trues, preds)
        precision = precision_score(trues, preds, average='macro', zero_division=0)
        recall = recall_score(trues, preds, average='macro', zero_division=0)
        f1 = f1_score(trues, preds, average='macro', zero_division=0)

        try:
            auc = roc_auc_score(trues, preds, multi_class='ovr', average='macro')
        except ValueError:
            auc = None

        if verbose:
            logger.info(f"Acc {acc:.4f} | Precision {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f} | AUC {auc}")

        return EvaluationResult(
            accuracy=acc, precision=precision, recall=recall,
            f1=f1, auc=auc, preds=preds
        )

    # ------------------------------
    # Fine-tune for Link Prediction
    # ------------------------------
    def finetune_link_prediction(self, model, data, rem_edge_list, epochs=30,
                                 lr=0.01, weight_decay=5e-4, log_every=10):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        bce_loss = torch.nn.BCEWithLogitsLoss()

        pos_edges = rem_edge_list[0][0]
        n = data.num_nodes

        def score(u, v):
            return (u * v).sum(dim=1)

        logger.info("Fine-tuning on Link Prediction")
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            emb = model(data.x, data.edge_index)

            neg_edges = sample_negative_edges(pos_edges, n).to(data.x.device)

            pos_scores = score(emb[pos_edges[:, 0]], emb[pos_edges[:, 1]])
            neg_scores = score(emb[neg_edges[:, 0]], emb[neg_edges[:, 1]])

            logits = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

            loss = bce_loss(logits, labels)
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == epochs:
                logger.info(f"Epoch {epoch:03d} | LP Loss {loss.item():.4f}")
        return model

    # ------------------------------
    # Evaluate Link Prediction
    # ------------------------------
    def evaluate_link_prediction(self, model, data, rem_edge_list, verbose=False) -> EvaluationResult:
        model.eval()
        with torch.no_grad():
            emb = model(data.x, data.edge_index)

        pos_edges = rem_edge_list[0][0]
        n = data.num_nodes
        neg_edges = sample_negative_edges(pos_edges, n).to(data.x.device)

        def score(u, v):
            return (u * v).sum(dim=1)

        pos_scores = score(emb[pos_edges[:, 0]], emb[pos_edges[:, 1]])
        neg_scores = score(emb[neg_edges[:, 0]], emb[neg_edges[:, 1]])

        scores = torch.cat([pos_scores, neg_scores]).cpu()
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu()
        preds = (scores > 0).float()

        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        if verbose:
            logger.info(
                f"LP Eval | Acc {acc:.4f} | Prec {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f} | AUC {auc:.4f} | AP {ap:.4f}")

        return EvaluationResult(
            accuracy=acc, precision=precision, recall=recall,
            f1=f1, auc=auc, ap=ap, preds=preds
        )

# pipeline.py (continue after DefaultPipeline)

class TransferLearningPipeline(DefaultPipeline):
    """
    A pipeline that implements transfer learning by pretraining
    the model on a separate dataset (source_data, source_labels)
    before fine-tuning on the target dataset.
    """

    def __init__(self, source_data, source_labels,
                 seed=None, device=None,
                 train_ratio=0.6, val_ratio=0.2):
        super().__init__(seed=seed, device=device,
                         train_ratio=train_ratio, val_ratio=val_ratio)
        self.source_data = source_data.to(self.device)
        self.source_labels = source_labels.to(self.device)

    def pretrain(self, model, data, epochs=50,
                 lr=0.01, weight_decay=5e-4, log_every=10):
        """
        Pretrain the model on a different dataset (self.source_data, self.source_labels).
        This is standard supervised training — NOT structural pretraining.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        logger.info("Pretraining on source dataset (transfer learning).")
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(self.source_data.x, self.source_data.edge_index)
            loss = criterion(out[self.source_data.train_mask],
                             self.source_labels[self.source_data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == epochs:
                val_result = self.evaluate_classification(
                    model, self.source_data, self.source_labels,
                    mask=self.source_data.val_mask
                )
                logger.info(f"[Pretrain Epoch {epoch:03d}] "
                            f"Loss {loss.item():.4f} | Val Acc {val_result.accuracy:.4f}")

        return model

import random
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from experiments.experiment_utils import sample_negative_edges, split_edges_for_link_prediction, EvaluationResult
from utils import get_device


# ------------------------
# Helper Functions
# ------------------------
def set_seeds(seed: int = 42):
    """Sets seeds for reproducibility across torch, numpy, and python's random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_masks(num_nodes: int, train_ratio: float = 0.6, val_ratio: float = 0.8, device=None):
    """
    Creates boolean masks for train, validation, and test splits.

    Args:
        num_nodes: Total number of nodes.
        train_ratio: Fraction for training.
        val_ratio: Fraction (cumulative) for training + validation.
        device: Device on which to create the masks.

    Returns:
        Tuple of (train_mask, val_mask, test_mask).
    """
    if device is None:
        device = get_device()
    indices = torch.randperm(num_nodes, device=device)
    train_cut = int(train_ratio * num_nodes)
    val_cut = int(val_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    train_mask[indices[:train_cut]] = True
    val_mask[indices[train_cut:val_cut]] = True
    test_mask[indices[val_cut:]] = True

    return train_mask, val_mask, test_mask


# ------------------------
# Model Initialization
# ------------------------
def init_structural_gnn(
        data,
        hidden_dim: int,
        output_dim: int,
        embedding_dim: int,
        num_layers: int,
        do_featrec: bool,
        device,
        num_classes: int = None,
        use_gate: bool = True,
        use_gat: bool = True
):
    """
    Initializes the StructuralGNN model.

    Args:
        data: A PyG data object.
        hidden_dim: Hidden dimension parameter.
        output_dim: Output dimension (for intermediate embeddings).
        embedding_dim: Final embedding dimension.
        num_layers: Number of layers.
        do_featrec: Whether to include feature reconstruction.
        device: Computation device.
        num_classes: Number of classes for classification head.
        use_gate: Whether to use the input gating mechanism.
        use_gat: Whether to use GAT final layer.

    Returns:
        model: An instance of StructuralGNN moved to device.
    """
    from models.struct_g import StructuralGNN
    model = StructuralGNN(
        num_nodes=data.num_nodes,
        edge_index=data.edge_index,
        input_dim=data.x.size(1),
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        use_gat=use_gat,
        use_gate=use_gate,
        num_classes=num_classes,
        feat_reconstruction=do_featrec
    ).to(device)
    return model


# ------------------------
# Phase 1: Pre-training Node2Vec Embeddings
# ------------------------
def pretrain_node2vec(model, node2vec_pretrain_epochs: int, batch_size: int = 128, lr: float = 0.01,
                      verbose: bool = True):
    """
    Pretrains Node2Vec embeddings.

    Args:
        model: The StructuralGNN model.
        node2vec_pretrain_epochs: Number of epochs to run Node2Vec pre-training.
        batch_size: Batch size.
        lr: Learning rate.
        verbose: Whether to print progress.

    Returns:
        model: The model after Node2Vec pre-training.
    """
    if verbose:
        print("\n=== Phase 1: Pre-training Node2Vec embeddings ===")
    model.train_node2vec(
        num_epochs=node2vec_pretrain_epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=verbose
    )
    return model


# ------------------------
# Phase 2: Full Pre-training with Self-Supervision
# ------------------------
def pretrain_full_model(
        model, classifier, data, labels, train_mask, full_pretrain_epochs: int,
        do_linkpred: bool, do_n2v_align: bool, do_featrec: bool, device, log_every: int = 10):
    """
    Pretrains the full Structural GNN using self-supervised tasks along with node classification.

    Args:
        model: The StructuralGNN model.
        classifier: The classification head (e.g., a Linear layer).
        data: The graph data object.
        labels: Node labels tensor.
        train_mask: Training mask.
        full_pretrain_epochs: Number of epochs.
        do_linkpred: Whether to include link prediction loss.
        do_n2v_align: Whether to include Node2Vec alignment loss.
        do_featrec: Whether to include feature reconstruction loss.
        device: Computation device.
        log_every: Logging frequency.

    Returns:
        Tuple (model, classifier) after pre-training.
    """
    print("\n=== Phase 2: Pre-training the Structural GNN (with self-supervision) ===")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=0.01, weight_decay=5e-4
    )

    for epoch in range(full_pretrain_epochs):
        model.train()
        classifier.train()
        optimizer.zero_grad()

        embeddings, pretrain_loss = model.forward_and_loss(
            data,
            neg_sample_size=5,
            do_node_class=True,
            do_linkpred=do_linkpred,
            do_featrec=do_featrec,
            do_n2v_align=do_n2v_align
        )
        logits = classifier(embeddings)
        cls_loss = criterion(logits[train_mask], labels.to(logits.device)[train_mask])
        total_loss = pretrain_loss + cls_loss

        total_loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == full_pretrain_epochs - 1:
            print(
                f"[Pretrain Epoch {epoch:03d}] Total Loss: {total_loss.item():.4f} | Cls: {cls_loss.item():.4f} | SSL: {pretrain_loss.item():.4f}")

    return model, classifier


def copy_model_weights(from_model, to_model):
    print("Copying weights from pre-trained model to new model...")
    to_model.load_state_dict(from_model.state_dict(), strict=False)
    return to_model


def pretrain_ssl_only(model, data, epochs, device, do_linkpred=True, do_n2v_align=True, do_featrec=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    print("\n=== Phase 2: Pretraining with Structure Only ===")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        _, loss = model.forward_and_loss(
            data,
            neg_sample_size=5,
            do_node_class=False,  # <- no labels!
            do_linkpred=do_linkpred,
            do_featrec=do_featrec,
            do_n2v_align=do_n2v_align
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch:03d}] SSL Loss: {loss.item():.4f}")

    return model


# ------------------------
# Evaluation Functions
# ------------------------
def evaluate_classification(model, data, labels, mask, device, verbose: bool = True) -> EvaluationResult:
    model.eval()
    labels = labels.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        embeddings = model(data.x.to(device), data.edge_index.to(device))
        logits = model.classify_nodes(embeddings)
        preds = logits[mask].argmax(dim=1)
        true = labels[mask]

    acc = accuracy_score(true.cpu(), preds.cpu())
    precision = precision_score(true.cpu(), preds.cpu(), average='macro', zero_division=0)
    recall = recall_score(true.cpu(), preds.cpu(), average='macro', zero_division=0)
    f1 = f1_score(true.cpu(), preds.cpu(), average='macro', zero_division=0)

    try:
        auc = roc_auc_score(true.cpu(), preds.cpu(), multi_class='ovr', average='macro')
    except ValueError:
        auc = None

    if verbose:
        print(f"  → Accuracy:  {acc:.4f}")
        print(f"  → Precision: {precision:.4f}")
        print(f"  → Recall:    {recall:.4f}")
        print(f"  → F1 Score:  {f1:.4f}")
        if auc is not None:
            print(f"  → AUC (OvR): {auc:.4f}")

    return EvaluationResult(
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        preds=preds
    )


def evaluate_link_prediction(model, data, rem_edge_list, device) -> EvaluationResult:
    """
    Evaluates link prediction performance on removed edges using dot product scoring.
    Returns an EvaluationResult object.
    """
    model.eval()
    with torch.no_grad():
        node_indices = torch.arange(data.num_nodes, device=device)
        gnn_emb = model(data.x.to(device), data.edge_index.to(device), node_indices)
        n2v_emb = model.node2vec_layer(node_indices)

    # Positive edges from removed edge list
    pos_edges = rem_edge_list[0][0].to(device)

    # Sample negative edges
    neg_edges = sample_negative_edges(pos_edges, data.num_nodes).to(device)

    def score(u, v):
        return model._pairwise_score(
            gnn_emb[u], gnn_emb[v],
            n2v_emb[u], n2v_emb[v]
        ).squeeze()

    pos_scores = score(pos_edges[:, 0], pos_edges[:, 1])
    neg_scores = score(neg_edges[:, 0], neg_edges[:, 1])
    scores = torch.cat([pos_scores, neg_scores])
    lp_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    print(
        f"\n→ Pos scores: mean={pos_scores.mean().item():.4f}, min={pos_scores.min().item():.4f}, max={pos_scores.max().item():.4f}")
    print(
        f"→ Neg scores: mean={neg_scores.mean().item():.4f}, min={neg_scores.min().item():.4f}, max={neg_scores.max().item():.4f}")
    print(
        f"→ Sigmoid scores range: [{torch.sigmoid(scores).min().item():.4f}, {torch.sigmoid(scores).max().item():.4f}]")

    preds = (torch.sigmoid(scores) > 0.5).float()

    # Compute metrics
    acc = accuracy_score(lp_labels.cpu(), preds.cpu())
    precision = precision_score(lp_labels.cpu(), preds.cpu(), zero_division=0)
    recall = recall_score(lp_labels.cpu(), preds.cpu(), zero_division=0)
    f1 = f1_score(lp_labels.cpu(), preds.cpu(), zero_division=0)
    auc = roc_auc_score(lp_labels.cpu().detach(), scores.cpu().detach())
    ap = average_precision_score(lp_labels.cpu().detach(), scores.cpu().detach())

    print("\n=== Link Prediction Evaluation ===")
    print(f"  → Accuracy:  {acc:.4f}")
    print(f"  → Precision: {precision:.4f}")
    print(f"  → Recall:    {recall:.4f}")
    print(f"  → F1 Score:  {f1:.4f}")
    print(f"  → AUC:       {auc:.4f}")
    print(f"  → AP:        {ap:.4f}")

    return EvaluationResult(
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        ap=ap,
        preds=preds
    )


# ------------------------
# Phase 3: Fine-tuning for Node Classification
# ------------------------
def finetune_classification(model, data, labels, train_mask, finetune_epochs: int, device,
                            log_every: int = 10):
    """
    Fine-tunes the StructuralGNN's internal classification head.
    """
    assert model.num_classes is not None, "Model must be re-initialized with `num_classes` for classification."

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n=== Phase 3: Fine-tuning for Node Classification ===")
    for epoch in range(finetune_epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(data.x.to(device), data.edge_index.to(device))
        logits = model.classify_nodes(embeddings)
        loss = criterion(logits[train_mask], labels.to(device)[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == finetune_epochs - 1:
            print(f"[Fine-tune Epoch {epoch:03d}] Loss: {loss.item():.4f}")
            _ = evaluate_classification(model, data, labels, data.val_mask, device, verbose=True)

    return model


# Fine Tune Link Prediction

def finetune_link_prediction(
        model,
        data,
        rem_edge_list,
        finetune_epochs: int,
        neg_sample_size: int = 5,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        device=None,
        log_every: int = 10
):
    """
    Fine-tunes StructuralGNN for link prediction using supervised link supervision.

    Args:
        model: The StructuralGNN model.
        data: A PyG data object.
        rem_edge_list: Held-out edge list from split_edges_for_link_prediction.
        finetune_epochs: Number of fine-tuning epochs.
        neg_sample_size: Number of in-batch negatives per positive.
        lr: Learning rate.
        weight_decay: Weight decay for Adam optimizer.
        device: Device to run on.
        log_every: Print frequency.

    Returns:
        model: Fine-tuned model.
    """
    print("\n=== Phase 3: Fine-tuning for Link Prediction ===")

    if device is None:
        device = get_device()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    node_indices = torch.arange(data.num_nodes, device=device)
    data.edge_index = data.edge_index.to(device)
    data.x = data.x.to(device)

    for epoch in range(finetune_epochs):
        model.train()
        optimizer.zero_grad()

        # Get updated embeddings
        embeddings = model(data.x, data.edge_index, node_indices)

        # Supervised link prediction loss on held-out edges
        loss = model.link_prediction_loss(embeddings, rem_edge_list[0][0].T.to(device), neg_sample_size=neg_sample_size)

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == finetune_epochs - 1:
            print(f"[Fine-tune LP Epoch {epoch:03d}] Loss: {loss.item():.4f}")
            _ = evaluate_link_prediction(model, data, rem_edge_list, device)

    return model


# ------------------------
# Main Pipeline Function
# ------------------------
def run_structg_pipeline(
        data,
        labels,
        hidden_dim: int = 64,
        output_dim: int = 32,
        embedding_dim: int = 128,
        num_layers: int = 2,
        pretrain_epochs: int = 100,
        finetune_epochs: int = 30,
        do_linkpred: bool = True,
        do_n2v_align: bool = False,
        do_featrec: bool = True,
        use_gate: bool = True,  # <-- NEW
        use_gat: bool = True,  # <-- NEW
        seed: int = 42,
        num_classes: int = None,
):
    from experiments.experiment_utils import set_global_seed

    set_global_seed(seed)
    device = get_device()
    print(f"Using device: {device} | Seed: {seed}")

    # Mask creation
    num_nodes = data.num_nodes
    train_mask, val_mask, test_mask = create_masks(num_nodes, device=device)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Edge split
    data.edge_index, rem_edge_list = split_edges_for_link_prediction(data.edge_index, removal_ratio=0.3)

    # Model with classification head
    num_classes = labels.unique().numel() if num_classes is None else num_classes
    model = init_structural_gnn(
        data,
        hidden_dim,
        output_dim,
        embedding_dim,
        num_layers,
        do_featrec,
        device,
        num_classes=num_classes,
        use_gate=use_gate,
        use_gat=use_gat
    )

    # === Phase 1: Pretrain Node2Vec ===
    pretrain_start_time = time.time()
    model = pretrain_node2vec(model, node2vec_pretrain_epochs=pretrain_epochs, batch_size=128, lr=0.01, verbose=True)
    pretrain_time = time.time() - pretrain_start_time

    # === Phase 2: Fine-tuning for Classification ===
    fine_tune_start_time = time.time()
    model = finetune_classification(model, data, labels, train_mask, finetune_epochs, device, log_every=10)
    finetune_time = time.time() - fine_tune_start_time

    # === Evaluation ===
    classifier_evaluation_start_time = time.time()
    classifier_results = evaluate_classification(model, data, labels, test_mask, device, verbose=True)
    classifier_evaluation_time = time.time() - classifier_evaluation_start_time

    # === Optional Link Prediction Fine-tune ===
    if do_linkpred:
        print("\n=== Fine-tuning and evaluating for link prediction ===")
        link_pred_start_time = time.time()
        model = finetune_link_prediction(model, data, rem_edge_list, finetune_epochs, device=device)
        link_pred_time = time.time() - link_pred_start_time

        lp_evaluation_start_time = time.time()
        lp_results = evaluate_link_prediction(model, data, rem_edge_list, device)
        lp_evaluation_time = time.time() - lp_evaluation_start_time
    else:
        lp_results = None
        lp_evaluation_time = 0
        link_pred_time = 0

    # Total Time
    total_time = time.time() - fine_tune_start_time - classifier_evaluation_time - lp_evaluation_time
    print(f"\n→ Total Training Time (excluding eval): {total_time:.2f} seconds")

    # Metadata
    classifier_results.metadata.update({
        "seed": seed,
        "pretrain_time": pretrain_time,
        "classifier_time": finetune_time,
        "total_time": total_time,
        "device": str(device),
        "model": "Struct-G",
        "using_internal_classifier": True
    })

    if lp_results:
        lp_results.metadata.update({
            "seed": seed,
            "pretrain_time": pretrain_time,
            "link_pred_time": link_pred_time,
            "total_time": total_time,
            "device": str(device),
            "model": "Struct-G",
            "using_internal_classifier": True
        })

    return model, classifier_results, lp_results


