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
    model = pretrain_node2vec(model, node2vec_pretrain_epochs=pretrain_epochs, batch_size=128, lr=0.01, verbose=True)

    # === Phase 2: Fine-tuning for Classification ===
    start_time = time.time()
    model = finetune_classification(model, data, labels, train_mask, finetune_epochs, device, log_every=10)

    # === Evaluation ===
    classifier_evaluation_start_time = time.time()
    classifier_results = evaluate_classification(model, data, labels, test_mask, device, verbose=True)
    classifier_evaluation_time = time.time() - classifier_evaluation_start_time

    # === Optional Link Prediction Fine-tune ===
    if do_linkpred:
        print("\n=== Fine-tuning and evaluating for link prediction ===")
        model = finetune_link_prediction(model, data, rem_edge_list, finetune_epochs, device=device)

        lp_evaluation_start_time = time.time()
        lp_results = evaluate_link_prediction(model, data, rem_edge_list, device)
        lp_evaluation_time = time.time() - lp_evaluation_start_time
    else:
        lp_results = None
        lp_evaluation_time = 0

    # Total Time
    total_time = time.time() - start_time - classifier_evaluation_time - lp_evaluation_time
    print(f"\n→ Total Training Time (excluding eval): {total_time:.2f} seconds")

    # Metadata
    classifier_results.metadata.update({
        "seed": seed,
        "train_time": total_time,
        "device": str(device),
        "model": "StructuralGNN"
    })
    if lp_results:
        lp_results.metadata.update({
            "seed": seed,
            "train_time": total_time,
            "device": str(device),
            "model": "StructuralGNN"
        })

    return model, classifier_results, lp_results
