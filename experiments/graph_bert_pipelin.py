import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from experiments.experiment_utils import (
    split_edges_for_link_prediction,
    sample_negative_edges,
    EvaluationResult,
    set_global_seed,
)
from utils import get_device
from graph_bert import GraphBERT


def create_masks(num_nodes: int, train_ratio=0.6, val_ratio=0.8, device=None):
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


def evaluate_classification(model, data_inputs, labels, mask, device):
    model.eval()
    with torch.no_grad():
        logits = model(**data_inputs)
        preds = logits[mask].argmax(dim=-1)
        true = labels[mask]

    acc = accuracy_score(true.cpu(), preds.cpu())
    precision = precision_score(true.cpu(), preds.cpu(), average='macro', zero_division=0)
    recall = recall_score(true.cpu(), preds.cpu(), average='macro', zero_division=0)
    f1 = f1_score(true.cpu(), preds.cpu(), average='macro', zero_division=0)
    try:
        auc = roc_auc_score(true.cpu(), preds.cpu(), multi_class='ovr', average='macro')
    except ValueError:
        auc = None

    return EvaluationResult(
        accuracy=acc, precision=precision, recall=recall, f1=f1, auc=auc, preds=preds
    )


def evaluate_link_prediction(model, data_inputs, rem_edge_list, device):
    model.eval()
    with torch.no_grad():
        outputs = model(**data_inputs, return_token_repr=True)
        token_repr = outputs["token_repr"]

    # Positive edges
    pos_edges = rem_edge_list[0][0].to(device)
    neg_edges = sample_negative_edges(pos_edges, data_inputs["x"].shape[0]).to(device)

    def score_pairs(edges):
        u, v = edges[:, 0], edges[:, 1]
        sims = (token_repr[u] * token_repr[v]).sum(dim=-1)  # dot product similarity
        return sims

    pos_scores = score_pairs(pos_edges)
    neg_scores = score_pairs(neg_edges)
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    probs = torch.sigmoid(scores)

    preds = (probs > 0.5).float()

    acc = accuracy_score(labels.cpu(), preds.cpu())
    precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
    recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
    f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)
    auc = roc_auc_score(labels.cpu(), scores.cpu())
    ap = average_precision_score(labels.cpu(), scores.cpu())

    return EvaluationResult(
        accuracy=acc, precision=precision, recall=recall, f1=f1, auc=auc, ap=ap, preds=preds
    )


def run_graphbert_pipeline(
    data,
    labels,
    d_model=128,
    n_heads=4,
    n_layers=3,
    dropout=0.1,
    lr=0.001,
    weight_decay=5e-4,
    epochs=50,
    do_linkpred=True,
    seed=42
):
    """
    Train + evaluate GraphBERT for classification and optional link prediction.
    """
    set_global_seed(seed)
    device = get_device()
    print(f"Using device: {device} | Seed: {seed}")

    num_nodes = data.num_nodes
    train_mask, val_mask, test_mask = create_masks(num_nodes, device=device)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    data.edge_index, rem_edge_list = split_edges_for_link_prediction(data.edge_index, removal_ratio=0.3)

    num_classes = labels.unique().numel()

    # Prepare GraphBERT
    model = GraphBERT(
        feat_dim=data.x.size(1),
        num_classes=num_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        num_role_buckets=16,
        num_dist_buckets=6,
        use_anchor_pos=False
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    data_inputs = {
        "x": data.x.to(device),
        "mask": torch.ones(num_nodes, dtype=torch.bool, device=device),
        "role_ids": torch.zeros((num_nodes,), dtype=torch.long, device=device),  # placeholder
        "attn_bias": None,
        "anchor_pos": None,
        "center_idx": None
    }

    print("\n=== Training GraphBERT ===")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(**data_inputs)
        loss = criterion(logits[train_mask], labels[train_mask].to(device))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch:03d}] Loss: {loss.item():.4f}")
            eval_res = evaluate_classification(model, data_inputs, labels.to(device), val_mask, device)
            print(f" → Val Acc: {eval_res.accuracy:.4f}")

    classifier_results = evaluate_classification(model, data_inputs, labels.to(device), test_mask, device)

    if do_linkpred:
        lp_results = evaluate_link_prediction(model, data_inputs, rem_edge_list, device)
    else:
        lp_results = None

    return model, classifier_results, lp_results
