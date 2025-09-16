# graph_bert_pipeline.py (replace your run_graphbert_pipeline with this version)

import math
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
from models.graph_bert import GraphBERT


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


def _iter_batches(index_tensor: torch.Tensor, batch_size: int):
    n = index_tensor.numel()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield index_tensor[start:end]


def _forward_nodes_as_len1_seqs(model, x_batch, device, role_id_value=0,
                                return_token_repr=False):
    """
    x_batch: [B, F] -> model expects [B, 1, F]
    Builds matching mask/role_ids with S=1 and calls model.
    """
    B, F = x_batch.shape
    x = x_batch.unsqueeze(1).to(device)                     # [B,1,F]
    mask = torch.ones((B, 1), dtype=torch.bool, device=device)
    role_ids = torch.full((B, 1), int(role_id_value), dtype=torch.long, device=device)

    out = model(
        x=x,
        mask=mask,
        role_ids=role_ids,
        attn_bias=None,
        anchor_pos=None,
        center_idx=None,
        return_token_repr=return_token_repr
    )
    return out  # logits [B,C] or dict with logits/token_repr/pooled


def _evaluate_split(model, data, labels, idx_split, device, batch_size=1024):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for idx_batch in _iter_batches(idx_split, batch_size):
            xb = data.x[idx_batch]  # [B,F]
            logits = _forward_nodes_as_len1_seqs(model, xb, device)
            logits = logits.squeeze(1) if logits.dim() == 3 else logits  # [B,C]
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_true.append(labels[idx_batch].cpu())
    preds = torch.cat(all_preds)
    true = torch.cat(all_true)

    acc = accuracy_score(true, preds)
    precision = precision_score(true, preds, average='macro', zero_division=0)
    recall = recall_score(true, preds, average='macro', zero_division=0)
    f1 = f1_score(true, preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(true, preds, multi_class='ovr', average='macro')
    except ValueError:
        auc = None
    return EvaluationResult(accuracy=acc, precision=precision, recall=recall, f1=f1, auc=auc, preds=preds)


def _get_all_node_embeddings(model, data, device, batch_size=2048):
    """
    Returns a dense tensor [N, d] of pooled embeddings for each node, computed in batches with S=1.
    This avoids O(N^2) attention memory.
    """
    model.eval()
    N, F = data.x.shape
    embs = torch.empty((N, model.d_model), dtype=torch.float32, device=device)
    with torch.no_grad():
        for idx_batch in _iter_batches(torch.arange(N, device=device), batch_size):
            xb = data.x[idx_batch]  # [B,F]
            out = _forward_nodes_as_len1_seqs(model, xb, device, return_token_repr=True)
            # Using the pooled representation (equivalent to the token since S=1)
            pooled = out["pooled"]  # [B,d]
            embs[idx_batch] = pooled
    return embs  # [N,d]


def _evaluate_link_prediction_batched(model, data, rem_edge_list, device, batch_size=2048):
    # Build node embeddings safely (no N^2 attention)
    token_repr = _get_all_node_embeddings(model, data, device, batch_size=batch_size)  # [N,d]

    # Positive edges
    pos_edges = rem_edge_list[0][0].to(device)          # [E_pos, 2]
    # Negatives sampled w.r.t. graph size
    neg_edges = sample_negative_edges(pos_edges, data.x.shape[0]).to(device)

    def score_pairs(edges):
        u, v = edges[:, 0], edges[:, 1]
        sims = (token_repr[u] * token_repr[v]).sum(dim=-1)  # dot product
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
    seed=42,
    batch_size=1024,            # NEW: mini-batch size over nodes
    role_id_value=0             # NEW: constant role id bucket unless you have real buckets
):
    """
    Train + evaluate GraphBERT for node classification (batched, S=1) and optional link prediction.
    """
    set_global_seed(seed)
    device = get_device()
    print(f"Using device: {device} | Seed: {seed}")

    assert hasattr(data, "x"), "data.x (node feature matrix) required"
    assert data.x.dim() == 2, f"Expected data.x 2-D [N,F], got {data.x.shape}"
    num_nodes, feat_dim = data.x.shape

    # Masks
    train_mask, val_mask, test_mask = create_masks(num_nodes, device=device)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    # For LP, remove a subset of edges
    data.edge_index, rem_edge_list = split_edges_for_link_prediction(
        data.edge_index, removal_ratio=0.3
    )

    num_classes = labels.unique().numel()

    # Model
    model = GraphBERT(
        feat_dim=feat_dim,
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

    # ----- Training (batched over nodes; each node as a len-1 sequence) -----
    print("\n=== Training GraphBERT (batched nodes; S=1) ===")
    model.train()
    for epoch in range(epochs):
        # iterate only over training nodes
        train_idx = torch.nonzero(train_mask, as_tuple=False).flatten()
        # shuffle each epoch
        perm = train_idx[torch.randperm(train_idx.numel(), device=train_idx.device)]
        epoch_loss = 0.0
        num_seen = 0

        for idx_batch in _iter_batches(perm, batch_size):
            optimizer.zero_grad()
            xb = data.x[idx_batch]                      # [B,F]
            logits = _forward_nodes_as_len1_seqs(model, xb, device)  # [B,C]
            logits = logits.squeeze(1) if logits.dim() == 3 else logits
            loss = criterion(logits, labels[idx_batch].to(device))
            loss.backward()
            optimizer.step()

            batch_size_actual = idx_batch.numel()
            epoch_loss += loss.item() * batch_size_actual
            num_seen += batch_size_actual

        epoch_loss /= max(1, num_seen)

        if epoch % 10 == 0 or epoch == epochs - 1:
            val_idx = torch.nonzero(val_mask, as_tuple=False).flatten()
            val_res = _evaluate_split(model, data, labels, val_idx, device, batch_size=batch_size)
            print(f"[Epoch {epoch:03d}] Loss: {epoch_loss:.4f}  |  Val Acc: {val_res.accuracy:.4f}")

    # ----- Final classification evaluation -----
    test_idx = torch.nonzero(test_mask, as_tuple=False).flatten()
    classifier_results = _evaluate_split(model, data, labels, test_idx, device, batch_size=batch_size)

    # ----- Link prediction (uses safe batched embeddings) -----
    if do_linkpred:
        lp_results = _evaluate_link_prediction_batched(
            model, data, rem_edge_list, device, batch_size=max(1024, batch_size)
        )
    else:
        lp_results = None

    return model, classifier_results, lp_results
