import math
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
from models.graph_bert import GraphBERT


def create_masks(num_nodes: int, train_ratio=0.6, val_ratio=0.8, device=None):
    print(f"[create_masks] Creating masks for {num_nodes} nodes "
          f"(train_ratio={train_ratio}, val_ratio={val_ratio})")
    if device is None:
        device = get_device()
        print(f"[create_masks] Using device from get_device(): {device}")

    indices = torch.randperm(num_nodes, device=device)
    print("[create_masks] Shuffled indices created")

    train_cut = int(train_ratio * num_nodes)
    val_cut = int(val_ratio * num_nodes)
    print(f"[create_masks] train_cut={train_cut}, val_cut={val_cut}")

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    train_mask[indices[:train_cut]] = True
    val_mask[indices[train_cut:val_cut]] = True
    test_mask[indices[val_cut:]] = True

    print(f"[create_masks] Done. "
          f"Train nodes={train_mask.sum().item()}, "
          f"Val nodes={val_mask.sum().item()}, "
          f"Test nodes={test_mask.sum().item()}")
    return train_mask, val_mask, test_mask


def _iter_batches(index_tensor: torch.Tensor, batch_size: int):
    n = index_tensor.numel()
    print(f"[_iter_batches] Splitting {n} indices into batches of {batch_size}")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if start % (batch_size * 50) == 0:  # print every ~50 batches
            print(f"[_iter_batches] Yielding batch indices {start}:{end}")
        yield index_tensor[start:end]
    print("[_iter_batches] All batches yielded")


def _forward_nodes_as_len1_seqs(model, x_batch, device, role_id_value=0,
                                return_token_repr=False):
    B, F = x_batch.shape
    x = x_batch.unsqueeze(1).to(device)
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
    return out



def _evaluate_split(model, data, labels, idx_split, device, batch_size=256):
    print(f"[Eval] Starting evaluation on {idx_split.numel()} nodes (batch_size={batch_size})")
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for i, idx_batch in enumerate(_iter_batches(idx_split, batch_size)):
            if i % 20 == 0:  # print progress every 20 batches
                start_node = idx_batch[0].item()
                end_node = idx_batch[-1].item()
            xb = data.x[idx_batch]
            logits = _forward_nodes_as_len1_seqs(model, xb, device)
            logits = logits.squeeze(1) if logits.dim() == 3 else logits
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_true.append(labels[idx_batch].cpu())

    print("[Eval] Concatenating predictions and labels...")
    preds = torch.cat(all_preds)
    true = torch.cat(all_true)

    print("[Eval] Computing metrics...")
    acc = accuracy_score(true, preds)
    precision = precision_score(true, preds, average='macro', zero_division=0)
    recall = recall_score(true, preds, average='macro', zero_division=0)
    f1 = f1_score(true, preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(true, preds, multi_class='ovr', average='macro')
    except ValueError:
        auc = None

    print(f"[Eval] Done. Acc={acc:.4f}, F1={f1:.4f}, AUC={'N/A' if auc is None else f'{auc:.4f}'}")
    return EvaluationResult(accuracy=acc, precision=precision, recall=recall, f1=f1, auc=auc, preds=preds)

def _get_all_node_embeddings(model, data, device, batch_size=256):
    print(f"[Embeddings] Start extracting embeddings for {data.x.shape[0]} nodes")
    model.eval()
    N, F = data.x.shape
    embs = torch.empty((N, model.d_model), dtype=torch.float32, device=device)
    with torch.no_grad():
        for i, idx_batch in enumerate(_iter_batches(torch.arange(N, device=device), batch_size)):
            if i % 50 == 0:
                print(f"[Embeddings] Batch {i}, nodes {idx_batch[0].item()}–{idx_batch[-1].item()}")
            xb = data.x[idx_batch]
            out = _forward_nodes_as_len1_seqs(model, xb, device, return_token_repr=True)
            pooled = out["pooled"]
            embs[idx_batch] = pooled
    print("[Embeddings] Done")
    return embs


def _evaluate_link_prediction_batched(model, data, rem_edge_list, device, batch_size=256):
    print("[LinkPred] Start")
    token_repr = _get_all_node_embeddings(model, data, device, batch_size=batch_size)
    print("[LinkPred] Got embeddings")

    pos_edges = rem_edge_list[0][0].to(device)
    print(f"[LinkPred] Positive edges: {pos_edges.shape}")

    print("[LinkPred] Sampling negative edges...")
    neg_edges = sample_negative_edges(pos_edges, data.x.shape[0]).to(device)
    print(f"[LinkPred] Negative edges: {neg_edges.shape}")

    def score_pairs(edges):
        u, v = edges[:, 0], edges[:, 1]
        sims = (token_repr[u] * token_repr[v]).sum(dim=-1)
        return sims

    print("[LinkPred] Scoring positive edges...")
    pos_scores = score_pairs(pos_edges)
    print("[LinkPred] Scoring negative edges...")
    neg_scores = score_pairs(neg_edges)

    print("[LinkPred] Concatenating scores and computing metrics...")
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
    print("[LinkPred] Done")
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
    lr=0.01,
    weight_decay=5e-4,
    epochs=30,
    do_linkpred=True,
    seed=42,
    batch_size=256,
    role_id_value=0
):
    print("[Pipeline] Starting GraphBERT pipeline")
    set_global_seed(seed)
    device = get_device()
    print(f"[Pipeline] Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_start = time.perf_counter()

    num_nodes, feat_dim = data.x.shape
    print(f"[Pipeline] Data: {num_nodes} nodes, {feat_dim} features")

    print("[Pipeline] Creating train/val/test masks...")
    train_mask, val_mask, test_mask = create_masks(num_nodes, device=device)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    print("[Pipeline] Moving features and labels to device...")
    data.x = data.x.to(device)
    labels = labels.to(device)

    print("[Pipeline] Splitting edges for link prediction...")
    data.edge_index, rem_edge_list = split_edges_for_link_prediction(
        data.edge_index, removal_ratio=0.3
    )
    print("[Pipeline] Edge split complete")

    num_classes = labels.unique().numel()
    print(f"[Pipeline] Number of classes: {num_classes}")

    print("[Pipeline] Initializing GraphBERT model...")
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
    print("[Pipeline] Optimizer and loss ready")

    print("\n=== Training GraphBERT (batched nodes; S=1) ===")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_start = time.perf_counter()

    model.train()
    for epoch in range(epochs):
        train_idx = torch.nonzero(train_mask, as_tuple=False).flatten()
        perm = train_idx[torch.randperm(train_idx.numel(), device=train_idx.device)]
        epoch_loss = 0.0
        num_seen = 0

        for b, idx_batch in enumerate(_iter_batches(perm, batch_size)):
            if b % 50 == 0:
                print(f"[Train] Epoch {epoch}, batch {b}, size {idx_batch.numel()}")

            optimizer.zero_grad()
            xb = data.x[idx_batch]
            logits = _forward_nodes_as_len1_seqs(model, xb, device)
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
            print(f"[Epoch {epoch:03d}] Loss: {epoch_loss:.4f} | Val Acc: {val_res.accuracy:.4f}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time = time.perf_counter() - train_start
    print(f"[Pipeline] Training complete in {train_time:.2f}s")

    print("[Pipeline] Running classifier evaluation on test split...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cls_eval_start = time.perf_counter()
    test_idx = torch.nonzero(test_mask, as_tuple=False).flatten()
    classifier_results = _evaluate_split(model, data, labels, test_idx, device, batch_size=batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    classifier_eval_time = time.perf_counter() - cls_eval_start
    print(f"[Pipeline] Classifier evaluation done in {classifier_eval_time:.2f}s")

    lp_results, lp_eval_time, link_pred_time = None, 0.0, 0.0
    if do_linkpred:
        print("[Pipeline] Running link prediction evaluation...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lp_eval_start = time.perf_counter()
        lp_results = _evaluate_link_prediction_batched(
            model, data, rem_edge_list, device, batch_size=max(256, batch_size)
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lp_eval_time = time.perf_counter() - lp_eval_start
        print(f"[Pipeline] Link prediction evaluation done in {lp_eval_time:.2f}s")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - total_start
    print(f"[Pipeline] Finished. Total runtime: {total_time:.2f}s")

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Pipeline] Parameters: total={num_params}, trainable={num_trainable}")

    classifier_results.metadata.update({
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "device": str(device),
        "model": "GraphBERT",
        "train_time": train_time,
        "classifier_eval_time": classifier_eval_time,
        "link_pred_time": link_pred_time,
        "lp_eval_time": lp_eval_time,
        "total_time": total_time,
        "num_parameters": int(num_params),
        "num_trainable_parameters": int(num_trainable)
    })
    if lp_results:
        lp_results.metadata.update(classifier_results.metadata)

    return model, classifier_results, lp_results

