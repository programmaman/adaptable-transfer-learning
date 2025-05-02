import torch
import torch.nn.functional as f
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, classification_report
from experiments.experiment_utils import split_edges_for_link_prediction, sample_negative_edges

from experiments.experiment_utils import EvaluationResult
from models.deep_gcn import (
    StructuralGcn,
    GnnClassifierHead,
    train_structural_feature_predictor,
    fine_tune_model,
    fine_tune_link_prediction
)
from utils import get_device


def evaluate_model(model, data, labels, device, verbose=True) -> EvaluationResult:
    """
    General-purpose model evaluation for node classification.
    Returns an EvaluationResult object.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        probs = f.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        true = labels.cpu()
        pred = preds.cpu()
        prob = probs.cpu()

        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, average='macro', zero_division=0)
        recall = recall_score(true, pred, average='macro', zero_division=0)
        f1 = f1_score(true, pred, average='macro', zero_division=0)

        try:
            auc = roc_auc_score(true, prob, multi_class='ovr', average='macro')
        except ValueError:
            auc = None

        if verbose:
            print(f"\n=== Node Classification ===")
            print(f"  → Accuracy:  {acc:.4f}")
            print(f"  → Precision: {precision:.4f}")
            print(f"  → Recall:    {recall:.4f}")
            print(f"  → F1 Score:  {f1:.4f}")
            if auc is not None:
                print(f"  → AUC (OvR): {auc:.4f}")
            print("  → Classification Report:")
            print(classification_report(true, pred, digits=4))

        return EvaluationResult(
            accuracy=acc,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
            preds=pred
        )


def evaluate_link_prediction(model, data, rem_edge_list, device='cpu') -> EvaluationResult:
    """
    Evaluates link prediction performance using only held-out edges from rem_edge_list (fair evaluation).
    Returns an EvaluationResult object.
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        emb = model(data.x, data.edge_index)

    if emb.dim() == 1:
        emb = emb.unsqueeze(-1)

    # Use held-out edges for positive samples
    pos_edges = rem_edge_list[0][0].to(device)
    n = data.num_nodes

    # Sample negative edges matching the count and shape
    neg_edges = sample_negative_edges(pos_edges, n).to(device)

    def dot_score(u, v):
        return (u * v).sum(dim=1)

    pos_scores = dot_score(emb[pos_edges[:, 0]], emb[pos_edges[:, 1]])
    neg_scores = dot_score(emb[neg_edges[:, 0]], emb[neg_edges[:, 1]])

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

    preds = (scores > 0).float()

    # Compute metrics
    acc = accuracy_score(labels.cpu(), preds.cpu())
    precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
    recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
    f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)
    auc = roc_auc_score(labels.cpu(), scores.cpu())
    ap = average_precision_score(labels.cpu(), scores.cpu())

    print(f"\n=== GCN Link Prediction (Fair Evaluation) ===")
    print(f"  → Accuracy:       {acc:.4f}")
    print(f"  → Precision:      {precision:.4f}")
    print(f"  → Recall:         {recall:.4f}")
    print(f"  → F1 Score:       {f1:.4f}")
    print(f"  → ROC-AUC:        {auc:.4f}")
    print(f"  → Avg Precision:  {ap:.4f}")

    return EvaluationResult(
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        ap=ap,
        preds=preds
    )


def run_structural_gcn_pipeline(data, labels, hidden_dim=64, mid_dim=32, pretrain_epochs=100, finetune_epochs=30, seed=None):
    from experiments.experiment_utils import set_global_seed
    import time

    if seed is not None:
        set_global_seed(seed)

    start_time = time.time()
    device = get_device()
    print(f"Using device: {device} | Seed: {seed}")

    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_ratio, val_ratio = 0.6, 0.2
    train_cut = int(train_ratio * num_nodes)
    val_cut = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_cut]] = True
    val_mask[indices[train_cut:val_cut]] = True
    test_mask[indices[val_cut:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    in_dim = data.x.size(1)
    pretrain_start_time = time.time()
    pretrain_model = StructuralGcn(in_channels=in_dim, hidden_channels=hidden_dim, mid_channels=mid_dim)
    pretrain_model = train_structural_feature_predictor(pretrain_model, data, epochs=pretrain_epochs, device=device)
    pretrain_time = time.time() - pretrain_start_time

    classifier_model = GnnClassifierHead(pretrained_model=pretrain_model, out_channels=len(labels.unique()))

    classifier_start_time = time.time()
    classifier_model = fine_tune_model(
        classifier_model,
        data,
        task_labels=labels,
        epochs=finetune_epochs,
        device=device
    )
    classifier_time = time.time() - classifier_start_time

    classifer_eval_start_time = time.time()
    classification_results = evaluate_model(classifier_model, data, labels, device=device)
    classifer_eval_time = time.time() - classifer_eval_start_time

    original_edges = data.edge_index
    data.edge_index, rem_edge_list = split_edges_for_link_prediction(original_edges)
    link_pred_start_time = time.time()
    classifier_model = fine_tune_link_prediction(
        classifier_model, data, rem_edge_list=rem_edge_list, epochs=finetune_epochs, device=device
    )
    link_pred_time = time.time() - link_pred_start_time
    lp_eval_start_time = time.time()
    lp_results = evaluate_link_prediction(classifier_model, data, rem_edge_list, device=device)
    lp_eval_time = time.time() - lp_eval_start_time

    runtime = time.time() - start_time - classifer_eval_time - lp_eval_time
    print(f"\nTotal Runtime: {runtime:.2f} seconds")

    classification_results.metadata.update({
        "seed": seed,
        "classifier_time": classifier_time,
        "pretrain_time": pretrain_time,
        "link_pred_time": link_pred_time,
        "device": str(device),
        "model": "StructuralGCN"
    })

    lp_results.metadata.update({
        "seed": seed,
        "classifier_time": classifier_time,
        "pretrain_time": pretrain_time,
        "link_pred_time": link_pred_time,
        "device": str(device),
        "model": "StructuralGCN"
    })

    return classifier_model, classification_results, lp_results

