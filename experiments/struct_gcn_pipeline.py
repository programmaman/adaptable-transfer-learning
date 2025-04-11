import torch
import torch.nn.functional as f
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, classification_report

from experiments.experiment_utils import EvaluationResult
from models.structural_gcn import (
    StructuralGcn,
    GnnClassifierHead,
    train_structural_feature_predictor,
    fine_tune_model
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


def evaluate_link_prediction(model, data, num_samples=1000, device='cpu') -> EvaluationResult:
    """
    Evaluates link prediction performance using dot product scores between positive and negative edges.
    Returns an EvaluationResult object.
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        emb = model(data.x, data.edge_index)

    if emb.dim() == 1:
        emb = emb.unsqueeze(-1)

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # Sample positive edges
    pos_idx = torch.randperm(edge_index.size(1))[:num_samples]
    pos_src, pos_dst = edge_index[0, pos_idx], edge_index[1, pos_idx]

    # Sample negative edges
    neg_src = torch.randint(0, num_nodes, (num_samples,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_samples,), device=device)

    def dot_score(u, v): return (u * v).sum(dim=1)

    pos_score = dot_score(emb[pos_src], emb[pos_dst])
    neg_score = dot_score(emb[neg_src], emb[neg_dst])

    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([
        torch.ones_like(pos_score),
        torch.zeros_like(neg_score)
    ])

    preds = (scores > 0).float()

    # Compute metrics
    acc = accuracy_score(labels.cpu(), preds.cpu())
    precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
    recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
    f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)
    auc = roc_auc_score(labels.cpu(), scores.cpu())
    ap = average_precision_score(labels.cpu(), scores.cpu())

    print(f"\n=== Link Prediction ===")
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


def run_structural_gcn_pipeline(data, labels, hidden_dim=64, mid_dim=32, pretrain_epochs=100, finetune_epochs=50):
    device = get_device()

    # Print Device
    print(f"Using device: {device}")

    # Create train/val/test split
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

    # === Pretraining Phase ===
    in_dim = data.x.size(1)
    pretrain_model = StructuralGcn(in_channels=in_dim, hidden_channels=hidden_dim, mid_channels=mid_dim)
    pretrain_model = train_structural_feature_predictor(pretrain_model, data, epochs=pretrain_epochs, device=device)

    # === Fine-tuning Phase ===
    classifier_model = GnnClassifierHead(pretrained_model=pretrain_model, out_channels=len(labels.unique()))
    classifier_model = fine_tune_model(
        classifier_model,
        data,
        task_labels=labels,
        epochs=finetune_epochs,
        device=device
    )

    # === Evaluation ===
    results = evaluate_model(classifier_model, data, labels, device=device)
    lp_results = evaluate_link_prediction(classifier_model, data, device=device)
    print(f"\nFinal Evaluation — Acc: {results.accuracy:.4f}")

    return classifier_model, results, lp_results
