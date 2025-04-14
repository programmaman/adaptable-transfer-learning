import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from experiments.experiment_utils import EvaluationResult
from models.baselines import SimpleGraphSAGE
from utils import get_device


def prepare_data(data, labels, train_ratio=0.6, val_ratio=0.2, seed=None):
    """
    Moves data and labels to device, splits nodes into train/val/test masks,
    and attaches masks to the data object.
    """
    device = get_device()
    data = data.to(device)
    labels = labels.to(device)
    if seed is not None:
        torch.manual_seed(seed)

    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes, device=device)
    train_cut = int(train_ratio * num_nodes)
    val_cut = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[perm[:train_cut]] = True
    val_mask[perm[train_cut:val_cut]] = True
    test_mask[perm[val_cut:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data, labels, device


def initialize_models(in_dim, num_classes, device):
    """
    Creates two GraphSAGE models: one for regression pretraining,
    one for classification fine-tuning, and moves them to device.
    """
    pre_model = SimpleGraphSAGE(in_channels=in_dim, out_channels=1).to(device)
    class_model = SimpleGraphSAGE(in_channels=in_dim, out_channels=num_classes).to(device)
    return pre_model, class_model


def pretrain(model, data, epochs=100, lr=0.01, weight_decay=5e-4, log_every=10):
    """
    Pretrains the model on structural regression targets.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    print("\n=== Pretraining on Structural Regression ===")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index).squeeze()
        loss = criterion(output, data.structural_targets)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Pretrain Loss: {loss.item():.4f}")
    return model


def evaluate_pretrain(model, data):
    """
    Evaluates pretraining by computing final MSE loss on structural targets.
    """
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index).squeeze()
        loss = torch.nn.functional.mse_loss(output, data.structural_targets)
    print(f"Final Pretrain MSE Loss: {loss.item():.4f}")
    return loss.item()


def fine_tune(class_model, pretrain_model, data, labels,
              epochs=100, lr=0.01, weight_decay=5e-4, log_every=10):
    """
    Fine-tunes the classification model, loading pretrained weights
    (except for the final layer), and returns the trained model.
    """
    # Load pretrained weights
    pre_dict = pretrain_model.state_dict()
    class_dict = class_model.state_dict()
    filtered = {k: v for k, v in pre_dict.items()
                if k in class_dict and v.shape == class_dict[k].shape}
    class_dict.update(filtered)
    class_model.load_state_dict(class_dict)

    optimizer = torch.optim.Adam(class_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n=== Fine-tuning on Node Classification ===")
    for epoch in range(1, epochs + 1):
        class_model.train()
        optimizer.zero_grad()
        out = class_model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], labels[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs:
            metrics = evaluate_classification(class_model, data, labels, data.val_mask, verbose=False)
            print(
                f"Epoch {epoch:03d} | Fine-tune Loss: {loss.item():.4f} | Val Acc: {metrics.accuracy:.4f} | Val F1: {metrics.f1:.4f}")
    return class_model

def fine_tune_link_prediction(model, data, epochs=50, lr=0.01, weight_decay=5e-4, num_samples=1000, log_every=10):
    """
    Fine-tunes the GraphSAGE model for link prediction using a dot-product-based binary classification loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    src, dst = data.edge_index
    n_edges = src.size(0)
    n = data.num_nodes

    def score(u, v):
        return (u * v).sum(dim=1)

    print("\n=== Fine-tuning for Link Prediction ===")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        emb = model(data.x, data.edge_index)
        idx = torch.randperm(n_edges)[:num_samples]

        pos_u, pos_v = src[idx], dst[idx]
        neg_u = torch.randint(0, n, (num_samples,), device=data.x.device)
        neg_v = torch.randint(0, n, (num_samples,), device=data.x.device)

        pos_scores = score(emb[pos_u], emb[pos_v])
        neg_scores = score(emb[neg_u], emb[neg_v])
        logits = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ])

        loss = bce_loss(logits, labels)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | LP Fine-tune Loss: {loss.item():.4f}")

    return model


def evaluate_classification(model, data, labels, mask, verbose=True) -> EvaluationResult:
    """
    Computes accuracy, precision, recall, and F1 on the given mask.
    Returns an EvaluationResult object.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)[mask].cpu()
        trues = labels[mask].cpu()

    accuracy = accuracy_score(trues, preds)
    precision = precision_score(trues, preds, average='macro', zero_division=0)
    recall = recall_score(trues, preds, average='macro', zero_division=0)
    f1 = f1_score(trues, preds, average='macro', zero_division=0)

    if verbose:
        print(f"  → Accuracy:  {accuracy:.4f}")
        print(f"  → Precision: {precision:.4f}")
        print(f"  → Recall:    {recall:.4f}")
        print(f"  → F1 Score:  {f1:.4f}")

    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        preds=preds
    )

from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_link_prediction(model, data, num_samples=1000) -> EvaluationResult:
    """
    Evaluates link prediction using dot-product similarity of node embeddings.
    """
    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index)

    src, dst = data.edge_index
    idx = torch.randperm(src.size(0))[:num_samples]
    pos_u, pos_v = src[idx], dst[idx]

    # Negative samples
    n = data.num_nodes
    neg_u = torch.randint(0, n, (num_samples,), device=data.x.device)
    neg_v = torch.randint(0, n, (num_samples,), device=data.x.device)

    # Scoring function (dot product)
    def score(u, v):
        return (u * v).sum(dim=1)

    pos_scores = score(emb[pos_u], emb[pos_v])
    neg_scores = score(emb[neg_u], emb[neg_v])

    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu()
    scores = torch.cat([pos_scores, neg_scores]).cpu()
    preds = (scores > 0).float()

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    print("\n--- Link Prediction Metrics ---")
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


def run_graphsage_pipeline(data, labels,
                           pretrain_epochs=100,
                           finetune_epochs=30,
                           seed=None):
    """
    Orchestrates the full GraphSAGE workflow using modular steps.
    Returns the trained classification model and test metrics.
    """
    from experiments.experiment_utils import set_global_seed
    import time

    start_time = time.time()
    device = get_device()
    print(f"Device: {device} | Seed: {seed}")

    if seed is not None:
        set_global_seed(seed)

    data, labels, device = prepare_data(data, labels, seed=seed)
    in_dim = data.x.size(1)
    num_classes = len(labels.unique())

    pre_model, class_model = initialize_models(in_dim, num_classes, device)

    pre_model = pretrain(pre_model, data, epochs=pretrain_epochs)
    evaluate_pretrain(pre_model, data)

    class_model = fine_tune(class_model, pre_model, data, labels, epochs=finetune_epochs)

    classifer_eval_start_time = time.time()
    classification_results = evaluate_classification(class_model, data, labels, data.test_mask)
    classifier_eval_runtime = time.time() - classifer_eval_start_time

    class_model = fine_tune_link_prediction(class_model, data, epochs=finetune_epochs)
    lp_eval_start_time = time.time()
    lp_results = evaluate_link_prediction(class_model, data)
    lp_eval_runtime = time.time() - lp_eval_start_time

    runtime = time.time() - start_time - classifier_eval_runtime - lp_eval_runtime

    classification_results.metadata.update({
        "seed": seed,
        "train_time": runtime,
        "device": str(device),
        "model": "GraphSAGE"
    })
    lp_results.metadata.update({
        "seed": seed,
        "train_time": runtime,
        "device": str(device),
        "model": "GraphSAGE"
    })

    return class_model, classification_results, lp_results

