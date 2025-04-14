import time

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score

from experiments.experiment_utils import EvaluationResult, set_global_seed
from models.baselines import SimpleGNN


def prepare_data(data, train_ratio=0.6, val_ratio=0.2, seed=None):
    """
    Splits nodes into train/val/test masks and attaches them to the data object.
    """
    if seed is not None:
        torch.manual_seed(seed)
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    train_cut = int(train_ratio * num_nodes)
    val_cut = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:train_cut]] = True
    val_mask[perm[train_cut:val_cut]] = True
    test_mask[perm[val_cut:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def initialize_models(in_dim, num_classes):
    """
    Creates two SimpleGNN instances: one for regression pretraining, one for classification fine-tuning.
    """
    pretrain_model = SimpleGNN(in_channels=in_dim, out_channels=1)
    class_model = SimpleGNN(in_channels=in_dim, out_channels=num_classes)
    return pretrain_model, class_model


def pretrain(model, data, epochs=100, lr=0.01, weight_decay=5e-4, log_every=10):
    """
    Pretrains the GNN model on structural regression targets.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    print("\n=== Pretraining on Clustering Coefficient ===")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index).squeeze()
        loss = criterion(out, data.structural_targets)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Pretrain Loss: {loss.item():.4f}")
    return model


def evaluate_pretrain(model, data):
    """
    Evaluates the pretraining model by reporting final regression loss.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze()
        loss = torch.nn.functional.mse_loss(out, data.structural_targets)
    print(f"Final Pretrain MSE Loss: {loss.item():.4f}")
    return loss.item()


def fine_tune(class_model, pretrain_model, data, labels, epochs=1, lr=0.01, weight_decay=5e-4, log_every=10):
    """
    Fine-tunes the classification model, loading pretrained weights (except final layer).
    """
    # Load pretrained weights (except final layer)
    pre_dict = pretrain_model.state_dict()
    class_dict = class_model.state_dict()
    filtered = {k: v for k, v in pre_dict.items() if k in class_dict and v.shape == class_dict[k].shape}
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
            print(f"Epoch {epoch:03d} | Fine-tune Loss: {loss.item():.4f} | Val Acc: {metrics.accuracy:.4f}")
    return class_model


def finetune_link_prediction(model, data, epochs=50, lr=0.01, weight_decay=5e-4, num_samples=1000, log_every=10):
    """
    Fine-tunes the GNN for link prediction using a simple dot-product loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    src, dst = data.edge_index
    num_edges = src.size(0)
    n = data.num_nodes

    def score(u, v):
        return (u * v).sum(dim=1)

    bce_loss = torch.nn.BCEWithLogitsLoss()

    print("\n=== Fine-tuning for Link Prediction ===")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        emb = model(data.x, data.edge_index)

        idx = torch.randperm(num_edges)[:num_samples]
        pos_u, pos_v = src[idx], dst[idx]
        neg_u = torch.randint(0, n, (num_samples,))
        neg_v = torch.randint(0, n, (num_samples,))

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



def evaluate_classification(model, data, labels, mask, verbose=False) -> EvaluationResult:
    """
    Evaluates classification performance on the given mask.
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

    try:
        auc = roc_auc_score(trues, preds, multi_class='ovr', average='macro')
    except ValueError:
        auc = None

    if verbose:
        print(f"  → Accuracy:  {accuracy:.4f}")
        print(f"  → Precision: {precision:.4f}")
        print(f"  → Recall:    {recall:.4f}")
        print(f"  → F1 Score:  {f1:.4f}")
        if auc is not None:
            print(f"  → AUC (OvR): {auc:.4f}")

    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        preds=preds
    )


def evaluate_link_prediction(model, data, num_samples=1000) -> EvaluationResult:
    """
    Evaluates link prediction by sampling positive and negative edges.
    Returns an EvaluationResult with binary classification metrics.
    """
    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index)

    src, dst = data.edge_index
    # Positive samples
    idx = torch.randperm(src.size(0))[:num_samples]
    pos_u, pos_v = src[idx], dst[idx]

    # Negative samples
    n = data.num_nodes
    neg_u = torch.randint(0, n, (num_samples,))
    neg_v = torch.randint(0, n, (num_samples,))

    def score(u, v):
        return (u * v).sum(dim=1)

    pos_scores = score(emb[pos_u], emb[pos_v])
    neg_scores = score(emb[neg_u], emb[neg_v])

    # Ground truth labels and scores
    labels = torch.cat([
        torch.ones_like(pos_scores),
        torch.zeros_like(neg_scores)
    ]).cpu()
    scores = torch.cat([pos_scores, neg_scores]).cpu()
    preds = (scores > 0).float()  # Threshold at 0

    # Compute metrics
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print(f"\n=== Link Prediction ===")
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


def run_pipeline(data, labels,
                 pretrain_epochs=100, finetune_epochs=30,
                 seed=None):
    from experiments.experiment_utils import set_global_seed
    import time

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {seed}")

    if seed is not None:
        set_global_seed(seed)

    data = prepare_data(data, seed=seed)
    in_dim = data.x.size(1)
    num_classes = len(labels.unique())
    pre_model, class_model = initialize_models(in_dim, num_classes)

    pre_model = pretrain(pre_model, data, epochs=pretrain_epochs)
    pretrain_loss = evaluate_pretrain(pre_model, data)

    class_model = fine_tune(class_model, pre_model, data, labels, epochs=finetune_epochs)
    classification_results = evaluate_classification(class_model, data, labels, data.test_mask)

    class_model = finetune_link_prediction(class_model, data, epochs=finetune_epochs)
    link_prediction_results = evaluate_link_prediction(class_model, data)

    runtime = time.time() - start_time

    classification_results.metadata.update({
        "seed": seed,
        "runtime": runtime,
        "device": str(device),
        "model": "SimpleGNN"
    })

    link_prediction_results.metadata.update({
        "seed": seed,
        "runtime": runtime,
        "device": str(device),
        "model": "SimpleGNN"
    })

    return class_model, classification_results, link_prediction_results

