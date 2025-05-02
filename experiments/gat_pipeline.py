import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from experiments.experiment_utils import EvaluationResult, sample_negative_edges, split_edges_for_link_prediction
from models.baselines import SimpleGAT


# ------------------------
# 1. Data Preparation
# ------------------------
def prepare_data(data, labels, train_ratio=0.6, val_ratio=0.2, seed=None, device=None):
    """
    Moves data and labels to device and creates train/val/test masks.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    labels = labels.to(device)
    if seed is not None:
        torch.manual_seed(seed)

    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes, device=device)

    train_cut = int(train_ratio * num_nodes)
    val_cut = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    train_mask[indices[:train_cut]] = True
    val_mask[indices[train_cut:val_cut]] = True
    test_mask[indices[val_cut:]] = True

    # Attach masks to data for convenience
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data, labels, device


# ------------------------
# 2. Model Initialization
# ------------------------
def initialize_models(in_dim, num_classes, heads, device):
    """
    Creates two GAT models:
      - A pretraining model (regression) with a single output.
      - A classification model with output dimension equals to num_classes.
    Moves both models to device.
    """
    pretrain_model = SimpleGAT(in_channels=in_dim, out_channels=1, heads=heads).to(device)
    class_model = SimpleGAT(in_channels=in_dim, out_channels=num_classes, heads=heads).to(device)
    return pretrain_model, class_model


# ------------------------
# 3. Pretraining Stage (Structural Regression)
# ------------------------
def pretrain(pretrain_model, data, epochs=100, lr=0.01, weight_decay=5e-4, log_every=10):
    """
    Pretrains the model on a structural regression task.
    """
    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    print("\n=== Pretraining on Structural Regression ===")
    for epoch in range(1, epochs + 1):
        pretrain_model.train()
        optimizer.zero_grad()
        output = pretrain_model(data.x, data.edge_index).squeeze()
        loss = criterion(output, data.structural_targets)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Pretrain Loss: {loss.item():.4f}")
    return pretrain_model


def evaluate_pretrain(pretrain_model, data):
    """
    Evaluates the pretraining stage by computing the final MSE loss.
    """
    pretrain_model.eval()
    with torch.no_grad():
        output = pretrain_model(data.x, data.edge_index).squeeze()
        loss = torch.nn.functional.mse_loss(output, data.structural_targets)
    print(f"Final Pretrain MSE Loss: {loss.item():.4f}")
    return loss.item()


# ------------------------
# 4. Fine-tuning Stage (Node Classification)
# ------------------------
def fine_tune(class_model, pretrain_model, data, labels, epochs=100, lr=0.01, weight_decay=5e-4, log_every=10):
    """
    Fine-tunes a classification model by transferring pretrained weights
    (except for the final layer if the shapes do not match).
    """
    # Transfer pretrained weights from pretrain_model to class_model
    pretrain_dict = pretrain_model.state_dict()
    class_dict = class_model.state_dict()
    filtered_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in class_dict and v.shape == class_dict[k].shape
    }
    class_dict.update(filtered_dict)
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

def finetune_link_prediction(model, data, rem_edge_list, epochs=50, lr=0.01, weight_decay=5e-4, log_every=10):
    """
    Fine-tunes GAT for link prediction using held-out edges only.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    pos_edges = rem_edge_list[0][0]
    n = data.num_nodes

    def score(u, v):
        return (u * v).sum(dim=1)

    print("\n=== Fine-tuning GAT for Link Prediction (Fair) ===")
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
            print(f"Epoch {epoch:03d} | LP Fine-tune Loss: {loss.item():.4f}")
    return model



# ------------------------
# 5. Evaluation
# ------------------------
def evaluate_classification(model, data, labels, mask, verbose=False) -> EvaluationResult:
    """
    Evaluates the classification performance over the provided mask.
    Returns an EvaluationResult object.
    """
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

from sklearn.metrics import average_precision_score

def evaluate_link_prediction(model, data, rem_edge_list) -> EvaluationResult:
    """
    Evaluates on held-out edges using dot product.
    """
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

    print(f"\n=== GAT Link Prediction (Fair) ===")
    print(f"  → Accuracy:       {acc:.4f}")
    print(f"  → Precision:      {precision:.4f}")
    print(f"  → Recall:         {recall:.4f}")
    print(f"  → F1 Score:       {f1:.4f}")
    print(f"  → ROC-AUC:        {auc:.4f}")
    print(f"  → Avg Precision:  {ap:.4f}")

    return EvaluationResult(
        accuracy=acc, precision=precision, recall=recall,
        f1=f1, auc=auc, ap=ap, preds=preds
    )




# ------------------------
# 6. Full Pipeline Orchestration
# ------------------------
def run_gat_pipeline(data, labels, heads=1, pretrain_epochs=100, finetune_epochs=30, seed=None):
    from experiments.experiment_utils import set_global_seed
    import time

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {seed}")

    if seed is not None:
        set_global_seed(seed)

    data, labels, device = prepare_data(data, labels, seed=seed, device=device)
    in_dim = data.x.size(1)
    num_classes = len(labels.unique())

    pretrain_model, class_model = initialize_models(in_dim, num_classes, heads, device)

    # pretrain_model = pretrain(pretrain_model, data, epochs=pretrain_epochs) #No Pretrain because it muddies the experiment argument
    # evaluate_pretrain(pretrain_model, data)

    classifier_train_start_time = time.time()
    class_model = fine_tune(class_model, pretrain_model, data, labels, epochs=finetune_epochs)
    classifier_train_time = time.time() - classifier_train_start_time

    classifier_eval_start_time = time.time()
    classification_results = evaluate_classification(class_model, data, labels, data.test_mask)
    classifier_eval_time = time.time() - classifier_eval_start_time

    original_edges = data.edge_index
    data.edge_index, rem_edge_list = split_edges_for_link_prediction(original_edges)
    link_prediction_train_start_time = time.time()
    class_model = finetune_link_prediction(class_model, data, rem_edge_list, epochs=finetune_epochs)
    link_prediction_train_time = time.time() - link_prediction_train_start_time


    link_prediction_eval_start_time = time.time()
    link_prediction_results = evaluate_link_prediction(class_model, data, rem_edge_list)
    link_prediction_eval_time = time.time() - link_prediction_eval_start_time

    runtime = time.time() - start_time - classifier_eval_time - link_prediction_eval_time

    classification_results.metadata.update({
        "seed": seed,
        "train_time": runtime,
        "device": str(device),
        "model": "SimpleGAT"
    })

    link_prediction_results.metadata.update({
        "seed": seed,
        "train_time": runtime,
        "device": str(device),
        "model": "SimpleGAT"
    })

    return class_model, classification_results, link_prediction_results


