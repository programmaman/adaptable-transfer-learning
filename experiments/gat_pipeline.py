import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from experiments.experiment_utils import EvaluationResult
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


# ------------------------
# 6. Full Pipeline Orchestration
# ------------------------
def run_gat_pipeline(data, labels, heads=1, pretrain_epochs=100, finetune_epochs=50, seed=None):
    """
    Orchestrates the complete pipeline for a GAT model: data preparation, pretraining,
    evaluation of pretraining, fine-tuning for classification, and final evaluation.
    """
    # Prepare data and masks on device
    data, labels, device = prepare_data(data, labels, seed=seed)

    # Define dimensions
    in_dim = data.x.size(1)
    num_classes = len(labels.unique())

    # Initialize the models
    pretrain_model, class_model = initialize_models(in_dim, num_classes, heads, device)

    # Pretraining stage
    pretrain_model = pretrain(pretrain_model, data, epochs=pretrain_epochs)
    evaluate_pretrain(pretrain_model, data)

    # Fine-tuning stage
    class_model = fine_tune(class_model, pretrain_model, data, labels, epochs=finetune_epochs)

    # Final evaluation on test set
    print("\n--- Final Test Classification Metrics ---")
    test_metrics = evaluate_classification(class_model, data, labels, data.test_mask)
    print(f"\nFinal Test Accuracy: {test_metrics.accuracy:.4f}")

    return class_model, test_metrics
