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
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[perm[:train_cut]] = True
    val_mask[perm[train_cut:val_cut]] = True
    test_mask[perm[val_cut:]] = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask
    return data, labels, device

def initialize_models(in_dim, num_classes, device):
    """
    Creates two GraphSAGE models: one for regression pretraining,
    one for classification fine-tuning, and moves them to device.
    """
    pre_model   = SimpleGraphSAGE(in_channels=in_dim, out_channels=1).to(device)
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
    pre_dict   = pretrain_model.state_dict()
    class_dict = class_model.state_dict()
    filtered   = {k: v for k, v in pre_dict.items()
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
            print(f"Epoch {epoch:03d} | Fine-tune Loss: {loss.item():.4f} | Val Acc: {metrics.accuracy:.4f} | Val F1: {metrics.f1:.4f}")
    return class_model

def evaluate_classification(model, data, labels, mask, verbose=True) -> EvaluationResult:
    """
    Computes accuracy, precision, recall, and F1 on the given mask.
    Returns an EvaluationResult object.
    """
    model.eval()
    with torch.no_grad():
        out   = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)[mask].cpu()
        trues = labels[mask].cpu()

    accuracy  = accuracy_score(trues, preds)
    precision = precision_score(trues, preds, average='macro', zero_division=0)
    recall    = recall_score(trues, preds, average='macro', zero_division=0)
    f1        = f1_score(trues, preds, average='macro', zero_division=0)

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

def run_graphsage_pipeline(data, labels,
                            pretrain_epochs=100,
                            finetune_epochs=50,
                            seed=None):
    """
    Orchestrates the full GraphSAGE workflow using modular steps.
    Returns the trained classification model and test metrics.
    """
    # 1) Prepare data & masks
    data, labels, device = prepare_data(data, labels, seed=seed)

    # 2) Initialize models
    in_dim      = data.x.size(1)
    num_classes = len(labels.unique())
    pre_model, class_model = initialize_models(in_dim, num_classes, device)

    # 3) Pretraining
    pre_model = pretrain(pre_model, data, epochs=pretrain_epochs)
    evaluate_pretrain(pre_model, data)

    # 4) Fine-tuning
    class_model = fine_tune(class_model, pre_model, data, labels, epochs=finetune_epochs)

    # 5) Final evaluation on test set
    print("\n--- Final Test Classification Metrics ---")
    test_metrics = evaluate_classification(class_model, data, labels, data.test_mask)
    print(f"\nFinal Test Accuracy: {test_metrics.accuracy:.4f}")

    return class_model, test_metrics
