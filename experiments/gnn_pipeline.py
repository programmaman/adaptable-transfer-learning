from experiments.experiment_utils import split_edges_for_link_prediction, sample_negative_edges
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


def finetune_link_prediction(model, data, rem_edge_list, epochs=50, lr=0.01, weight_decay=5e-4, log_every=10):
    """
    Fine-tunes GNN for link prediction using *only* held-out (removed) edges.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_edges = rem_edge_list[0][0]
    n = data.num_nodes

    def score(u, v):
        return (u * v).sum(dim=1)

    bce_loss = torch.nn.BCEWithLogitsLoss()

    print("\n=== Fine-tuning GNN for Link Prediction ===")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        emb = model(data.x, data.edge_index)

        # Sample negatives
        neg_edges = sample_negative_edges(pos_edges, n).to(data.x.device)

        pos_scores = score(emb[pos_edges[:, 0]], emb[pos_edges[:, 1]])
        neg_scores = score(emb[neg_edges[:, 0]], emb[neg_edges[:, 1]])

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


def evaluate_link_prediction(model, data, rem_edge_list) -> EvaluationResult:
    """
    Evaluates on the held-out edge list using dot product link prediction.
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
    labels = torch.cat([
        torch.ones_like(pos_scores),
        torch.zeros_like(neg_scores)
    ]).cpu()

    preds = (scores > 0).float()
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print(f"\n=== GNN Link Prediction (Fair Evaluation) ===")
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

    # pre_model = pretrain(pre_model, data, epochs=pretrain_epochs) #No Pretrain because it muddies the experiment argument
    # evaluate_pretrain(pre_model, data)

    class_model = fine_tune(class_model, pre_model, data, labels, epochs=finetune_epochs)

    classifer_eval_start_time = time.time()
    classification_results = evaluate_classification(class_model, data, labels, data.test_mask)
    classifer_eval_time = time.time() - classifer_eval_start_time

    # Fair edge split for LP
    original_edges = data.edge_index
    data.edge_index, rem_edge_list = split_edges_for_link_prediction(original_edges)

    class_model = finetune_link_prediction(class_model, data, rem_edge_list, epochs=finetune_epochs)

    lp_eval_start_time = time.time()
    link_prediction_results = evaluate_link_prediction(class_model, data, rem_edge_list)
    lp_eval_time = time.time() - lp_eval_start_time

    runtime = time.time() - start_time - classifer_eval_time - lp_eval_time

    classification_results.metadata.update({
        "seed": seed,
        "train_time": runtime,
        "device": str(device),
        "model": "SimpleGNN"
    })

    link_prediction_results.metadata.update({
        "seed": seed,
        "train_time": runtime,
        "device": str(device),
        "model": "SimpleGNN"
    })

    return class_model, classification_results, link_prediction_results

