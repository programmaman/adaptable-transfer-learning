import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from experiments.experiment_utils import sample_negative_edges, EvaluationResult
from models.gpt_gnn import GPT_GNN, GNN, Classifier


# ------------------------
# Data & Edge Preparation
# ------------------------
def prepare_gpt_data(data):
    """
    Converts a homogeneous PyG Data object into a GPT-GNN-friendly format.
    Adds node_type, edge_type, and edge_time attributes.
    """
    data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
    data.edge_type = torch.zeros(data.edge_index.size(1), dtype=torch.long)
    data.edge_time = torch.zeros(data.edge_index.size(1), dtype=torch.long)
    return data


def build_rem_edge_list(edge_index):
    """
    Splits the edge list for link prediction training.
    Returns:
        rem_edge_list: dict for GPT-GNN link prediction (removed edges)
        ori_edge_list: full original edge list for context
    """
    edge_array = edge_index.t().cpu().numpy()
    num_edges = edge_array.shape[0]
    perm = np.random.permutation(num_edges)
    keep_portion = int(0.8 * num_edges)

    # Keep 80%, remove 20%
    keep_edges = edge_array[perm[:keep_portion]]
    rem_edges = edge_array[perm[keep_portion:]]

    rem_edge_list = {0: {0: torch.tensor(rem_edges, dtype=torch.long)}}
    ori_edge_list = {0: {0: torch.tensor(edge_array, dtype=torch.long)}}
    return rem_edge_list, ori_edge_list


def create_masks(num_nodes, train_ratio=0.6, val_ratio=0.2, device=None):
    """
    Create train, validation, and test masks on the given device.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    indices = torch.randperm(num_nodes, device=device)
    train_cut = int(train_ratio * num_nodes)
    val_cut = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[indices[:train_cut]] = True
    val_mask[indices[train_cut:val_cut]] = True
    test_mask[indices[val_cut:]] = True

    return train_mask, val_mask, test_mask


# ------------------------
# Model Building
# ------------------------
def build_gpt_gnn_model(data, hidden_dim=64, num_layers=2, num_heads=2, device=None):
    """
    Constructs the GNN backbone and wraps it in a GPT-GNN model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a GPT-friendly data structure if not already present
    data = prepare_gpt_data(data)

    # Build edge lists for link prediction
    rem_edge_list, ori_edge_list = build_rem_edge_list(data.edge_index)
    node_dict = {0: (0, data.num_nodes)}  # For link prediction; modify as needed
    target_type = 0

    # Create the GNN backbone
    gnn = GNN(
        in_dim=data.x.size(1),
        n_hid=hidden_dim,
        num_types=1,
        num_relations=1,
        n_heads=num_heads,
        n_layers=num_layers
    )

    # Wrap in GPT-GNN
    model = GPT_GNN(
        gnn=gnn,
        rem_edge_list=rem_edge_list,
        attr_decoder=None,  # Not using text prediction in this instance
        types=[0],
        neg_samp_num=5,
        device=device
    ).to(device)

    return model, rem_edge_list, ori_edge_list, node_dict, target_type


# ------------------------
# Pretraining Stage (Link Prediction)
# ------------------------
def pretrain_link_prediction(model, data, node_type, edge_time, edge_type,
                             rem_edge_list, ori_edge_list, node_dict, target_type,
                             epochs=100, lr=0.005, weight_decay=5e-4, log_every=10, device=None):
    """
    Pretrains GPT-GNN using a link prediction loss.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("\n=== Pretraining GPT-GNN (Link Prediction) ===")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass (make sure to move all inputs to the correct device)
        node_emb = model(
            data.x.to(device),
            node_type.to(device),
            edge_time.to(device),
            data.edge_index.to(device),
            edge_type.to(device)
        )
        loss, _ = model.link_loss(
            node_emb, rem_edge_list, ori_edge_list, node_dict,
            target_type, use_queue=False, update_queue=False
        )
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Link Prediction Loss: {loss.item():.4f}")
    return model


# ------------------------
# Fine-Tuning Stage (Node Classification)
# ------------------------
def fine_tune_classifier(model, data, node_type, edge_time, edge_type,
                         labels, hidden_dim, finetune_epochs=100, lr=0.01, weight_decay=5e-4,
                         log_every=10, device=None):
    """
    Fine-tunes a classifier on top of the frozen/backbone GPT-GNN model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create train/val/test splits (using the full graph)
    num_nodes = data.num_nodes
    train_mask, val_mask, test_mask = create_masks(num_nodes, device=device)

    # Initialize the classifier model
    classifier = Classifier(n_hid=hidden_dim, n_out=len(labels.unique())).to(device)
    clf_optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n=== Fine-tuning on Node Classification ===")
    for epoch in range(finetune_epochs):
        classifier.train()
        model.eval()  # Freeze/backbone remains unchanged

        # Obtain node embeddings from GPT-GNN
        node_emb = model(
            data.x.to(device),
            node_type.to(device),
            edge_time.to(device),
            data.edge_index.to(device),
            edge_type.to(device)
        )
        out = classifier(node_emb)
        loss = criterion(out[train_mask], labels[train_mask].to(device))

        clf_optimizer.zero_grad()
        loss.backward()
        clf_optimizer.step()

        if epoch % log_every == 0 or epoch == finetune_epochs - 1:
            eval_metrics = evaluate_classifier(classifier, model, data, labels, val_mask, device, verbose=False)
            print(
                f"Epoch {epoch:03d} | Fine-tune Loss: {loss.item():.4f} | Val Accuracy: {eval_metrics.accuracy}")

    return classifier, train_mask, val_mask, test_mask


# ------------------------
# Evaluation Functions
# ------------------------
def evaluate_classifier(classifier, gnn_model, data, labels, mask, device, verbose=True) -> EvaluationResult:
    """
    Evaluates node classification performance using the classifier on top of GPT-GNN embeddings.
    Returns an EvaluationResult object.
    """
    classifier.eval()
    with torch.no_grad():
        # Compute node embeddings from gnn_model
        node_emb = gnn_model(
            data.x.to(device),
            data.node_type.to(device),
            data.edge_time.to(device),
            data.edge_index.to(device),
            data.edge_type.to(device)
        )
        logits = classifier(node_emb)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        # Apply mask
        true = labels[mask].cpu()
        pred_masked = preds[mask].cpu()
        probs_masked = probs[mask].cpu()

        acc = accuracy_score(true, pred_masked)
        precision = precision_score(true, pred_masked, average='macro', zero_division=0)
        recall = recall_score(true, pred_masked, average='macro', zero_division=0)
        f1 = f1_score(true, pred_masked, average='macro', zero_division=0)

        # Attempt to compute multi-class AUC
        try:
            auc = roc_auc_score(true, probs_masked, multi_class='ovr', average='macro')
        except ValueError:
            auc = None

        if verbose:
            print("\n=== GPT-GNN Node Classification ===")
            print(f"  → Accuracy:  {acc:.4f}")
            print(f"  → Precision: {precision:.4f}")
            print(f"  → Recall:    {recall:.4f}")
            print(f"  → F1 Score:  {f1:.4f}")
            if auc is not None:
                print(f"  → AUC (OvR): {auc:.4f}")
            print("\n  → Classification Report:")
            print(classification_report(true, pred_masked, digits=4))

        return EvaluationResult(
            accuracy=acc,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
            preds=pred_masked
        )


def evaluate_gpt_link_prediction(model, data, rem_edge_list, ori_edge_list, device) -> EvaluationResult:
    """
    Evaluates GPT-GNN link prediction performance by computing scores for
    positive and negative edges. Returns an EvaluationResult object.
    """
    model.eval()
    with torch.no_grad():
        emb = model(
            data.x.to(device),
            data.node_type.to(device),
            data.edge_time.to(device),
            data.edge_index.to(device),
            data.edge_type.to(device)
        )

    # Retrieve positive edges from removed edge list
    pos_edges = rem_edge_list[0][0].to(device)

    # Sample negative edges
    neg_edges = sample_negative_edges(pos_edges, data.num_nodes).to(device)

    def score(u, v):
        return (emb[u] * emb[v]).sum(dim=-1)

    pos_scores = score(pos_edges[:, 0], pos_edges[:, 1])
    neg_scores = score(neg_edges[:, 0], neg_edges[:, 1])

    scores = torch.cat([pos_scores, neg_scores])
    labels_lp = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    preds_lp = (scores > 0).float()

    # Compute metrics
    acc = accuracy_score(labels_lp.cpu(), preds_lp.cpu())
    precision = precision_score(labels_lp.cpu(), preds_lp.cpu(), zero_division=0)
    recall = recall_score(labels_lp.cpu(), preds_lp.cpu(), zero_division=0)
    f1 = f1_score(labels_lp.cpu(), preds_lp.cpu(), zero_division=0)
    auc = roc_auc_score(labels_lp.cpu(), scores.cpu())
    ap = average_precision_score(labels_lp.cpu(), scores.cpu())

    print("\n=== GPT-GNN Link Prediction ===")
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
        preds=preds_lp
    )

# ------------------------
# Pipeline Orchestration
# ------------------------
def run_gpt_gnn_pipeline(data, labels, hidden_dim=64, num_layers=2, num_heads=2,
                         pretrain_epochs=100, finetune_epochs=1):
    """
    Orchestrates the full GPT-GNN workflow:
      1. Preprocess the graph data.
      2. Build edge splits for link prediction.
      3. Construct the GNN backbone and wrap it in GPT-GNN.
      4. Pretrain the GPT-GNN using link prediction.
      5. Fine-tune a classifier on node classification.
      6. Evaluate node classification and link prediction performance.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Preprocess data (adds node_type, edge_type, and edge_time)
    data = prepare_gpt_data(data)

    # Build train/val/test masks (for node classification fine-tuning later)
    num_nodes = data.num_nodes
    train_mask, val_mask, test_mask = create_masks(num_nodes, device=device)
    # Attach masks to data for use in evaluation
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Build GPT-GNN model & supporting structures for link prediction
    model, rem_edge_list, ori_edge_list, node_dict, target_type = build_gpt_gnn_model(
        data, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, device=device
    )

    # Pretraining: Link Prediction
    # (Grab the additional attributes from data)
    node_type = data.node_type
    edge_type = data.edge_type
    edge_time = data.edge_time
    model = pretrain_link_prediction(
        model, data, node_type, edge_time, edge_type,
        rem_edge_list, ori_edge_list, node_dict, target_type,
        epochs=pretrain_epochs, lr=0.005, weight_decay=5e-4, log_every=10, device=device
    )

    # Fine-tuning: Node Classification
    classifier, train_mask, val_mask, test_mask = fine_tune_classifier(
        model, data, node_type, edge_time, edge_type,
        labels, hidden_dim, finetune_epochs=finetune_epochs, lr=0.01, weight_decay=5e-4,
        log_every=10, device=device
    )

    # Final Evaluation: Node Classification
    print("\n=== Final Evaluation on Test Set ===")
    classification_results = evaluate_classifier(classifier, model, data, labels, test_mask, device)
    print("\n=== Classification Report ===")
    print(classification_report(labels[test_mask].cpu(), classification_results.preds.cpu(), digits=4))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(labels[test_mask].cpu(), classification_results.preds.cpu()))
    print(f"\nFinal Test Accuracy: {classification_results.accuracy:.4f}")

    # Evaluation: Link Prediction
    lp_results = evaluate_gpt_link_prediction(model, data, rem_edge_list, ori_edge_list, device)

    return classifier, classification_results, lp_results
