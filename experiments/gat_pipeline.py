import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, \
    roc_auc_score
from torch_geometric.utils import train_test_split_edges
import torch.nn.functional as f

from models.baselines import SimpleGAT
from utils import get_device


def run_gat_pipeline(data, labels, heads=1, pretrain_epochs=100, finetune_epochs=100):
    # Setup
    # Setup
    device = get_device()
    data = data.to(device)

    # Preserve original edge_index for structural regression + node classification
    original_edge_index = data.edge_index.clone()

    # Prepare link prediction edges
    data = train_test_split_edges(data)
    data.edge_index = original_edge_index  # Restore for message passing

    labels = labels.to(device)
    in_dim = data.x.size(1)
    num_classes = len(labels.unique())

    # Train/Val/Test Split
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes, device=device)
    train_cut = int(0.6 * num_nodes)
    val_cut = int(0.8 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[indices[:train_cut]] = True
    val_mask[indices[train_cut:val_cut]] = True
    test_mask[indices[val_cut:]] = True

    # -------------------------
    # Unified Model for Both Stages
    # -------------------------
    model = SimpleGAT(in_channels=in_dim, out_channels=1, heads=heads).to(device)

    # -------------------------
    # Stage 1: Pretraining (Structural Regression)
    # -------------------------
    print("\n=== Pretraining on Structural Regression ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    regression_loss = torch.nn.MSELoss()

    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index).squeeze()
        loss = regression_loss(output, data.structural_targets)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Pretrain Loss: {loss.item():.4f}")

    # -------------------------
    # Stage 2: Fine-tuning (Node Classification)
    # -------------------------
    print("\n=== Fine-tuning on Node Classification ===")

    # Save pretrained conv1 weights
    pretrained_conv_state = model.conv1.state_dict()

    # Re-init model for classification
    model = SimpleGAT(in_channels=in_dim, out_channels=num_classes, heads=heads).to(device)

    # Load only matching weights
    conv1_state = model.conv1.state_dict()
    filtered_state = {k: v for k, v in pretrained_conv_state.items() if
                      k in conv1_state and v.shape == conv1_state[k].shape}
    conv1_state.update(filtered_state)
    model.conv1.load_state_dict(conv1_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report
    )

    def evaluate(model, data, labels, mask, verbose=True):
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = f.softmax(logits, dim=1)

            pred = probs.argmax(dim=1)
            true = labels[mask]
            pred_masked = pred[mask]
            probs_masked = probs[mask]

            # Move to CPU
            true_cpu = true.cpu()
            pred_cpu = pred_masked.cpu()
            probs_cpu = probs_masked.cpu()

            # Metrics
            acc = accuracy_score(true_cpu, pred_cpu)
            precision = precision_score(true_cpu, pred_cpu, average='macro', zero_division=0)
            recall = recall_score(true_cpu, pred_cpu, average='macro', zero_division=0)
            f1 = f1_score(true_cpu, pred_cpu, average='macro', zero_division=0)

            # Optional AUC (only if more than 1 class and 1-vs-rest setup)
            try:
                auc = roc_auc_score(true_cpu, probs_cpu, multi_class='ovr', average='macro')
            except ValueError:
                auc = None

            if verbose:
                print(f"  → Accuracy:  {acc:.4f}")
                print(f"  → Precision: {precision:.4f}")
                print(f"  → Recall:    {recall:.4f}")
                print(f"  → F1 Score:  {f1:.4f}")
                if auc is not None:
                    print(f"  → AUC (OvR): {auc:.4f}")
                print("\n  → Classification Report:")
                print(classification_report(true_cpu, pred_cpu, digits=4))

            return {
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "AUC": auc
            }

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    def evaluate_link_prediction(model, data):
        model.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index)

            pos_edge_index = data.test_pos_edge_index
            neg_edge_index = data.test_neg_edge_index

            pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
            neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(pos_scores.size(0)),
                torch.zeros(neg_scores.size(0))
            ])

            # Apply threshold
            preds = (scores > 0).float()

            # Compute metrics
            acc = accuracy_score(labels.cpu(), preds.cpu())
            precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
            recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
            f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)

            auc = roc_auc_score(labels.cpu(), scores.cpu())
            ap = average_precision_score(labels.cpu(), scores.cpu())

            print(f"\nLink Prediction:")
            print(f"  → Accuracy:  {acc:.4f}")
            print(f"  → Precision: {precision:.4f}")
            print(f"  → Recall:    {recall:.4f}")
            print(f"  → F1 Score:  {f1:.4f}")
            print(f"  → AUC:       {auc:.4f}")
            print(f"  → AP:        {ap:.4f}")

            return {
                "LP-Accuracy": acc,
                "LP-Precision": precision,
                "LP-Recall": recall,
                "LP-F1": f1,
                "LP-AUC": auc,
                "LP-AP": ap
            }

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Fine-tune Loss: {loss.item():.4f}")
            val_metrics = evaluate(model, data, labels, val_mask, verbose=False)
            print(f"Epoch {epoch:03d} | Val Accuracy: {val_metrics['Accuracy']:.4f}")

    results = evaluate(model, data, labels, test_mask)
    lp_results = evaluate_link_prediction(model, data)

    print(f"\nClassification Results:")
    for k, v in results.items():
        print(f"  → {k}: {v:.4f}")

    print(f"\nLink Prediction Results:")
    for k, v in lp_results.items():
        print(f"  → {k}: {v:.4f}")
    return model, results, lp_results
