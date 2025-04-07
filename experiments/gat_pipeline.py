import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, \
    roc_auc_score
from torch_geometric.utils import train_test_split_edges

from models.baselines import SimpleGAT
from utils import get_device


def run_gat_pipeline(data, labels, heads=1, pretrain_epochs=100, finetune_epochs=100):
    # Setup
    device = get_device()
    data = data.to(device)
    data = train_test_split_edges(data)
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

    # Replace the output head for classification
    model.out_proj = torch.nn.Linear(model.out_proj.in_features, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def evaluate(model, data, labels, mask, verbose=True):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            true = labels[mask]
            pred_masked = pred[mask]

            # Convert to CPU
            pred_masked_cpu = pred_masked.cpu()
            true_cpu = true.cpu()

            acc = accuracy_score(true_cpu, pred_masked_cpu)
            precision = precision_score(true_cpu, pred_masked_cpu, average='macro', zero_division=0)
            recall = recall_score(true_cpu, pred_masked_cpu, average='macro', zero_division=0)
            f1 = f1_score(true_cpu, pred_masked_cpu, average='macro', zero_division=0)

            if verbose:
                print(f"  → Accuracy:  {acc:.4f}")
                print(f"  → Precision: {precision:.4f}")
                print(f"  → Recall:    {recall:.4f}")
                print(f"  → F1 Score:  {f1:.4f}")

            return acc

    def evaluate_link_prediction(model, data):
        model.eval()
        with torch.no_grad():
            # Get node embeddings
            z = model(data.x, data.edge_index)

            # Positive edge scores
            pos_edge_index = data.test_pos_edge_index
            pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)

            # Negative edge scores
            neg_edge_index = data.test_neg_edge_index
            neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

            # Combine
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])

            auc = roc_auc_score(labels.cpu(), scores.cpu())
            ap = average_precision_score(labels.cpu(), scores.cpu())

            print(f"\nLink Prediction → AUC: {auc:.4f}, AP: {ap:.4f}")
            return auc, ap

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Fine-tune Loss: {loss.item():.4f}")
            val_acc = evaluate(model, data, labels, val_mask)
            print(f"Epoch {epoch:03d} | Val Acc: {val_acc:.4f}")

    test_acc = evaluate(model, data, labels, test_mask)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    return model, test_acc
