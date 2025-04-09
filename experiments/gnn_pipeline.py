import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    average_precision_score, roc_auc_score
from models.baselines import SimpleGNN
import torch.nn.functional as f


def run_gnn_pipeline(data, labels, pretrain_epochs=100, finetune_epochs=100):
    # Split for downstream classification
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

    # Initialize model
    in_dim = data.x.size(1)
    out_dim = 1  # For pretraining regression
    num_classes = len(labels.unique())
    gnn_model = SimpleGNN(in_channels=in_dim, out_channels=out_dim)

    # Pretraining — Structural regression
    pretrain_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01, weight_decay=5e-4)
    regression_loss = torch.nn.MSELoss()

    print("\n=== Pretraining on Clustering Coefficient ===")
    for epoch in range(1, pretrain_epochs + 1):
        gnn_model.train()
        pretrain_optimizer.zero_grad()
        output = gnn_model(data.x, data.edge_index).squeeze()
        loss = regression_loss(output, data.structural_targets)
        loss.backward()
        pretrain_optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Pretrain Loss: {loss.item():.4f}")

    # Fine-tuning — Classification
    class_model = SimpleGNN(in_channels=in_dim, out_channels=num_classes)

    # Load pretrained weights except for the final layer
    pretrained_dict = gnn_model.state_dict()
    model_dict = class_model.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_dict)
    class_model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(class_model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

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

            try:
                auc = roc_auc_score(true_cpu, probs_cpu, multi_class='ovr', average='macro')
            except ValueError:
                auc = None

            if verbose:
                print("\n=== Node Classification Evaluation ===")
                print(f"  → Accuracy:  {acc:.4f}")
                print(f"  → Precision: {precision:.4f}")
                print(f"  → Recall:    {recall:.4f}")
                print(f"  → F1 Score:  {f1:.4f}")
                if auc is not None:
                    print(f"  → AUC (OvR): {auc:.4f}")
                print("  → Classification Report:")
                print(classification_report(true_cpu, pred_cpu, digits=4))

            return {
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "AUC": auc
            }

    def evaluate_link_prediction(model, data, num_samples=1000):
        model.eval()
        with torch.no_grad():
            emb = model(data.x, data.edge_index)

        edge_index = data.edge_index
        src_pos, dst_pos = edge_index[0], edge_index[1]
        pos_idx = torch.randperm(src_pos.size(0))[:num_samples]
        pos_edges = (src_pos[pos_idx], dst_pos[pos_idx])

        # Generate negative samples
        num_nodes = data.num_nodes
        neg_src = torch.randint(0, num_nodes, (num_samples,))
        neg_dst = torch.randint(0, num_nodes, (num_samples,))
        neg_edges = (neg_src, neg_dst)

        # Score function
        def link_score(u, v):
            return (u * v).sum(dim=1)

        pos_score = link_score(emb[pos_edges[0]], emb[pos_edges[1]])
        neg_score = link_score(emb[neg_edges[0]], emb[neg_edges[1]])

        # Combine scores and labels
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ])

        preds = (scores > 0).float()

        # Metrics
        acc = accuracy_score(labels.cpu(), preds.cpu())
        precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
        recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
        f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)
        auc = roc_auc_score(labels.cpu(), scores.cpu())
        ap = average_precision_score(labels.cpu(), scores.cpu())

        print("\n=== Link Prediction Evaluation ===")
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

    print("\n=== Fine-tuning on Node Classification ===")
    for epoch in range(1, finetune_epochs + 1):
        class_model.train()
        optimizer.zero_grad()
        out = class_model(data.x, data.edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            val_acc = evaluate(class_model, data, labels, val_mask)
            print(f"Epoch {epoch:03d} | Fine-tune Loss: {loss.item():.4f} | Validation Accuracy: {val_acc['Accuracy']:.4f}")

    test_results = evaluate(class_model, data, labels, test_mask)
    lp_test_results = evaluate_link_prediction(class_model, data)
    print(f"\nFinal Test Accuracy: {test_results['Accuracy']:.4f}")

    return class_model, test_results, lp_test_results
