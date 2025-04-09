import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from models.baselines import SimpleGraphSAGE
from utils import get_device


def run_graphsage_pipeline(data, labels, pretrain_epochs=100, finetune_epochs=100):
    # Set device and move data and labels to the same device
    device = get_device()
    data = data.to(device)
    labels = labels.to(device)

    # Split the nodes into training, validation, and test sets
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes, device=device)
    train_ratio, val_ratio = 0.6, 0.2
    train_cut = int(train_ratio * num_nodes)
    val_cut = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[indices[:train_cut]] = True
    val_mask[indices[train_cut:val_cut]] = True
    test_mask[indices[val_cut:]] = True

    # Define dimensions
    in_dim = data.x.size(1)
    out_dim = 1  # Pretraining: regression output (e.g., predicting a structural target)
    num_classes = len(labels.unique())

    # ---------------------
    # Pretraining Stage
    # ---------------------
    print("\n=== Pretraining on Structural Regression ===")
    pretrain_model = SimpleGraphSAGE(in_channels=in_dim, out_channels=out_dim).to(device)
    pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=0.01, weight_decay=5e-4)
    regression_loss = torch.nn.MSELoss()

    for epoch in range(1, pretrain_epochs + 1):
        pretrain_model.train()
        pretrain_optimizer.zero_grad()
        output = pretrain_model(data.x, data.edge_index).squeeze()
        loss = regression_loss(output, data.structural_targets)
        loss.backward()
        pretrain_optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Pretrain Loss: {loss.item():.4f}")

    # ---------------------
    # Fine-tuning Stage (Node Classification)
    # ---------------------
    print("\n=== Fine-tuning on Node Classification ===")
    class_model = SimpleGraphSAGE(in_channels=in_dim, out_channels=num_classes).to(device)

    # Transfer pretrained weights except for parameters with mismatched shapes (e.g., final layer)
    pretrained_dict = pretrain_model.state_dict()
    model_dict = class_model.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_dict)
    class_model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(class_model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def evaluate(model, data, labels, mask, verbose=True):
        """
        Evaluates node classification performance.
        """
        model.eval()
        with torch.no_grad():
            # Get model output (logits) and make predictions
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)  # Get the predicted class
            true = labels[mask]
            pred_masked = pred[mask]

            # Move to CPU for sklearn metrics
            pred_masked = pred_masked.cpu()
            true = true.cpu()

            # Classification metrics
            acc = accuracy_score(true, pred_masked)
            precision = precision_score(true, pred_masked, average='macro', zero_division=0)
            recall = recall_score(true, pred_masked, average='macro', zero_division=0)
            f1 = f1_score(true, pred_masked, average='macro', zero_division=0)

            # Optional AUC calculation (only if multi-class)
            try:
                auc = roc_auc_score(true, out.cpu(), multi_class='ovr', average='macro')
            except ValueError:
                auc = None

            if verbose:
                print(f"  → Accuracy:  {acc:.4f}")
                print(f"  → Precision: {precision:.4f}")
                print(f"  → Recall:    {recall:.4f}")
                print(f"  → F1 Score:  {f1:.4f}")
                if auc is not None:
                    print(f"  → AUC (OvR): {auc:.4f}")

            return {
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "AUC": auc
            }

    def evaluate_link_prediction(model, data, rem_edge_list, ori_edge_list, device):
        """
        Evaluates link prediction performance.
        """
        model.eval()
        with torch.no_grad():
            # Get node embeddings from the model
            emb = model(data.x.to(device), data.node_type.to(device), data.edge_time.to(device),
                        data.edge_index.to(device), data.edge_type.to(device))

        # Retrieve positive and negative test edges (binary classification)
        pos_edges = rem_edge_list[0][0].to(device)
        neg_edges = model.sample_negative_edges(pos_edges, data.x.size(0)).to(device)

        # Define a function to score edges (dot product of node embeddings)
        def score(u, v):
            return (emb[u] * emb[v]).sum(dim=-1)

        # Compute scores for positive and negative edges
        pos_scores = score(pos_edges[:, 0], pos_edges[:, 1])
        neg_scores = score(neg_edges[:, 0], neg_edges[:, 1])

        # Combine scores and labels (1 for positive edges, 0 for negative edges)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

        # Predict based on the score threshold (positive if score > 0)
        preds = (scores > 0).float()

        # Compute metrics
        acc = accuracy_score(labels.cpu(), preds.cpu())
        precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
        recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
        f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)
        auc = roc_auc_score(labels.cpu(), scores.cpu())
        ap = average_precision_score(labels.cpu(), scores.cpu())

        print(f"\n=== Link Prediction ===")
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
        class_model.train()
        optimizer.zero_grad()
        out = class_model(data.x, data.edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Fine-tune Loss: {loss.item():.4f}")
            val_acc = evaluate(class_model, data, labels, val_mask)
            print(f"Epoch {epoch:03d} | Val Acc: {val_acc:.4f}")

    classifier_results = evaluate(class_model, data, labels, test_mask)
    lp_results = evaluate_link_prediction(class_model, data, data.rem_edge_list, data.ori_edge_list, device)
    print(f"\nFinal Classifier Test Accuracy: {classifier_results['Accuracy']:.4f}")
    print(f"Final Link Prediction Test Accuracy: {lp_results['LP-Accuracy']:.4f}")

    return class_model, classifier_results, lp_results
