import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.baselines import SimpleGAT  # Ensure this imports your SimpleGAT model

from models.baselines import SimpleGAT

def run_gat_pipeline(data, labels, heads=1, pretrain_epochs=100, finetune_epochs=100):
    # Create train, validation, and test masks
    # Set device and move data and labels to the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    labels = labels.to(device)

    # Split for downstream classification
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    indices = torch.randperm(num_nodes, device=device)
    train_ratio, val_ratio = 0.6, 0.2
    train_cut = int(train_ratio * num_nodes)
    val_cut = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # Create masks directly on the target device
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[indices[:train_cut]] = True
    val_mask[indices[train_cut:val_cut]] = True
    test_mask[indices[val_cut:]] = True

    # Define dimensions
    in_dim = data.x.size(1)
    out_dim = 1  # For pretraining regression (e.g., predicting clustering coefficient)
    out_dim = 1  # For pretraining regression (e.g., predicting a structural target)
    num_classes = len(labels.unique())

    # ---------------------
    # Pretraining Stage
    # ---------------------
    # -------------------------
    # Pretraining Stage (Structural Regression)
    # -------------------------
    print("\n=== Pretraining on Structural Regression ===")
    pretrain_model = SimpleGAT(in_channels=in_dim, out_channels=out_dim, heads=heads)
    pretrain_model = SimpleGAT(in_channels=in_dim, out_channels=out_dim, heads=heads).to(device)
    pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=0.01, weight_decay=5e-4)
    regression_loss = torch.nn.MSELoss()

    for epoch in range(1, pretrain_epochs + 1):
        pretrain_model.train()
        pretrain_optimizer.zero_grad()
        # Forward pass and compute regression loss against structural targets
        output = pretrain_model(data.x, data.edge_index).squeeze()
        loss = regression_loss(output, data.structural_targets)
        loss.backward()
        pretrain_optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Pretrain Loss: {loss.item():.4f}")

    # ---------------------
    # -------------------------
    # Fine-tuning Stage (Node Classification)
    # ---------------------
    # -------------------------
    print("\n=== Fine-tuning on Node Classification ===")
    # Initialize the classification model with output dimension equal to the number of classes
    class_model = SimpleGAT(in_channels=in_dim, out_channels=num_classes, heads=heads)
    class_model = SimpleGAT(in_channels=in_dim, out_channels=num_classes, heads=heads).to(device)

    # Transfer pretrained weights (except the final layer weights)
    # Transfer pretrained weights except for the final layer weights (if shapes match)
    pretrained_dict = pretrain_model.state_dict()
    model_dict = class_model.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_dict)
    class_model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(class_model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Define evaluation function
    def evaluate(model, data, labels, mask, verbose=True):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            true = labels[mask]
            pred_masked = pred[mask]

            # Compute metrics using sklearn
            acc = accuracy_score(true.cpu(), pred_masked.cpu())
            precision = precision_score(true.cpu(), pred_masked.cpu(), average='macro', zero_division=0)
            recall = recall_score(true.cpu(), pred_masked.cpu(), average='macro', zero_division=0)
            f1 = f1_score(true.cpu(), pred_masked.cpu(), average='macro', zero_division=0)
            # Move to CPU for sklearn metrics
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

    # Fine-tuning loop for node classification
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
            print(f"Epoch {epoch:03d} | Fine-tune Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
            print(f"Epoch {epoch:03d} | Val Acc: {val_acc:.4f}")

    test_acc = evaluate(class_model, data, labels, test_mask)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    return class_model, test_acc

