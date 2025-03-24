import torch
from sklearn.metrics import accuracy_score
from models.simple_gnn import SimpleGNN
from experiment_utils import generate_synthetic_graph, generate_task_labels

# Prepare data
data = generate_synthetic_graph()
labels = generate_task_labels(data)

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
hidden_dim = 64
out_dim = 1  # For pretraining regression
num_classes = len(labels.unique())
pretrain_model = SimpleGNN(in_channels=in_dim, hidden_channels=hidden_dim, out_channels=out_dim)

# Pretraining — Structural regression
pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=0.01, weight_decay=5e-4)
regression_loss = torch.nn.MSELoss()

print("\n=== Pretraining on Clustering Coefficient ===")
for epoch in range(1, 101):
    pretrain_model.train()
    pretrain_optimizer.zero_grad()
    output = pretrain_model(data.x, data.edge_index).squeeze()
    loss = regression_loss(output, data.structural_targets)
    loss.backward()
    pretrain_optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Pretrain Loss: {loss.item():.4f}")

# Fine-tuning — Classification
class_model = SimpleGNN(in_channels=in_dim, hidden_channels=hidden_dim, out_channels=num_classes)

# Load pretrained weights except for the final layer
pretrained_dict = pretrain_model.state_dict()
model_dict = class_model.state_dict()

# Filter out layers with mismatched shapes (e.g., final layer)
filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

# Update and load
model_dict.update(filtered_dict)
class_model.load_state_dict(model_dict)

# Optimizer and loss
optimizer = torch.optim.Adam(class_model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def evaluate(model, data, labels, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc = accuracy_score(labels[mask].cpu(), pred[mask].cpu())
    return acc

print("\n=== Fine-tuning on Node Classification ===")
for epoch in range(1, 101):
    class_model.train()
    optimizer.zero_grad()
    out = class_model(data.x, data.edge_index)
    loss = criterion(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        val_acc = evaluate(class_model, data, labels, val_mask)
        print(f"Epoch {epoch:03d} | Fine-tune Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

# Final test accuracy
test_acc = evaluate(class_model, data, labels, test_mask)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
