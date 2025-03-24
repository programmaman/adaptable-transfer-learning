import torch
from models.structural_gcn import StructuralGcn, GnnClassifierHead, train_structural_feature_predictor, fine_tune_model, evaluate_model
from experiment_utils import generate_synthetic_graph, generate_task_labels

# === Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate synthetic data
data = generate_synthetic_graph()
labels = generate_task_labels(data)

# Create train/val/test split
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

# === Pretraining Phase ===
in_dim = data.x.size(1)
hidden_dim = 64
mid_dim = 32

pretrain_model = StructuralGcn(in_channels=in_dim, hidden_channels=hidden_dim, mid_channels=mid_dim)
pretrain_model = train_structural_feature_predictor(pretrain_model, data, epochs=100, device=device)

# === Fine-tuning Phase ===
classifier_model = GnnClassifierHead(pretrained_model=pretrain_model, out_channels=len(labels.unique()))
classifier_model = fine_tune_model(
    classifier_model,
    data,
    task_labels=labels,
    epochs=100,
    device=device
)

# === Evaluation ===
val_loss, val_acc = evaluate_model(classifier_model, data, labels, device=device)
print(f"\nFinal Evaluation — Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
