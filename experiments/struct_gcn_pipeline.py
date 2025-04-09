import torch

from models.structural_gcn import (
    StructuralGcn,
    GnnClassifierHead,
    train_structural_feature_predictor,
    fine_tune_model,
    evaluate_model, evaluate_link_prediction
)
from utils import get_device


def run_structural_gcn_pipeline(data, labels, hidden_dim=64, mid_dim=32, pretrain_epochs=100, finetune_epochs=100):
    device = get_device()

    #Print Device
    print(f"Using device: {device}")

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
    pretrain_model = StructuralGcn(in_channels=in_dim, hidden_channels=hidden_dim, mid_channels=mid_dim)
    pretrain_model = train_structural_feature_predictor(pretrain_model, data, epochs=pretrain_epochs, device=device)

    # === Fine-tuning Phase ===
    classifier_model = GnnClassifierHead(pretrained_model=pretrain_model, out_channels=len(labels.unique()))
    classifier_model = fine_tune_model(
        classifier_model,
        data,
        task_labels=labels,
        epochs=finetune_epochs,
        device=device
    )

    # === Evaluation ===
    results = evaluate_model(classifier_model, data, labels, device=device)
    lp_results = evaluate_link_prediction(classifier_model, data, device=device)
    print(f"\nFinal Evaluation — Loss: {results['loss']:.4f}, Acc: {results['acc']:.4f}")

    return classifier_model, results, lp_results