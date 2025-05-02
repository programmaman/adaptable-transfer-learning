import random
import time
import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

from experiments.struct_g_internal_pipeline import create_masks
from utils import get_device


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_scratch(
    model_class,
    data,
    labels,
    train_mask,
    test_mask,
    hidden_dim,
    output_dim,
    embedding_dim,
    num_layers,
    device,
    num_classes,
    finetune_epochs,
):
    set_random_seeds()
    device = device
    model = model_class(
        num_nodes=data.num_nodes,
        edge_index=data.edge_index,
        input_dim=data.x.size(1),
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        use_gate=True,
        use_gat=True,
        num_classes=num_classes,
        feat_reconstruction=False
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    data = data.to(device)
    labels = labels.to(device)

    start_time = time.time()
    for epoch in range(finetune_epochs):
        model.train()
        optimizer.zero_grad()
        node_indices = torch.arange(data.num_nodes, device=device)
        embeddings = model(data.x, data.edge_index, node_indices)
        logits = model.classify_nodes(embeddings)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
    total_time = time.time() - start_time

    # Evaluation
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index, node_indices)
        logits = model.classify_nodes(embeddings)
        preds = logits[test_mask].argmax(dim=1)
        true = labels[test_mask]

    acc = accuracy_score(true.cpu(), preds.cpu())
    f1 = f1_score(true.cpu(), preds.cpu(), average='macro', zero_division=0)

    return acc, f1, total_time


def pretrain_then_finetune(
    model_class,
    pretrain_fn,
    data,
    labels,
    train_mask,
    test_mask,
    hidden_dim,
    output_dim,
    embedding_dim,
    num_layers,
    device,
    num_classes,
    pretrain_epochs,
    finetune_epochs,
):
    set_random_seeds()
    device = device
    model = model_class(
        num_nodes=data.num_nodes,
        edge_index=data.edge_index,
        input_dim=data.x.size(1),
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        use_gate=True,
        use_gat=True,
        num_classes=num_classes,
        feat_reconstruction=False
    ).to(device)

    # Pretraining phase
    model = pretrain_fn(model, data, pretrain_epochs, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    data = data.to(device)
    labels = labels.to(device)

    start_time = time.time()
    for epoch in range(finetune_epochs):
        model.train()
        optimizer.zero_grad()
        node_indices = torch.arange(data.num_nodes, device=device)
        embeddings = model(data.x, data.edge_index, node_indices)
        logits = model.classify_nodes(embeddings)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
    total_time = time.time() - start_time

    # Evaluation
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index, node_indices)
        logits = model.classify_nodes(embeddings)
        preds = logits[test_mask].argmax(dim=1)
        true = labels[test_mask]

    acc = accuracy_score(true.cpu(), preds.cpu())
    f1 = f1_score(true.cpu(), preds.cpu(), average='macro', zero_division=0)

    return acc, f1, total_time


def run_full_experiment(
    model_class,
    pretrain_fn,
    data,
    labels,
    hidden_dim=64,
    output_dim=32,
    embedding_dim=128,
    num_layers=2,
    device=None,
    pretrain_epochs=100,
    finetune_epochs_list=[5, 10, 20, 50, 100],
):
    if device is None:
        device = get_device()

    num_classes = labels.unique().numel()

    # Create masks
    train_mask, val_mask, test_mask = create_masks(data.num_nodes, device=device)

    print("\n=== Simple GNN (No Pretrain) ===")
    scratch_results = []
    for ft_epochs in finetune_epochs_list:
        acc, f1, t = train_scratch(
            model_class,
            data,
            labels,
            train_mask,
            test_mask,
            hidden_dim,
            output_dim,
            embedding_dim,
            num_layers,
            device,
            num_classes,
            ft_epochs,
        )
        scratch_results.append((ft_epochs, acc, f1, t))
        print(f"  Finetune {ft_epochs} epochs → Acc: {acc:.4f}, F1: {f1:.4f}, Time: {t:.2f}s")

    print("\n=== Pretrained GNN (Node2Vec + SSL) ===")
    pretrain_results = []
    for ft_epochs in finetune_epochs_list:
        acc, f1, t = pretrain_then_finetune(
            model_class,
            pretrain_fn,
            data,
            labels,
            train_mask,
            test_mask,
            hidden_dim,
            output_dim,
            embedding_dim,
            num_layers,
            device,
            num_classes,
            pretrain_epochs,
            ft_epochs,
        )
        pretrain_results.append((ft_epochs, acc, f1, t))
        print(f"  Finetune {ft_epochs} epochs → Acc: {acc:.4f}, F1: {f1:.4f}, Time: {t:.2f}s")

    return scratch_results, pretrain_results
