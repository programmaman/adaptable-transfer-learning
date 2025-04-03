import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from models.gpt_gnn import GPT_GNN, GNN, Matcher, Classifier

def prepare_gpt_data(data):
    """
    Converts a homogeneous PyG Data object into the GPT-GNN-friendly format.
    Adds node_type, edge_type, and edge_time attributes with default values.
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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_classifier(classifier, gnn_model, data, labels, mask, device, verbose=True):
    classifier.eval()
    with torch.no_grad():
        node_emb = gnn_model(
            data.x.to(device),
            data.node_type.to(device),
            data.edge_time.to(device),
            data.edge_index.to(device),
            data.edge_type.to(device)
        )
        out = classifier(node_emb)
        pred = out.argmax(dim=1).cpu()
        mask = mask.cpu()
        true = labels.cpu()[mask]
        pred_masked = pred[mask]

        acc = accuracy_score(true, pred_masked)
        precision = precision_score(true, pred_masked, average='macro', zero_division=0)
        recall = recall_score(true, pred_masked, average='macro', zero_division=0)
        f1 = f1_score(true, pred_masked, average='macro', zero_division=0)

        if verbose:
            print(f"  → Accuracy:  {acc:.4f}")
            print(f"  → Precision: {precision:.4f}")
            print(f"  → Recall:    {recall:.4f}")
            print(f"  → F1 Score:  {f1:.4f}")

        return acc, pred_masked, true


def run_gpt_gnn_pipeline(data, labels, hidden_dim=64, num_layers=2, num_heads=2,


                          pretrain_epochs=100, finetune_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Print Device
    print(f"Using device: {device}")

    # === Preprocess Graph for GPT-GNN ===
    node_type = torch.zeros(data.num_nodes, dtype=torch.long)  # Homogeneous nodes
    edge_type = torch.zeros(data.edge_index.size(1), dtype=torch.long)  # Homogeneous edges
    edge_time = torch.zeros(data.edge_index.size(1), dtype=torch.long)  # Dummy time

    data.node_type = node_type
    data.edge_type = edge_type
    data.edge_time = edge_time

    # === Create edge splits for link prediction ===
    rem_edge_list, ori_edge_list = build_rem_edge_list(data.edge_index)
    node_dict = {0: (0, data.num_nodes)}
    target_type = 0

    # === Create GNN Backbone ===
    gnn = GNN(
        in_dim=data.x.size(1),
        n_hid=hidden_dim,
        num_types=1,
        num_relations=1,
        n_heads=num_heads,
        n_layers=num_layers
    )

    # === Wrap in GPT-GNN ===
    model = GPT_GNN(
        gnn=gnn,
        rem_edge_list=rem_edge_list,
        attr_decoder=None,  # Not using text prediction
        types=[0],
        neg_samp_num=5,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # === Pretraining: Link Prediction ===
    print("\n=== Pretraining GPT-GNN (Link Prediction) ===")
    for epoch in range(pretrain_epochs):
        model.train()
        optimizer.zero_grad()
        node_emb = model(data.x.to(device), node_type.to(device), edge_time.to(device), data.edge_index.to(device), edge_type.to(device))
        loss, _ = model.link_loss(
            node_emb, rem_edge_list, ori_edge_list, node_dict,
            target_type, use_queue=False, update_queue=False
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Link Prediction Loss: {loss.item():.4f}")

    # === Fine-tuning: Node Classification ===
    print("\n=== Fine-tuning on Node Classification ===")
    classifier = Classifier(n_hid=hidden_dim, n_out=len(labels.unique())).to(device)
    clf_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Train/val/test split
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_cut = int(0.6 * num_nodes)
    val_cut = int(0.8 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_cut]] = True
    val_mask[indices[train_cut:val_cut]] = True
    test_mask[indices[val_cut:]] = True

    # Fine-tune
    for epoch in range(finetune_epochs):
        classifier.train()
        model.eval()  # GNN is frozen or shared

        node_emb = model(data.x.to(device), node_type.to(device), edge_time.to(device), data.edge_index.to(device), edge_type.to(device))
        out = classifier(node_emb)
        loss = criterion(out[train_mask], labels[train_mask].to(device))

        clf_optimizer.zero_grad()
        loss.backward()
        clf_optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                classifier.eval()
                val_acc, _, _ = evaluate_classifier(classifier, model, data, labels, val_mask, device, verbose=False)
                print(f"Epoch {epoch:03d} | Fine-tune Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

    # Final evaluation
    print("\n=== Final Evaluation on Test Set ===")
    test_acc, test_preds, test_labels = evaluate_classifier(classifier, model, data, labels, test_mask, device)

    print("\n=== Classification Report ===")
    print(classification_report(test_labels, test_preds, digits=4))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(test_labels, test_preds))

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    return classifier, test_acc
