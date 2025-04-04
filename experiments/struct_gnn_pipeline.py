import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_structural_node2vec_pipeline(
        data,
        labels,
        hidden_dim=64,
        output_dim=32,
        embedding_dim=128,
        num_layers=2,
        node2vec_pretrain_epochs=100,
        full_pretrain_epochs=50,
        finetune_epochs=100,
        do_linkpred=True,
        do_n2v_align=True,
        do_featrec=False,
):
    """
    data:       A PyG data object with
                  - data.x            [num_nodes, input_dim]
                  - data.edge_index   [2, num_edges]
    labels:     A 1D LongTensor [num_nodes] of class labels.
    hidden_dim, output_dim, embedding_dim, num_layers: GNN architecture params
    node2vec_pretrain_epochs:   epochs for Node2Vec only
    full_pretrain_epochs:       epochs for structural GNN pretraining
    finetune_epochs:            epochs for final node classification
    do_linkpred, do_n2v_align,
    do_featrec:  booleans indicating which self-supervised tasks to include
                 in Phase 2. You can add or remove tasks easily.

    Returns:
        model, classifier, test_acc
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Print Device
    print(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # (A) Train/val/test split
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # (B) Initialize Model
    # --------------------------------------------------------------------------
    # If you'd like to unify classification in the same model,
    # you can pass num_classes=labels.unique().numel() to your ImprovedStructuralGNN.
    # Below, we show the approach with a separate classifier:
    from models.structural_gnn import StructuralGNN

    model = StructuralGNN(
        num_nodes=num_nodes,
        edge_index=data.edge_index,
        input_dim=data.x.size(1),
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        use_gat=True,  # or False
        num_classes=None,  # We do classification externally
        feat_reconstruction=do_featrec
    ).to(device)

    node_indices = torch.arange(num_nodes, device=device)

    # --------------------------------------------------------------------------
    # Phase 1: Pre-train Node2Vec Embeddings
    # --------------------------------------------------------------------------
    print("\n=== Phase 1: Pre-training Node2Vec embeddings ===")
    model.train_node2vec(
        num_epochs=node2vec_pretrain_epochs,
        batch_size=128,
        lr=0.01,
        verbose=True
    )

    # --------------------------------------------------------------------------
    # Phase 2: Pre-train Full Model (Self-Supervised)
    # Using link prediction, optional feature reconstruction, alignment, etc.
    # --------------------------------------------------------------------------
    print("\n=== Phase 2: Self-Supervised Pre-training the GNN ===")
    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(full_pretrain_epochs):
        model.train()
        pretrain_optimizer.zero_grad()
        # This call returns (embeddings, total_loss) for the tasks you enable:
        _, pretrain_loss = model.forward_and_loss(
            data,
            neg_sample_size=5,  # e.g. 5 negative edges per positive
            do_node_class=False,  # no classification yet
            do_linkpred=do_linkpred,  # link prediction
            do_featrec=do_featrec,  # feature reconstruction
            do_n2v_align=do_n2v_align  # align GNN & node2vec embeddings
        )
        pretrain_loss.backward()
        pretrain_optimizer.step()

        if epoch % 10 == 0:
            print(f"[Pretrain Epoch {epoch:03d}] Loss: {pretrain_loss.item():.4f}")

    def evaluate(model, classifier, data_x, edge_index, node_indices, labels, mask, verbose=True):
        model.eval()
        classifier.eval()
        with torch.no_grad():
            logits = classifier(model(data_x, edge_index, node_indices))
            preds = logits[mask].argmax(dim=1)
            true = labels[mask]

            preds = preds.cpu()
            true = true.cpu()

            acc = accuracy_score(true, preds)
            precision = precision_score(true, preds, average='macro', zero_division=0)
            recall = recall_score(true, preds, average='macro', zero_division=0)
            f1 = f1_score(true, preds, average='macro', zero_division=0)

            if verbose:
                print(f"  → Accuracy:  {acc:.4f}")
                print(f"  → Precision: {precision:.4f}")
                print(f"  → Recall:    {recall:.4f}")
                print(f"  → F1 Score:  {f1:.4f}")

            return acc  # Keep returning acc unless you want to return all metrics

    # --------------------------------------------------------------------------
    # Phase 3: Fine-tune for Node Classification
    # --------------------------------------------------------------------------
    print("\n=== Phase 3: Fine-tuning for Node Classification ===")
    classifier = torch.nn.Linear(output_dim, labels.unique().numel()).to(device)

    finetune_optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=0.01, weight_decay=5e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    data_x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    labels = labels.to(device)

    for epoch in range(finetune_epochs):
        model.train()
        classifier.train()
        finetune_optimizer.zero_grad()

        # Generate embeddings
        embeddings = model(data_x, edge_index, node_indices)
        # Classify
        logits = classifier(embeddings)

        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        finetune_optimizer.step()

        if epoch % 10 == 0:
            # Validation
            model.eval()
            classifier.eval()
            with torch.no_grad():
                print(f"[Fine-tune Epoch {epoch:03d}] Loss: {loss.item():.4f}")
                evaluate(model, classifier, data_x, edge_index, node_indices, labels, val_mask)

    # --------------------------------------------------------------------------
    # Final Test Evaluation
    # --------------------------------------------------------------------------
    model.eval()
    classifier.eval()
    with torch.no_grad():
        test_embeddings = model(data_x, edge_index, node_indices)
        test_logits = classifier(test_embeddings)
        test_preds = test_logits[test_mask].argmax(dim=1)
        test_acc = evaluate(model, classifier, data_x, edge_index, node_indices, labels, test_mask)
    print("\nFinal Test Evaluation:")

    return model, classifier, test_acc
