import time
import dgl
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn.functional as f

import utils  # Utility functions: seed_torch, evaluate, get_init_info
from models.gppt import GraphSAGE


def pyg_to_dgl(data):
    edge_index = data.edge_index
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=data.num_nodes)
    g.ndata['feat'] = data.x
    g.ndata['label'] = data.y
    return g


def run_gppt_pipeline(data, labels,
                      hidden_dim=64,
                      n_layers=2,
                      dropout=0.2,
                      aggregator_type='mean',
                      center_num=4,
                      lr=0.005,
                      weight_decay=5e-4,
                      lr_c=0.01,
                      batch_size=512,
                      n_epochs=100,
                      sample_list=[10, 10],
                      seed=42,
                      dataset_name='dataset',
                      file_id='exp01'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert PyG Data to DGL
    g = pyg_to_dgl(data)
    g = g.to(device)
    features = g.ndata['feat']
    labels = g.ndata['label']

    # Train/val/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_idx, test_val_idx = next(sss.split(np.zeros(labels.shape[0]), labels.cpu()))
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(sss_val.split(np.zeros(len(test_val_idx)), labels[test_val_idx].cpu()))

    train_nid = torch.tensor(train_idx).to(device)
    val_nid = torch.tensor(test_val_idx[val_idx]).to(device)
    test_nid = torch.tensor(test_val_idx[test_idx]).to(device)

    # DGL sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_list)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid.int(),
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    # Init model
    model = GraphSAGE(
        in_feats=features.shape[1],
        n_hidden=hidden_dim,
        n_classes=len(labels.unique()),
        n_layers=n_layers,
        activation=torch.relu,
        dropout=dropout,
        aggregator_type=aggregator_type,
        center_num=center_num
    ).to(device)

    # Initialize weights
    model.weigth_init(g, features, labels, train_nid)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    acc_all = []
    loss_all = []

    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()

        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            inputs = mfgs[0].srcdata['feat']
            lab = mfgs[-1].dstdata['label']
            logits = model(mfgs, inputs)
            loss = f.cross_entropy(logits, lab)

            loss = loss + lr_c * utils.constraint(device, model.get_mul_prompt())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_prompt_weight(model.get_mid_h())

        acc = utils.evaluate(model, g, test_nid, batch_size, device, sample_list)
        acc_all.append(acc)
        loss_all.append(loss.item())

        print(f"Epoch {epoch:03d} | Time(s) {time.time() - t0:.4f} | Loss {loss.item():.4f} | Accuracy {acc:.4f}")

    # Final test accuracy
    test_acc = np.mean(acc_all[-10:])
    print("Test Accuracy {:.4f}".format(test_acc))
    return model, test_acc
