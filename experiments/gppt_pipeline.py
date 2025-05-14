"""
experiments/promptsage_pipeline.py
Run Prompt‑SAGE (DGL implementation) inside the PyG‑based experiment framework.
"""

from __future__ import annotations
import gc, random, time
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import dgl
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from experiments.experiment_utils import (
    EvaluationResult,
    create_masks,                       # same helper you showed earlier
    sample_negative_edges,
    split_edges_for_link_prediction,
    set_global_seed                     # wrapper around np / torch / random seeding
)
from utils import get_device
from your_model_file import GraphSAGE   # <-- put real module path here


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def pyg_to_dgl(data):
    """Convert a PyG Data object to a (homogeneous) DGLGraph and copy node feats."""
    g = dgl.graph((data.edge_index[0], data.edge_index[1]),
                  num_nodes=data.num_nodes)
    g = dgl.to_simple(g)                      # remove multi‑edges
    g.ndata['feat'] = data.x
    return g


def dotprod_score(z, edges):
    """z: (N,D) tensor; edges: (...,2) long tensor → raw dot‑product scores."""
    return (z[edges[:, 0]] * z[edges[:, 1]]).sum(dim=-1)


def evaluate_classification(model: GraphSAGE,
                            g: dgl.DGLGraph,
                            labels: torch.Tensor,
                            mask: torch.Tensor,
                            device) -> EvaluationResult:
    model.eval()
    with torch.no_grad():
        emb = model.inference(g, g.ndata['feat'], device,
                              batch_size=4096, num_workers=0)
    preds = emb.argmax(dim=1)[mask].cpu()
    true  = labels[mask].cpu()

    acc  = accuracy_score(true, preds)
    prec = precision_score(true, preds, average="macro", zero_division=0)
    rec  = recall_score(true, preds, average="macro", zero_division=0)
    f1   = f1_score(true, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(true, preds, multi_class="ovr", average="macro")
    except ValueError:
        auc = None

    return EvaluationResult(acc, prec, rec, f1, auc)


def evaluate_link_prediction(model: GraphSAGE,
                             g: dgl.DGLGraph,
                             rem_edge_list,
                             device) -> EvaluationResult:
    model.eval()
    with torch.no_grad():
        emb = model.inference(g, g.ndata['feat'], device,
                              batch_size=4096, num_workers=0)

    pos_edges = rem_edge_list[0][0].to(device)          # (E_pos, 2)
    neg_edges = sample_negative_edges(pos_edges, g.num_nodes()).to(device)

    pos_scores = dotprod_score(emb, pos_edges)
    neg_scores = dotprod_score(emb, neg_edges)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones_like(pos_scores),
                        torch.zeros_like(neg_scores)])

    preds_bin = (torch.sigmoid(scores) > 0.5).float()

    acc  = accuracy_score(labels.cpu(), preds_bin.cpu())
    prec = precision_score(labels.cpu(), preds_bin.cpu(), zero_division=0)
    rec  = recall_score(labels.cpu(), preds_bin.cpu(), zero_division=0)
    f1   = f1_score(labels.cpu(), preds_bin.cpu(), zero_division=0)
    auc  = roc_auc_score(labels.cpu(), scores.cpu())
    ap   = average_precision_score(labels.cpu(), scores.cpu())

    return EvaluationResult(acc, prec, rec, f1, auc, ap)


def finetune_link_prediction(model: GraphSAGE,
                             g: dgl.DGLGraph,
                             rem_edge_list,
                             epochs: int,
                             lr: float,
                             weight_decay: float,
                             device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    pos_edges = rem_edge_list[0][0].to(device)

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        emb = model.forward_smc(g, g.ndata['feat'])
        neg_edges = sample_negative_edges(pos_edges, g.num_nodes()).to(device)
        pos_s = dotprod_score(emb, pos_edges)
        neg_s = dotprod_score(emb, neg_edges)
        loss  = F.binary_cross_entropy_with_logits(
            torch.cat([pos_s, neg_s]),
            torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)]))
        loss.backward()
        optimizer.step()
    return model


# --------------------------------------------------------------------------- #
# main public entry‑point                                                     #
# --------------------------------------------------------------------------- #
def run_promptsage_pipeline(
        data,
        labels,
        seed: int = 42,
        n_hidden: int = 128,
        n_layers: int = 2,
        center_num: int = 4,
        dropout: float = 0.2,
        sample_list=(15, 10),
        cls_epochs: int = 50,
        lp_finetune_epochs: int = 20,
        lr: float = 2e-3,
        weight_decay: float = 5e-4,
        do_linkpred: bool = True
) -> Tuple[GraphSAGE, EvaluationResult, Optional[EvaluationResult]]:

    # ---------- setup ------------------------------------------------------
    set_global_seed(seed)
    device = get_device()
    print(f"Prompt‑SAGE | device={device} | seed={seed}")

    # masks & edge split (PyG tensors)
    train_mask, val_mask, test_mask = create_masks(data.num_nodes, device=device)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    data.edge_index, rem_edge_list = split_edges_for_link_prediction(
        data.edge_index, removal_ratio=0.3)

    # ---------- convert to DGL --------------------------------------------
    g = pyg_to_dgl(data).to(device)
    g.ndata['label'] = labels.to(device)

    # ---------- model ------------------------------------------------------
    model = GraphSAGE(
        in_feats=data.x.size(1),
        n_hidden=n_hidden,
        n_classes=int(labels.max()) + 1,
        classes=int(labels.max()) + 1,
        n_layers=n_layers,
        activation=F.relu,
        dropout=dropout,
        aggregator_type="mean",
        center_num=center_num
    ).to(device)

    # minimal args stub so we can reuse original helper methods
    class _Args: pass
    args = _Args()
    args.dataset = "tmp"; args.file_id = "0"; args.seed = seed
    args.n_hidden = n_hidden; args.n_layers = n_layers
    args.dropout = dropout; args.aggregator_type = "mean"
    args.center_num = center_num
    args.batch_size = 1024
    args.sample_list = list(sample_list)
    args.lr = lr; args.weight_decay = weight_decay
    args.n_epochs = cls_epochs; args.lr_c = 0.
    model.load_parameters(args)
    model.weigth_init(g, g.ndata['feat'], g.ndata['label'],
                      train_mask.nonzero(as_tuple=False).squeeze())

    # ---------- supervised training for classification ---------------------
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_list)
    train_loader = dgl.dataloading.NodeDataLoader(
        g, train_mask.nonzero().squeeze(), sampler,
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(cls_epochs):
        model.train()
        for _, _, blocks in train_loader:
            inputs = blocks[0].srcdata['feat']
            tgt    = blocks[-1].dstdata['label']
            logits = model(blocks, inputs)
            loss   = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()

    # ---------- evaluation -------------------------------------------------
    cls_res = evaluate_classification(model, g, labels, test_mask, device)

    # ---------- optional LP fine‑tune & eval -------------------------------
    if do_linkpred:
        model = finetune_link_prediction(
            model, g, rem_edge_list,
            epochs=lp_finetune_epochs,
            lr=lr, weight_decay=weight_decay,
            device=device)
        lp_res = evaluate_link_prediction(model, g, rem_edge_list, device)
    else:
        lp_res = None

    # ---------- tidy up ----------------------------------------------------
    torch.cuda.empty_cache(); gc.collect()
    return model, cls_res, lp_res
