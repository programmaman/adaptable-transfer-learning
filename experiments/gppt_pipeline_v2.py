"""
GPPT Experiment Script - Following Official Implementation
Implements two-phase training: Pre-training + Prompt Tuning
Based on official GPPT repository: https://github.com/MingChen-Sun/GPPT
"""

import argparse
import csv
import os
import sys
import time
import logging
import yaml

import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import sklearn.metrics as skm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gppt import GraphSAGE, SAGE, NegativeSampler
from tasks.NodeClassificationTask import NodeClassificationTask
from Utilities import (
    generate_synthetic_graph,
    get_device,
    generate_dgl_compatible_synthetic_graph,
    add_synthetic_labels_if_missing,
    prepare_graph_for_dgl,
    create_dgl_graph_with_reverse_mapping,
    load_experiment_config,
    setup_logging,
    set_global_seed,
    log_experiment_metadata,
    save_model_checkpoint,
    load_model_checkpoint,
)

# Set up unified logger
logger = setup_logging()


def seed_torch(seed=42):
    """Set random seed for reproducibility"""
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dgl_graph_from_synthetic(num_nodes=200, num_edges=500, feature_dim=16):
    """Create a DGL graph from synthetic data"""
    # Use DGL-compatible synthetic graph generation
    g = generate_dgl_compatible_synthetic_graph(num_nodes, num_edges, feature_dim)
    g = add_synthetic_labels_if_missing(g, num_classes=2)

    # Extract features and labels
    node_features = g.ndata["feat"]
    if "label" in g.ndata:
        labels = g.ndata["label"]
    else:
        labels = torch.randint(0, 2, (g.number_of_nodes(),))

    # Create reverse edge mapping for DGL
    src_nodes, dst_nodes = g.edges()
    g_with_mapping, reverse_eid_map = create_dgl_graph_with_reverse_mapping(
        src_nodes, dst_nodes, num_nodes
    )

    # Copy node data to the new graph
    g_with_mapping.ndata["feat"] = node_features
    g_with_mapping.ndata["label"] = labels

    # Create train/val/test splits (following official implementation)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 60% train, 20% val, 20% test
    indices = torch.randperm(num_nodes)
    train_end = int(0.6 * num_nodes)
    val_end = int(0.8 * num_nodes)

    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    g_with_mapping.ndata["train_mask"] = train_mask
    g_with_mapping.ndata["val_mask"] = val_mask
    g_with_mapping.ndata["test_mask"] = test_mask

    return g_with_mapping, node_features, labels, reverse_eid_map


class CrossEntropyLoss(torch.nn.Module):
    """Cross entropy loss for edge prediction (from official implementation)"""

    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata["h"] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            pos_score = pos_graph.edata["score"]
        with neg_graph.local_scope():
            neg_graph.ndata["h"] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            neg_score = neg_graph.edata["score"]

        score = torch.cat([pos_score, neg_score])
        label = torch.cat(
            [torch.ones_like(pos_score), torch.zeros_like(neg_score)]
        ).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss


def run_pretraining(g, features, device, args, reverse_eid_map=None):
    """Phase 1: Unsupervised pre-training using edge prediction"""
    logger.info("=== Starting Pre-training Phase ===")

    # Add self-loops and prepare graph (following official implementation)
    g = dgl.add_self_loop(g)

    # Update reverse_eid_map if we added self-loops
    if reverse_eid_map is not None:
        # Extend reverse_eid_map for self-loops
        num_self_loops = g.number_of_nodes()
        self_loop_map = torch.arange(
            g.number_of_edges() - num_self_loops, g.number_of_edges()
        )
        reverse_eid_map = torch.cat([reverse_eid_map, self_loop_map])

    # Create SAGE model for pre-training (following official implementation)
    model = SAGE(
        features.shape[1],
        args.n_hidden,
        args.n_hidden,  # Output dimension for pre-training
        args.n_layers,
        F.relu,
        args.dropout,
        args.aggregator_type,
    )
    model = model.to(device)

    # Create edge data loader for unsupervised training
    n_edges = g.number_of_edges()
    train_seeds = torch.arange(n_edges)

    # Create sampler for mini-batch training
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )

    # Create negative sampler
    neg_sampler = NegativeSampler(g, args.num_negs, args.neg_share)

    # Attach reverse_eid_map to the graph for DGL to use
    logger.debug(f"reverse_eid_map type: {type(reverse_eid_map)}")
    logger.debug(f"reverse_eid_map shape: {reverse_eid_map.shape}")
    logger.debug(f"reverse_eid_map is None: {reverse_eid_map is None}")
    logger.debug(f"Graph num_edges: {g.number_of_edges()}")

    g.edata["reverse_id"] = reverse_eid_map

    # Verify the reverse mapping is attached
    logger.debug(f"Graph edata keys: {list(g.edata.keys())}")
    if "reverse_id" in g.edata:
        logger.debug(f"Graph reverse_id shape: {g.edata['reverse_id'].shape}")
    else:
        logger.warning("reverse_id not found in graph edata!")

    # Create data loader with proper DGL EdgeDataLoader arguments
    # Try without reverse_id exclusion first to test basic functionality
    dataloader = dgl.dataloading.EdgeDataLoader(
        g,
        train_seeds,
        sampler,
        negative_sampler=neg_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # Initialize loss function and optimizer
    loss_fcn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Pre-training loop
    model.train()
    for epoch in range(args.pretrain_epochs):
        total_loss = 0
        num_batches = 0

        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
            batch_inputs = features[input_nodes].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.int().to(device) for block in blocks]

            # Forward pass
            batch_pred = model(blocks, batch_inputs)

            # Compute loss
            loss = loss_fcn(batch_pred, pos_graph, neg_graph)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if step % args.log_every == 0:
                logger.info(
                    f"Epoch {epoch:03d} | Step {step:05d} | Loss {loss.item():.4f}"
                )

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch:03d} | Avg Loss {avg_loss:.4f}")

    # Save pre-trained model checkpoint (standardized)
    save_model_checkpoint(model, f"./logs/pretrained/synthetic_model_{args.file_id}.pt")
    logger.info(
        f"Pre-trained model saved to ./logs/pretrained/synthetic_model_{args.file_id}.pt"
    )

    return model


def run_prompt_tuning(g, features, labels, device, args):
    """Phase 2: Prompt tuning using pre-trained model"""
    logger.info("=== Starting Prompt Tuning Phase ===")

    # Get split masks
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    train_nid = torch.nonzero(train_mask, as_tuple=False).squeeze().to(torch.int64)
    val_nid = torch.nonzero(val_mask, as_tuple=False).squeeze().to(torch.int64)
    test_nid = torch.nonzero(test_mask, as_tuple=False).squeeze().to(torch.int64)

    n_classes = len(torch.unique(labels))

    # Create GraphSAGE model for prompt tuning
    model = GraphSAGE(
        features.shape[1],
        args.n_hidden,
        n_classes,
        args.n_layers,
        F.relu,
        args.dropout,
        args.aggregator_type,
        args.center_num,
    )
    model = model.to(device)

    # Load pre-trained parameters
    model_path = f"./logs/pretrained/synthetic_model_{args.file_id}.pt"
    if os.path.exists(model_path):
        logger.info(f"Loading pre-trained model from {model_path}")
        pretrained_dict = torch.load(model_path, map_location=device)
        model_dict = model.state_dict()

        # Filter out prompt-related parameters and load only pre-trained backbone
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and "prompt" not in k and "pp" not in k
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    else:
        logger.warning(f"Pre-trained model not found at {model_path}")

    # Initialize prompt weights
    model.weigth_init(g, features, labels, train_nid)

    # Create data loader for prompt tuning
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args.sample_list)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid.long(),
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # Initialize optimizer for prompt tuning
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Training loop
    model.train()
    acc_history = []
    loss_history = []

    for epoch in range(args.n_epochs):
        total_loss = 0
        num_batches = 0

        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            inputs = mfgs[0].srcdata["feat"]
            lab = mfgs[-1].dstdata["label"]

            # Forward pass
            logits = model(mfgs, inputs)

            # Compute loss with constraint regularization
            loss = F.cross_entropy(logits, lab)

            # Add constraint regularization for prompt orthogonality
            constraint_loss = constraint_regularization(device, model.get_mul_prompt())
            loss = loss + args.lr_c * constraint_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update prompt weights using K-means
            model.update_prompt_weight(model.get_mid_h())

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Evaluation
        model.eval()
        with torch.no_grad():
            acc = evaluate_model(model, g, features, labels, test_nid, device, args)
        model.train()

        acc_history.append(acc)
        loss_history.append(avg_loss)

        logger.info(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | Test Acc {acc:.4f}")

    return model, acc_history, loss_history


def constraint_regularization(device, prompt_list):
    """Constraint regularization for prompt orthogonality (from official implementation)"""
    if isinstance(prompt_list, list):
        total_constraint = 0
        for p in prompt_list:
            total_constraint += torch.norm(
                torch.mm(p, p.transforms) - torch.eye(p.shape[0]).to(device)
            )
        return total_constraint / len(prompt_list)
    else:
        return torch.norm(
            torch.mm(prompt_list, prompt_list.transforms)
            - torch.eye(prompt_list.shape[0]).to(device)
        )


def evaluate_model(model, g, features, labels, test_nid, device, args):
    """Evaluate model accuracy"""
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args.sample_list)
    test_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        test_nid.long(),
        sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        device=device,
    )

    predictions = []
    true_labels = []

    with torch.no_grad():
        for input_nodes, output_nodes, mfgs in test_dataloader:
            inputs = mfgs[0].srcdata["feat"]
            labels_batch = mfgs[-1].dstdata["label"]

            logits = model(mfgs, inputs)
            predictions.append(logits.argmax(1).cpu().numpy())
            true_labels.append(labels_batch.cpu().numpy())

    predictions = np.concatenate(predictions)
    true_labels = np.concatenate(true_labels)

    accuracy = skm.accuracy_score(true_labels, predictions)
    return accuracy


def main(args):
    """Main function to run GPPT experiment"""
    logger.info("Starting GPPT Experiment")
    logger.info(f"Arguments: {args}")
    # Optionally load config from YAML if specified
    config_path = getattr(args, "config", None)
    if config_path and os.path.exists(config_path):
        config = load_experiment_config(config_path)
        logger.info(f"Loaded experiment config from {config_path}")
        # Override args with config values
        for k, v in config.items():
            setattr(args, k, v)
    # Set global seed for reproducibility
    set_global_seed(args.seed)
    # Log experiment metadata for reproducibility
    log_experiment_metadata(log_dir="./logs/test_gppt", config=vars(args))
    # Get device
    device = get_device(args.gpu)
    logger.info(f"Using device: {device}")

    # Create synthetic graph
    g, features, labels, reverse_eid_map = create_dgl_graph_from_synthetic(
        num_nodes=args.num_nodes, num_edges=args.num_edges, feature_dim=args.feature_dim
    )

    logger.info(
        f"Created graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges"
    )
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Number of classes: {len(torch.unique(labels))}")

    # Move data to device
    features = features.to(device)
    labels = labels.to(device)
    g = g.to(device)

    # Phase 1: Pre-training
    pretrained_model = run_pretraining(g, features, device, args, reverse_eid_map)

    # Phase 2: Prompt tuning
    model, acc_history, loss_history = run_prompt_tuning(
        g, features, labels, device, args
    )

    # Save results
    results = {
        "accuracy_history": acc_history,
        "loss_history": loss_history,
        "final_accuracy": acc_history[-1] if acc_history else 0.0,
        "args": vars(args),
    }
    # Save to CSV
    os.makedirs("./logs/test_gppt", exist_ok=True)
    results_df = pd.DataFrame(
        {
            "epoch": range(len(acc_history)),
            "accuracy": acc_history,
            "loss": loss_history,
        }
    )
    results_df.to_csv("./logs/test_gppt/results.csv", index=False)
    logger.info(f"Final test accuracy: {acc_history[-1]:.4f}")
    logger.info("Results saved to ./logs/test_gppt/results.csv")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPPT Experiment")

    # Data parameters
    parser.add_argument(
        "--num_nodes", type=int, default=200, help="Number of nodes in synthetic graph"
    )
    parser.add_argument(
        "--num_edges", type=int, default=500, help="Number of edges in synthetic graph"
    )
    parser.add_argument("--feature_dim", type=int, default=16, help="Feature dimension")

    # Model parameters
    parser.add_argument(
        "--n_hidden", type=int, default=128, help="Number of hidden units"
    )
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--aggregator_type", type=str, default="gcn", help="Aggregator type"
    )
    parser.add_argument(
        "--center_num", type=int, default=5, help="Number of prompt centers"
    )

    # Training parameters
    parser.add_argument(
        "--pretrain_epochs", type=int, default=10, help="Number of pre-training epochs"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=30, help="Number of prompt tuning epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--lr_c", type=float, default=0.01, help="Constraint loss weight"
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")

    # Sampling parameters
    parser.add_argument(
        "--fan_out", type=str, default="10,25", help="Fan-out for sampling"
    )
    parser.add_argument(
        "--sample_list",
        type=list,
        default=[4, 4],
        help="Sample list for neighbor sampling",
    )
    parser.add_argument(
        "--num_negs", type=int, default=2, help="Number of negative samples"
    )
    parser.add_argument(
        "--neg_share",
        default=False,
        action="store_true",
        help="Whether to share negative samples",
    )

    # System parameters
    parser.add_argument("--gpu", type=int, default=-1, help="GPU to use, -1 for CPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for data loading"
    )
    parser.add_argument("--log_every", type=int, default=20, help="Log every N steps")
    parser.add_argument(
        "--file_id", type=str, default="synthetic", help="File ID for saving model"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to experiment config YAML file"
    )

    args = parser.parse_args()
    main(args)
