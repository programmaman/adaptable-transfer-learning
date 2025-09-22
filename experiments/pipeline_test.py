# experiments/run_twitch_transfer.py

import os
import torch
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

from experiments.experiment_utils import split_edges_for_link_prediction
from models.baselines import SimpleGraphSAGE, SimpleGAT, SimpleGNN
from models.struct_g import StructuralGNN
from pipeline import StructGPipeline, TransferLearningPipeline, DefaultPipeline  # your StructuralPipeline class


# ------------------------------
# Dataset Loader (Twitch TL format)
# ------------------------------
import pandas as pd
import torch
import json
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder


def load_twitch_dataset(prefix, label_col="mature"):
    """
    Loads Twitch-TL dataset given a prefix like 'musae_ES' or 'musae_RU'.
    Expects files:
      {prefix}_edges.csv
      {prefix}_features.json
      {prefix}_target.csv

    Args:
        label_col (str): which column in *_target.csv to use as classification label.
                         (e.g., 'mature' or 'partner')
    """
    edge_path = f"{prefix}_edges.csv"
    features_path = f"{prefix}_features.json"
    target_path = f"{prefix}_target.csv"

    # ------------------------
    # Load edges
    # ------------------------
    edges_df = pd.read_csv(edge_path)
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)

    # ------------------------
    # Load features
    # ------------------------
    with open(features_path, "r") as f:
        features_dict = json.load(f)

    node_ids = sorted(set(int(k) for k in features_dict.keys()))
    node_id_map = {nid: i for i, nid in enumerate(node_ids)}

    num_nodes = len(node_ids)
    num_features = max(f for feats in features_dict.values() for f in feats) + 1
    x = torch.zeros((num_nodes, num_features))
    for raw_id, feats in features_dict.items():
        mapped_id = node_id_map[int(raw_id)]
        x[mapped_id, feats] = 1.0

    # ------------------------
    # Load labels
    # ------------------------
    target_df = pd.read_csv(target_path)

    # Use new_id instead of id for mapping
    if "new_id" not in target_df.columns:
        raise ValueError(f"'new_id' column not found in {target_path}")

    target_df = target_df[target_df["new_id"].isin(node_ids)]
    target_df["mapped_id"] = target_df["new_id"].map(node_id_map)

    if label_col not in target_df.columns:
        raise ValueError(
            f"Column '{label_col}' not found in {target_path}. "
            f"Available columns: {list(target_df.columns)}"
        )

    label_encoder = LabelEncoder()
    labels = torch.full((num_nodes,), -1, dtype=torch.long)

    encoded_labels = label_encoder.fit_transform(target_df[label_col].astype(str))
    labels[target_df["mapped_id"]] = torch.tensor(encoded_labels, dtype=torch.long)

    # ------------------------
    # Filter edges to mapped nodes
    # ------------------------
    edge_list = edge_index.t().tolist()
    filtered_edges = [
        [node_id_map[src], node_id_map[dst]]
        for src, dst in edge_list
        if src in node_id_map and dst in node_id_map
    ]
    edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()

    # ------------------------
    # Build Data object
    # ------------------------
    data = Data(x=x, edge_index=edge_index)

    print(f"Loaded graph {prefix}")
    print(f"  → {data.num_nodes} nodes, {data.num_edges} edges, {x.size(1)} features")
    print(f"  → Label coverage: {(labels >= 0).sum().item()} / {len(labels)} nodes labeled")
    print(f"  → Label column: {label_col} | Classes: {list(label_encoder.classes_)}")

    return data, labels, label_encoder



# ------------------------------
# Main Experiment
# ------------------------------
def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "twitch-tl"))
    source_prefix = os.path.join(base_dir, "ES", "musae_ES")
    target_prefix = os.path.join(base_dir, "RU", "musae_RU")

    # Load source (ES) and target (RU) datasets
    source_data, source_labels, _ = load_twitch_dataset(source_prefix, label_col="mature")
    target_data, target_labels, _ = load_twitch_dataset(target_prefix, label_col="mature")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===========================================================
    # StructuralGNN runs (what you already had)
    # ===========================================================
    pipeline = StructGPipeline(seed=42, device=device)

    print("\n=== Running Transfer Learning (StructuralGNN): Pretrain on ES → Finetune on RU ===")
    transfer_model = StructuralGNN(
        num_nodes=source_data.num_nodes,
        edge_index=source_data.edge_index,
        input_dim=source_data.x.size(1),
        hidden_dim=64,
        output_dim=32,
        embedding_dim=128,
        num_layers=2,
        use_gat=True,
        use_gate=True,
        num_classes=target_labels.unique().numel(),
        feat_reconstruction=True,
    ).to(device)

    transfer_model, transfer_cls_results, transfer_lp_results = pipeline.transfer_learning_run(
        source_data, source_labels,
        target_data, target_labels,
        transfer_model,
        pretrain_epochs=50,
        finetune_epochs=30,
    )

    print("\n=== StructuralGNN Transfer Learning Results ===")
    print("Classification:", transfer_cls_results)
    print("Link Prediction:", transfer_lp_results)

    print("\n=== Running Regular Training (StructuralGNN): Train/Eval on RU only ===")
    regular_model = StructuralGNN(
        num_nodes=target_data.num_nodes,
        edge_index=target_data.edge_index,
        input_dim=target_data.x.size(1),
        hidden_dim=64,
        output_dim=32,
        embedding_dim=128,
        num_layers=2,
        use_gat=True,
        use_gate=True,
        num_classes=target_labels.unique().numel(),
        feat_reconstruction=True,
    ).to(device)

    regular_model, regular_cls_results, regular_lp_results = pipeline.run(
        data=target_data,
        labels=target_labels,
        model=regular_model,
        pretrain_epochs=50,
        finetune_epochs=30,
    )

    print("\n=== StructuralGNN Regular Training Results ===")
    print("Classification:", regular_cls_results)
    print("Link Prediction:", regular_lp_results)

    # ===========================================================
    # Simple Models (DefaultPipeline + TransferLearningPipeline)
    # ===========================================================
    for ModelClass, name in [(SimpleGNN, "SimpleGNN"),
                             (SimpleGAT, "SimpleGAT"),
                             (SimpleGraphSAGE, "SimpleGraphSAGE")]:
        print(f"\n=== Running {name} Transfer Learning: Pretrain on ES → Finetune on RU ===")
        transfer_pipeline = TransferLearningPipeline(
            source_data, source_labels, seed=42, device=device
        )
        transfer_model = ModelClass(
            in_channels=source_data.x.size(1),
            out_channels=target_labels.unique().numel()
        ).to(device)

        transfer_model, transfer_cls_results, transfer_lp_results = transfer_pipeline.transfer_learning_run(
            source_data, source_labels,
            target_data, target_labels,
            transfer_model,
            pretrain_epochs=50,
            finetune_epochs=30,
        )
        print(f"\n=== {name} Transfer Learning Results ===")
        print("Classification:", transfer_cls_results)
        print("Link Prediction:", transfer_lp_results)

        print(f"\n=== Running {name} Regular Training: Train/Eval on RU only ===")
        regular_pipeline = DefaultPipeline(seed=42, device=device)
        regular_model = ModelClass(
            in_channels=target_data.x.size(1),
            out_channels=target_labels.unique().numel()
        ).to(device)

        regular_model, regular_cls_results, regular_lp_results = regular_pipeline.run(
            data=target_data,
            labels=target_labels,
            model=regular_model,
            pretrain_epochs=0,   # no pretraining for DefaultPipeline
            finetune_epochs=30,
        )
        print(f"\n=== {name} Regular Training Results ===")
        print("Classification:", regular_cls_results)
        print("Link Prediction:", regular_lp_results)


if __name__ == "__main__":
    main()
