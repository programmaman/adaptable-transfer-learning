#!/usr/bin/env python3
import os
import time
import yaml
import torch
import pandas as pd
from utils import get_device
from yaml import SafeLoader
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.transforms import SVDFeatureReduction

# Import dataset loaders
from experiments.experiment_utils import (
    generate_synthetic_graph,
    load_musae_facebook_dataset,
    load_email_eu_core_dataset,
    load_twitch_gamers_dataset,
    load_deezer_europe_dataset,
    load_musae_github_dataset,
)

# Import model and utils
from models.GNNLorA import transfer, GNN, act, get_parameter, GNNLoRA


def create_masks(num_nodes: int, train_ratio: float = 0.6, val_ratio: float = 0.8, device=None):
    if device is None:
        device = get_device()
    indices = torch.randperm(num_nodes, device=device)
    train_cut = int(train_ratio * num_nodes)
    val_cut = int(val_ratio * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[indices[:train_cut]] = True
    val_mask[indices[train_cut:val_cut]] = True
    test_mask[indices[val_cut:]] = True
    return train_mask, val_mask, test_mask

def build_args():
    import argparse
    parser = argparse.ArgumentParser(description="GraphLoRA Multi-Dataset Experiments")
    parser.add_argument("--pretrain_dataset", type=str, default="synthetic_pretrain")
    parser.add_argument("--test_dataset", type=str, default="synthetic_test")
    parser.add_argument("--pretext", type=str, default="GraphLoRA")
    parser.add_argument("--para_config", type=str, default="./config.yaml")
    parser.add_argument("--is_reduction", action="store_true")
    parser.add_argument("--few", action="store_true")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=2000)
    parser.add_argument("--num_edges", type=int, default=4000)
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--force_pretrain", action="store_true")
    parser.add_argument("--sup_weight", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--output_file", type=str, default="/app/results/graphlora_results.xlsx")
    return parser.parse_args()


def pretrain_and_save(args, config, pretrain_data):
    """Runs pretraining step and saves GNN weights for transfer() to load."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    start_time = time.time()

    pretrain_data = pretrain_data.to(device)

    gnn = GNN(
        pretrain_data.x.shape[1],
        config["output_dim"],
        act(config["activation"]),
        config["gnn_type"],
        config["num_layers"]
    ).to(device)

    decoder = torch.nn.Linear(config["output_dim"], pretrain_data.x.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(decoder.parameters()),  # ✅ optimize both
        lr=0.01,
        weight_decay=5e-4
    )

    gnn.train()
    decoder.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = gnn(pretrain_data.x, pretrain_data.edge_index)  # now both are on GPU
        recon = decoder(out)
        loss = torch.nn.functional.mse_loss(recon, pretrain_data.x)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"[Pretrain] Epoch {epoch + 1:03d} | Loss: {loss.item():.4f}")

    os.makedirs("./pre_trained_gnn", exist_ok=True)
    model_path = f"./pre_trained_gnn/{args.pretrain_dataset}.{args.pretext}.{config['gnn_type']}.{args.is_reduction}.pth"
    torch.save(gnn.state_dict(), model_path)
    print(f"[Pretrain] Saved GNN checkpoint to {model_path}")
    print(f"[Pretrain] Completed in {time.time() - start_time:.2f} seconds")



def run_graphlora_on_dataset(args, config, dataset_name, data, labels):
    print(f"\n========== [GraphLoRA] {dataset_name} ==========")
    start_time = time.time()
    if not hasattr(data, "y") or data.y is None:
        data.y = labels
    args.test_dataset = dataset_name
    args.pretrain_dataset = f"{dataset_name.lower()}_pretrain"
    setting = "few" if args.few else "public"
    if setting in config and dataset_name in config[setting]:
        args = get_parameter(args)
    else:
        args.wd1, args.wd2, args.wd3 = 0.0, 0.0, 0.0
        args.lr1, args.lr2, args.lr3 = 0.01, 0.01, 0.01
        args.l1, args.l2, args.l3, args.l4 = 1.0, 1.0, 1.0, 1.0
        args.num_epochs = 30
    model_path = f"./pre_trained_gnn/{args.pretrain_dataset}.{args.pretext}.{config['gnn_type']}.{args.is_reduction}.pth"
    if args.force_pretrain or not os.path.exists(model_path):
        pretrain_and_save(args, config, data)
    else:
        checkpoint = torch.load(model_path, map_location="cpu")
        first_key = next(iter(checkpoint))
        first_weight = checkpoint[first_key]
        if first_weight.shape[1] != data.x.shape[1]:
            pretrain_and_save(args, config, data)
    device = get_device()
    train_mask, val_mask, test_mask = create_masks(data.num_nodes, device=device)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    metrics = transfer(
        args,
        config,
        args.gpu,
        args.is_reduction,
        pretrain_dataset=data,
        test_dataset=data,
    )
    result = {
        "Experiment": dataset_name,
        "Pipeline": "GraphLoRA",
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "auc": metrics.get("auc"),
        "ap": metrics.get("ap"),
        "runtime_sec": time.time() - start_time,
    }
    return result

def save_results_to_excel(results, output_file):
    # Normalize all dicts so they have the same keys
    df = pd.json_normalize(results)

    # Convert problematic types (like tensors, lists, custom objects) to strings
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if not isinstance(x, (int, float, str, bool, type(None))) else x)

    # Write safely with ExcelWriter
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, index=False)
    print(f"[Info] Results successfully saved to {output_file}")

def main():
    args = build_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.para_config, "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    results = []
    for run in range(1):
        print(f"\n=================== GraphLoRA Run {run} ===================")
        # Synthetic Dataset
        data, labels = generate_synthetic_graph(
            num_nodes=args.num_nodes, num_edges=args.num_edges, feature_dim=args.feature_dim
        )
        synth_result = run_graphlora_on_dataset(args, config, "Synthetic", data, labels)
        synth_result.update({"Experiment": "Synthetic", "Run": run})
        results.append(synth_result)

        # Facebook Dataset
        fb_dir = os.path.join(os.path.dirname(__file__), "../datasets/facebook_large")
        fb_data, fb_labels, _ = load_musae_facebook_dataset(
            os.path.join(fb_dir, "musae_facebook_edges.csv"),
            os.path.join(fb_dir, "musae_facebook_features.json"),
            os.path.join(fb_dir, "musae_facebook_target.csv"),
        )
        fb_result = run_graphlora_on_dataset(args, config, "Facebook", fb_data, fb_labels)
        fb_result.update({"Experiment": "Facebook", "Run": run})
        results.append(fb_result)

        # Email-EU-Core
        email_dir = os.path.join(os.path.dirname(__file__), "../datasets/email-eu-core")
        email_data, email_labels = load_email_eu_core_dataset(
            os.path.join(email_dir, "email-Eu-core.txt"),
            os.path.join(email_dir, "email-Eu-core-department-labels.txt"),
        )
        email_result = run_graphlora_on_dataset(args, config, "Email-EU-Core", email_data, email_labels)
        email_result.update({"Experiment": "Email-EU-Core", "Run": run})
        results.append(email_result)

        # GitHub
        gh_dir = os.path.join(os.path.dirname(__file__), "../datasets/git_web_ml")
        gh_data, gh_labels, _ = load_musae_github_dataset(
            os.path.join(gh_dir, "musae_git_edges.csv"),
            os.path.join(gh_dir, "musae_git_features.json"),
            os.path.join(gh_dir, "musae_git_target.csv"),
        )
        gh_result = run_graphlora_on_dataset(args, config, "GitHub", gh_data, gh_labels)
        gh_result.update({"Experiment": "GitHub", "Run": run})
        results.append(gh_result)

        # Deezer
        deezer_dir = os.path.join(os.path.dirname(__file__), "../datasets/deezer_europe")
        dz_data, dz_labels = load_deezer_europe_dataset(
            os.path.join(deezer_dir, "deezer_europe_edges.csv"),
            os.path.join(deezer_dir, "deezer_europe_features.json"),
            os.path.join(deezer_dir, "deezer_europe_target.csv"),
        )
        dz_result = run_graphlora_on_dataset(args, config, "Deezer", dz_data, dz_labels)
        dz_result.update({"Experiment": "Deezer Europe", "Run": run})
        results.append(dz_result)

        print(f"Run {run} completed.")
        time.sleep(30)  # allow GPU to cool down

    # Save results to Excel
    save_results_to_excel(results, args.output_file)
    print(f"\nAll GraphLoRA results saved to {args.output_file}")


if __name__ == "__main__":
    main()
