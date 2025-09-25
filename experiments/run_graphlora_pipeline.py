import os
import time

import pandas as pd
import yaml
import torch
from yaml import SafeLoader

from experiments.graph_lora import save_results_to_excel, build_args
from experiments.pipeline import DefaultPipeline, GraphLoRAPipeline

from experiments.experiment_utils import (
    generate_synthetic_graph,
    load_musae_facebook_dataset,
    load_email_eu_core_dataset,
    load_musae_github_dataset,
    load_deezer_europe_dataset,

)
from models.GNNLorA import GraphLoRAWrapped

import pandas as pd
import os


def summarize_results(input_file="results.csv", output_file="reduced_results_summary.csv"):
    ext = os.path.splitext(input_file)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(input_file)
    elif ext == ".xlsx":
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .csv and .xlsx are supported.")

    grouped = df.groupby("Dataset")
    metrics = [
        "Classification_Acc", "Classification_F1",
        "LinkPred_AUC", "LinkPred_AP",
        "Metadata.classifier_time", "Metadata.link_pred_time", "Metadata.total_time"
    ]

    mean_df = grouped[metrics].mean().rename(columns=lambda x: x + "_mean")
    std_df = grouped[metrics].std().rename(columns=lambda x: x + "_std")
    summary_df = pd.concat([mean_df, std_df], axis=1)

    def format_metric(mean, std):
        return f"{mean:.4f} ± {std:.4f}"

    for metric in metrics:
        summary_df[metric] = [
            format_metric(m, s)
            for m, s in zip(summary_df[metric + "_mean"], summary_df[metric + "_std"])
        ]

    summary_df = summary_df[metrics]
    summary_df.reset_index(inplace=True)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(output_file, index=False)


def run_graphlora_pipeline(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load config file
    with open(args.para_config, "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    results = []

    NUM_RUNS = 3

    for run in range(NUM_RUNS):
        print(f"\n=================== GraphLoRA Pipeline Run {run} ===================")

        # ---------------- Synthetic ----------------
        data, labels = generate_synthetic_graph(
            num_nodes=args.num_nodes,
            num_edges=args.num_edges,
            feature_dim=args.feature_dim,
        )
        synth_result = run_single_experiment(args, config, "Synthetic", data, labels, device)
        synth_result.update({"Experiment": "Synthetic", "Run": run})
        results.append(synth_result)

        # ---------------- Facebook ----------------
        fb_dir = os.path.join(os.path.dirname(__file__), "../datasets/facebook_large")
        fb_data, fb_labels, _ = load_musae_facebook_dataset(
            os.path.join(fb_dir, "musae_facebook_edges.csv"),
            os.path.join(fb_dir, "musae_facebook_features.json"),
            os.path.join(fb_dir, "musae_facebook_target.csv"),
        )
        fb_result = run_single_experiment(args, config, "Facebook", fb_data, fb_labels, device)
        fb_result.update({"Experiment": "Facebook", "Run": run})
        results.append(fb_result)

        # ---------------- Email-EU-Core ----------------
        email_dir = os.path.join(os.path.dirname(__file__), "../datasets/email-eu-core")
        email_data, email_labels = load_email_eu_core_dataset(
            os.path.join(email_dir, "email-Eu-core.txt"),
            os.path.join(email_dir, "email-Eu-core-department-labels.txt"),
        )
        email_result = run_single_experiment(args, config, "Email-EU-Core", email_data, email_labels, device)
        email_result.update({"Experiment": "Email-EU-Core", "Run": run})
        results.append(email_result)

        # ---------------- GitHub ----------------
        gh_dir = os.path.join(os.path.dirname(__file__), "../datasets/git_web_ml")
        gh_data, gh_labels, _ = load_musae_github_dataset(
            os.path.join(gh_dir, "musae_git_edges.csv"),
            os.path.join(gh_dir, "musae_git_features.json"),
            os.path.join(gh_dir, "musae_git_target.csv"),
        )
        gh_result = run_single_experiment(args, config, "GitHub", gh_data, gh_labels, device)
        gh_result.update({"Experiment": "GitHub", "Run": run})
        results.append(gh_result)

        # ---------------- Deezer Europe ----------------
        deezer_dir = os.path.join(os.path.dirname(__file__), "../datasets/deezer_europe")
        dz_data, dz_labels = load_deezer_europe_dataset(
            os.path.join(deezer_dir, "deezer_europe_edges.csv"),
            os.path.join(deezer_dir, "deezer_europe_features.json"),
            os.path.join(deezer_dir, "deezer_europe_target.csv"),
        )
        dz_result = run_single_experiment(args, config, "Deezer Europe", dz_data, dz_labels, device)
        dz_result.update({"Experiment": "Deezer Europe", "Run": run})
        results.append(dz_result)

        print(f"Run {run} completed.")
        time.sleep(30)

    # Save results
    save_results_to_excel(results, args.output_file)
    print(f"\nAll GraphLoRA pipeline results saved to {args.output_file}")

    # Run summary
    summarize_results(input_file=args.output_file, output_file="summary_" + args.output_file)




def run_single_experiment(args, config, name, data, labels, device):
    in_dim = data.x.shape[1]
    out_dim = config["output_dim"]
    num_classes = len(labels.unique())

    base_model_path = f"./pre_trained_gnn/{name.lower()}_backbone.pth"

    model = GraphLoRAWrapped(
        in_dim=in_dim,
        out_dim=out_dim,
        num_classes=num_classes,
        base_model_path=base_model_path,  # now local, not args
        gnn_type=config.get("gnn_type", "GCN"),
        num_layers=2,
        r=args.r,
        activation=config.get("activation", "relu"),
    ).to(device)

    random_int = int(time.time())

    # Use GraphLoRAPipeline instead of DefaultPipeline
    pipeline = GraphLoRAPipeline(base_model_path, seed=random_int, device=device)
    model, class_results, lp_results = pipeline.run(
        data, labels, model,
        pretrain_epochs=100,
        finetune_epochs=30,
    )

    # Collect results into dict
    return {
        "Dataset": name,
        "Classification_Acc": class_results.accuracy,
        "Classification_F1": class_results.f1,
        "LinkPred_AUC": lp_results.auc,
        "LinkPred_AP": lp_results.ap,
        "Metadata": pipeline.metadata,
    }





if __name__ == "__main__":
    # args = build_args()
    # run_graphlora_pipeline(args)
    # Just excel
    summarize_results(input_file="/app/results/graphlora_results.xlsx", output_file="results/summary_graphlora.csv")
