import os
import pandas as pd
import torch

from experiments.graph_bert_pipeline import run_graphbert_pipeline
from experiments.experiment_utils import (
    generate_synthetic_graph,
    load_musae_facebook_dataset,
    load_email_eu_core_dataset,
    load_musae_github_dataset,
    load_deezer_europe_dataset,
)
from torch_geometric.utils import to_networkx
import networkx as nx


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_graphbert_experiment(data, labels, seed=42):
    """Helper to run GraphBERT on a dataset and return results dicts."""
    print(f"\n=== Running GraphBERT Experiment (seed={seed}) ===")
    model, cls_results, lp_results = run_graphbert_pipeline(
        data=data,
        labels=labels,
        d_model=128,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
        epochs=30,
        lr=1e-3,
        weight_decay=5e-4,
        do_linkpred=True,
        seed=seed,
    )
    cls_dict = {"Pipeline": "GraphBERT", **cls_results.as_dict(), "Experiment": "GraphBERT"}
    lp_dict = {"Pipeline": "GraphBERT", **lp_results.as_dict(), "Experiment": "GraphBERT"} if lp_results else None
    return cls_dict, lp_dict


def run_all_graphbert_experiments(output_file="/app/results/graphbert_results.xlsx"):
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        results_cls, results_lp = [], []

        # 1) Synthetic
        print("\n=================== Synthetic Dataset ===================")
        data, labels = generate_synthetic_graph()
        cls, lp = run_graphbert_experiment(data, labels)
        results_cls.append({"Experiment": "Synthetic", **cls})
        if lp: results_lp.append({"Experiment": "Synthetic", **lp})

        # 2) Facebook
        print("\n=================== Facebook Dataset ===================")
        fb_dir = os.path.join(BASE_DIR, "../datasets/facebook_large")
        data, labels, _ = load_musae_facebook_dataset(
            os.path.join(fb_dir, "musae_facebook_edges.csv"),
            os.path.join(fb_dir, "musae_facebook_features.json"),
            os.path.join(fb_dir, "musae_facebook_target.csv"),
        )

        cls, lp = run_graphbert_experiment(data, labels)
        results_cls.append({"Experiment": "Facebook", **cls})
        if lp: results_lp.append({"Experiment": "Facebook", **lp})

        # 3) Email-EU-Core
        print("\n=================== Email-EU-Core Dataset ===================")
        email_dir = os.path.join(BASE_DIR, "../datasets/email-eu-core")
        data, labels = load_email_eu_core_dataset(
            os.path.join(email_dir, "email-Eu-core.txt"),
            os.path.join(email_dir, "email-Eu-core-department-labels.txt")
        )

        cls, lp = run_graphbert_experiment(data, labels)
        results_cls.append({"Experiment": "Email-EU-Core", **cls})
        if lp: results_lp.append({"Experiment": "Email-EU-Core", **lp})

        # 4) GitHub
        print("\n=================== GitHub Dataset ===================")
        gh_dir = os.path.join(BASE_DIR, "../datasets/git_web_ml")
        data, labels, _ = load_musae_github_dataset(
            os.path.join(gh_dir, "musae_git_edges.csv"),
            os.path.join(gh_dir, "musae_git_features.json"),
            os.path.join(gh_dir, "musae_git_target.csv"),
        )

        cls, lp = run_graphbert_experiment(data, labels)
        results_cls.append({"Experiment": "GitHub", **cls})
        if lp: results_lp.append({"Experiment": "GitHub", **lp})

        # 5) Deezer
        print("\n=================== Deezer Dataset ===================")
        dz_dir = os.path.join(BASE_DIR, "../datasets/deezer_europe")
        data, labels = load_deezer_europe_dataset(
            os.path.join(dz_dir, "deezer_europe_edges.csv"),
            os.path.join(dz_dir, "deezer_europe_features.json"),
            os.path.join(dz_dir, "deezer_europe_target.csv"),
        )

        cls, lp = run_graphbert_experiment(data, labels)
        results_cls.append({"Experiment": "Deezer Europe", **cls})
        if lp: results_lp.append({"Experiment": "Deezer Europe", **lp})

        # Save results
        pd.DataFrame(results_cls).to_excel(writer, sheet_name="Classification", index=False)
        if results_lp:
            pd.DataFrame(results_lp).to_excel(writer, sheet_name="LinkPrediction", index=False)

    print(f"\nAll GraphBERT experiment results saved to {output_file}")


if __name__ == "__main__":
    run_all_graphbert_experiments()
