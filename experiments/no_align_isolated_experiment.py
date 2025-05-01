import os
import time
import torch
import pandas as pd
from torch_geometric.utils import to_networkx
import networkx as nx

from experiments.experiment_runner import print_run_results
from experiments.struct_g_internal_pipeline import run_structg_pipeline_internal
from experiments.experiment_utils import (
    load_musae_facebook_dataset,
    load_email_eu_core_dataset,
    load_musae_github_dataset,
    load_deezer_europe_dataset,
    generate_synthetic_graph
)
from utils import get_device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_structg_noalign(name, load_func, args: dict, num_runs=5):
    print(f"\n=================== {name} Struct-G NoAlign Experiments ===================")
    cls_results, lp_results = [], []

    for run in range(1, num_runs + 1):
        print(f"\n--- [{name}] Run {run} ---")
        seed = 42 + run
        data, labels = load_func(**args)[0:2]  # Load data and labels

        if isinstance(data, tuple):  # (data, labels, encoder)
            data, labels = data[0], data[1]

        device = get_device()
        data = data.to(device)
        labels = labels.to(device)

        if name.lower() != "synthetic":
            nx_g = to_networkx(data, to_undirected=True)
            clustering = nx.clustering(nx_g)
            data.structural_targets = torch.tensor(
                [clustering.get(i, 0.0) for i in range(data.num_nodes)],
                dtype=torch.float
            )

        # Run pipeline without Node2Vec alignment
        _, cls, lp = run_structg_pipeline_internal(
            data, labels, seed=seed, do_n2v_align=False, do_linkpred=True
        )

        cls_results.append({
            "Experiment": name, "Run": run, "Pipeline": "Struct-G No Align", **cls.as_dict()
        })
        lp_results.append({
            "Experiment": name, "Run": run, "Pipeline": "Struct-G No Align", **lp.as_dict()
        })

        print_run_results(run, [cls_results[-1]])
        print_run_results(run, [lp_results[-1]])
        time.sleep(20)

    return cls_results, lp_results


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(BASE_DIR, "../results/structg_noalign_results.xlsx")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        datasets = {
            "Synthetic": (generate_synthetic_graph, {}),

            "Facebook": (load_musae_facebook_dataset, {
                "edge_path": os.path.join(BASE_DIR, "../datasets/facebook_large/musae_facebook_edges.csv"),
                "features_path": os.path.join(BASE_DIR, "../datasets/facebook_large/musae_facebook_features.json"),
                "target_path": os.path.join(BASE_DIR, "../datasets/facebook_large/musae_facebook_target.csv"),
            }),

            "Email-EU-Core": (load_email_eu_core_dataset, {
                "edge_path": os.path.join(BASE_DIR, "../datasets/email-eu-core/email-Eu-core.txt"),
                "label_path": os.path.join(BASE_DIR, "../datasets/email-eu-core/email-Eu-core-department-labels.txt"),
            }),

            "GitHub": (load_musae_github_dataset, {
                "edge_path": os.path.join(BASE_DIR, "../datasets/git_web_ml/musae_git_edges.csv"),
                "features_path": os.path.join(BASE_DIR, "../datasets/git_web_ml/musae_git_features.json"),
                "target_path": os.path.join(BASE_DIR, "../datasets/git_web_ml/musae_git_target.csv"),
            }),

            "Deezer Europe": (load_deezer_europe_dataset, {
                "edge_path": os.path.join(BASE_DIR, "../datasets/deezer_europe/deezer_europe_edges.csv"),
                "features_path": os.path.join(BASE_DIR, "../datasets/deezer_europe/deezer_europe_features.json"),
                "target_path": os.path.join(BASE_DIR, "../datasets/deezer_europe/deezer_europe_target.csv"),
            }),
        }

        for name, (loader, args) in datasets.items():
            cls, lp = run_structg_noalign(name, loader, args, num_runs=5)
            pd.DataFrame(cls).to_excel(writer, sheet_name=f"{name}_Classification", index=False)
            pd.DataFrame(lp).to_excel(writer, sheet_name=f"{name}_LinkPrediction", index=False)

    print(f"\n✓ All Struct-G NoAlign results saved to: {output_path}")

