###############################################################################
# promptsage_runner.py
# Run Prompt‑SAGE (+ dot‑product LP head) on all datasets and save XLSX.
###############################################################################
import os, time, gc
import networkx as nx
import torch
import pandas as pd
from torch_geometric.utils import to_networkx

# --------------------------------------------------------------------------- #
# dataset loaders (unchanged)                                                 #
# --------------------------------------------------------------------------- #
from experiment_utils import (
    generate_synthetic_graph,
    load_musae_facebook_dataset,
    load_email_eu_core_dataset,
    load_deezer_europe_dataset,
)
from experiments.experiment_utils import load_musae_github_dataset
from experiments.gppt_pipeline import run_promptsage_pipeline


# --------------------------------------------------------------------------- #
# our new pipeline                                                            #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# pretty‑printer                                                              #
# --------------------------------------------------------------------------- #
def print_run_results(run, results):
    print(f"\nRun {run} results:")
    header = "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Pipeline", "Accuracy", "Precision", "Recall", "F1", "AUC", "AP")
    print(header)

    def _fmt(x):
        return "{:<10}".format("N/A" if x is None else f"{x:0.4f}")
    for r in results:
        row = "{:<20} ".format(r["Pipeline"]) + "".join(_fmt(r.get(k)) for k in
                                                        ("accuracy", "precision",
                                                         "recall", "f1", "auc", "ap"))
        print(row)

# --------------------------------------------------------------------------- #
# generic experiment driver                                                   #
# --------------------------------------------------------------------------- #
def run_dataset(name, data_loader, num_runs=5, extra_setup=None):
    """
    data_loader -> (data, labels)  OR (data, labels, encoder)
    extra_setup  -> optional callable(data) for structural_targets, etc.
    """
    cls_rows, lp_rows = [], []
    for run in range(1, num_runs + 1):
        print(f"\n--- {name} Run {run} ---")
        data_tuple = data_loader()
        data, labels = data_tuple[0], data_tuple[1]

        if extra_setup is not None:
            extra_setup(data)

        seed = 42 + run
        print(f"\n========== [{name}] Prompt‑SAGE ==========")
        _, cls_res, lp_res = run_promptsage_pipeline(
            data, labels, seed=seed, do_linkpred=True)

        cls_rows.append({"Experiment": name, "Run": run,
                         "Pipeline": "Prompt‑SAGE", **cls_res.as_dict()})
        lp_rows.append({"Experiment": name, "Run": run,
                        "Pipeline": "Prompt‑SAGE", **lp_res.as_dict()})

        print_run_results(run, [cls_rows[-1]])
        print_run_results(run, [lp_rows[-1]])

        # give GPU a breather on large graphs
        torch.cuda.empty_cache(); gc.collect(); time.sleep(30)
    return cls_rows, lp_rows

# --------------------------------------------------------------------------- #
# per‑dataset wrappers to satisfy the call signature                          #
# --------------------------------------------------------------------------- #
def _add_clustering_targets(data):
    nx_g = to_networkx(data, to_undirected=True)
    clustering = nx.clustering(nx_g)
    data.structural_targets = torch.tensor(
        [clustering.get(i, 0.0) for i in range(data.num_nodes)], dtype=torch.float)

def run_synthetic(num_runs):
    return run_dataset("Synthetic",
                       lambda: generate_synthetic_graph(),
                       num_runs)

def run_facebook(num_runs):
    fb_dir = os.path.join(BASE_DIR, "../datasets/facebook_large")
    return run_dataset(
        "Facebook",
        lambda: load_musae_facebook_dataset(
            os.path.join(fb_dir, "musae_facebook_edges.csv"),
            os.path.join(fb_dir, "musae_facebook_features.json"),
            os.path.join(fb_dir, "musae_facebook_target.csv")),
        num_runs, _add_clustering_targets)

def run_email(num_runs):
    em_dir = os.path.join(BASE_DIR, "../datasets/email-eu-core")
    return run_dataset(
        "Email‑EU‑Core",
        lambda: load_email_eu_core_dataset(
            os.path.join(em_dir, "email-Eu-core.txt"),
            os.path.join(em_dir, "email-Eu-core-department-labels.txt")),
        num_runs, _add_clustering_targets)

def run_github(num_runs):
    gh_dir = os.path.join(BASE_DIR, "../datasets/git_web_ml")
    return run_dataset(
        "GitHub",
        lambda: load_musae_github_dataset(
            os.path.join(gh_dir, "musae_git_edges.csv"),
            os.path.join(gh_dir, "musae_git_features.json"),
            os.path.join(gh_dir, "musae_git_target.csv")),
        num_runs, _add_clustering_targets)

def run_deezer(num_runs):
    dz_dir = os.path.join(BASE_DIR, "../datasets/deezer_europe")
    return run_dataset(
        "Deezer Europe",
        lambda: load_deezer_europe_dataset(
            os.path.join(dz_dir, "deezer_europe_edges.csv"),
            os.path.join(dz_dir, "deezer_europe_features.json"),
            os.path.join(dz_dir, "deezer_europe_target.csv")),
        num_runs, _add_clustering_targets)

# --------------------------------------------------------------------------- #
# main orchestration                                                          #
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_all_experiments(num_runs=5,
                        output_file="./promptsage_results.xlsx"):
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for run_fn, cls_sheet, lp_sheet in [
            (run_synthetic, "Synthetic_Classification", "Synthetic_LinkPrediction"),
            (run_facebook,  "Facebook_Classification",  "Facebook_LinkPrediction"),
            (run_email,     "Email_Classification",     "Email_LinkPrediction"),
            (run_github,    "GitHub_Classification",    "GitHub_LinkPrediction"),
            (run_deezer,    "Deezer_Classification",    "Deezer_LinkPrediction"),
        ]:
            cls_rows, lp_rows = run_fn(num_runs)
            pd.DataFrame(cls_rows).to_excel(writer, sheet_name=cls_sheet, index=False)
            pd.DataFrame(lp_rows ).to_excel(writer, sheet_name=lp_sheet, index=False)
            del cls_rows, lp_rows  # free memory

    print(f"\nAll Prompt‑SAGE results saved to {output_file}")


if __name__ == "__main__":
    run_all_experiments(num_runs=5)
