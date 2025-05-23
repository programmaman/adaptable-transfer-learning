import os
import time

import networkx as nx
import torch
from torch_geometric.utils import to_networkx
import pandas as pd  # For storing and saving results

from .experiment_utils import (
    generate_synthetic_graph,
    load_musae_facebook_dataset,
    load_email_eu_core_dataset, load_twitch_gamers_dataset, load_deezer_europe_dataset,
)
from experiments.experiment_utils import load_musae_github_dataset
from experiments.gat_pipeline import run_gat_pipeline
from experiments.gnn_pipeline import run_pipeline
from experiments.graph_sage_pipeline import run_graphsage_pipeline
from experiments.struct_g_internal_pipeline import run_structg_pipeline_internal
from experiments.deep_gcn_pipeline import run_structural_gcn_pipeline
from experiments.gpt_gnn_pipeline import run_gpt_gnn_pipeline
from experiments.struct_g_pipeline import run_structg_pipeline


def print_run_results(run, results):
    """
    Prints formatted results (link prediction or classification) for a given run.
    For any numerical field that is None, prints "N/A".

    Parameters:
        run (int): The current run number.
        results (list of dict): A list of dictionaries with the following keys:
            "Pipeline", "accuracy", "precision", "recall", "f1", "auc", "ap".
            (For classification results, "auc" and "ap" can also be provided or skipped.)
    """

    print(f"\nRun {run} results:")
    header = "{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Pipeline", "Accuracy", "Precision", "Recall", "F1", "AUC", "AP"
    )
    print(header)

    def fmt_value(val):
        # If value is None, return "N/A" left-justified in a field width of 10;
        # otherwise, format as a float with 4 decimals.
        if val is None:
            return "{:<10}".format("N/A")
        else:
            return "{:<10.4f}".format(val)

    for result in results:
        pipeline = result.get("Pipeline", "N/A")
        # Use .get(key) so missing fields also default to None
        accuracy = result.get("accuracy")
        precision = result.get("precision")
        recall = result.get("recall")
        f1 = result.get("f1")
        auc = result.get("auc")
        ap = result.get("ap")
        # Build the row using the safe formatter for each numerical value
        row = "{:<15} ".format(pipeline) + \
              fmt_value(accuracy) + \
              fmt_value(precision) + \
              fmt_value(recall) + \
              fmt_value(f1) + \
              fmt_value(auc) + \
              fmt_value(ap)
        print(row)



def run_synthetic_experiments(num_runs=5):
    print("\n=================== Synthetic Graph Experiments ===================")
    synthetic_cls_results = []
    synthetic_lp_results = []
    for run in range(1, num_runs + 1):
        print(f"\n--- Synthetic Experiment Run {run} ---")
        data, labels = generate_synthetic_graph()
        seed = 42 + run

        # SimpleGNN
        print("\n========== [Synthetic] SimpleGNN ==========")
        _, simplegnn_cls, simplegnn_lp = run_pipeline(data, labels, seed=seed)

        # Deep GCN
        print("\n========== [Synthetic] Deep GCN ==========")
        _, sgcn_cls, sgcn_lp = run_structural_gcn_pipeline(data, labels, seed=seed)

        # GPT-GNN
        print("\n========== [Synthetic] GPT-GNN ==========")
        _, gpt_cls, gpt_lp = run_gpt_gnn_pipeline(data, labels, seed=seed)

        # Struct-G Structural Only Pretrain (Node2Vec)
        print("\n========== [Synthetic] Struct-G Structural Only Pretrain (Node2Vec) ==========")
        _, structgnn_ssl_cls, structgnn_ssl_lp = run_structg_pipeline(data, labels, seed=seed, do_linkpred=True)

        #Struct-G Internal Classifier
        print("\n========== [Synthetic] Struct-G Internal Classifier ==========")
        _, structgnn_cls, structgnn_lp = run_structg_pipeline_internal(data, labels, seed=seed, do_linkpred=True)

        # SimpleGraphSAGE
        print("\n========== [Synthetic] SimpleGraphSAGE ==========")
        _, graphsage_cls, graphsage_lp = run_graphsage_pipeline(data, labels, seed=seed)

        # SimpleGAT
        print("\n========== [Synthetic] SimpleGAT ==========")
        _, simplegat_cls, simplegat_lp = run_gat_pipeline(data, labels, seed=seed)

        # Store the classification results
        synthetic_cls_results.extend([
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "SimpleGNN", **simplegnn_cls.as_dict()},
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "Deep GCN", **sgcn_cls.as_dict()},
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "GPT-GNN", **gpt_cls.as_dict()},
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "Struct-G Structural Only Pretrain", **structgnn_ssl_cls.as_dict()},
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "Struct-G Internal Classifier", **structgnn_cls.as_dict()},
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "SimpleGraphSAGE", **graphsage_cls.as_dict()},
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "SimpleGAT", **simplegat_cls.as_dict()},
        ])

        # Store the link prediction results
        synthetic_lp_results.extend([
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "SimpleGNN", **simplegnn_lp.as_dict()},
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "Deep GCN", **sgcn_lp.as_dict()},
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "GPT-GNN", **gpt_lp.as_dict()},
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "Struct-G Structural Only Pretrain", **structgnn_ssl_lp.as_dict()},
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "Struct-G Internal Classifier", **structgnn_lp.as_dict()},
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "SimpleGraphSAGE", **graphsage_lp.as_dict()},
            {"Experiment": "Synthetic", "Run": run, "Pipeline": "SimpleGAT", **simplegat_lp.as_dict()},
        ])

        #Print CLS
        print_run_results(run, synthetic_cls_results[-7:])
        # Print LP results
        print_run_results(run, synthetic_lp_results[-7:])


    return synthetic_cls_results, synthetic_lp_results


def run_facebook_experiments(edge_path, features_path, target_path, num_runs=5):
    print("\n=================== Facebook Graph Experiments ===================")
    facebook_cls_results = []
    facebook_lp_results = []
    for run in range(1, num_runs + 1):
        print(f"\n--- Facebook Experiment Run {run} ---")
        data, labels, label_encoder = load_musae_facebook_dataset(edge_path, features_path, target_path)
        seed = 42 + run

        # Compute the clustering for structural targets
        nx_g = to_networkx(data, to_undirected=True)
        clustering = nx.clustering(nx_g)
        data.structural_targets = torch.tensor(
            [clustering.get(i, 0.0) for i in range(data.num_nodes)], dtype=torch.float
        )

        print("\n========== [Facebook] SimpleGNN ==========")
        _, simplegnn_cls, simplegnn_lp = run_pipeline(data, labels, seed=seed)
        print("\n========== [Facebook] Deep GCN ==========")
        _, structural_cls, structural_lp = run_structural_gcn_pipeline(data, labels, seed=seed)
        print("\n========== [Facebook] GPT-GNN ==========")
        _, gpt_cls, gpt_lp = run_gpt_gnn_pipeline(data, labels, seed=seed)
        print("\n========== [Facebook] Struct-G Structural Only Pretrain (Node2Vec) ==========")
        _, structgnn_ssl_cls, structgnn_ssl_lp = run_structg_pipeline(data, labels, seed=seed)
        print("\n========== [Facebook] Struct-G Internal Classifier ==========")
        _, structgnn_cls, structgnn_lp = run_structg_pipeline_internal(data, labels, seed=seed, do_linkpred=True)
        print("\n========== [Facebook] SimpleGAT ==========")
        _, simplegat_cls, simplegat_lp = run_gat_pipeline(data, labels, seed=seed)
        print("\n========== [Facebook] SimpleGraphSAGE ==========")
        _, graphsage_cls, graphsage_lp = run_graphsage_pipeline(data, labels, seed=seed)

        facebook_cls_results.extend([
            {"Experiment": "Facebook", "Run": run, "Pipeline": "SimpleGNN", **simplegnn_cls.as_dict()},
            {"Experiment": "Facebook", "Run": run, "Pipeline": "Deep GCN", **structural_cls.as_dict()},
            {"Experiment": "Facebook", "Run": run, "Pipeline": "GPT-GNN", **gpt_cls.as_dict()},
            {"Experiment": "Facebook", "Run": run, "Pipeline": "Struct-G Structural Only Pretrain", **structgnn_ssl_cls.as_dict()},
            {"Experiment": "Facebook", "Run": run, "Pipeline": "Struct-G Internal Classifier", **structgnn_cls.as_dict()},
            {"Experiment": "Facebook", "Run": run, "Pipeline": "SimpleGAT", **simplegat_cls.as_dict()},
            {"Experiment": "Facebook", "Run": run, "Pipeline": "SimpleGraphSAGE", **graphsage_cls.as_dict()},
        ])

        facebook_lp_results.extend([
            {"Experiment": "Facebook", "Run": run, "Pipeline": "SimpleGNN", **simplegnn_lp.as_dict()},
            {"Experiment": "Facebook", "Run": run, "Pipeline": "Deep GCN", **structural_lp.as_dict()},
            {"Experiment": "Facebook", "Run": run, "Pipeline": "GPT-GNN", **gpt_lp.as_dict()},
            {"Experiment": "Facebook", "Run": run, "Pipeline": "Struct-G Structural Only Pretrain", **structgnn_ssl_lp.as_dict()},
            {"Experiment": "Facebook", "Run": run, "Pipeline": "Struct-G Internal Classifier", **structgnn_lp.as_dict()},
            {"Experiment": "Facebook", "Run": run, "Pipeline": "SimpleGAT", **simplegat_lp.as_dict()},
            {"Experiment": "Facebook", "Run": run, "Pipeline": "SimpleGraphSAGE", **graphsage_lp.as_dict()},
        ])

        #print results
        print_run_results(run, facebook_cls_results[-7:])
        print_run_results(run, facebook_lp_results[-7:])
        #wait to allow GPU to cool down
        time.sleep(60)


    return facebook_cls_results, facebook_lp_results


def run_email_eu_core_experiments(edge_path, label_path, num_runs=5):
    print("\n=================== Email-EU-Core Experiments ===================")
    email_cls_results = []
    email_lp_results = []
    for run in range(1, num_runs + 1):
        print(f"\n--- Email-EU-Core Experiment Run {run} ---")
        data, labels = load_email_eu_core_dataset(edge_path, label_path)
        seed = 42 + run

        # Compute the clustering for structural targets
        nx_g = to_networkx(data, to_undirected=True)
        clustering = nx.clustering(nx_g)
        data.structural_targets = torch.tensor(
            [clustering.get(i, 0.0) for i in range(data.num_nodes)], dtype=torch.float
        )

        print("\n========== [Email-EU-Core] SimpleGNN ==========")
        _, simplegnn_cls, simplegnn_lp = run_pipeline(data, labels, seed=seed)
        print("\n========== [Email-EU-Core] Deep GCN ==========")
        _, structural_cls, structural_lp = run_structural_gcn_pipeline(data, labels, seed=seed)
        print("\n========== [Email-EU-Core] GPT-GNN ==========")
        _, gpt_cls, gpt_lp = run_gpt_gnn_pipeline(data, labels, seed=seed)
        print("\n========== [Email-EU-Core] Struct-G Structural Only Pretrain (Node2Vec) ==========")
        _, structgnn_ssl_cls, structgnn_ssl_lp = run_structg_pipeline(data, labels, seed=seed)
        print("\n========== [Email-EU-Core] Struct-G Internal Classifier ==========")
        _, structgnn_cls, structgnn_lp = run_structg_pipeline_internal(data, labels, seed=seed, do_linkpred=True)
        print("\n========== [Email-EU-Core] SimpleGAT ==========")
        _, simplegat_cls, simplegat_lp = run_gat_pipeline(data, labels, seed=seed)
        print("\n========== [Email-EU-Core] SimpleGraphSAGE ==========")
        _, graphsage_cls, graphsage_lp = run_graphsage_pipeline(data, labels, seed=seed)

        email_cls_results.extend([
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "SimpleGNN", **simplegnn_cls.as_dict()},
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "Deep GCN", **structural_cls.as_dict()},
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "GPT-GNN", **gpt_cls.as_dict()},
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "Struct-G Structural Only Pretrain", **structgnn_ssl_cls.as_dict()},
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "Struct-G Internal Classifier", **structgnn_cls.as_dict()},
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "SimpleGAT", **simplegat_cls.as_dict()},
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "SimpleGraphSAGE", **graphsage_cls.as_dict()},
        ])

        email_lp_results.extend([
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "SimpleGNN", **simplegnn_lp.as_dict()},
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "Deep GCN", **structural_lp.as_dict()},
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "GPT-GNN", **gpt_lp.as_dict()},
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "Struct-G Structural Only Pretrain", **structgnn_ssl_lp.as_dict()},
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "Struct-G Internal Classifier", **structgnn_lp.as_dict()},
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "SimpleGAT", **simplegat_lp.as_dict()},
            {"Experiment": "Email-EU-Core", "Run": run, "Pipeline": "SimpleGraphSAGE", **graphsage_lp.as_dict()},
        ])

        # Print results
        print_run_results(run, email_cls_results[-7:])
        print_run_results(run, email_lp_results[-7:])

        #wait to allow GPU to cool down
        time.sleep(60)

    return email_cls_results, email_lp_results


def run_github_experiments(edge_path, features_path, target_path, num_runs=5):
    print("\n=================== GitHub Graph Experiments ===================")
    github_cls_results = []
    github_lp_results = []
    for run in range(1, num_runs + 1):
        print(f"\n--- GitHub Experiment Run {run} ---")
        data, labels, label_encoder = load_musae_github_dataset(edge_path, features_path, target_path)
        seed = 42 + run

        # Compute the clustering for structural targets
        nx_g = to_networkx(data, to_undirected=True)
        clustering = nx.clustering(nx_g)
        data.structural_targets = torch.tensor(
            [clustering.get(i, 0.0) for i in range(data.num_nodes)], dtype=torch.float
        )

        print("\n========== [GitHub] SimpleGNN ==========")
        _, simplegnn_cls, simplegnn_lp = run_pipeline(data, labels, seed=seed)
        print("\n========== [GitHub] Deep GCN ==========")
        _, structural_cls, structural_lp = run_structural_gcn_pipeline(data, labels, seed=seed)
        print("\n========== [GitHub] GPT-GNN ==========")
        _, gpt_cls, gpt_lp = run_gpt_gnn_pipeline(data, labels, seed=seed)
        print("\n========== [GitHub] Struct-G Structural Only Pretrain (Node2Vec) ==========")
        _, structgnn_ssl_cls, structgnn_ssl_lp = run_structg_pipeline(data, labels, seed=seed)
        print("\n========== [GitHub] Struct-G Internal Classifier ==========")
        _, structgnn_cls, structgnn_lp = run_structg_pipeline_internal(data, labels, seed=seed, do_linkpred=True)
        print("\n========== [GitHub] SimpleGAT ==========")
        _, simplegat_cls, simplegat_lp = run_gat_pipeline(data, labels, seed=seed)
        print("\n========== [GitHub] SimpleGraphSAGE ==========")
        _, graphsage_cls, graphsage_lp = run_graphsage_pipeline(data, labels, seed=seed)

        github_cls_results.extend([
            {"Experiment": "GitHub", "Run": run, "Pipeline": "SimpleGNN", **simplegnn_cls.as_dict()},
            {"Experiment": "GitHub", "Run": run, "Pipeline": "Deep GCN", **structural_cls.as_dict()},
            {"Experiment": "GitHub", "Run": run, "Pipeline": "GPT-GNN", **gpt_cls.as_dict()},
            {"Experiment": "GitHub", "Run": run, "Pipeline": "Struct-G Structural Only Pretrain", **structgnn_ssl_cls.as_dict()},
            {"Experiment": "GitHub", "Run": run, "Pipeline": "Struct-G Internal Classifier", **structgnn_cls.as_dict()},
            {"Experiment": "GitHub", "Run": run, "Pipeline": "SimpleGAT", **simplegat_cls.as_dict()},
            {"Experiment": "GitHub", "Run": run, "Pipeline": "SimpleGraphSAGE", **graphsage_cls.as_dict()},
        ])

        github_lp_results.extend([
            {"Experiment": "GitHub", "Run": run, "Pipeline": "SimpleGNN", **simplegnn_lp.as_dict()},
            {"Experiment": "GitHub", "Run": run, "Pipeline": "Deep GCN", **structural_lp.as_dict()},
            {"Experiment": "GitHub", "Run": run, "Pipeline": "GPT-GNN", **gpt_lp.as_dict()},
            {"Experiment": "GitHub", "Run": run, "Pipeline": "Struct-G Structural Only Pretrain", **structgnn_ssl_lp.as_dict()},
            {"Experiment": "GitHub", "Run": run, "Pipeline": "Struct-G Internal Classifier", **structgnn_lp.as_dict()},
            {"Experiment": "GitHub", "Run": run, "Pipeline": "SimpleGAT", **simplegat_lp.as_dict()},
            {"Experiment": "GitHub", "Run": run, "Pipeline": "SimpleGraphSAGE", **graphsage_lp.as_dict()},
        ])

        # Print results
        print_run_results(run, github_cls_results[-7:])
        print_run_results(run, github_lp_results[-7:])
        #wait to allow GPU to cool down
        time.sleep(60)

    return github_cls_results, github_lp_results

def run_deezer_experiments(edge_path, features_path, target_path, num_runs=5):
    print("\n=================== Deezer Europe Experiments ===================")
    deezer_cls_results = []
    deezer_lp_results = []
    for run in range(1, num_runs + 1):
        print(f"\n--- Deezer Europe Experiment Run {run} ---")
        data, labels = load_deezer_europe_dataset(edge_path, features_path, target_path)
        seed = 42 + run

        nx_g = to_networkx(data, to_undirected=True)
        clustering = nx.clustering(nx_g)
        data.structural_targets = torch.tensor([clustering.get(i, 0.0) for i in range(data.num_nodes)], dtype=torch.float)

        print("\n========== [Deezer] SimpleGNN ==========")
        _, simplegnn_cls, simplegnn_lp = run_pipeline(data, labels, seed=seed)
        print("\n========== [Deezer] Deep GCN ==========")
        _, structural_cls, structural_lp = run_structural_gcn_pipeline(data, labels, seed=seed)
        print("\n========== [Deezer] GPT-GNN ==========")
        _, gpt_cls, gpt_lp = run_gpt_gnn_pipeline(data, labels, seed=seed)
        print("\n========== [Deezer] Struct-G Structural Only Pretrain (Node2Vec) ==========")
        _, structgnn_ssl_cls, structgnn_ssl_lp = run_structg_pipeline(data, labels, seed=seed)
        print("\n========== [Deezer] Struct-G Internal Classifier ==========")
        _, structgnn_cls, structgnn_lp = run_structg_pipeline_internal(data, labels, seed=seed, do_linkpred=True)
        print("\n========== [Deezer] SimpleGAT ==========")
        _, simplegat_cls, simplegat_lp = run_gat_pipeline(data, labels, seed=seed)
        print("\n========== [Deezer] SimpleGraphSAGE ==========")
        _, graphsage_cls, graphsage_lp = run_graphsage_pipeline(data, labels, seed=seed)

        deezer_cls_results.extend([
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "SimpleGNN", **simplegnn_cls.as_dict()},
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "Deep GCN", **structural_cls.as_dict()},
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "GPT-GNN", **gpt_cls.as_dict()},
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "Struct-G Structural Only Pretrain", **structgnn_ssl_cls.as_dict()},
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "Struct-G Internal Classifier", **structgnn_cls.as_dict()},
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "SimpleGAT", **simplegat_cls.as_dict()},
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "SimpleGraphSAGE", **graphsage_cls.as_dict()},
        ])

        deezer_lp_results.extend([
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "SimpleGNN", **simplegnn_lp.as_dict()},
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "Deep GCN", **structural_lp.as_dict()},
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "GPT-GNN", **gpt_lp.as_dict()},
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "Struct-G Structural Only Pretrain", **structgnn_ssl_lp.as_dict()},
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "Struct-G Internal Classifier", **structgnn_lp.as_dict()},
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "SimpleGAT", **simplegat_lp.as_dict()},
            {"Experiment": "Deezer Europe", "Run": run, "Pipeline": "SimpleGraphSAGE", **graphsage_lp.as_dict()},
        ])

        print_run_results(run, deezer_cls_results[-7:])
        print_run_results(run, deezer_lp_results[-7:])
        time.sleep(60)

    return deezer_cls_results, deezer_lp_results


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_all_experiments(num_runs=5,
                        output_file="/app/results/experiment_results.xlsx"):
    # Open the Excel writer up front:
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

        # 1) Synthetic
        synth_cls, synth_lp = run_synthetic_experiments(num_runs)
        pd.DataFrame(synth_cls).to_excel(
            writer, sheet_name="Synthetic_Classification", index=False
        )
        pd.DataFrame(synth_lp).to_excel(
            writer, sheet_name="Synthetic_LinkPrediction", index=False
        )
        # drop refs so memory can be freed
        del synth_cls, synth_lp

        # 2) Facebook
        fb_dir = os.path.join(BASE_DIR, "../datasets/facebook_large")
        fb_cls, fb_lp = run_facebook_experiments(
            edge_path=os.path.join(fb_dir, "musae_facebook_edges.csv"),
            features_path=os.path.join(fb_dir, "musae_facebook_features.json"),
            target_path=os.path.join(fb_dir, "musae_facebook_target.csv"),
            num_runs=num_runs,
        )
        pd.DataFrame(fb_cls).to_excel(
            writer, sheet_name="Facebook_Classification", index=False
        )
        pd.DataFrame(fb_lp).to_excel(
            writer, sheet_name="Facebook_LinkPrediction", index=False
        )
        del fb_cls, fb_lp

        # 3) Email-EU-Core
        email_dir = os.path.join(BASE_DIR, "../datasets/email-eu-core")
        email_cls, email_lp = run_email_eu_core_experiments(
            edge_path=os.path.join(email_dir, "email-Eu-core.txt"),
            label_path=os.path.join(email_dir, "email-Eu-core-department-labels.txt"),
            num_runs=num_runs,
        )
        pd.DataFrame(email_cls).to_excel(
            writer, sheet_name="Email_Classification", index=False
        )
        pd.DataFrame(email_lp).to_excel(
            writer, sheet_name="Email_LinkPrediction", index=False
        )
        del email_cls, email_lp

        # 4) GitHub
        github_dir = os.path.join(BASE_DIR, "../datasets/git_web_ml")
        gh_cls, gh_lp = run_github_experiments(
            edge_path=os.path.join(github_dir, "musae_git_edges.csv"),
            features_path=os.path.join(github_dir, "musae_git_features.json"),
            target_path=os.path.join(github_dir, "musae_git_target.csv"),
            num_runs=num_runs,
        )
        pd.DataFrame(gh_cls).to_excel(
            writer, sheet_name="GitHub_Classification", index=False
        )
        pd.DataFrame(gh_lp).to_excel(
            writer, sheet_name="GitHub_LinkPrediction", index=False
        )
        del gh_cls, gh_lp

        # 5) Deezer
        deezer_dir = os.path.join(BASE_DIR, "../datasets/deezer_europe")
        dz_cls, dz_lp = run_deezer_experiments(
            edge_path=os.path.join(deezer_dir, "deezer_europe_edges.csv"),
            features_path=os.path.join(deezer_dir, "deezer_europe_features.json"),
            target_path=os.path.join(deezer_dir, "deezer_europe_target.csv"),
            num_runs=num_runs,
        )
        pd.DataFrame(dz_cls).to_excel(
            writer, sheet_name="Deezer_Classification", index=False
        )
        pd.DataFrame(dz_lp).to_excel(
            writer, sheet_name="Deezer_LinkPrediction", index=False
        )
        del dz_cls, dz_lp

        # when the 'with' block ends, writer.save() is implicitly called
    print(f"\nAll experiment results saved to {output_file}")


def main():
    run_all_experiments(num_runs=5)

if __name__ == "__main__":
    main()
