import os

import networkx as nx
import torch
from torch_geometric.utils import to_networkx

from experiment_utils import generate_synthetic_graph, generate_task_labels, load_musae_facebook_dataset, load_email_eu_core_dataset
from experiments.synthetic_graph_experiment_simple_gnn import run_gnn_pipeline
from experiments.synthetic_graph_experiment_structural_gcn import run_structural_gcn_pipeline
from experiments.synthetic_graph_experiment_gpt_gnn import run_gpt_gnn_pipeline
from experiments.synthetic_graph_experiment_struct_gnn import run_structural_node2vec_pipeline


def run_synthetic_experiments():
    print("\n=================== Synthetic Graph Experiments ===================")

    # === Generate Synthetic Data and Labels ===
    data, labels = generate_synthetic_graph()

    print("\n========== [Synthetic] SimpleGNN ==========")
    simplegnn_model, simplegnn_test_acc = run_gnn_pipeline(data, labels)

    print("\n========== [Synthetic] StructuralGCN ==========")
    structural_model, structural_loss, structural_gcn_acc = run_structural_gcn_pipeline(data, labels)

    print("\n========== [Synthetic] GPT-GNN ==========")
    gpt_model, gpt_test_acc = run_gpt_gnn_pipeline(data, labels)

    print("\n========== [Synthetic] StructuralGNN (Node2Vec) ==========")
    structural_model, structural_classifier, structural_acc = run_structural_node2vec_pipeline(data, labels)

    print("\n========== [Synthetic] Summary ==========")
    print(f"[Synthetic] SimpleGNN Final Test Accuracy:       {simplegnn_test_acc:.4f}")
    print(f"[Synthetic] StructuralGCN Final Val Accuracy:    {structural_gcn_acc:.4f} | Loss: {structural_loss:.4f}")
    print(f"[Synthetic] GPT-GNN Final Test Accuracy:         {gpt_test_acc:.4f}")
    print(f"[Synthetic] StructuralGNN Final Test Accuracy:   {structural_acc:.4f}")


def run_facebook_experiments(edge_path, features_path, target_path):
    print("\n=================== Facebook Graph Experiments ===================")

    data, labels, label_encoder = load_musae_facebook_dataset(edge_path, features_path, target_path)

    nx_g = to_networkx(data, to_undirected=True)
    clustering = nx.clustering(nx_g)

    data.structural_targets = torch.tensor(
        [clustering.get(i, 0.0) for i in range(data.num_nodes)],
        dtype=torch.float
    )

    print("\n========== [Facebook] SimpleGNN ==========")
    simplegnn_model, simplegnn_test_acc = run_gnn_pipeline(data, labels)

    print("\n========== [Facebook] StructuralGCN ==========")
    structural_model, structural_loss, structural_gcn_acc = run_structural_gcn_pipeline(data, labels)

    print("\n========== [Facebook] GPT-GNN ==========")
    gpt_model, gpt_test_acc = run_gpt_gnn_pipeline(data, labels)

    print("\n========== [Facebook] StructuralGNN (Node2Vec) ==========")
    structural_model, structural_classifier, structural_acc = run_structural_node2vec_pipeline(data, labels)

    print("\n========== [Facebook] Summary ==========")
    print(f"[Facebook] SimpleGNN Final Test Accuracy:       {simplegnn_test_acc:.4f}")
    print(f"[Facebook] StructuralGCN Final Val Accuracy:    {structural_gcn_acc:.4f} | Loss: {structural_loss:.4f}")
    print(f"[Facebook] GPT-GNN Final Test Accuracy:         {gpt_test_acc:.4f}")
    print(f"[Facebook] StructuralGNN Final Test Accuracy:   {structural_acc:.4f}")


def run_email_eu_core_experiments(edge_path, label_path):
    print("\n=================== Email-EU-Core Experiments ===================")

    data, labels = load_email_eu_core_dataset(edge_path, label_path)

    # Compute clustering coefficient for structural pretraining
    nx_g = to_networkx(data, to_undirected=True)
    clustering = nx.clustering(nx_g)
    data.structural_targets = torch.tensor(
        [clustering.get(i, 0.0) for i in range(data.num_nodes)],
        dtype=torch.float
    )

    print("\n========== [Email-EU-Core] SimpleGNN ==========")
    simplegnn_model, simplegnn_test_acc = run_gnn_pipeline(data, labels)

    print("\n========== [Email-EU-Core] StructuralGCN ==========")
    structural_model, structural_loss, structural_gcn_acc = run_structural_gcn_pipeline(data, labels)

    print("\n========== [Email-EU-Core] GPT-GNN ==========")
    gpt_model, gpt_test_acc = run_gpt_gnn_pipeline(data, labels)

    print("\n========== [Email-EU-Core] StructuralGNN (Node2Vec) ==========")
    structural_model, structural_classifier, structural_acc = run_structural_node2vec_pipeline(data, labels)

    print("\n========== [Email-EU-Core] Summary ==========")
    print(f"[Email-EU-Core] SimpleGNN Final Test Accuracy:       {simplegnn_test_acc:.4f}")
    print(f"[Email-EU-Core] StructuralGCN Final Val Accuracy:    {structural_gcn_acc:.4f} | Loss: {structural_loss:.4f}")
    print(f"[Email-EU-Core] GPT-GNN Final Test Accuracy:         {gpt_test_acc:.4f}")
    print(f"[Email-EU-Core] StructuralGNN Final Test Accuracy:   {structural_acc:.4f}")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # this file's dir

def run_experiments():
    run_synthetic_experiments()

    # fb_dir = os.path.join(BASE_DIR, "../datasets/facebook_large")
    # run_facebook_experiments(
    #     edge_path=os.path.join(fb_dir, "musae_facebook_edges.csv"),
    #     features_path=os.path.join(fb_dir, "musae_facebook_features.json"),
    #     target_path=os.path.join(fb_dir, "musae_facebook_target.csv")
    # )

    email_dir = os.path.join(BASE_DIR, "../datasets/email-eu-core")
    run_email_eu_core_experiments(
        edge_path=os.path.join(email_dir, "email-Eu-core.txt"),
        label_path=os.path.join(email_dir, "email-Eu-core-department-labels.txt")
    )



if __name__ == '__main__':
    run_experiments()

