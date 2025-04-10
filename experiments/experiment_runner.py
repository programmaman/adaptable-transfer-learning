import os

import networkx as nx
import torch
from torch_geometric.utils import to_networkx

from experiment_utils import generate_synthetic_graph, generate_task_labels, load_musae_facebook_dataset, load_email_eu_core_dataset
from experiments.gat_pipeline import run_gat_pipeline
from experiments.gnn_pipeline import run_pipeline
from experiments.graph_sage_pipeline import run_graphsage_pipeline
from experiments.struct_gcn_pipeline import run_structural_gcn_pipeline
from experiments.gpt_gnn_pipeline import run_gpt_gnn_pipeline
from experiments.struct_gnn_pipeline import run_structural_node2vec_pipeline


def run_synthetic_experiments():
    print("\n=================== Synthetic Graph Experiments ===================")

    # === Generate Synthetic Data and Labels ===
    data, labels = generate_synthetic_graph()

    print("\n========== [Synthetic] SimpleGNN ==========")
    _, simplegnn_cls_results, simplegnn_lp_results = run_pipeline(data, labels)

    print("\n========== [Synthetic] StructuralGCN ==========")
    _, sgcn_cls_results, sgcn_lp_results = run_structural_gcn_pipeline(data, labels)

    print("\n========== [Synthetic] GPT-GNN ==========")
    _, gpt_cls_results, gpt_lp_results = run_gpt_gnn_pipeline(data, labels)

    print("\n========== [Synthetic] StructuralGNN (Node2Vec) ==========")
    _, structgnn_classifier, structgnn_cls_results, structgnn_lp_results = run_structural_node2vec_pipeline(data, labels)

    print("\n========== [Synthetic] SimpleGraphSAGE ==========")
    _, graphsage_cls_result = run_graphsage_pipeline(data, labels)

    print("\n========== [Synthetic] SimpleGAT ==========")
    _, simplegat_cls_results = run_gat_pipeline(data, labels)

    print("\n=================== Classification Summary ===================")
    print(f"[SimpleGNN]         {simplegnn_cls_results.summary()}")
    print(f"[StructuralGCN]     {sgcn_cls_results.summary()}")
    print(f"[GPT-GNN]           {gpt_cls_results.summary()}")
    print(f"[StructuralGNN]     {structgnn_cls_results.summary()}")
    print(f"[SimpleGraphSAGE]   {graphsage_cls_result.summary()}")
    print(f"[SimpleGAT]         {simplegat_cls_results.summary()}")

    print("\n=================== Link Prediction Summary ===================")
    print(f"[SimpleGNN]         {simplegnn_lp_results.summary()}")
    print(f"[StructuralGCN]     {sgcn_lp_results.summary()}")
    print(f"[GPT-GNN]           {gpt_lp_results.summary()}")
    print(f"[StructuralGNN]     {structgnn_lp_results.summary()}")



def run_facebook_experiments(edge_path, features_path, target_path):
    print("\n=================== Facebook Graph Experiments ===================")

    data, labels, label_encoder = load_musae_facebook_dataset(edge_path, features_path, target_path)

    # Generate structural targets from clustering coefficient
    nx_g = to_networkx(data, to_undirected=True)
    clustering = nx.clustering(nx_g)
    data.structural_targets = torch.tensor(
        [clustering.get(i, 0.0) for i in range(data.num_nodes)],
        dtype=torch.float
    )

    print("\n========== [Facebook] SimpleGNN ==========")
    simplegnn_model, simplegnn_cls_results, simplegnn_lp_results = run_pipeline(data, labels)

    print("\n========== [Facebook] StructuralGCN ==========")
    structural_model, structural_cls_results, structural_lp_results = run_structural_gcn_pipeline(data, labels)

    print("\n========== [Facebook] GPT-GNN ==========")
    gpt_model, gpt_cls_results, gpt_lp_results = run_gpt_gnn_pipeline(data, labels)

    print("\n========== [Facebook] StructuralGNN (Node2Vec) ==========")
    structural_model, structural_classifier, structgnn_cls_results, structgnn_lp_results = run_structural_node2vec_pipeline(data, labels)

    print("\n========== [Facebook] SimpleGAT ==========")
    simplegat_model, simplegat_cls_results = run_gat_pipeline(data, labels)

    print("\n========== [Facebook] SimpleGraphSAGE ==========")
    graphsage_model, graphsage_cls_results = run_graphsage_pipeline(data, labels)

    print("\n=================== Classification Summary ===================")
    print(f"[SimpleGNN]         {simplegnn_cls_results.summary()}")
    print(f"[StructuralGCN]     {structural_cls_results.summary()}")
    print(f"[GPT-GNN]           {gpt_cls_results.summary()}")
    print(f"[StructuralGNN]     {structgnn_cls_results.summary()}")
    print(f"[SimpleGAT]         {simplegat_cls_results.summary()}")
    print(f"[SimpleGraphSAGE]   {graphsage_cls_results.summary()}")

    print("\n=================== Link Prediction Summary ===================")
    print(f"[SimpleGNN]         {simplegnn_lp_results.summary()}")
    print(f"[StructuralGCN]     {structural_lp_results.summary()}")
    print(f"[GPT-GNN]           {gpt_lp_results.summary()}")
    print(f"[StructuralGNN]     {structgnn_lp_results.summary()}")




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
    simplegnn_model, simplegnn_test_acc = run_pipeline(data, labels)

    print("\n========== [Email-EU-Core] StructuralGCN ==========")
    structural_model, structural_loss, structural_gcn_acc = run_structural_gcn_pipeline(data, labels)

    print("\n========== [Email-EU-Core] GPT-GNN ==========")
    gpt_model, gpt_test_acc = run_gpt_gnn_pipeline(data, labels)

    print("\n========== [Email-EU-Core] StructuralGNN (Node2Vec) ==========")
    structural_model, structural_classifier, structural_acc = run_structural_node2vec_pipeline(data, labels)

    print("\n========== [Email-EU-Core] SimpleGAT ==========")
    simplegat_model, simplegat_test_acc = run_gat_pipeline(data, labels)

    print("\n========== [Email-EU-Core] SimpleGraphSAGE ==========")
    graphsage_model, graphsage_test_acc = run_graphsage_pipeline(data, labels)

    print("\n========== [Email-EU-Core] Summary ==========")
    print(f"[Email-EU-Core] SimpleGNN Final Test Accuracy:       {simplegnn_test_acc:.4f}")
    print(f"[Email-EU-Core] StructuralGCN Final Val Accuracy:    {structural_gcn_acc:.4f} | Loss: {structural_loss:.4f}")
    print(f"[Email-EU-Core] GPT-GNN Final Test Accuracy:         {gpt_test_acc:.4f}")
    print(f"[Email-EU-Core] StructuralGNN Final Test Accuracy:   {structural_acc:.4f}")
    print(f"[Email-EU-Core] SimpleGAT Final Test Accuracy:         {simplegat_test_acc:.4f}")
    print(f"[Email-EU-Core] SimpleGraphSAGE Final Test Accuracy:   {graphsage_test_acc:.4f}")



BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # this file's dir

def run_experiments():
    run_synthetic_experiments()

    # fb_dir = os.path.join(BASE_DIR, "../datasets/facebook_large")
    # run_facebook_experiments(
    #     edge_path=os.path.join(fb_dir, "musae_facebook_edges.csv"),
    #     features_path=os.path.join(fb_dir, "musae_facebook_features.json"),
    #     target_path=os.path.join(fb_dir, "musae_facebook_target.csv")
    # )
    #
    # email_dir = os.path.join(BASE_DIR, "../datasets/email-eu-core")
    # run_email_eu_core_experiments(
    #     edge_path=os.path.join(email_dir, "email-Eu-core.txt"),
    #     label_path=os.path.join(email_dir, "email-Eu-core-department-labels.txt")
    # )



if __name__ == '__main__':
    run_experiments()

