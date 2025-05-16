import os
import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx
from experiments.experiment_utils import (
    generate_synthetic_graph,
    load_email_eu_core_dataset,
    load_musae_facebook_dataset,
    load_musae_github_dataset, load_deezer_europe_dataset,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_homophily(data, labels):
    edge_index = data.edge_index
    same_label = (labels[edge_index[0]] == labels[edge_index[1]])
    return same_label.float().mean().item()


def summarize_centrality(metric_dict):
    values = list(metric_dict.values())
    series = pd.Series(values)
    return {
        "mean": series.mean(),
        "max": series.max(),
        "std": series.std(),
    }


import time

def analyze_graph_structure(name, data, labels):
    print(f"\n🔍 Analyzing graph: {name}")
    G = to_networkx(data, to_undirected=True)

    if not nx.is_connected(G):
        print(f"ℹ️  Graph '{name}' is not connected. Using largest connected component.")
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    print(f"✅ Graph '{name}' stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    start = time.time()
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    assortativity = nx.degree_assortativity_coefficient(G)

    try:
        diameter = nx.diameter(G)
    except Exception:
        print("⚠️  Diameter computation failed (likely due to graph size).")
        diameter = None

    # homophily = compute_homophily(data, labels)
    #
    # print(f"⏳ Computing centrality metrics for '{name}'...")
    #
    # # Timed blocks
    # t0 = time.time()
    # bc = nx.betweenness_centrality(G)
    # print(f"  ✅ Betweenness: {time.time() - t0:.2f}s")
    #
    # t1 = time.time()
    # cc = nx.closeness_centrality(G)
    # print(f"  ✅ Closeness: {time.time() - t1:.2f}s")
    #
    # t2 = time.time()
    # try:
    #     ec = nx.eigenvector_centrality(G, max_iter=500)
    # except Exception:
    #     print("  ⚠️  Eigenvector centrality failed. Defaulting to 0.0s.")
    #     ec = {n: 0.0 for n in G.nodes()}
    # print(f"  ✅ Eigenvector: {time.time() - t2:.2f}s")
    #
    # t3 = time.time()
    # pr = nx.pagerank(G)
    # print(f"  ✅ PageRank: {time.time() - t3:.2f}s")
    #
    # total_time = time.time() - start
    # print(f"✅ Finished '{name}' in {total_time:.2f} seconds.")
    #
    # bc_stats = summarize_centrality(bc)
    # cc_stats = summarize_centrality(cc)
    # ec_stats = summarize_centrality(ec)
    # pr_stats = summarize_centrality(pr)

    return {
        "dataset": name,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "density": density,
        "avg_clustering": avg_clustering,
        "assortativity": assortativity,
        "diameter": diameter,
        # "homophily": homophily,
        # "bc_mean": bc_stats["mean"],
        # "bc_max": bc_stats["max"],
        # "bc_std": bc_stats["std"],
        # "cc_mean": cc_stats["mean"],
        # "cc_max": cc_stats["max"],
        # "cc_std": cc_stats["std"],
        # "ec_mean": ec_stats["mean"],
        # "ec_max": ec_stats["max"],
        # "ec_std": ec_stats["std"],
        # "pr_mean": pr_stats["mean"],
        # "pr_max": pr_stats["max"],
        # "pr_std": pr_stats["std"],
    }



def run_all_graph_analyses():
    rows = []

    # # --- Synthetic ---
    # data, labels = generate_synthetic_graph()
    # rows.append(analyze_graph_structure("Synthetic", data, labels))

    # # --- Facebook ---
    # fb_dir = os.path.join(BASE_DIR, "../datasets/facebook_large")
    # data, labels, _ = load_musae_facebook_dataset(
    #     edge_path=os.path.join(fb_dir, "musae_facebook_edges.csv"),
    #     features_path=os.path.join(fb_dir, "musae_facebook_features.json"),
    #     target_path=os.path.join(fb_dir, "musae_facebook_target.csv"),
    # )
    # rows.append(analyze_graph_structure("Facebook", data, labels))
    #
    # # --- Email ---
    # email_dir = os.path.join(BASE_DIR, "../datasets/email-eu-core")
    # data, labels = load_email_eu_core_dataset(
    #     edge_path=os.path.join(email_dir, "email-Eu-core.txt"),
    #     label_path=os.path.join(email_dir, "email-Eu-core-department-labels.txt"),
    # )
    # rows.append(analyze_graph_structure("Email", data, labels))
    #
    # # --- GitHub ---
    # github_dir = os.path.join(BASE_DIR, "../datasets/git_web_ml")
    # data, labels, _ = load_musae_github_dataset(
    #     edge_path=os.path.join(github_dir, "musae_git_edges.csv"),
    #     features_path=os.path.join(github_dir, "musae_git_features.json"),
    #     target_path=os.path.join(github_dir, "musae_git_target.csv"),
    # )
    # rows.append(analyze_graph_structure("GitHub", data, labels))

    # --- Deezer ---
    deezer_dir = os.path.join(BASE_DIR, "../datasets/deezer_europe")
    data, labels = load_deezer_europe_dataset(
        edge_path=os.path.join(deezer_dir, "deezer_europe_edges.csv"),
        features_path=os.path.join(deezer_dir, "deezer_europe_features.json"),
        target_path=os.path.join(deezer_dir, "deezer_europe_target.csv"),
    )
    rows.append(analyze_graph_structure("Deezer", data, labels))


    df = pd.DataFrame(rows)
    df.to_csv("graph_structure_stats.csv", index=False)
    print("✅ Saved graph stats to graph_structure_stats.csv")


if __name__ == "__main__":
    run_all_graph_analyses()
