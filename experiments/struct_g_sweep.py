# Unified Structural Graph Sweep Script
# Purpose: Analyze how graph structure affects GNN model performance

import itertools

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import from_networkx
from tqdm import tqdm

# --- Sweep Configuration ---
CONFIG_SPACE = {
    'num_nodes': [200, 500, 1000],  # Vary graph size explicitly
    'edge_factor': [1, 2, 4],  # New: Edge multiplier for node count
    'homophily': [0.1, 0.5, 0.9],
    'clustering': [0.0, 0.2],
    'diameter': [3, 5],
    'assortativity': [-0.5, 0.0, 0.5],
}

LABEL_NOISE = 0.33
MAX_CLUSTERING_ITERS = 300
FAST_MODE = False  # If True: skips rewiring/diameter mods for speed

# --- Pipeline Imports ---
from pipeline_1 import run_structg_pipeline_internal
from pipeline_2 import run_structg_pipeline as run_structg_pipeline_external


# --- Graph Utilities ---
def generate_config_grid(config_dict):
    keys, values = zip(*config_dict.items())
    product = list(itertools.product(*values))
    return [dict(zip(keys, v)) for v in product]


def assign_labels(G, num_nodes, noise_prob=LABEL_NOISE):
    labels = []
    for node in G.nodes():
        true_block = int(node < num_nodes // 2)
        noisy_label = true_block if np.random.rand() > noise_prob else 1 - true_block
        labels.append(noisy_label)
    for i, label in enumerate(labels):
        G.nodes[i]['y'] = label
    return G


def increase_clustering(G, target, max_iter=MAX_CLUSTERING_ITERS):
    for _ in range(max_iter):
        if nx.transitivity(G) >= target:
            break
        w = np.random.choice(G.nodes())
        neighbors = list(G.neighbors(w))
        if len(neighbors) >= 2:
            u, v = np.random.choice(neighbors, 2, replace=False)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
    return G


def rewire_for_assortativity(G, target, max_iter=1000):
    for _ in range(max_iter):
        nx.double_edge_swap(G, nswap=1, max_tries=100)
        if abs(nx.degree_assortativity_coefficient(G) - target) < 0.05:
            break
    return G


def generate_synthetic_graph(config):
    num_nodes = config['num_nodes']
    target_edges = config['edge_factor'] * num_nodes
    sizes = [num_nodes // 2, num_nodes // 2]
    p = config['homophily']
    probs = [[p, 1 - p], [1 - p, p]]
    G = nx.stochastic_block_model(sizes, probs)

    if not FAST_MODE:
        G = increase_clustering(G, config['clustering'])
        if nx.is_connected(G):
            try:
                while nx.diameter(G) > config['diameter']:
                    u, v = np.random.choice(G.nodes(), 2, replace=False)
                    if G.has_edge(u, v):
                        G.remove_edge(u, v)
            except:
                pass
        G = rewire_for_assortativity(G, config['assortativity'])

    while G.number_of_edges() < target_edges:
        u, v = np.random.choice(G.nodes(), 2, replace=False)
        G.add_edge(u, v)

    G = assign_labels(G, num_nodes)
    for node in G.nodes():
        G.nodes[node]['x'] = np.random.randn(10).astype(np.float32)

    data = from_networkx(G)
    data.x = data.x.float()
    return data, G


# --- Main Sweep Loop ---
def run_structural_sweep(debug=False):
    configs = generate_config_grid(CONFIG_SPACE)
    results = []

    for i, config in enumerate(tqdm(configs, desc="Sweeping Graph Configurations")):
        print(f"\n[CONFIG {i + 1}/{len(configs)}] {config}")
        try:
            graph_data, G_nx = generate_synthetic_graph(config)

            labels = graph_data.y
            if labels.dtype != torch.long:
                labels = LabelEncoder().fit_transform(labels.cpu().numpy())
                labels = torch.tensor(labels, dtype=torch.long)
            graph_data.y = labels

            # --- Run Internal Classifier Pipeline ---
            _, int_cls, int_lp = run_structg_pipeline_internal(
                data=graph_data.clone(),
                labels=labels.clone(),
                do_linkpred=True,
                do_featrec=False,
                do_n2v_align=True,
                pretrain_epochs=30 if not debug else 5,
                finetune_epochs=15 if not debug else 3,
            )

            # --- Run External Classifier Pipeline ---
            _, ext_cls, ext_lp = run_structg_pipeline_external(
                data=graph_data.clone(),
                labels=labels.clone(),
                do_linkpred=True,
                do_featrec=True,
                do_n2v_align=True,
                pretrain_epochs=30 if not debug else 5,
                finetune_epochs=15 if not debug else 3,
            )

            def safe(v):
                return v if v is not None else -1

            row = {
                **config,
                'num_edges': G_nx.number_of_edges(),
                'avg_degree': 2 * G_nx.number_of_edges() / G_nx.number_of_nodes(),
                'int_cls_acc': int_cls.accuracy, 'int_cls_f1': int_cls.f1, 'int_cls_auc': safe(int_cls.auc),
                'int_lp_acc': safe(int_lp.accuracy), 'int_lp_f1': safe(int_lp.f1), 'int_lp_auc': safe(int_lp.auc),
                'ext_cls_acc': ext_cls.accuracy, 'ext_cls_f1': ext_cls.f1, 'ext_cls_auc': safe(ext_cls.auc),
                'ext_lp_acc': safe(ext_lp.accuracy), 'ext_lp_f1': safe(ext_lp.f1), 'ext_lp_auc': safe(ext_lp.auc),
                'real_density': nx.density(G_nx),
                'real_assortativity': nx.degree_assortativity_coefficient(G_nx),
                'real_clustering': nx.transitivity(G_nx),
                'real_diameter': nx.diameter(G_nx) if nx.is_connected(G_nx) else -1
            }

            results.append(row)

            if debug:
                print("[DEBUG] Stopping after one configuration.")
                break

        except Exception as e:
            print(f"[ERROR] Config {i + 1}: {e}")
            continue

    df = pd.DataFrame(results)
    df.to_csv("structural_sweep_results.csv", index=False)
    print("\n[SAVED] Results saved to 'structural_sweep_results.csv'")


if __name__ == '__main__':
    run_structural_sweep(debug=False)
