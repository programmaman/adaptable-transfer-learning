import itertools
import numpy as np
import pandas as pd
import networkx as nx
import torch
import time

from networkx.algorithms.community.label_propagation import label_propagation_communities
from torch_geometric.utils import from_networkx, to_networkx
from experiments.struct_g_pipeline import run_structg_pipeline
from sklearn.preprocessing import LabelEncoder

# =====================
# 1. EXPERIMENT CONFIGURATION
# =====================

CONFIG_SPACE = {
    'homophily':     [0.1, 0.3, 0.5, 0.7, 0.9],
    'clustering':    [0.0, 0.2, 0.4],  # Keep clustering limited to reduce runtime
    'diameter':      [3, 5, 7],
    'assortativity': [-0.5, 0.0, 0.5],
    'density':       [0.005, 0.01, 0.05],
}

N_NODES = 500
MAX_CLUSTERING_ITERS = 500


# =====================
# 2. CONFIG GENERATION
# =====================

def generate_config_grid(config_dict):
    keys, values = zip(*config_dict.items())
    product = list(itertools.product(*values))
    print(f"Generated {len(product)} configurations.")
    return [dict(zip(keys, v)) for v in product]


def assign_community_labels(G):
    communities = list(label_propagation_communities(G))
    comm_map = {}
    for label, nodes in enumerate(communities):
        for node in nodes:
            comm_map[node] = label
    nx.set_node_attributes(G, comm_map, 'y')
    return G


# =====================
# 3. GRAPH GENERATION
# =====================

def rewire_for_assortativity(G, target, max_iter=1000):
    for _ in range(max_iter):
        nx.double_edge_swap(G, nswap=1, max_tries=100)
        current = nx.degree_assortativity_coefficient(G)
        if abs(current - target) < 0.05:
            break
    return G


def increase_clustering(G, target, max_iter=MAX_CLUSTERING_ITERS):
    for i in range(max_iter):
        current = nx.transitivity(G)
        if current >= target:
            break
        w = np.random.choice(G.nodes())
        neighbors = list(G.neighbors(w))
        if len(neighbors) < 2:
            continue
        u, v = np.random.choice(neighbors, 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
    return G


def generate_synthetic_graph(config):
    sizes = [N_NODES // 2] * 2
    p = config['homophily']
    probs = [[p, 1 - p], [1 - p, p]]
    G = nx.stochastic_block_model(sizes, probs)

    G = increase_clustering(G, config['clustering'])

    if nx.is_connected(G):
        try:
            while nx.diameter(G) > config['diameter']:
                u, v = np.random.choice(G.nodes(), 2, replace=False)
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
        except Exception:
            pass

    G = rewire_for_assortativity(G, config['assortativity'])

    while nx.density(G) < config['density']:
        u, v = np.random.choice(G.nodes(), 2, replace=False)
        G.add_edge(u, v)

    G = assign_community_labels(G)
    for node in G.nodes():
        G.nodes[node]['x'] = np.random.randn(10).astype(np.float32)

    data = from_networkx(G)
    data.x = data.x.float()
    return data


# =====================
# 4. STRUCTURAL SWEEP
# =====================

def run_structural_sweep(debug=False):
    configs = generate_config_grid(CONFIG_SPACE)
    results = []

    for i, config in enumerate(configs):
        print(f"\nRunning configuration {i + 1}/{len(configs)}: {config}")
        try:
            graph_data = generate_synthetic_graph(config)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Convert labels
            labels = graph_data.y
            if labels.dtype != torch.long:
                labels = LabelEncoder().fit_transform(labels.cpu().numpy())
                labels = torch.tensor(labels, dtype=torch.long)

            graph_data.y = labels.to(device)
            graph_data.x = graph_data.x.to(device)
            graph_data.edge_index = graph_data.edge_index.to(device)

            num_classes = int(graph_data.y.max().item() + 1)

            _, _, clf_res, lp_res = run_structg_pipeline(
                data=graph_data,
                labels=graph_data.y,
                do_linkpred=True,
                do_featrec=False,
                do_n2v_align=True,
                pretrain_epochs=10 if debug else 30,
                finetune_epochs=5 if debug else 15,
                num_classes=num_classes
            )

            G_nx = to_networkx(graph_data, to_undirected=True)
            row = {
                **config,
                # Node classification
                'cls_accuracy': clf_res.accuracy,
                'cls_f1': clf_res.f1,
                'cls_precision': clf_res.precision,
                'cls_recall': clf_res.recall,
                'cls_auc': clf_res.auc,
                # Link prediction
                'lp_accuracy': lp_res.accuracy if lp_res else None,
                'lp_f1': lp_res.f1 if lp_res else None,
                'lp_precision': lp_res.precision if lp_res else None,
                'lp_recall': lp_res.recall if lp_res else None,
                'lp_auc': lp_res.auc if lp_res else None,
                'lp_ap': lp_res.ap if lp_res else None,
                # Graph stats
                'real_density': nx.density(G_nx),
                'real_assortativity': nx.degree_assortativity_coefficient(G_nx),
                'real_clustering': nx.transitivity(G_nx),
                'real_diameter': nx.diameter(G_nx) if nx.is_connected(G_nx) else -1
            }
            results.append(row)

            if debug:
                print("Debug mode active — exiting early.")
                break

        except Exception as e:
            print(f"Error during config {i + 1}: {e}")
            continue

    df = pd.DataFrame(results)
    df.to_csv("structuralgnn_sweep_results.csv", index=False)
    print("Results saved to structuralgnn_sweep_results.csv")


if __name__ == '__main__':
    run_structural_sweep(debug=False)
