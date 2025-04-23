import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.utils import to_networkx
from experiments.struct_g_pipeline import run_structg_pipeline

# ---------------------------
# Ablation Configurations
# ---------------------------
ABLATION_MODES = {
    'full':         dict(do_linkpred=True,  do_n2v_align=True,  do_featrec=True),
    'no_align':     dict(do_linkpred=True,  do_n2v_align=False, do_featrec=True),
    'no_linkpred':  dict(do_linkpred=False, do_n2v_align=True,  do_featrec=True),
    'no_featrec':   dict(do_linkpred=True,  do_n2v_align=True,  do_featrec=False),
    'no_ssl':       dict(do_linkpred=False, do_n2v_align=False, do_featrec=False),
}


# ---------------------------
# Save Utilities
# ---------------------------
def save_metrics(metrics, out_dir):
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

def save_embeddings(embeddings, name, out_dir):
    torch.save(embeddings, os.path.join(out_dir, f"{name}.pt"))

def plot_tsne(embeddings, labels, out_dir):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    X_2d = tsne.fit_transform(embeddings.cpu().numpy())
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels.cpu().numpy(), cmap="tab10", alpha=0.7)
    plt.legend(*scatter.legend_elements(), loc="best")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsne.png"))
    plt.close()


# ---------------------------
# Main Runner
# ---------------------------
def run_analysis_on_graph(name, data, labels, base_output_dir="results", seed=42):
    for mode, cfg in ABLATION_MODES.items():
        print(f"\n[Running {mode}] on {name}")
        out_dir = os.path.join(base_output_dir, f"{name}_{mode}")
        os.makedirs(out_dir, exist_ok=True)

        model, _, clf_res, lp_res = run_structg_pipeline(
            data=data,
            labels=labels,
            seed=seed,
            pretrain_epochs=30,
            finetune_epochs=15,
            num_classes=int(labels.max().item() + 1),
            **cfg
        )

        with torch.no_grad():
            node_idx = torch.arange(data.num_nodes).to(labels.device)
            e_v = model(data.x, data.edge_index, node_idx)
            z_n2v = model.node2vec_layer(node_idx)

        save_embeddings(e_v, "e_v", out_dir)
        save_embeddings(z_n2v, "z_n2v", out_dir)

        save_metrics({
            "classification": clf_res.to_dict(),
            "link_prediction": lp_res.to_dict() if lp_res else None
        }, out_dir)

        plot_tsne(e_v, labels, out_dir)


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    from experiments.synthetic_generator import generate_synthetic_graph

    config = {
        'homophily': 0.5,
        'clustering': 0.2,
        'diameter': 5,
        'assortativity': 0.0,
        'density': 0.01,
    }
    data = generate_synthetic_graph(config)
    run_analysis_on_graph("synthetic1", data, data.y)
