import json
import os

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from experiments.experiment_utils import load_musae_facebook_dataset
from utils import get_device

# ---------------------------
# Ablation Configurations
# ---------------------------
ABLATION_MODES = {
    'full': dict(do_linkpred=True, do_n2v_align=True, do_featrec=True, do_classification=True),
    'no_align': dict(do_linkpred=True, do_n2v_align=False, do_featrec=True, do_classification=True),
    'no_linkpred': dict(do_linkpred=False, do_n2v_align=True, do_featrec=True, do_classification=True),
    'no_featrec': dict(do_linkpred=True, do_n2v_align=True, do_featrec=False, do_classification=True),
    'no_ssl': dict(do_linkpred=False, do_n2v_align=False, do_featrec=False, do_classification=True),
    'no_classification': dict(do_linkpred=True, do_n2v_align=True, do_featrec=True, do_classification=False),
    # Architecture ablations:
    'no_gat': dict(do_linkpred=True, do_n2v_align=True, do_featrec=True, do_classification=True, use_gat=False),
    'shallow_gnn': dict(do_linkpred=True, do_n2v_align=True, do_featrec=True, do_classification=True, num_layers=1),
    'no_gate': dict(do_linkpred=True, do_n2v_align=True, do_featrec=True, do_classification=True, use_gate=False),
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
from experiments.struct_g_internal_pipeline import run_structg_pipeline_internal
from experiments.struct_g_pipeline import run_structg_pipeline


def run_analysis_on_graph(name, data, labels, base_output_dir="results", seed=42):
    for mode, mode_cfg in ABLATION_MODES.items():
        print(f"\n[Running {mode}] on {name}")
        out_dir = os.path.join(base_output_dir, f"{name}_{mode}")
        os.makedirs(out_dir, exist_ok=True)

        cfg = mode_cfg.copy()
        do_classification = cfg.pop("do_classification", True)

        # NEW: extract optional overrides (and remove from cfg to avoid double-passing)
        use_gate = cfg.pop("use_gate", True)
        use_gat = cfg.pop("use_gat", True)
        num_layers = cfg.pop("num_layers", 2)

        # Ensure all data and model inputs are on the same device
        device = get_device()
        data = data.to(device)
        labels = labels.to(device)

        if do_classification:
            model, clf_res, lp_res = run_structg_pipeline_internal(
                data=data,
                labels=labels,
                seed=seed,
                pretrain_epochs=100,
                finetune_epochs=30,
                use_gate=use_gate,  # <-- pass explicitly
                use_gat=use_gat,  # <-- pass explicitly
                num_layers=num_layers,  # <-- pass explicitly
                **cfg
            )
        else:
            model, clf_res, lp_res = run_structg_pipeline(
                data=data,
                labels=labels,
                seed=seed,
                pretrain_epochs=100,
                finetune_epochs=30,
                num_classes=None,  # force no classifier
                use_gate=use_gate,  # <-- pass explicitly
                use_gat=use_gat,  # <-- pass explicitly
                num_layers=num_layers,  # <-- pass explicitly
                **cfg
            )

        with torch.no_grad():
            node_idx = torch.arange(data.num_nodes, device=device)
            e_v = model(data.x, data.edge_index, node_idx)
            z_n2v = model.node2vec_layer(node_idx)

        save_embeddings(e_v, "e_v", out_dir)
        save_embeddings(z_n2v, "z_n2v", out_dir)

        save_metrics({
            "classification": clf_res.as_dict() if clf_res else None,
            "link_prediction": lp_res.as_dict() if lp_res else None
        }, out_dir)

        if do_classification and clf_res:
            plot_tsne(e_v, labels, out_dir)


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    # Load Facebook graph (or other)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    fb_dir = os.path.join(BASE_DIR, "../datasets/facebook_large")
    edge_path = os.path.join(fb_dir, "musae_facebook_edges.csv")
    features_path = os.path.join(fb_dir, "musae_facebook_features.json")
    target_path = os.path.join(fb_dir, "musae_facebook_target.csv")
    data, labels, _ = load_musae_facebook_dataset(edge_path, features_path, target_path)
    run_analysis_on_graph("facebook", data, labels)
