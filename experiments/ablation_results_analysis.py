import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.patches as patches
import seaborn as sns

# ─── STYLE ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams.update({
    "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False,
})

# ─── PATHS / MODES ─────────────────────────────────────────────────────
BASE_RESULTS_DIR = "results"
ABLATION_MODES = [
    "full", "no_align", "no_align_no_featrec", "no_linkpred", "no_featrec",
    "linkpred_only", "no_ssl", "no_classification",
    "no_gat", "shallow_gnn", "no_gate"
]

# ─── HELPERS ───────────────────────────────────────────────────────────
def load_metrics(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  metrics file not found: {path}")
        return None


def collect_ablation_metrics(graph, base_dir="results"):
    rows = []
    for mode in ABLATION_MODES:
        p = os.path.join(base_dir, f"{graph}_{mode}", "metrics.json")
        if not os.path.exists(p):
            print(f"→ skipping missing mode: {mode}")
            continue
        m = load_metrics(p)
        if not m:
            continue
        cls, lp = m.get("classification"), m.get("link_prediction")
        rows.append({
            "mode": mode,
            "clf_accuracy": cls.get("accuracy") if cls else None,
            "lp_auc":       lp.get("auc")      if lp  else None,
            "lp_ap":        lp.get("ap")       if lp  else None,
        })
    return pd.DataFrame(rows)

# ─── PRETTY BAR (shadow + rounded) ─────────────────────────────────────
def _shadow_bar(ax, x, h, w, color, z=2):
    ax.add_patch(patches.FancyBboxPatch(      # shadow
        (x-0.02, 0), w, h,
        boxstyle="round,pad=0.02", facecolor="grey", alpha=.25,
        linewidth=0, zorder=z-1))
    ax.add_patch(patches.FancyBboxPatch(      # main bar
        (x, 0), w, h,
        boxstyle="round,pad=0.02", facecolor=color,
        edgecolor="black", linewidth=0, zorder=z))

# ─── PLOTTING ──────────────────────────────────────────────────────────
def plot_ablation_results(df, metric, title=None, save_png=None):
    d      = df.dropna(subset=[metric]).sort_values(metric, ascending=False)
    modes  = d["mode"].values
    values = d[metric].values

    palette = sns.color_palette("Set2", len(modes))

    bar_w   = 0.5
    x_ticks = np.arange(len(modes))

    fig_w   = max(10, len(modes) * 1.1)   # ← ② wider canvas
    fig, ax = plt.subplots(figsize=(fig_w, 7))

    for i, (v, col) in enumerate(zip(values, palette)):
        xpos = i - bar_w/2 + 0.05         # keep tiny centre shift
        _shadow_bar(ax, xpos, v, bar_w, col)

    # axes & labels ------------------------------------------------------
    pad = bar_w * len(modes) / 2
    ax.set_xlim(-pad-.2, len(modes)-1 + pad + .2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(modes, rotation=35, ha='right', weight="medium")
    ax.set_ylabel(metric.replace("_", " ").title(), weight="bold")
    ax.set_title(title or f"Ablation Study – {metric.replace('_',' ').title()}",
                 weight="bold", pad=20)
    ax.set_ylim(0, values.max() * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=.4)
    plt.tight_layout()

    # high-res export ----------------------------------------------------
    if save_png:
        os.makedirs(os.path.dirname(save_png), exist_ok=True)
        plt.savefig(save_png, dpi=1200, bbox_inches="tight")  # ← ③
        print(f"✅ saved {save_png}")

    plt.show()


# ─── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    GRAPH = "facebook"
    df_ablation = collect_ablation_metrics(GRAPH)
    print(df_ablation)

    plot_ablation_results(df_ablation, "clf_accuracy",
                          "Classification Accuracy",
                          save_png=f"plots/{GRAPH}_clf_acc.png")

    plot_ablation_results(df_ablation, "lp_auc",
                          "Link-Prediction AUC",
                          save_png=f"plots/{GRAPH}_lp_auc.png")
