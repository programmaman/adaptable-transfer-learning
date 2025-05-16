import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import matplotlib.patches as patches, numpy as np, os

# ─── style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams.update({
    "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

# ─── load & tidy ──────────────────────────────────────────────────────
df = pd.read_csv("gnn_summary_statistics.csv")
if "pipeline" in df.columns:
    df = df.rename(columns={"pipeline": "model"})
df["model"] = df["model"].str.replace("Simple", "", regex=False)
df["task"]  = df["dataset"].apply(
    lambda d: "classification" if "Classification" in d else "link_prediction"
)

# ─── plotting helper ──────────────────────────────────────────────────
def plot_metric_bar(metric="accuracy_mean", *, save=False, dpi=600):
    is_lp   = metric in ["auc_mean", "ap_mean"]
    task    = "link_prediction" if is_lp else "classification"

    plot_df = df[df["task"] == task].copy()
    datasets = sorted(plot_df["dataset"].unique())
    palette  = dict(zip(datasets, sns.color_palette("Set2", len(datasets))))

    model_order = (
        plot_df.groupby("model")[metric].mean()
        .sort_values(ascending=False).index
    )

    # dynamic canvas width ------------------------------------------------
    fig_width  = max(8, len(model_order) * 0.9)          # ❶
    fig_height = 9
    _, ax = plt.subplots(figsize=(fig_width, fig_height))

    plot_df["model"] = pd.Categorical(plot_df["model"],
                                      categories=model_order, ordered=True)
    plot_df.sort_values(["model", "dataset"], inplace=True)

    bar_w   = 0.8 / len(datasets)
    x_ticks = np.arange(len(model_order))

    for i, ds in enumerate(datasets):
        ds_df   = plot_df[plot_df["dataset"] == ds]
        heights = ds_df[metric].values
        for j, h in enumerate(heights):
            xpos = x_ticks[j] - 0.4 + bar_w * i + 0.05    # ❷ tiny shift
            # drop-shadow
            ax.add_patch(patches.FancyBboxPatch(
                (xpos - 0.02, 0), bar_w, h,
                boxstyle="round,pad=0.02", linewidth=0,
                facecolor="gray", alpha=0.2, zorder=1))
            # main bar
            ax.add_patch(patches.FancyBboxPatch(
                (xpos, 0), bar_w, h,
                boxstyle="round,pad=0.02", linewidth=0,
                facecolor=palette[ds], edgecolor="black", zorder=2))

    pad = bar_w * len(datasets) / 2
    ax.set_xlim(-pad - .2, len(model_order)-1 + pad + .2) # ❸ symmetric space
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(model_order, rotation=30, ha="right",
                       fontsize=13, weight="medium")
    ax.set_ylabel(metric.replace("_mean", "").capitalize(),
                  fontsize=16, weight="bold", labelpad=10)
    ax.set_xlabel("Model", fontsize=16, weight="bold", labelpad=10)
    ax.set_title(f"{metric.replace('_mean','').capitalize()} by Model "
                 f"({task.capitalize()})",
                 fontsize=22, weight="bold", pad=30)
    ax.set_ylim(0, plot_df[metric].max() * 1.2)

    # legend
    handles = [patches.Patch(color=palette[d], label=d) for d in datasets]
    ax.legend(handles=handles, title="Dataset",
              bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    sns.despine(left=True)
    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        fname = f"plots/{metric}_4k.png"
        plt.savefig(fname, dpi=dpi, bbox_inches="tight")
        print(f"✅ saved {fname}")

    plt.show()

# ─── run all metrics ──────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    for m in ["accuracy_mean", "f1_mean", "precision_mean",
              "recall_mean", "auc_mean", "ap_mean"]:
        if m in df.columns:
            print(f"📊 plotting {m} …")
            plot_metric_bar(m, save=True)
