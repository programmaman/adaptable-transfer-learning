import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Seaborn and Matplotlib tuning
sns.set(style="whitegrid", font_scale=1.4)
plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

# Load and prep data
df = pd.read_csv("gnn_summary_statistics.csv")
df['model'] = df['model'].str.replace("Simple", "", regex=False)

datasets = sorted(df['dataset'].unique())
dataset_palette = dict(zip(datasets, sns.color_palette("Set2", len(datasets))))

def plot_metric_bar(metric="accuracy_mean", save=False, dpi=600):
    # Sort models by mean metric performance
    model_order = (
        df.groupby("model")[metric]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    plot_data = df.copy()
    plot_data["model"] = pd.Categorical(plot_data["model"], categories=model_order, ordered=True)
    plot_data.sort_values(["model", "dataset"], inplace=True)

    # === Dynamic Figure Sizing ===
    n_models = len(model_order)
    n_datasets = len(datasets)
    total_bars = n_models * n_datasets

    bar_width_px = 90   # width per bar (in pixels)
    fig_width_px = total_bars * bar_width_px
    fig_height_px = 1200  # fixed vertical size

    fig_width_in = fig_width_px / dpi
    fig_height_in = fig_height_px / dpi

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    # Bar spacing
    bar_width = 0.8 / n_datasets
    x_ticks = np.arange(n_models)

    # Draw each bar + shadow
    for i, dataset in enumerate(datasets):
        dataset_df = plot_data[plot_data['dataset'] == dataset]
        heights = dataset_df[metric].values

        for j, h in enumerate(heights):
            xpos = x_ticks[j] - 0.4 + bar_width * i

            ax.add_patch(patches.FancyBboxPatch(
                (xpos - 0.015, 0), bar_width, h,
                boxstyle="round,pad=0.01",
                facecolor='gray', alpha=0.2,
                linewidth=0, zorder=1
            ))
            ax.add_patch(patches.FancyBboxPatch(
                (xpos, 0), bar_width, h,
                boxstyle="round,pad=0.01",
                facecolor=dataset_palette[dataset],
                edgecolor='black',
                linewidth=0.5,
                zorder=2
            ))

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(model_order, rotation=30, ha="right", fontsize=14, weight="medium")
    ax.set_ylabel(metric.replace("_mean", "").capitalize(), fontsize=16, weight="bold", labelpad=10)
    ax.set_xlabel("Model", fontsize=16, weight="bold", labelpad=10)
    ax.set_title(f"{metric.replace('_mean', '').capitalize()} by Model and Dataset", fontsize=22, weight="bold", pad=30)
    ax.set_ylim(0, plot_data[metric].max() * 1.15)

    handles = [patches.Patch(color=dataset_palette[ds], label=ds) for ds in datasets]
    ax.legend(handles=handles, title="Dataset", bbox_to_anchor=(1.005, 1), loc='upper left', frameon=False)

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine(left=True)

    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        path = f"plots/{metric}_ultrawide.png"
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"✅ Saved full-size {metric} plot to {path}")

    plt.show()

# Run the plots at ultra-wide resolution
if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    plot_metric_bar("accuracy_mean", save=True)
    plot_metric_bar("f1_mean", save=True)
    plot_metric_bar("auc_mean", save=True)
    plot_metric_bar("ap_mean", save=True)
