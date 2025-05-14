import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# --- Style ---
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

# --- Load & Prep ---
df = pd.read_csv("gnn_summary_statistics.csv")

# Rename pipeline to model
if "pipeline" in df.columns:
    df = df.rename(columns={"pipeline": "model"})

# Normalize model names
df['model'] = df['model'].str.replace("Simple", "", regex=False)

# Categorize tasks
df['task'] = df['dataset'].apply(lambda d: 'classification' if 'Classification' in d else 'link_prediction')

# --- Plotting ---
def plot_metric_bar(metric="accuracy_mean", save=False, dpi=600):
    # Determine task type
    is_lp_metric = metric in ["auc_mean", "ap_mean"]
    task_filter = 'link_prediction' if is_lp_metric else 'classification'

    plot_data = df[df['task'] == task_filter].copy()
    datasets = sorted(plot_data['dataset'].unique())
    dataset_palette = dict(zip(datasets, sns.color_palette("Set2", len(datasets))))

    model_order = (
        plot_data.groupby("model")[metric]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    fig_width = 16
    fig_height = 9

    _, ax = plt.subplots(figsize=(fig_width, fig_height))

    plot_data["model"] = pd.Categorical(plot_data["model"], categories=model_order, ordered=True)
    plot_data.sort_values(["model", "dataset"], inplace=True)

    bar_width = 0.8 / len(datasets)
    x_ticks = np.arange(len(model_order))

    for i, dataset in enumerate(datasets):
        dataset_df = plot_data[plot_data['dataset'] == dataset]
        heights = dataset_df[metric].values

        for j, h in enumerate(heights):
            xpos = x_ticks[j] - 0.4 + bar_width * i

            # Drop shadow
            ax.add_patch(patches.FancyBboxPatch(
                (xpos - 0.02, 0),
                width=bar_width,
                height=h,
                boxstyle="round,pad=0.02",
                linewidth=0,
                facecolor='gray',
                alpha=0.2,
                zorder=1,
            ))

            # Main bar
            ax.add_patch(patches.FancyBboxPatch(
                (xpos, 0),
                width=bar_width,
                height=h,
                boxstyle="round,pad=0.02",
                linewidth=0,
                facecolor=dataset_palette[dataset],
                edgecolor='black',
                zorder=2,
            ))

    padding = bar_width * len(datasets) / 2
    ax.set_xlim(-padding, len(model_order) - 1 + padding)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(model_order, rotation=30, ha="right", fontsize=13, weight="medium")
    ax.set_ylabel(metric.replace("_mean", "").capitalize(), fontsize=16, weight="bold", labelpad=10)
    ax.set_xlabel("Model", fontsize=16, weight="bold", labelpad=10)
    ax.set_title(f"{metric.replace('_mean', '').capitalize()} by Model ({task_filter.capitalize()})", fontsize=22, weight="bold", pad=30)
    ax.set_ylim(0, plot_data[metric].max() * 1.2)

    # Legend
    handles = [patches.Patch(color=dataset_palette[ds], label=ds) for ds in datasets]
    ax.legend(handles=handles, title="Dataset", bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)

    ax.grid(axis='y', linestyle='--', alpha=0.4)
    sns.despine(left=True)
    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        filename = f"plots/{metric}_4k.png"
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        print(f"✅ Saved {metric} plot to {filename}")

    plt.show()


# --- Run All Task-Specific Plots ---
if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    metrics = ["accuracy_mean", "f1_mean", "precision_mean", "recall_mean", "auc_mean", "ap_mean"]

    for metric in metrics:
        if metric in df.columns:
            print(f"📊 Plotting {metric}...")
            plot_metric_bar(metric, save=True)
