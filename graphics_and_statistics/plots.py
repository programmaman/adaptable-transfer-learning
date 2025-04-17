import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Set plot style
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
})

# Load summary statistics
df = pd.read_csv("gnn_summary_statistics.csv")

# Normalize model naming
df['model'] = df['model'].str.replace("Simple", "", regex=False)

# Consistent color palette per model
model_order = df['model'].unique()
palette = sns.color_palette("colorblind", len(model_order))
model_palette = dict(zip(model_order, palette))

def plot_metric_bar(metric="accuracy_mean", std=None, save=False):
    fig, ax = plt.subplots(figsize=(12, 6))
    order = (
        df.groupby('model')[metric]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    sns.barplot(
        data=df,
        x="model",
        y=metric,
        hue="dataset",
        hue_order=sorted(df['dataset'].unique()),
        order=order,
        palette=model_palette,
        ci=None,
        edgecolor=".2"
    )
    ax.set_title(f"Model Comparison: {metric.replace('_mean', '').capitalize()}")
    ax.set_ylabel(metric.replace('_mean', '').capitalize())
    ax.set_xlabel("Model")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save:
        plt.savefig(f"plots/{metric}_barplot.png", bbox_inches="tight")
    plt.show()

os.makedirs("plots", exist_ok=True)
plot_metric_bar("accuracy_mean")
plot_metric_bar("f1_mean")
plot_metric_bar("auc_mean")
plot_metric_bar("ap_mean")


