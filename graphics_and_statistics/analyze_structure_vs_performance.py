import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Load & Merge ---
graph_stats = pd.read_csv("graph_structure_stats.csv")
gnn_results = pd.read_csv("gnn_summary_statistics.csv")

# Normalize dataset names for merge
gnn_results["base_dataset"] = gnn_results["dataset"].apply(lambda x: x.split("_")[0])

# Merge the data
merged = pd.merge(
    gnn_results,
    graph_stats,
    left_on="base_dataset",
    right_on="dataset",
    suffixes=("_metric", "_graph")
).drop(columns=["dataset_graph"])

# --- Save Merged Data ---
merged.to_csv("merged_performance_and_graph_stats.csv", index=False)
print("✅ Merged dataset saved to 'merged_performance_and_graph_stats.csv'")


# --- Correlation Heatmap ---
def plot_correlation_matrix(df, title="Correlation Matrix"):
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()

    plt.figure(figsize=(18, 14))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(title, fontsize=18, weight="bold")
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/structure_vs_performance_corr.png", dpi=300)
    plt.show()

plot_correlation_matrix(merged, title="Correlation: GNN Performance vs Graph Structure")


def plot_scatter(x, y, hue="model", df=merged):
    g = sns.lmplot(
        data=df, x=x, y=y, hue=hue,
        height=6, aspect=1.6, markers="o", ci=None,
        scatter_kws={"s": 60, "edgecolor": "w", "linewidths": 0.5}
    )

    # Expand canvas space and ensure legend is shown and clear
    g.fig.subplots_adjust(right=0.8)  # Leave room for legend
    if g._legend:
        g._legend.set_bbox_to_anchor((1.05, 0.5))
        g._legend.set_title("Model")
        g._legend.set_frame_on(True)  # Optional: makes legend stand out
        g._legend.get_frame().set_edgecolor('black')

    g.set_titles("")  # Remove default FacetGrid title
    g.fig.suptitle(f"{y} vs {x}", fontsize=16, weight="bold")  # Main title

    fname = f"plots/{y}_vs_{x}.png"
    g.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {fname}")
    plt.show()


# --- Key structural predictors ---
structure_metrics = ["homophily", "avg_clustering", "diameter", "bc_std", "cc_mean", "pr_mean"]
performance_metrics = ["accuracy_mean", "f1_mean", "precision_mean", "recall_mean", "auc_mean", "ap_mean"]

for perf in performance_metrics:
    for struct in structure_metrics:
        if perf in merged.columns and struct in merged.columns:
            plot_scatter(x=struct, y=perf)

print("✅ All scatter plots generated.")
