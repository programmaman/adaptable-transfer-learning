import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# 1. Load the summary file
# ------------------------------------------------------------------
csv_path = Path("gnn_summary_statistics.csv")
assert csv_path.exists(), f"❌  Cannot find {csv_path}"
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# 2. Split into classification vs. link-prediction
# ------------------------------------------------------------------
is_linkpred = df["dataset"].str.contains("Link Prediction", case=False)
cls_df = df[~is_linkpred].copy()
lp_df = df[is_linkpred].copy()

# ------------------------------------------------------------------
# 3. Metrics
# ------------------------------------------------------------------
metrics_cls = ["accuracy", "f1"]
metric_names = {"accuracy": "Accuracy", "f1": "F1 Score"}

# ------------------------------------------------------------------
# 4. Output directory
# ------------------------------------------------------------------
out_dir = Path("latex_tables")
out_dir.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# 5. Bold function
# ------------------------------------------------------------------
def bold_max(tex, df):
    for row_lbl, row in df.iterrows():
        best = row.max()
        safe_lbl = str(row_lbl).replace('_', r'\_')
        tex = tex.replace(f"{safe_lbl} &", f"{safe_lbl} &").replace(
            f"{best:.3f}", rf"\textbf{{{best:.3f}}}")
    return tex


# ------------------------------------------------------------------
# 6. Classification tables (mean ± std, one per metric)
# ------------------------------------------------------------------
for metric in metrics_cls:
    mean_pivot = (
        cls_df
        .pivot(index="dataset", columns="pipeline", values=f"{metric}_mean")
        .sort_index(axis=1)
    )
    std_pivot = (
        cls_df
        .pivot(index="dataset", columns="pipeline", values=f"{metric}_std")
        .sort_index(axis=1)
    )

    formatted = mean_pivot.copy()
    for row in mean_pivot.index:
        for col in mean_pivot.columns:
            m = mean_pivot.loc[row, col]
            s = std_pivot.loc[row, col]
            if pd.notna(m) and pd.notna(s):
                formatted.loc[row, col] = f"{m:.3f} ± {s:.3f}"
            else:
                formatted.loc[row, col] = ""

    latex = formatted.to_latex(
        escape=True,
        caption=f"Node-classification {metric_names[metric]} (mean ± std over 5 runs; best mean per row in bold).",
        label=f"tab:cls_{metric}",
        bold_rows=False,
        column_format="l" + "r" * formatted.shape[1]
    )

    # Bold the best MEAN value per row
    latex = bold_max(latex, mean_pivot)
    (out_dir / f"table_cls_{metric}.tex").write_text(latex)


# ------------------------------------------------------------------
# 7. Link prediction tables: AUC and AP
# ------------------------------------------------------------------
lp_metrics = ["auc", "ap"]
lp_metric_names = {"auc": "AUC", "ap": "Average Precision"}

for metric in lp_metrics:
    mean_pivot = (
        lp_df
        .pivot(index="dataset", columns="pipeline", values=f"{metric}_mean")
        .sort_index(axis=1)
    )
    std_pivot = (
        lp_df
        .pivot(index="dataset", columns="pipeline", values=f"{metric}_std")
        .sort_index(axis=1)
    )

    formatted = mean_pivot.copy()
    for row in mean_pivot.index:
        for col in mean_pivot.columns:
            m = mean_pivot.loc[row, col]
            s = std_pivot.loc[row, col]
            if pd.notna(m) and pd.notna(s):
                formatted.loc[row, col] = f"{m:.3f} ± {s:.3f}"
            else:
                formatted.loc[row, col] = ""

    latex = formatted.to_latex(
        escape=True,
        caption=f"Link-prediction {lp_metric_names[metric]} (mean ± std over 5 runs; best mean per row in bold).",
        label=f"tab:lp_{metric}",
        bold_rows=False,
        column_format="l" + "r" * formatted.shape[1]
    )

    latex = bold_max(latex, mean_pivot)
    (out_dir / f"table_lp_{metric}.tex").write_text(latex)

# ------------------------------------------------------------------
# 8. Done
# ------------------------------------------------------------------
print("✅  Saved:")
for metric in metrics_cls:
    print(f"   • latex_tables/table_cls_{metric}.tex")
for metric in lp_metrics:
    print(f"   • latex_tables/table_lp_{metric}.tex")

