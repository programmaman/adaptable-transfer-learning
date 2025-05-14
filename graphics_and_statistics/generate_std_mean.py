import pandas as pd
import numpy as np

# Load Excel
file_path = "../results/experiment_results.xlsx"
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

# Metrics
metrics = ["accuracy", "precision", "recall", "f1", "auc", "ap"]
timing_components = ["classifier_time", "pretrain_time", "link_pred_time"]
timing_metrics = ["train_time"] + timing_components

# Alias mapping
time_aliases = {
    "classifier_train_time": "classifier_time",
    "link_prediction_time": "link_pred_time",
    "finetune_time": "classifier_time"
}

all_summaries = []

for sheet in sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = df.columns.str.strip().str.lower()
    df = df.loc[:, ~df.columns.duplicated()]

    if "pipeline" not in df.columns:
        print(f"[Warning] Skipping '{sheet}' — missing 'pipeline'")
        continue

    # Save original train_time (for fallback use)
    original_train_time = df["train_time"].copy() if "train_time" in df.columns else None

    # Unify alias columns into canonical time columns
    for alias, canonical in time_aliases.items():
        if alias in df.columns:
            if canonical in df.columns:
                df[canonical] = df[canonical].fillna(0) + df[alias].fillna(0)
            else:
                df[canonical] = df[alias]
            df.drop(columns=[alias], inplace=True)

    # Ensure all timing components exist
    for col in timing_components:
        if col not in df.columns:
            df[col] = 0.0

    # Compute total train_time from components
    computed_train_time = df[timing_components].sum(axis=1)

    # Use computed train time when available, fallback to original if needed
    if original_train_time is not None:
        df["train_time"] = computed_train_time.where(computed_train_time > 0, original_train_time)
    else:
        df["train_time"] = computed_train_time

    # Determine what metrics are available in this sheet
    available_metrics = [m for m in metrics + timing_metrics if m in df.columns]
    if not available_metrics:
        print(f"[Warning] Skipping '{sheet}' — no usable metric columns")
        continue

    # Group and summarize
    summary = df.groupby("pipeline")[available_metrics].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(col).rstrip('_') for col in summary.columns.values]
    summary.insert(0, "dataset", sheet.replace("_", " ").replace("LinkPrediction", "Link Prediction"))

    all_summaries.append(summary)

# Combine and export
final_summary = pd.concat(all_summaries, ignore_index=True)
final_summary.to_csv("gnn_summary_statistics.csv", index=False)
print("✅ Saved: gnn_summary_statistics.csv with full timing details")
