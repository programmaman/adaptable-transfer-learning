import pandas as pd

# Paths
input_file = "/app/results/graphbert_results.xlsx"
output_file = "/app/results/graphbert_summary.xlsx"

# Columns to summarize
numeric_cols = [
    "accuracy", "precision", "recall", "f1", "auc", "ap",
    "train_time", "classifier_eval_time", "link_pred_time",
    "lp_eval_time", "total_time"
]

# Function to summarize one sheet
def summarize_sheet(sheet_name):
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    # Only keep numeric cols that are actually present
    available_cols = [c for c in numeric_cols if c in df.columns]
    agg = df.groupby("Experiment")[available_cols].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg

# Summarize both sheets
summary_cls = summarize_sheet("Classification")
summary_lp = summarize_sheet("LinkPrediction")

# Save to Excel with multiple sheets
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    summary_cls.to_excel(writer, sheet_name="Classification_Summary")
    summary_lp.to_excel(writer, sheet_name="LinkPrediction_Summary")

print(f"Summaries with mean/std saved to {output_file}")
print("Classification Summary:")
print(summary_cls)
print("\nLink Prediction Summary:")
print(summary_lp)
