import pandas as pd

# Path to your Excel file
file_path = "../results/experiment_results_v2.xlsx"
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

# Metrics to summarize
metrics = ["accuracy", "precision", "recall", "f1", "auc", "ap"]

# Collect summaries here
all_summaries = []

for sheet in sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)

    # Standardize and clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Remove duplicate columns (e.g., multiple 'model' columns)
    df = df.loc[:, ~df.columns.duplicated()]

    # Rename 'pipeline' to 'model' if needed
    if 'pipeline' in df.columns and 'model' not in df.columns:
        df.rename(columns={'pipeline': 'model'}, inplace=True)



    # Verify required columns exist
    if 'model' not in df.columns:
        print(f"[Warning] Skipping sheet '{sheet}': missing 'model' or 'pipeline' column.")
        continue

    existing_metrics = [m for m in metrics if m in df.columns]
    if not existing_metrics:
        print(f"[Warning] Skipping sheet '{sheet}': no valid metric columns found.")
        continue

    print(f"Processing sheet: {sheet}")
    print("Columns:", df.columns.tolist())
    print(df[['model']].head())

    try:
        # Group by model and compute mean & std
        summary = df.groupby("model")[existing_metrics].agg(['mean', 'std']).reset_index()

        # Flatten multi-index column names
        summary.columns = ['_'.join(col).rstrip('_') for col in summary.columns.values]

        # Add dataset name as a column
        summary.insert(0, 'dataset', sheet)

        # Replace "_" with " " in 'dataset' row
        summary['dataset'] = summary['dataset'].str.replace("_", " ")
        #Space LinkPrediction to Link Prediction
        summary['dataset'] = summary['dataset'].str.replace("LinkPrediction", "Link Prediction")

        all_summaries.append(summary)

    except Exception as e:
        print(f"[Error] Failed to process sheet '{sheet}': {e}")
        continue

# Combine all dataset summaries
final_summary = pd.concat(all_summaries, ignore_index=True)

# Save or inspect
final_summary.to_csv("gnn_summary_statistics.csv", index=False)
print("✅ Summary saved to gnn_summary_statistics.csv")
