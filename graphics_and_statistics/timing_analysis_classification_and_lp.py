"""
Timing tables (Classification vs Link-Prediction)
─────────────────────────────────────────────────
Reads  : ./results/timing_by_dataset.csv
Writes : results/latex_tables/tab_runtime_classification.tex
         results/latex_tables/tab_runtime_link_prediction.tex
"""

import math
import os
from pathlib import Path

import pandas as pd

# ───────────────────────── configuration ──────────────────────────────
CSV_FILE = "./results/timing_by_dataset.csv"
OUT_DIR  = Path("results/latex_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────── helpers ────────────────────────────────────
def _valid(v) -> bool:
    """True if v is numeric, non-NaN and > 0."""
    return (v is not None) and (not isinstance(v, float) or not math.isnan(v)) and v > 0


def runtime_for_row(row):
    """Pick the most relevant timing for each experiment row."""
    if "Classification" in row["dataset"]:
        preferred = row["classifier_time_mean"]
    else:                                   # Link-Prediction
        preferred = row["link_pred_time_mean"]
    return preferred if _valid(preferred) else row["train_time_mean"]


def to_latex_matrix(df_wide, caption, label):
    """
    Convert a *wide* DF (index=model, columns=datasets) into a LaTeX booktabs table,
    bold-facing the fastest model per dataset (column minimum).
    """
    # String-format and bold the minima
    df_fmt = df_wide.copy()
    for col in df_fmt.columns:
        min_idx = df_fmt[col].idxmin()
        df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "--")
        df_fmt.loc[min_idx, col] = r"\textbf{" + df_fmt.loc[min_idx, col] + "}"

    # Build LaTeX
    header_cols = " & ".join(df_fmt.columns) + r" \\"
    tex = (
        r"\begin{table}[ht]\centering" "\n"
        + rf"\caption{{{caption}}}"    "\n"
        + rf"\label{{{label}}}"        "\n"
        + r"\begin{tabular}{l " + " ".join(["r"] * len(df_fmt.columns)) + r"}" "\n"
        + r"\toprule"  "\n"
        + r"Model & " + header_cols + "\n"
        + r"\midrule" "\n"
    )

    for model, row in df_fmt.iterrows():
        tex += model + " & " + " & ".join(row.values) + r" \\" + "\n"

    tex += (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"  "\n"
    )
    return tex


# ───────────────────────── main ───────────────────────────────────────
df = pd.read_csv(CSV_FILE)

# Canonical model names
if "pipeline" in df.columns:
    df = df.rename(columns={"pipeline": "model"})
df["model"] = df["model"].str.replace("Simple", "", regex=False)

# Compute the runtime for every row
df["time_sec"] = df.apply(runtime_for_row, axis=1)

# Add a simple task label
df["task"] = df["dataset"].apply(
    lambda s: "Classification" if "Classification" in s else "Link Prediction"
)

# Build one wide table per task
for task_name, sub in df.groupby("task", sort=False):
    wide = (
        sub.pivot_table(
            index="model",
            columns="dataset",
            values="time_sec",
            aggfunc="mean",
        )
        .round(2)
        .sort_index()
    )

    caption = f"Runtime comparison by dataset – {task_name}"
    label   = f"tab_runtime_{task_name.lower().replace(' ', '_')}"
    tex     = to_latex_matrix(wide, caption, label)

    out_file = OUT_DIR / f"{label}.tex"
    out_file.write_text(tex)

    # Console preview
    print(f"\n{out_file}:")
    print(tex)
