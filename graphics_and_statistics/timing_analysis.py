"""
Timing-only summary for the GNN benchmark table
──────────────────────────────────────────────
Writes:
    results/timing_by_model.csv
    results/timing_by_dataset.csv
"""

import os
from pathlib import Path
import pandas as pd

IN_FILE  = "gnn_summary_statistics.csv"
OUT_DIR  = Path("results")          # relative folder
OUT_DIR.mkdir(exist_ok=True)        # make sure it exists

TIME_COLS = [
    "train_time_mean", "train_time_std",
    "classifier_time_mean", "classifier_time_std",
    "pretrain_time_mean",  "pretrain_time_std",
    "link_pred_time_mean", "link_pred_time_std",
]

TARGET_MODELS = [
    "GPT-GNN",
    "Struct-G Internal Classifier",
    "Struct-G Structural Only Pretrain",
]

# ─── load & tidy ──────────────────────────────────────────────────────
df = pd.read_csv(IN_FILE)

# normalise model names
if "pipeline" in df.columns:
    df = df.rename(columns={"pipeline": "model"})
df["model"] = df["model"].str.replace("Simple", "", regex=False)

# ─── LP → CLS timing-fix for the three target models ──────────────────
# add a helper column with the dataset *prefix* (strip task suffix)
df["base_ds"] = df["dataset"].str.replace(
    r"\s+(Classification|Link Prediction)$", "", regex=True)

for model in TARGET_MODELS:
    for base in df["base_ds"].unique():
        lp_mask  = (df["base_ds"] == base) & df["dataset"].str.endswith("Link Prediction") & (df["model"] == model)
        cls_mask = (df["base_ds"] == base) & df["dataset"].str.endswith("Classification") & (df["model"] == model)

        if not lp_mask.any() or not cls_mask.any():
            continue   # no matching pair → skip

        lp_vals = df.loc[lp_mask, TIME_COLS].iloc[0]      # first (only) LP row
        df.loc[cls_mask, TIME_COLS] = lp_vals.values      # copy into CLS row(s)

# drop helper
df = df.drop(columns="base_ds")

# ─── 1) per-dataset table ─────────────────────────────────────────────
by_ds = (
    df[["dataset", "model", *TIME_COLS]]
      .sort_values(["dataset", "model"])
)
by_ds.to_csv(OUT_DIR / "timing_by_dataset.csv", index=False)

# ─── 2) overall table ────────────────────────────────────────────────
by_model = (
    df.groupby("model")[TIME_COLS]
      .mean()
      .round(3)
      .reset_index()
      .sort_values("train_time_mean")
)
by_model.to_csv(OUT_DIR / "timing_by_model.csv", index=False)

print("✓ wrote:")
print(f"   • {OUT_DIR / 'timing_by_model.csv'}")
print(f"   • {OUT_DIR / 'timing_by_dataset.csv'}")
