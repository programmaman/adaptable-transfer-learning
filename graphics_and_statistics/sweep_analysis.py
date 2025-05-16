#!/usr/bin/env python3
"""
struct_sweep_analysis.py – Analysis of Structural-G sweep (325 configs)

Outputs (written to OUT_DIR):
  lm_node_cls.txt                # OLS summary for int_cls_f1
  lm_link_auc.txt                # OLS summary for int_lp_auc
  table_main_effects_cls.tex     # main-effects coef table (LaTeX)
  table_top10_delta_cls.tex      # 10 best Δ-F1 configs  (LaTeX)
  table_bottom10_delta_cls.tex   # 10 worst Δ-F1 configs (LaTeX)
  heatmap_corr.png               # Spearman correlation heat-map
  line_homophily_clustering.png  # Homophily × clustering faceted lines
  manifest.json                  # handy file list
"""
import os, json, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# -------------------------------------------------------------------------
CSV_PATH = "../results/structural_sweep_results.csv"     # <-- change if needed
OUT_DIR  = "out"                              # <-- change if needed
os.makedirs(OUT_DIR, exist_ok=True)
# -------------------------------------------------------------------------

# 1) LOAD & CLEAN ----------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df.replace(-1, np.nan, inplace=True)          # sentinel → NaN
df["diameter"]  = df["diameter"].astype("category")
df["num_nodes"] = df["num_nodes"].astype("category")

# internal – external deltas (optional but insightful)
df["delta_cls_f1"] = df["int_cls_f1"] - df["ext_cls_f1"]
df["delta_lp_auc"] = df["int_lp_auc"] - df["ext_lp_auc"]

# 2) REGRESSION MODELS -----------------------------------------------------
f_cls = "int_cls_f1 ~ homophily + clustering + C(diameter) + assortativity + edge_factor + C(num_nodes)"
f_lp  = "int_lp_auc ~ homophily + clustering + C(diameter) + assortativity + edge_factor + C(num_nodes)"

mdl_cls = smf.ols(f_cls, data=df).fit()
mdl_lp  = smf.ols(f_lp , data=df).fit()

open(f"{OUT_DIR}/lm_node_cls.txt","w").write(mdl_cls.summary().as_text())
open(f"{OUT_DIR}/lm_link_auc.txt","w").write(mdl_lp .summary().as_text())

# LaTeX table – main effects (classification)
(mdl_cls.summary2().tables[1]
 .reset_index()
 .rename(columns={"index":"factor"})
 .to_latex(f"{OUT_DIR}/table_main_effects_cls.tex",
           index=False, float_format="%.3f"))

# 3) VISUALS ---------------------------------------------------------------
corr_cols = ["homophily","clustering","assortativity","edge_factor",
             "delta_cls_f1","delta_lp_auc"]
sns.heatmap(df[corr_cols].corr("spearman"), annot=True,
            cmap="vlag", fmt=".2f", center=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/heatmap_corr.png", dpi=300)
plt.close()

sns.relplot(
    data=df, x="homophily", y="int_cls_f1",
    hue="clustering", style="diameter",
    col="num_nodes", kind="line", markers=True,
    facet_kws={"sharey":False}
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/line_homophily_clustering.png", dpi=300)
plt.close()

# 4) BEST / WORST CONFIGS --------------------------------------------------
df.nlargest(10,"delta_cls_f1").to_latex(
    f"{OUT_DIR}/table_top10_delta_cls.tex", index=False, float_format="%.3f")
df.nsmallest(10,"delta_cls_f1").to_latex(
    f"{OUT_DIR}/table_bottom10_delta_cls.tex", index=False, float_format="%.3f")

# 5) MANIFEST --------------------------------------------------------------
json.dump({
    "regression": ["lm_node_cls.txt","lm_link_auc.txt"],
    "latex_tables": ["table_main_effects_cls.tex",
                     "table_top10_delta_cls.tex",
                     "table_bottom10_delta_cls.tex"],
    "figures": ["heatmap_corr.png","line_homophily_clustering.png"]
}, open(f"{OUT_DIR}/manifest.json","w"), indent=2)

print(f"✅ Analysis complete – see '{OUT_DIR}/' for outputs.")
