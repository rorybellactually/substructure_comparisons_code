"""
 File Purpose: Generate a heatmap summarizing pairwise agreement metrics between two segmentation models per sub-structure.
 What this script does: Reads a CSV, computes Dice (DSC) and volume similarity per sub-structure, and saves a heatmap image with annotated values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Section: Inputs
# Change these paths to match your environment if needed.
csv_path = Path("/Users/shenminghao/Desktop/Master Project/result/comparison_results_Lymphoma_PlatiPy_Limbus/combined_detailed_metrics_Lymphoma_PlatiPy_Limbus.csv")
out_png = Path("/Users/shenminghao/Desktop/Master Project/result/Lymphoma_pairwise_agreement_heatmap_rvol_dsc.png")

# Optional: desired display order (missing items will be ignored).
desired_order = [
    "Heart",
    "Left Ventricle",
    "Left Atrium",
    "Right Atrium",
    "Right Ventricle",
    "Atrioventricular Node",
    "Pulmonary Artery",
    "Left Anterior Descending",
    "Superior Vena Cava",
    "Aorta",
]

# Section: Load data
df = pd.read_csv(csv_path)

# Section: Basic column checks
required = ["sub_structure", "DSC", "Volume Platipy (mL)", "Volume Limbus (mL)"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Section: Compute volume similarity
# VS = 1 - |Vp - Vl| / (Vp + Vl); if denominator is 0 then set to NaN.
vp = df["Volume Platipy (mL)"].astype(float)
vl = df["Volume Limbus (mL)"].astype(float)
den = vp + vl
vs = 1.0 - (vp - vl).abs() / den.replace(0, np.nan)
df["Volume Similarity"] = vs

# Section: Aggregate metrics per sub-structure
metrics = (
    df.groupby("sub_structure")
      .agg(DSC=("DSC", "mean"),
           Volume_Similarity=("Volume Similarity", "mean"),
           n=("DSC", "size"))
)

# Section: Apply optional sub-structure order
order = [s for s in desired_order if s in metrics.index]
if order:
    metrics = metrics.loc[order]
else:
    metrics = metrics.sort_index()

# Section: Build matrix for heatmap
heat_mat = metrics[["DSC", "Volume_Similarity"]].to_numpy()

# Both metrics are in [0, 1]; use a shared color scale for comparison.
vmin, vmax = 0.0, 1.0

# Section: Plot heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heat_mat, aspect="auto", vmin=vmin, vmax=vmax)

# Axis ticks and labels
ax.set_xticks([0, 1])
ax.set_xticklabels(["DSC", "Volume Similarity"])
ax.set_yticks(np.arange(metrics.shape[0]))
ax.set_yticklabels(metrics.index)

# Annotate each cell with the value; show "NA" for NaN
for i in range(heat_mat.shape[0]):
    for j in range(heat_mat.shape[1]):
        val = heat_mat[i, j]
        txt = "NA" if np.isnan(val) else f"{val:.2f}"
        ax.text(j, i, txt, ha="center", va="center")

# Title and colorbar
title = ""
ax.set_title(title)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Value")

plt.tight_layout()
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.show()

print("Saved to:", out_png.resolve())
print("\nPer-structure sample counts (n):")
print(metrics["n"])
