"""
File Purpose: Generate Paediatrics cohort figures comparing PlatiPy vs Limbus DSC per sub-structure
"""

# Section: Imports
from __future__ import annotations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Section: User paths
PL_PATH = Path("/Users/shenminghao/Desktop/Master Project/result/comparison_results_Paediatrics_PlatiPy_Manual/combined_detailed_metrics_PlatiPy.csv")
LM_PATH = Path("/Users/shenminghao/Desktop/Master Project/result/comparison_results_Paediatrics_Limbus_Manual/combined_detailed_metrics_Limbus.csv")

# Section: Output directory
OUT_DIR = Path("/Users/shenminghao/Desktop/Master Project/result/paeds_substructure_plots_seaborn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Section: Seaborn aesthetics
sns.set_theme(style="whitegrid", context="talk")
# sns.set(font="DejaVu Sans")

# Section: Helpers
def sanitize(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name).strip())
    return s.strip("_")


def dsc_limits(x, y, pad=0.02):
    lo = max(0.0, min(np.nanmin(x), np.nanmin(y)) - pad)
    hi = min(1.0, max(np.nanmax(x), np.nanmax(y)) + pad)
    lo = min(lo, 0.0)
    hi = max(hi, 1.0)
    return lo, hi


# Section: Load and pair data
pl = pd.read_csv(PL_PATH)
lm = pd.read_csv(LM_PATH)
pl.columns = [c.strip() for c in pl.columns]
lm.columns = [c.strip() for c in lm.columns]
for df in (pl, lm):
    if "DSC" in df.columns:
        df["DSC"] = pd.to_numeric(df["DSC"], errors="coerce")

paired = pl.merge(
    lm,
    on=["patient_id", "sub_structure"],
    suffixes=("_platipy", "_limbus"),
    how="inner",
)
structures = sorted(paired["sub_structure"].dropna().unique().tolist())


# Section: Figure 1 — Identity scatter per sub-structure
def plot_identity_per_structure(structure: str) -> Path:
    df = paired.loc[paired["sub_structure"] == structure].copy()
    x = df["DSC_platipy"].values
    y = df["DSC_limbus"].values
    n = len(df)
    r = np.nan
    if n >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        r = float(np.corrcoef(x, y)[0, 1])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.scatterplot(
        data=df, x="DSC_platipy", y="DSC_limbus",
        ax=ax, s=70, alpha=0.75, edgecolor="white", linewidth=0.6
    )

    lo, hi = dsc_limits(x, y, pad=0.02)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("DSC — PlatiPy")
    ax.set_ylabel("DSC — Limbus")

    title = f"Identity: {structure}  (N={n})"
    if n >= 2 and np.isfinite(r):
        title += f"  |  Pearson r={r:.2f}"
    ax.set_title(title)

    sns.despine(ax=ax)
    fig.tight_layout()
    out = OUT_DIR / f"identity_{sanitize(structure)}.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


# Section: Figure 2 — Paired line plot per sub-structure
def plot_paired_lines_per_structure(structure: str) -> Path:
    df = paired.loc[paired["sub_structure"] == structure, ["patient_id", "DSC_platipy", "DSC_limbus"]].copy()
    long = (
        df.rename(columns={"DSC_platipy": "PlatiPy", "DSC_limbus": "Limbus"})
          .melt(id_vars="patient_id", var_name="model", value_name="DSC")
    )
    order = ["PlatiPy", "Limbus"]
    long["model"] = pd.Categorical(long["model"], categories=order, ordered=True)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.lineplot(
        data=long, x="model", y="DSC",
        units="patient_id", estimator=None,
        lw=1.2, alpha=0.35, marker="o", markers=True, dashes=False, ax=ax
    )
    med = long.groupby("model")["DSC"].median().reindex(order)
    ax.scatter(order, med.values, s=90, zorder=5)

    ax.set_xlabel("")
    ax.set_ylabel("DSC")
    ax.set_title(f"Paired lines (per patient): {structure}  (N={df.shape[0]})")
    ax.set_ylim(0, 1)
    sns.despine(ax=ax)
    fig.tight_layout()
    out = OUT_DIR / f"paired_lines_{sanitize(structure)}.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


# Section: Batch generation
for s in structures:
    plot_identity_per_structure(s)
    plot_paired_lines_per_structure(s)


# Section: Figure 3 — Grouped boxplot over all sub-structures
pl_long = pl.loc[:, ["sub_structure", "DSC"]].copy()
pl_long["model"] = "PlatiPy"
lm_long = lm.loc[:, ["sub_structure", "DSC"]].copy()
lm_long["model"] = "Limbus"
both = pd.concat([pl_long, lm_long], ignore_index=True)
both = both[both["sub_structure"].isin(structures)].copy()
both["model"] = pd.Categorical(both["model"], categories=["PlatiPy", "Limbus"], ordered=True)

W = max(10, len(structures) * 0.9)
fig, ax = plt.subplots(figsize=(W, 6.8))
sns.boxplot(
    data=both, x="sub_structure", y="DSC", hue="model",
    dodge=True, showfliers=False, width=0.7, ax=ax
)
sns.stripplot(
    data=both, x="sub_structure", y="DSC", hue="model",
    dodge=True, jitter=0.15, alpha=0.25, linewidth=0, ax=ax
)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), title="Model", loc="best")

ax.set_xlabel("")
ax.set_ylabel("DSC")
ax.set_title("DSC by sub-structure — grouped boxplots (PlatiPy vs Limbus)")
ax.set_ylim(0, 1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
sns.despine(ax=ax)
fig.tight_layout()
grouped_box_path = OUT_DIR / "grouped_boxplot_all_structures.png"
fig.savefig(grouped_box_path, dpi=240)
plt.close(fig)

print(f"[OK] Saved figures to: {OUT_DIR}")
