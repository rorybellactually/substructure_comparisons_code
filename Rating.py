"""
 File Purpose: Build stacked bar charts of rating distributions (1/2/3) per organ for Platipy and Limbus.
 What this script does: Parses hard-coded ratings, computes counts and percentages by organ and model, plots stacked bars, and saves figures and CSVs.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from pathlib import Path

# Section: Raw data and organ list
patients_raw = [
    "CD00793","DC00289","ES00379","FW00138","FW00974",
    "FZ00679","GI01669","GU03241(adult)","HW01755","HW01755"
]

organs = [
    "Left Ventricle",
    "Right Ventricle",
    "Left Atrium",
    "Right Atrium",
    # "Aorta",
    # "Pulmonary Artery",
    # "Superior Vena Cava",
    # "Left Anterior Descending",
    # "Atrioventricular Node",
    "Heart"
]

rows = {
    "Left Ventricle":          ["2/2","2/3","2/3","2/2","3/2","3/3","3/2","3/2","2/2","2/2"],
    "Right Ventricle":         ["2/3","2/2","3/3","2/2","2/2","2/3","2/2","3/2","2/2","2/2"],
    "Left Atrium":             ["2/3","2/3","3/3","3/3","3/3","3/3","3/2","2/2","2/2","2/2"],
    "Right Atrium":            ["2/3","2/3","2/3","2/3","2/2","2/0","2/2","3/2","2/2","2/2"],
    # "Aorta":                   ["3/0","3/0","3/0","3/0","2/0","3/0","3/0","2/0","3/0","2/0"],
    # "Pulmonary Artery":        ["3/2","3/3","3/2","3/2","3/3","3/3","3/2","3/2","3/3","3/2"],
    # "Superior Vena Cava":      ["3/2","3/2","3/3","3/2","3/2","3/2","3/2","3/2","3/3","3/3"],
    # "Left Anterior Descending":["3/3","2/3","2/3","2/2","2/2","2/2","3/3","3/1","2/3","1/3"],
    # "Atrioventricular Node":   ["2/0","2/0","1/0","3/0","2/0","2/0","3/0","2/0","1/0","2/0"],
    "Heart":                   ["2/2","2/2","2/2","2/2","2/2","2/3","2/2","2/2","2/2","2/2"],
}

# Section: Clean patient labels for uniqueness and filesystem safety
patient_labels = []
seen = {}
for p in patients_raw:
    p_clean = re.sub(r'[^A-Za-z0-9]+', '_', p).strip('_')
    seen[p_clean] = seen.get(p_clean, 0) + 1
    if seen[p_clean] > 1:
        p_clean = f"{p_clean}_{seen[p_clean]}"
    patient_labels.append(p_clean)

# Section: Build long-form DataFrame (one row per Organ, Patient, Model)
records = []
for organ in organs:
    cells = rows[organ]
    for p_label, cell in zip(patient_labels, cells):
        cell = cell.split()[0]
        m = re.match(r'^\s*([0-3])\s*/\s*([0-3])', cell)
        assert m, f"Bad cell: {organ} - {p_label} - {cell}"
        platipy_score = int(m.group(1))
        limbus_score  = int(m.group(2))
        records.append({"Organ": organ, "Patient": p_label, "Model": "Platipy", "Score": platipy_score})
        records.append({"Organ": organ, "Patient": p_label, "Model": "Limbus",  "Score": limbus_score})

df = pd.DataFrame.from_records(records)

# Section: Compute counts and percentages per organ and model (exclude score=0 from denominator)
counts = (
    df.groupby(["Organ","Model","Score"])
      .size()
      .unstack("Score", fill_value=0)
      .reindex(columns=[0,1,2,3], fill_value=0)
)
den = counts[[1,2,3]].sum(axis=1).replace(0, np.nan)
pct = counts[[1,2,3]].div(den, axis=0).fillna(0.0) * 100.0

# Section: Plotting function for a single model
def plot_model(pct_table, model_name, outfile):
    sub = pct_table.xs(model_name, level="Model").reindex(organs)
    x = np.arange(len(sub.index))
    bottom = np.zeros_like(x, dtype=float)

    plt.figure(figsize=(10, 5.5))
    colors  = {1:"#2ca02c", 2:"#ffbf00", 3:"#d62728"}
    hatches = {1:None,       2:None,      3:".."}

    labels = {
        1: "1: no clinical significance change",
        2: "2: minor changes",
        3: "3: major changes"
    }

    for score in [1, 2, 3]:
        vals = sub[score].to_numpy()
        plt.bar(
            x, vals, bottom=bottom,
            label=labels[score],
            color=colors[score],
            edgecolor="black",
            linewidth=0.5,
            hatch=hatches[score]
        )
        bottom += vals

    plt.xticks(x, sub.index, rotation=35, ha='right')
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)

    legend_handles = [
        mpatches.Patch(facecolor=colors[1], edgecolor="black", label=labels[1]),
        mpatches.Patch(facecolor=colors[2], edgecolor="black", label=labels[2]),
        mpatches.Patch(facecolor=colors[3], edgecolor="black", hatch=hatches[3], label=labels[3])
    ]
    plt.legend(handles=legend_handles, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

# Section: Output figures and CSVs
out = Path("out_figs"); out.mkdir(exist_ok=True)
plot_model(pct, "Platipy", out / "Segmentation ratings per organ – Platipy.png")
plot_model(pct, "Limbus",  out / "Segmentation ratings per organ – Limbus.png")

counts.reset_index().to_csv(out / "ratings_counts_by_organ_model.csv", index=False)
pct.reset_index().to_csv(out / "ratings_percent_by_organ_model.csv", index=False)

print("Done. Figures & CSVs are in:", out.resolve())
