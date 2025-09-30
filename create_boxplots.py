"""
File Purpose: Build grouped boxplots of DSC values per organ for Platipy and Limbus.
What this script does: Parses hard-coded DSC values, builds a long-form DataFrame,
and plots grouped boxplots (Platipy vs Limbus) for each organ.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def read_patients(file_path: Path) -> np.ndarray | None:
    if not file_path.exists():
        print(f"Patient list not found at: {file_path}")
        return
    try:
        patient_ids = np.loadtxt(file_path, dtype=str, ndmin=1)
        print(f"Loaded {len(patient_ids)} patient IDs.")
        return patient_ids
    except Exception as e:
        print(f"Failed to load patient list: {e}")
        return

PATIENT_LIST_PATH = Path("/Volumes/diskAshur2/data_msc/LymphomaLog/Patients.txt")
patient_list = read_patients(PATIENT_LIST_PATH)

PLATIPY_METRIC_PATH = Path("/Users/user/Documents/PhD/A.Work/CohortStudy/Minghao/platipy-vs-manual-contours-metrics.csv")
platipy_vs_manual_metric_csv = pd.read_csv(PLATIPY_METRIC_PATH)

LIMBUS_METRIC_PATH = Path("/Users/user/Documents/PhD/A.Work/CohortStudy/data/collated_spreadsheets_LimbusvsGoldstandard.csv")
limbus_vs_manual_csv = pd.read_csv(LIMBUS_METRIC_PATH)



# Section: Raw data and organ list (editable)
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

# filter platipy_vslimbus_metric_csv to only include organs in the organs list
platipy_vs_manual_metric_csv = platipy_vs_manual_metric_csv[
    platipy_vs_manual_metric_csv["sub_struct"].isin(organs)
]
limbus_vs_manual_csv = limbus_vs_manual_csv[
    limbus_vs_manual_csv["sub_structure"].isin(organs)
]

platipy_vs_manual_metric_csv = platipy_vs_manual_metric_csv.rename(columns={"sub_struct": "sub_structure"})


# get dsc values
rows = {}
for organ in organs:
    platipy_vals = platipy_vs_manual_metric_csv.loc[platipy_vs_manual_metric_csv["sub_structure"] == organ, "DSC"].astype(float).tolist()
    limbus_vals = limbus_vs_manual_csv.loc[limbus_vs_manual_csv["sub_structure"] == organ, "DSC"].astype(float).tolist()

    rows[organ] = list(zip(platipy_vals, limbus_vals))

# -----------------------------
# Section: Build long-form DataFrame
# -----------------------------
patient_labels = []
seen = {}
for p in patient_list:
    p_clean = re.sub(r'[^A-Za-z0-9]+', '_', p).strip('_')
    seen[p_clean] = seen.get(p_clean, 0) + 1
    if seen[p_clean] > 1:
        p_clean = f"{p_clean}_{seen[p_clean]}"
    patient_labels.append(p_clean)

records = []
for organ in organs:
    cells = rows[organ]
    if len(patient_labels) != len(cells):
        print(f"Warning: mismatch for {organ}: {len(patient_labels)} patients vs {len(cells)} DSC entries")
    for p_label, (platipy_dsc, limbus_dsc) in zip(patient_labels, cells):
        records.append({"Organ": organ, "Patient": p_label, "Model": "Platipy", "DSC": platipy_dsc})
        records.append({"Organ": organ, "Patient": p_label, "Model": "Limbus",  "DSC": limbus_dsc})

df = pd.DataFrame.from_records(records)

# -----------------------------
# Section: Plot grouped boxplots
# -----------------------------
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="Organ", y="DSC", hue="Model",
    showfliers=False,   # ✅ no small circles
    width=0.6
)

sns.stripplot(
    data=df,
    x="Organ", y="DSC", hue="Model",
    dodge=True, alpha=0.5, size=3, color="k", legend=False  # ✅ only one legend
)

plt.title("DSC")
plt.xticks(rotation=35, ha="right")
plt.ylim(0, 1)
plt.ylabel("DSC")

# keep only one legend (from boxplot)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title="Model", frameon=False)

plt.tight_layout()
out = Path("out_figs"); out.mkdir(exist_ok=True)
plt.savefig(out / "DSC_boxplots_by_organ_clean.png", dpi=200)
plt.show()

print("Done. Figure saved in:", out.resolve())
