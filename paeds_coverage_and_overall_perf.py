"""
File Purpose: Summarize paediatric cohort coverage and overall performance across segmentation models.
What this script does: Loads detailed metrics, computes coverage/overlap sizes, summarizes performance stats, writes CSVs, and prints a concise report.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

# Section: Configuration - user-defined file paths
PLATIPY_PATH = Path("/Users/shenminghao/Desktop/Master Project/result/comparison_results_Paediatrics_PlatiPy_Manual/combined_detailed_metrics_PlatiPy.csv")
LIMBUS_PATH  = Path("/Users/shenminghao/Desktop/Master Project/result/comparison_results_Paediatrics_Limbus_Manual/combined_detailed_metrics_Limbus.csv")
PYCERR_PATH  = Path("/Users/shenminghao/Desktop/Master Project/result/comparison_results_Paediatrics_PyCeRR_Manual/combined_detailed_metrics_PyCeRR.csv")

# Section: Output directory
OUT_DIR = Path("/Users/shenminghao/Desktop/Master Project/result/paeds_cov_overall")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Section: Helper constants
NUM_COLS = ["DSC", "Mean Absolute Distance (mm)", "HD_mm", "volume_ratio"]

# Section: Helper functions - IO and data cleaning
def read_and_clean(p: Path) -> pd.DataFrame:
    # Read CSV, strip column names, coerce key numeric columns
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# Section: Helper functions - coverage metrics
def basic_coverage(df: pd.DataFrame, model_name: str) -> dict:
    # Compute basic coverage stats per model
    return {
        "Model": model_name,
        "Unique patients": int(df["patient_id"].nunique() if "patient_id" in df.columns else np.nan),
        "Total pairs (patient×structure)": int(len(df)),
        "Unique substructures": int(df["sub_structure"].nunique() if "sub_structure" in df.columns else np.nan),
    }

# Section: Helper functions - performance summary
def overall_performance(df: pd.DataFrame, model_name: str) -> dict:
    # Compute overall performance summary for a model
    dsc = df["DSC"]
    perf = {
        "Model": model_name,
        "DSC mean": float(dsc.mean()),
        "DSC SD": float(dsc.std(ddof=1)),
        "Median DSC": float(dsc.median()),
        "DSC < 0.5 (%)": float((dsc < 0.5).mean() * 100),
        "DSC == 0 (count)": int((dsc == 0).sum()),
        "HD_mm median": float(df["HD_mm"].median()) if "HD_mm" in df.columns else np.nan,
        "MAD median (mm)": float(df["Mean Absolute Distance (mm)"].median()) if "Mean Absolute Distance (mm)" in df.columns else np.nan,
        "Volume ratio median": float(df["volume_ratio"].median()) if "volume_ratio" in df.columns else np.nan,
    }
    return perf

# Section: Helper functions - overlap calculations
def overlap_sizes(pl: pd.DataFrame, lm: pd.DataFrame, py: pd.DataFrame) -> pd.DataFrame:
    # Build two-model and three-model overlaps on patient_id and sub_structure
    pl_lm = pl.merge(lm, on=["patient_id", "sub_structure"], how="inner", suffixes=("_platipy","_limbus"))
    all3 = pl_lm.merge(py, on=["patient_id", "sub_structure"], how="inner")

    rows = []
    rows.append({
        "Overlap set": "PlatiPy ∩ Limbus",
        "Patients": int(pl_lm["patient_id"].nunique() if "patient_id" in pl_lm.columns else np.nan),
        "Substructures": int(pl_lm["sub_structure"].nunique() if "sub_structure" in pl_lm.columns else np.nan),
        "Total pairs": int(len(pl_lm)),
    })
    rows.append({
        "Overlap set": "PlatiPy ∩ Limbus ∩ PyCeRR",
        "Patients": int(all3["patient_id"].nunique() if "patient_id" in all3.columns else np.nan),
        "Substructures": int(all3["sub_structure"].nunique() if "sub_structure" in all3.columns else np.nan),
        "Total pairs": int(len(all3)),
    })
    return pd.DataFrame(rows)

# Section: Main workflow
def main() -> None:
    # Step 1: Read input CSVs
    pl = read_and_clean(PLATIPY_PATH)
    lm = read_and_clean(LIMBUS_PATH)
    py = read_and_clean(PYCERR_PATH)

    # Step 2: Coverage per model
    cov_rows = [
        basic_coverage(pl, "PlatiPy"),
        basic_coverage(lm, "Limbus"),
        basic_coverage(py, "PyCeRR"),
    ]
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(OUT_DIR / "coverage_per_model.csv", index=False)

    # Step 3: Overlap sizes
    ov_df = overlap_sizes(pl, lm, py)
    ov_df.to_csv(OUT_DIR / "coverage_overlaps.csv", index=False)

    # Step 4: Overall performance per model
    perf_rows = [
        overall_performance(pl, "PlatiPy"),
        overall_performance(lm, "Limbus"),
        overall_performance(py, "PyCeRR"),
    ]
    perf_df = pd.DataFrame(perf_rows)

    # Round selected columns for presentation only
    for c in ["DSC mean","DSC SD","Median DSC","DSC < 0.5 (%)","HD_mm median","MAD median (mm)","Volume ratio median"]:
        if c in perf_df.columns:
            perf_df[c] = perf_df[c].round(3)
    perf_df.to_csv(OUT_DIR / "overall_performance_summary.csv", index=False)

    # Step 5: Console report
    print("\n=== Coverage per model ===")
    print(cov_df.to_string(index=False))
    print("\n=== Overlap sizes ===")
    print(ov_df.to_string(index=False))
    print("\n=== Overall performance (gold standard vs auto-contours) ===")
    print(perf_df.to_string(index=False))

# Section: Entry point
if __name__ == "__main__":
    main()
