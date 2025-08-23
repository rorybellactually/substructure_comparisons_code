"""
Summarise segmentation outputs for three datasets (REQUITE, Lymphoma, Paediatrics)
and three models (Limbus, Platipy, PyCeRR_subseg), producing catalogs and aggregate CSV reports
"""

## Section: Imports
from pathlib import Path
import argparse
import csv
from collections import defaultdict, Counter
import sys
import pandas as pd

## Section: Constants
DATASETS = ["REQUITE", "Lymphoma", "Paediatrics"]
MODELS = ["Limbus", "Platipy", "PyCeRR_subseg"]


## Section: I/O helpers
def read_patient_list(root: Path, dataset: str) -> list[str]:
    log_dir = root / f"{dataset}Log"
    txt = log_dir / "Patients.txt"
    if not txt.exists():
        print(f"[WARN] Missing Patients.txt for {dataset}: {txt}", file=sys.stderr)
        return []
    patients = []
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patients.append(line)
    return patients


def scan_model_files(patient_dir: Path, model: str) -> list[Path]:
    model_dir = patient_dir / model
    if not model_dir.exists():
        return []
    paths = list(model_dir.rglob("*.nii")) + list(model_dir.rglob("*.nii.gz"))
    paths = [p for p in paths if p.is_file() and not p.name.startswith(".")]
    return paths


def relpath_safe(p: Path, start: Path) -> str:
    try:
        return str(p.relative_to(start))
    except Exception:
        return str(p)


## Section: Main entry point and argument parsing
def main():
    parser = argparse.ArgumentParser(description="Summarise segmentation outputs per dataset and model.")
    parser.add_argument("--root", type=str, default="/Volumes/diskAshur2/data_msc/",
                        help="Root folder containing REQUITE, Lymphoma, Paediatrics and their *Log folders.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output folder for CSVs. Default: <root>/segmentation_summary")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve() if args.out else (root / "segmentation_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    per_patient_counts = []

    for dataset in DATASETS:
        patients = read_patient_list(root, dataset)
        dataset_dir = root / dataset

        catalog_rows = []
        model_patient_has_any = {m: set() for m in MODELS}
        model_total_files = Counter({m: 0 for m in MODELS})
        model_substructures = {m: set() for m in MODELS}
        model_patient_struct_counts = defaultdict(lambda: Counter())

        for pid in patients:
            patient_dir = dataset_dir / pid
            if not patient_dir.exists():
                continue

            for model in MODELS:
                files = scan_model_files(patient_dir, model)
                if files:
                    model_patient_has_any[model].add(pid)
                model_total_files[model] += len(files)

                for f in files:
                    name = f.name
                    if name.endswith(".nii.gz"):
                        sub = name[:-7]
                    elif name.endswith(".nii"):
                        sub = name[:-4]
                    else:
                        sub = name
                    model_substructures[model].add(sub)
                    model_patient_struct_counts[model][pid] += 1

                    catalog_rows.append({
                        "dataset": dataset,
                        "patient_id": pid,
                        "model": model,
                        "substructure_name": sub,
                        "file_name": name,
                        "file_path_rel": relpath_safe(f, root),
                        "file_path_abs": str(f.resolve())
                    })

        catalog_path = out_dir / f"{dataset}_structures_catalog.csv"
        with catalog_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=[
                "dataset", "patient_id", "model", "substructure_name",
                "file_name", "file_path_rel", "file_path_abs"
            ])
            writer.writeheader()
            for row in catalog_rows:
                writer.writerow(row)

        total_from_list = len(patients)
        for model in MODELS:
            patients_with_any = len(model_patient_has_any[model])
            total_files = model_total_files[model]
            unique_subs = len(model_substructures[model])

            for pid, cnt in model_patient_struct_counts[model].items():
                per_patient_counts.append({
                    "dataset": dataset,
                    "model": model,
                    "patient_id": pid,
                    "n_substructures": cnt
                })

            overall_rows.append({
                "dataset": dataset,
                "model": model,
                "patients_total_from_list": total_from_list,
                "patients_with_any_output": patients_with_any,
                "patients_without_output": max(total_from_list - patients_with_any, 0),
                "total_files": total_files,
                "unique_substructures_across_patients": unique_subs
            })

        print(f"\n=== {dataset} ===")
        print(f"Patients in list: {total_from_list}")
        for model in MODELS:
            print(
                f"- {model:12s} | patients_with_any_output={len(model_patient_has_any[model]):3d} | "
                f"total_files={model_total_files[model]:4d} | "
                f"unique_substructures={len(model_substructures[model]):3d}"
            )
        print(f"Catalog CSV written: {catalog_path}")

    overall_path = out_dir / "segmentation_summary_by_dataset_model.csv"
    if overall_rows:
        with overall_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(overall_rows[0].keys()))
            writer.writeheader()
            for row in overall_rows:
                writer.writerow(row)

    dist_path = out_dir / "per_patient_substructure_counts.csv"
    with dist_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["dataset","model","patient_id","n_substructures"])
        writer.writeheader()
        for row in per_patient_counts:
            writer.writerow(row)

    matrix_path = out_dir / "segmentation_summary_matrix.csv"
    try:
        df = pd.DataFrame(overall_rows)
        required_cols = {"dataset","model","patients_total_from_list","patients_with_any_output"}
        if required_cols.issubset(df.columns):
            df["success_over_total"] = (
                df["patients_with_any_output"].astype(int).astype(str)
                + " / "
                + df["patients_total_from_list"].astype(int).astype(str)
            )
            pivot = (
                df.pivot(index="dataset", columns="model", values="success_over_total")
                  .reindex(index=DATASETS)
                  .reindex(columns=MODELS)
            )
            pivot.to_csv(matrix_path)
            print("\n=== Matrix (success/total) ===")
            print(pivot.fillna(""))
            print(f"Matrix CSV written: {matrix_path}")
        else:
            print("[WARN] Cannot build matrix, required columns missing.")
    except Exception as e:
        print(f"[WARN] Failed to create matrix CSV: {e}", file=sys.stderr)

    print("\n=== Overall CSV outputs ===")
    print(f"- Overall summary: {overall_path}")
    print(f"- Matrix summary:  {matrix_path}")
    print(f"- Per-patient counts: {dist_path}")
    print("Done.")


## Section: Script execution guard
if __name__ == "__main__":
    main()
