"""
File Purpose: Batch-rename specific sub-folders within each patient directory for a dataset.
What this script does: Reads a patient list, applies a name-mapping to sub-folders, and optionally performs the renames.
"""

# Section: Standard library imports
import os
import sys

# Section: Core renaming routine
def rename_subfolders(base_dir: str,
                      patient_list_file: str,
                      mapping: dict[str, str],
                      dry_run: bool = True) -> None:
    # Load patient identifiers from the provided list file
    try:
        with open(patient_list_file, "r", encoding="utf-8") as f:
            patients = [line.strip() for line in f if line.strip()]
    except FileNotFoundError as exc:
        sys.exit(f"Patient list file not found: {exc.filename}")

    # Iterate over each patient directory and apply renaming rules
    for patient in patients:
        patient_dir = os.path.join(base_dir, patient)
        if not os.path.isdir(patient_dir):
            print(f"Skipping {patient_dir} (directory not found).")
            continue

        for name in os.listdir(patient_dir):
            src_path = os.path.join(patient_dir, name)
            if not os.path.isdir(src_path):
                continue  # Skip files

            if name not in mapping:
                continue  # No rule for this folder

            new_name = mapping[name]
            dst_path = os.path.join(patient_dir, new_name)

            if os.path.exists(dst_path):
                print(f"Cannot rename {src_path}: target {dst_path} already exists.")
                continue

            action = "Would rename" if dry_run else "Renaming"
            print(f"{action} {src_path}  -->  {dst_path}")

            if not dry_run:
                os.rename(src_path, dst_path)

# Section: Script entry point and configuration
if __name__ == "__main__":
    # Base directory that contains one sub-directory per patient
    BASE_DIR = "/home/donal/data/server2/Msc_Minghao/REQUITE"

    # Path to a text file that lists patient identifiers, one per line
    PATIENT_LIST_FILE = "/home/donal/data/server2/Msc_Minghao/REQUITELog/Patients.txt"

    # Mapping rules: keys are current sub-folder names; values are desired new names
    MAPPING = {
        "RTSTRUCT_masks": "Limbus",
        "substructures": "Platipy",
        # "T1": "T1_pre",
    }

    # Execute the renaming (set dry_run=True to preview without making changes)
    rename_subfolders(BASE_DIR, PATIENT_LIST_FILE, MAPPING, dry_run=False)
