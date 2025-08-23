"""
File Purpose: Compare and report agreement between two cardiac segmentation model outputs across patients.
What this script does: Loads masks, computes metrics, generates per-patient plots and CSV reports, and saves combined summaries.
"""

# Section: Imports
import os
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from platipy.imaging.visualisation.visualiser import ImageVisualiser
from platipy.imaging.label.comparison import (
    compute_metric_dsc,
    compute_metric_masd,
    compute_metric_hd,
    compute_volume,
)
from platipy.imaging.utils.crop import label_to_roi
from tqdm import tqdm

# Section: Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

ROOT_DIR = Path("/Volumes/diskAshur2/data_msc/Lymphoma")
PATIENT_LIST_PATH = Path("/Volumes/diskAshur2/data_msc/LymphomaLog/Patients.txt")
# MODELS = ["Platipy", "Limbus", "PyCeRR_subseg"]
MODELS = ["Platipy", "Limbus"]
BASE_IMAGE_FILENAME = "ct.nii.gz"

# Section: Structure definitions and colors
CANONICAL_STRUCTURES: Dict[str, List[str]] = {
    "Left Ventricle":  ["Ventricle_L", "Left_Ventricle", "DL_LV", "LV", "Ventricle_L_LG"],
    "Right Ventricle": ["Ventricle_R", "Right_Ventricle", "DL_RV", "RV", "Ventricle_R_LG"],
    "Left Atrium":     ["Atrium_L", "Left_Atrium", "DL_LA", "LA", "Atrium_L_LG"],
    "Right Atrium":    ["Atrium_R", "Right_Atrium", "DL_RA", "RA", "Atrium_R_LG"],
    "Aorta":           ["A_Aorta", "Aorta", "DL_AORTA", "Aorta_LG"],
    "Pulmonary Artery":["A_Pulmonary", "Pulmonary_Artery", "DL_PA", "Pulmonary_Trunk_LG"],
    "Superior Vena Cava": ["V_Venacava_S", "Superior_Vena_Cava", "DL_SVC", "V_CavaSuperior_LG"],
    "Left Anterior Descending": ["A_LAD", "LAD", "Left_Anterior_Descending", "LAD_Artery", "LAD_LG"],
    "Atrioventricular Node":    ["Node_Atrioventricular", "CN_Atrioventricular", "AV_Node", "AV_node_LG"],
    "Heart": ["Heart", "ct_heartStructure", "Heart_LG"],
}

STRUCTURE_COLORS = {
    "Left Ventricle": "#F14A4A",
    "Right Ventricle": "#4ECDC4",
    "Left Atrium": "#45B7D1",
    "Right Atrium": "#96CEB4",
    "Aorta": "#FFEAA7",
    "Pulmonary Artery": "#DDA0DD",
    "Superior Vena Cava": "#98FB98",
    "Left Anterior Descending": "#F0A3FF",
    "Atrioventricular Node": "#FFB347",
    "Heart": "#D3D3D3",
}

# Section: File search utilities
def find_file(
    search_dir: Path,
    file_synonyms: List[str],
    case_insensitive: bool = True
) -> Optional[Path]:
    """
    Find a single .nii or .nii.gz file in a directory by matching any of the provided synonyms.
    Synonyms and filenames are normalized by stripping .nii/.nii.gz and optional lowercasing.
    """
    if not search_dir.is_dir():
        return None

    def norm_stem(name: str) -> str:
        s = name
        if case_insensitive:
            s = s.lower()
        if s.endswith(".nii.gz"):
            s = s[:-7]
        elif s.endswith(".nii"):
            s = s[:-4]
        return s

    synonyms_to_check = [norm_stem(s) for s in file_synonyms]
    found_paths: List[Path] = []

    for f in search_dir.iterdir():
        name = f.name
        name_cmp = name.lower() if case_insensitive else name

        if name_cmp.endswith((".nii", ".nii.gz")):
            stem = norm_stem(name)
            if stem in synonyms_to_check:
                found_paths.append(f)

    if not found_paths:
        return None
    if len(found_paths) > 1:
        for f in found_paths:
            if "LG" in f.name:
                logging.warning(
                    f"Found multiple candidates for synonyms {file_synonyms} in {search_dir}: {found_paths}. LG match."
                )
                return f
        raise ValueError(
            f"Found multiple candidates in {search_dir} for synonyms {file_synonyms}: {found_paths}"
        )
    return found_paths[0]


def find_base_image(patient_id: str, root_dir: Path) -> Optional[Path]:
    """Find the base anatomical image for a patient."""
    patient_dir = root_dir / patient_id
    image_path = find_file(patient_dir, [Path(BASE_IMAGE_FILENAME).stem])
    print("image path: ", image_path)
    if not image_path:
        logging.warning(f"Base image '{BASE_IMAGE_FILENAME}' not found for patient: {patient_id}")
    return image_path


def find_mask(
    patient_id: str,
    model: str,
    structure_synonyms: List[str],
    root_dir: Path
) -> Optional[Path]:
    """Find a segmentation mask for a given patient, model, and structure."""
    search_dir = root_dir / patient_id / model
    if not search_dir.is_dir():
        return None
    return find_file(search_dir, structure_synonyms)


def create_output_directories(patient_id: str, root_dir: Path, single_patient_mode: bool = False) -> tuple[Path, Path]:
    """Create output directories for visualization and reports."""
    if single_patient_mode:
        patient_dir = root_dir / patient_id
        base_output_dir = patient_dir / "comparison_results"
    else:
        base_output_dir = Path("./comparison_results")

    base_output_dir.mkdir(exist_ok=True, parents=True)

    if not single_patient_mode:
        patient_output_dir = base_output_dir / patient_id
        patient_output_dir.mkdir(exist_ok=True, parents=True)
        visualization_dir = patient_output_dir / "plots"
        report_dir = patient_output_dir / "reports"
    else:
        visualization_dir = base_output_dir / "plots"
        report_dir = base_output_dir / "reports"

    visualization_dir.mkdir(exist_ok=True, parents=True)
    report_dir.mkdir(exist_ok=True, parents=True)

    return visualization_dir, report_dir

# Section: Visualization
def create_enhanced_visualization(
    base_image_sitk,
    contour_dict_a: Dict,
    contour_dict_b: Dict,
    combined_mask,
    plot_title: str,
    output_path: Path
) -> None:
    """
    Create enhanced visualization with custom styling for different models.
    Platipy (Model A) contours are dashed, Limbus (Model B) contours are solid.
    """
    try:
        # Define cropping limits from combined mask
        (sag_size, cor_size, ax_size), (sag_0, cor_0, ax_0) = label_to_roi(
            combined_mask, expansion_mm=20
        )
        limits = [
            ax_0, ax_0 + ax_size,
            cor_0, cor_0 + cor_size,
            sag_0, sag_0 + sag_size,
        ]

        # Create visualiser with limits
        vis = ImageVisualiser(base_image_sitk, limits=limits)

        # Line styles per model
        style_a = {"linestyle": "--", "linewidth": 2.5}  # Dashed for Platipy
        style_b = {"linestyle": "-", "linewidth": 2.0}   # Solid for Limbus

        # Add contours for Model A
        for sub_structure, mask in contour_dict_a.items():
            color = STRUCTURE_COLORS.get(sub_structure)
            name = f"{sub_structure} ({MODELS[0]})"
            if hasattr(vis, "add_contour"):
                vis.add_contour(mask, name=name, color=color, **style_a)
            else:
                vis.add_label(mask, name=name, color=color, outline=True)

        # Add contours for Model B
        for sub_structure, mask in contour_dict_b.items():
            color = STRUCTURE_COLORS.get(sub_structure)
            name = f"{sub_structure} (Manual Contour)"
            # name = f"{sub_structure} ({MODELS[1]})"
            if hasattr(vis, "add_contour"):
                vis.add_contour(mask, name=name, color=color, **style_b)
            else:
                vis.add_label(mask, name=name, color=color, outline=True)

        # Render figure
        fig = vis.show()

        # Title and style legend
        fig.suptitle(plot_title, fontsize=14, fontweight='bold')

        ax = fig.get_axes()[0] if fig.get_axes() else None
        if ax:
            textstr = f'{MODELS[0]}: Dashed lines (--)\nManual Contours: Solid lines (—)'
            # textstr = f'{MODELS[0]}: Dashed lines (--)\n{MODELS[1]}: Solid lines (—)'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

            # Normalize legend labels by structure
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = {}
            for handle, label in zip(handles, labels):
                structure_name = label.split(' (')[0]
                if structure_name not in unique_labels and structure_name in STRUCTURE_COLORS:
                    handle.set_color(STRUCTURE_COLORS[structure_name])
                    unique_labels[structure_name] = handle

            # Optional explicit legend (left commented to preserve original behavior)
            # ax.legend(unique_labels.values(), unique_labels.keys(),
            #           loc='best', title="Structures", prop={'size': 8})

        # Save figure
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor='white')
        plt.close(fig)

        logging.info(f"Enhanced visualization saved successfully to: {output_path}")

    except Exception as e:
        logging.error(f"Failed to create enhanced visualization: {e}", exc_info=True)

# Section: Reporting helpers
def print_formatted_summary(summary_df: pd.DataFrame, patient_count: int, structure_count: int):
    """Print a formatted summary table for quick review in the console."""
    print("\n" + "="*120)
    print(" " * 40 + "Segmentation Comparison Summary")
    print("="*120)

    print(f"{'Structure':<25} {'DSC':<20} {'Mean Abs Dist (mm)':<25} {'Hausdorff Dist (mm)':<25} {'Volume Ratio':<20}")
    print("-" * 120)

    for structure in summary_df.index:
        dsc = summary_df.loc[structure, 'DSC']
        mad = summary_df.loc[structure, 'Mean Absolute Distance (mm)']
        hd = summary_df.loc[structure, 'HD_mm']
        vr = summary_df.loc[structure, 'volume_ratio']
        print(f"{structure:<25} {dsc:<20} {mad:<25} {hd:<25} {vr:<20}")

    print("="*120)
    print(f"Analysis complete. Processed {structure_count} structure instances from {patient_count} patient(s).")
    print("="*120)

# Section: Main workflow
def main():
    """Run the comparison workflow for a single patient or batch mode."""
    parser = argparse.ArgumentParser(
        description="Compare cardiac segmentation masks from two models using an optimized workflow with enhanced visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-p", "--patient_id",
        type=str,
        default=None,
        help="Specify a single patient ID to process. If not provided, runs in batch mode using the patient list file."
    )
    args = parser.parse_args()

    # Determine patient list
    single_patient_mode = args.patient_id is not None

    if single_patient_mode:
        patient_ids = [args.patient_id]
        logging.info(f"Running in single-patient mode for: {args.patient_id}")
    else:
        logging.info("Running in batch mode for all patients.")
        if not PATIENT_LIST_PATH.exists():
            logging.error(f"Patient list not found at: {PATIENT_LIST_PATH}")
            return
        try:
            patient_ids = np.loadtxt(PATIENT_LIST_PATH, dtype=str, ndmin=1)
            logging.info(f"Loaded {len(patient_ids)} patient IDs.")
        except Exception as e:
            logging.error(f"Failed to load patient list: {e}")
            return

    # Iterate patients
    all_metrics_dfs = []
    pbar_patients = tqdm(patient_ids, desc="Processing Patients")

    for patient_id in pbar_patients:
        pbar_patients.set_postfix_str(f"Patient: {patient_id}")

        visualization_dir, report_dir = create_output_directories(patient_id, ROOT_DIR, single_patient_mode)

        base_image_path = find_base_image(patient_id, ROOT_DIR)
        if not base_image_path:
            logging.warning(f"Skipping patient {patient_id} due to missing base image.")
            continue

        contour_dict_a = {}
        contour_dict_b = {}
        all_masks_for_roi = []
        mask_paths_a = {}
        mask_paths_b = {}

        for sub_structure, synonyms in CANONICAL_STRUCTURES.items():
            path_a = find_mask(patient_id, MODELS[0], synonyms, ROOT_DIR)
            path_b = find_mask(patient_id, MODELS[1], synonyms, ROOT_DIR)

            if path_a and path_b:
                try:
                    mask_a_sitk = sitk.Cast(sitk.ReadImage(str(path_a)), sitk.sitkUInt8)
                    mask_b_sitk = sitk.Cast(sitk.ReadImage(str(path_b)), sitk.sitkUInt8)
                    mask_b_sitk.CopyInformation(mask_a_sitk)

                    contour_dict_a[sub_structure] = mask_a_sitk
                    contour_dict_b[sub_structure] = mask_b_sitk
                    mask_paths_a[sub_structure] = str(path_a)
                    mask_paths_b[sub_structure] = str(path_b)
                    all_masks_for_roi.extend([mask_a_sitk, mask_b_sitk])
                except Exception as e:
                    logging.error(f"Error reading masks for {patient_id} | {sub_structure}: {e}")
            else:
                logging.warning(f"Missing mask pair for {patient_id} | {sub_structure}. Skipping this structure.")

        if not contour_dict_a:
            logging.warning(f"No common structures found for patient {patient_id}. Skipping plot generation.")
            continue

        try:
            base_image_sitk = sitk.ReadImage(str(base_image_path))

            rows = []
            for sub_structure in contour_dict_a.keys():
                la = contour_dict_a[sub_structure]
                lb = contour_dict_b[sub_structure]
                try:
                    dsc = compute_metric_dsc(la, lb)
                    masd = compute_metric_masd(la, lb)
                    hd = compute_metric_hd(la, lb)
                    vol_a = compute_volume(la)
                    vol_b = compute_volume(lb)
                    rows.append({
                        "sub_structure": sub_structure,
                        "DSC": dsc,
                        "Mean Absolute Distance (mm)": masd,
                        "HD_mm": hd,
                        f"Volume {MODELS[0]} (mL)": vol_a,
                        f"Volume {MODELS[1]} (mL)": vol_b,
                    })
                except Exception as e:
                    logging.error(f"Metric failure | {patient_id} | {sub_structure}: {e}")

            if not rows:
                logging.warning(f"No metrics computed for patient {patient_id}.")
                continue

            df_metrics = pd.DataFrame(rows).set_index("sub_structure")

            combined_mask = all_masks_for_roi[0]
            for m in all_masks_for_roi[1:]:
                combined_mask = sitk.Maximum(combined_mask, m)

            plot_filename = f"{patient_id}_all_structures_comparison_enhanced.png"
            output_plot_path = visualization_dir / plot_filename
            plot_title = ""
            # plot_title = f"Segmentation Comparison - Patient: {patient_id}"

            create_enhanced_visualization(
                base_image_sitk,
                contour_dict_a,
                contour_dict_b,
                combined_mask,
                plot_title,
                output_plot_path
            )

            df_metrics['patient_id'] = patient_id
            df_metrics['model_A_mask'] = df_metrics.index.map(mask_paths_a)
            df_metrics['model_B_mask'] = df_metrics.index.map(mask_paths_b)

            vol_a_col = f"Volume {MODELS[0]} (mL)"
            vol_b_col = f"Volume {MODELS[1]} (mL)"
            df_metrics['volume_ratio'] = np.where(
                df_metrics[vol_b_col] > 0,
                df_metrics[vol_a_col] / df_metrics[vol_b_col],
                np.nan
            )

            all_metrics_dfs.append(df_metrics)

        except Exception as e:
            logging.error(f"Failed to process or plot for patient {patient_id}: {e}", exc_info=True)

    if not all_metrics_dfs:
        logging.error("No valid patient data processed. Cannot generate reports. Exiting.")
        return

    logging.info("Metric computation complete. Generating reports...")

    detailed_df = pd.concat(all_metrics_dfs)
    cols_order = [
        "patient_id", "sub_structure",
        "DSC", "Mean Absolute Distance (mm)", "HD_mm", "volume_ratio",
        f"Volume {MODELS[0]} (mL)", f"Volume {MODELS[1]} (mL)",
        "model_A_mask", "model_B_mask"
    ]
    detailed_df = detailed_df.reset_index().rename(columns={'index': 'sub_structure'})
    detailed_df = detailed_df[cols_order]

    report_base_dir = Path("./comparison_results") if not single_patient_mode else ROOT_DIR / patient_ids[0] / "comparison_results"

    report_base_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Report base directory created at: {report_base_dir}")

    # Save reports
    if single_patient_mode:
        patient_id = patient_ids[0]
        _, report_dir = create_output_directories(patient_id, ROOT_DIR, True)
        detailed_df.to_csv(report_dir / f"{patient_id}_detailed_metrics.csv", index=False, float_format="%.3f")
        logging.info(f"Saved detailed results to '{report_dir / f'{patient_id}_detailed_metrics.csv'}'")
    else:
        for patient_id, group in detailed_df.groupby('patient_id'):
            _, report_dir = create_output_directories(patient_id, ROOT_DIR, False)
            group.to_csv(report_dir / "detailed_metrics.csv", index=False, float_format="%.3f")
            logging.info(f"Saved patient {patient_id} detailed results.")
        combined_detailed_path = report_base_dir / "combined_detailed_metrics.csv"
        detailed_df.to_csv(combined_detailed_path, index=False, float_format="%.3f")
        logging.info(f"Saved combined detailed results to '{combined_detailed_path}'")

    # Summary statistics
    metric_cols = ["DSC", "Mean Absolute Distance (mm)", "HD_mm", "volume_ratio"]
    summary_stats = detailed_df.groupby("sub_structure")[metric_cols].agg(['mean', 'std'])

    formatted_summary = pd.DataFrame(index=summary_stats.index)
    for metric in metric_cols:
        mean_col, std_col = (metric, 'mean'), (metric, 'std')
        if detailed_df['patient_id'].nunique() > 1:
            formatted_summary[metric] = (
                summary_stats[mean_col].map('{:.3f}'.format) + ' ± ' +
                summary_stats[std_col].map('{:.3f}'.format)
            )
        else:
            formatted_summary[metric] = summary_stats[mean_col].map('{:.3f}'.format)

    if single_patient_mode:
        patient_id = patient_ids[0]
        _, report_dir = create_output_directories(patient_id, ROOT_DIR, True)
        summary_csv_path = report_dir / "summary_metrics.csv"
    else:
        summary_csv_path = report_base_dir / "combined_summary_metrics.csv"

    formatted_summary.to_csv(summary_csv_path)
    logging.info(f"Saved summary statistics to '{summary_csv_path}'")

    patient_count = detailed_df['patient_id'].nunique()
    structure_count = len(detailed_df)

    print_formatted_summary(formatted_summary, patient_count, structure_count)


if __name__ == "__main__":
    main()
