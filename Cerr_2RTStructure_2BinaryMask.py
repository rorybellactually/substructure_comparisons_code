#!/usr/bin/env python3
"""
Extract binary masks from DICOM RTSTRUCT files
"""

import os
import sys
import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path
from skimage.draw import polygon as ski_polygon


def find_rtstruct_file(base_dir):
    """
    Find the first RTSTRUCT DICOM file within a directory tree.
    """
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".dcm") and "rtstruct" in f.lower():
                return os.path.join(root, f)
    return None


def find_nifti_file(base_dir):
    """
    Find the first NIfTI (.nii or .nii.gz) file within a directory tree.
    """
    for root, _, files in os.walk(base_dir):
        for f in files:
            name = f.lower()
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                return os.path.join(root, f)
    return None


def load_rtstruct_and_nifti(rtstruct_file, nifti_file):
    """
    Load RTSTRUCT DICOM dataset and the NIfTI reference image.
    """
    ds = pydicom.dcmread(rtstruct_file)
    nii_img = nib.load(nifti_file)
    return ds, nii_img


def polygons_to_mask(polygons, img_shape):
    """
    Convert a list of 2D polygons to a binary mask (for one slice).
    Each polygon is a Nx2 array of (row, col) coordinates in pixel space.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for poly in polygons:
        rr, cc = ski_polygon(poly[:, 0], poly[:, 1], shape=img_shape[:2])
        mask[rr, cc] = 1
    return mask


def extract_rtstruct_masks(rtstruct_file, nifti_file, output_dir):
    """
    Extract binary masks for all ROIs with contour data using the NIfTI geometry as reference.

    Args:
        rtstruct_file: path to the RTSTRUCT DICOM file
        nifti_file: path to the reference NIfTI file
        output_dir: directory to write per-ROI masks
    """
    print(f"Processing RTSTRUCT: {rtstruct_file}")
    print(f"Reference NIfTI file: {nifti_file}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        ds, nii_img = load_rtstruct_and_nifti(rtstruct_file, nifti_file)

        # Extract geometry info from NIfTI
        affine = nii_img.affine
        inv_affine = np.linalg.inv(affine)
        vol_shape = nii_img.shape  # (z, y, x) or (y, x, z) depending on orientation; we will map using affine

        # Create a lookup for ROI number -> ROI name
        roi_info = {}
        if hasattr(ds, "StructureSetROISequence"):
            for roi in ds.StructureSetROISequence:
                roi_number = roi.ROINumber
                roi_name = roi.ROIName if hasattr(roi, "ROIName") else f"ROI_{roi_number}"
                roi_info[roi_number] = roi_name

        if not hasattr(ds, "ROIContourSequence"):
            print("No ROIContourSequence found in RTSTRUCT.")
            return False

        success_count = 0

        for roi_contour in ds.ROIContourSequence:
            roi_number = roi_contour.ReferencedROINumber
            roi_name = roi_info.get(roi_number, f"ROI_{roi_number}")

            # Skip ROIs without contours
            if not hasattr(roi_contour, "ContourSequence"):
                print(f"Skipping ROI without contours: {roi_name}")
                continue

            print(f"Extracting ROI: {roi_name}")

            # Collect per-slice polygons mapped into voxel space
            slice_polygons = {}  # key: z_index (int), value: list of polygons (arrays)
            try:
                for contour in roi_contour.ContourSequence:
                    if not hasattr(contour, "ContourData"):
                        continue

                    data = contour.ContourData
                    points = np.array(data, dtype=float).reshape(-1, 3)  # in DICOM (x, y, z) mm

                    # Convert from world (mm) coordinates to voxel indices
                    # Append ones for affine multiplication (x, y, z, 1)
                    homog = np.column_stack([points, np.ones((points.shape[0],))])
                    vox = (inv_affine @ homog.T).T  # -> (N, 4)
                    vox = vox[:, :3]  # drop homogeneous component

                    # Round to nearest pixel for mask fill
                    # We take (y, x) from voxel coordinates: (z, y, x)
                    z_idx_float = vox[:, 2]
                    yx = vox[:, [1, 0]]

                    # Determine which slice these belong to by rounding the mean z
                    z_index = int(np.round(np.mean(z_idx_float)))

                    # Discard polygons that fall out of volume
                    if z_index < 0 or z_index >= vol_shape[2] if len(vol_shape) > 2 else z_index >= vol_shape[0]:
                        continue

                    # Ensure coordinates are within the plane bounds
                    # We will clip later after mask creation; here just store polygon
                    poly_rc = yx  # (row, col) = (y, x) in voxel space
                    slice_polygons.setdefault(z_index, []).append(poly_rc)

                if not slice_polygons:
                    print(f"No valid contours for ROI: {roi_name}")
                    continue

                # Create a 3D mask volume matching NIfTI shape
                # We assume NIfTI ordering (X, Y, Z) or (Y, X, Z) depending on library; we reconstruct per-slice.
                if len(vol_shape) == 3:
                    mask_vol = np.zeros(vol_shape, dtype=np.uint8)
                    height, width = vol_shape[0], vol_shape[1]
                    # Try to infer axis order: we mapped polygons using affine such that z_index indexes the third dim.
                    # We will fill as mask_vol[y, x, z] = 1 per polygon.
                    for z_index, polys in slice_polygons.items():
                        # Build a 2D mask for this slice
                        # Prepare polygon coordinates as (row=y, col=x)
                        # Clip polygon coords to image plane bounds
                        clipped_polys = []
                        for poly in polys:
                            poly_clipped = np.copy(poly)
                            poly_clipped[:, 0] = np.clip(poly_clipped[:, 0], 0, height - 1)
                            poly_clipped[:, 1] = np.clip(poly_clipped[:, 1], 0, width - 1)
                            clipped_polys.append(poly_clipped.astype(np.float64))

                        # Fill polygons onto the 2D plane
                        plane = np.zeros((height, width), dtype=np.uint8)
                        for poly in clipped_polys:
                            if poly.shape[0] >= 3:
                                rr, cc = ski_polygon(poly[:, 0], poly[:, 1], shape=plane.shape)
                                plane[rr, cc] = 1

                        # Assign into volume (y, x, z)
                        if 0 <= z_index < mask_vol.shape[2]:
                            mask_vol[:, :, z_index] = np.maximum(mask_vol[:, :, z_index], plane)

                else:
                    print(f"Unsupported NIfTI volume shape: {vol_shape}")
                    continue

                # Save mask as NIfTI alongside the reference affine/header
                roi_safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in roi_name)
                output_path = os.path.join(output_dir, f"{roi_safe}.nii.gz")

                # Use same header/affine as reference image
                mask_img = nib.Nifti1Image(mask_vol, affine=affine)
                nib.save(mask_img, output_path)
                print(f"Saved: {output_path}")
                success_count += 1

            except Exception as e:
                print(f"Error extracting ROI {roi_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\nSuccessfully extracted {success_count}/{len(ds.ROIContourSequence)} ROIs with contour data")
        return success_count > 0

    except Exception as e:
        print(f"Error processing RTSTRUCT: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_single_patient_from_output(patient_id, output_folder=None):
    """
    Process a single patient's RTSTRUCT data from an output folder.
    """
    print(f"Patient ID: {patient_id}")
    print(f"Output folder: {output_folder}")

    # Assume RTSTRUCT is in the output folder
    rtstruct_file = find_rtstruct_file(output_folder)
    if not rtstruct_file:
        print(f"Error: RTSTRUCT file not found in {output_folder}")
        return False

    # Find NIfTI file
    nifti_file = find_nifti_file(output_folder)
    if not nifti_file:
        print(f"Error: NIfTI file not found in {output_folder}")
        return False

    # Create masks directory
    masks_dir = os.path.join(output_folder, "RTSTRUCT_masks")
    os.makedirs(masks_dir, exist_ok=True)

    # Extract masks for all ROIs
    return extract_rtstruct_masks(rtstruct_file, nifti_file, masks_dir)


def process_single_patient(patient_dir):
    """
    Process a single patient's data given the patient directory.
    """
    print(f"Patient directory: {patient_dir}")

    # Find RTSTRUCT file
    rtstruct_file = find_rtstruct_file(patient_dir)
    if not rtstruct_file:
        print(f"Error: RTSTRUCT file not found in {patient_dir}")
        return False

    # Find NIfTI file
    nifti_file = find_nifti_file(patient_dir)
    if not nifti_file:
        print(f"Error: NIfTI file not found in {patient_dir}")
        return False

    # Output directory
    output_dir = os.path.join(patient_dir, "RTSTRUCT_masks")
    os.makedirs(output_dir, exist_ok=True)

    # Extract masks for all ROIs
    return extract_rtstruct_masks(rtstruct_file, nifti_file, output_dir)


def process_all_patients(base_dir, patient_list_file):
    """
    Batch process all patients listed in a file.
    """
    if not os.path.exists(patient_list_file):
        print(f"Error: Patient list file not found: {patient_list_file}")
        return

    # Read list of patient IDs
    with open(patient_list_file, 'r') as f:
        patient_ids = [line.strip() for line in f if line.strip()]

    print(f"Found {len(patient_ids)} patients to process")

    success_count = 0
    error_count = 0

    for i, patient_id in enumerate(patient_ids, 1):
        print(f"\n{'='*50}")
        print(f"Processing patient {i}/{len(patient_ids)}: {patient_id}")
        print(f"{'='*50}")

        # Patient directory
        patient_dir = os.path.join(base_dir, patient_id)
        if not os.path.exists(patient_dir):
            print(f"Patient directory not found: {patient_dir}")
            error_count += 1
            continue

        ok = process_single_patient(patient_dir)
        if ok:
            success_count += 1
        else:
            error_count += 1

    print("\nBatch processing summary")
    print(f"Total: {len(patient_ids)}, Success: {success_count}, Errors: {error_count}")


def main():
    """
    Entry point for interactive processing.
    """
    print("DICOM RTSTRUCT Binary Mask Extractor")
    print("="*50)

    # Update these paths as needed for your environment
    base_dir = "/Users/user/Documents/PhD/A.Work/CohortStudy/data/Paediatrics"
    patient_list_file = "/Users/user/Documents/PhD/A.Work/CohortStudy/data/paediatrics_patient_numbers.txt"

    print("Choose processing mode:")
    print("1. Process single patient")
    print("2. Process all patients from list")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == '1':
        patient_id_input = input("Enter patient ID: ").strip()
        patient_dir_single = os.path.join(base_dir, patient_id_input)
        if os.path.exists(patient_dir_single):
            print(f"\nProcessing single patient: {patient_id_input}")
            process_single_patient(patient_dir_single)
        else:
            print(f"Error: Patient directory not found for ID: {patient_id_input}")
    elif choice == '2':
        print("\nProcessing all patients from list...")
        process_all_patients(base_dir, patient_list_file)
    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
