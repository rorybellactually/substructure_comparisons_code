"""

Usage:
    1) Run directly:
        python dicom_to_nifti.py
       - If Patients.txt exists, processes only those IDs.
       - Otherwise, falls back to batch converting all subfolders under CT root.

    2) Import and use programmatically:
        from dicom_to_nifti import dicom_to_nifti_converter, batch_convert_dicom_to_nifti
        nifti_path = dicom_to_nifti_converter("/path/to/dicom/folder", patient_id="PID123")
        batch_convert_dicom_to_nifti("/path/to/root")

Notes:
    - Output overwriting behavior: If "ct.nii.gz" already exists, it will be reused (no re-conversion).
    - Basic sanity check warns when the chosen series has < 10 slices.
"""

import os
import sys
import logging
import SimpleITK as sitk

# Setup logging
log_file = r"/home/donal/data/server2/Msc_Minghao/PaediatricsLog/autoseg_processing_log.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add the path to the RaystationUtils.py file to the system path
utils_path = r"/home/donal/data/server2/Msc_Minghao/RaystationUtils.py"
utils_dir = os.path.dirname(utils_path)

if os.path.exists(utils_dir) and utils_dir not in sys.path:
    sys.path.append(utils_dir)
import RaystationUtils as utils  # noqa: E402

# Set paths
ctScansPath = r"/home/donal/data/server2/Msc_Minghao/Paediatrics_withCardiacSubstructs"


def dicom_to_nifti_converter(dicom_path, patient_id=None):
    """
    Convert a DICOM series to NIfTI (.nii.gz) and save it into the same directory.

    Args:
        dicom_path (str): Path to the DICOM folder.
        patient_id (str, optional): Patient ID; if not provided, folder name is used.

    Returns:
        str: Absolute path to the generated NIfTI file on success; None on failure.
    """
    try:
        # Use folder name as patient_id if not provided
        if patient_id is None:
            patient_id = os.path.basename(os.path.normpath(dicom_path))

        # NIfTI output path (saved into the DICOM folder)
        nifti_file_path = os.path.join(dicom_path, "ct.nii.gz")

        # Skip if the NIfTI file already exists
        if os.path.exists(nifti_file_path):
            logging.info(f"NIfTI file already exists: {nifti_file_path}")
            print(f"NIfTI file already exists: {nifti_file_path}")
            return nifti_file_path

        # Read DICOM series
        logging.info(f"Reading DICOM series from: {dicom_path}")
        print(f"Reading DICOM series from: {dicom_path}")
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(dicom_path)

        if not series_IDs:
            logging.error(f"No DICOM series found in: {dicom_path}")
            print(f"No DICOM series found in: {dicom_path}")
            return None

        # Select the series with the most slices
        best_series_id = ""
        max_slices = 0
        logging.info(f"Found {len(series_IDs)} series. Selecting the one with the most slices.")
        print(f"Found {len(series_IDs)} series. Selecting the one with the most slices.")

        for series_id in series_IDs:
            file_names = reader.GetGDCMSeriesFileNames(dicom_path, series_id)
            num_slices = len(file_names)
            logging.debug(f"Series ID: {series_id} has {num_slices} slices.")
            print(f"Series ID: {series_id} has {num_slices} slices.")
            if num_slices > max_slices:
                max_slices = num_slices
                best_series_id = series_id

        if max_slices < 10:
            logging.warning(f"Warning: Selected series has only {max_slices} slices; may cause issues.")
            print(f"Warning: Selected series has only {max_slices} slices; may cause issues.")

        logging.info(f"Selected series ID: {best_series_id} with {max_slices} slices")
        print(f"Selected series ID: {best_series_id} with {max_slices} slices")

        # Read the selected series
        file_names = reader.GetGDCMSeriesFileNames(dicom_path, best_series_id)
        reader.SetFileNames(file_names)
        image = reader.Execute()

        # Cast to float32 for downstream processing consistency
        image = sitk.Cast(image, sitk.sitkFloat32)

        # Write NIfTI
        sitk.WriteImage(image, nifti_file_path)
        logging.info(f"Successfully converted DICOM to NIfTI: {nifti_file_path}")
        print(f"Successfully converted DICOM to NIfTI: {nifti_file_path}")

        return nifti_file_path

    except Exception as e:
        logging.error(f"Failed to convert DICOM to NIfTI for {patient_id}: {e}")
        print(f"Failed to convert DICOM to NIfTI for {patient_id}: {e}")
        return None


def batch_convert_dicom_to_nifti(root_path):
    """
    Batch convert DICOM folders under a root directory to NIfTI files.

    Args:
        root_path (str): Root path containing multiple patient DICOM folders.
    """
    if not os.path.exists(root_path):
        print(f"Root path does not exist: {root_path}")
        return

    success_count = 0
    fail_count = 0

    # Iterate through all subfolders in the root directory
    for patient_folder in os.listdir(root_path):
        patient_path = os.path.join(root_path, patient_folder)

        # Skip non-directory entries
        if not os.path.isdir(patient_path):
            continue

        print(f"\nProcessing patient: {patient_folder}")
        result = dicom_to_nifti_converter(patient_path, patient_folder)

        if result:
            success_count += 1
        else:
            fail_count += 1

    print("\n=== Conversion Summary ===")
    print(f"Successfully converted: {success_count}")
    print(f"Failed conversions: {fail_count}")
    print(f"Total processed: {success_count + fail_count}")


# Example usage
if __name__ == "__main__":
    try:
        # Read patient IDs from file
        patient_ids = utils.txt_to_numpy_array(
            r"/home/donal/data/server2/Msc_Minghao/PaediatricsLog/Patients.txt"
        )

        success_count = 0
        fail_count = 0

        logging.info(f"Starting batch conversion for {len(patient_ids)} patients")
        print(f"Starting batch conversion for {len(patient_ids)} patients")

        for patient_id in patient_ids:
            patient_id = patient_id.strip()  # Remove potential whitespace
            if not patient_id:               # Skip empty lines
                continue

            patient_dicom_path = os.path.join(ctScansPath, patient_id)

            if not os.path.exists(patient_dicom_path):
                logging.warning(f"Patient folder does not exist: {patient_dicom_path}")
                print(f"Patient folder does not exist: {patient_dicom_path}")
                fail_count += 1
                continue

            logging.info(f"Processing patient: {patient_id}")
            print(f"\nProcessing patient: {patient_id}")

            result = dicom_to_nifti_converter(patient_dicom_path, patient_id)

            if result:
                success_count += 1
            else:
                fail_count += 1

        # Log and print summary
        logging.info("=== Conversion Summary ===")
        logging.info(f"Successfully converted: {success_count}")
        logging.info(f"Failed conversions: {fail_count}")
        logging.info(f"Total processed: {success_count + fail_count}")

        print("\n=== Conversion Summary ===")
        print(f"Successfully converted: {success_count}")
        print(f"Failed conversions: {fail_count}")
        print(f"Total processed: {success_count + fail_count}")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"Error in main execution: {e}")

        # Fallback: process all folders under ctScansPath if the patient list file is missing/unreadable
        print("\nFallback: Processing all folders in ctScansPath...")
        batch_convert_dicom_to_nifti(ctScansPath)
