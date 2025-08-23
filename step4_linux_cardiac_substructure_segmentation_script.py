"""
File Purpose: Run cardiac substructure auto-segmentation for a cohort and save results.
What this script does: Reads patient CT images, runs Platipy hybrid cardiac segmentation, writes NIfTI outputs, and generates a snapshot for QA.
"""

# Section: Imports
import os
import sys
import logging
from matplotlib import pyplot as plt
import SimpleITK as sitk
from platipy.imaging.tests.data import get_lung_nifti
from platipy.imaging.projects.cardiac.run import run_hybrid_segmentation
from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com
from os import makedirs

# Section: GPU and environment configuration
# Use 2nd GPU on Pepita server
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Run the code on a specified GPU

# Section: Logging setup
log_file = r"/home/donal/data/server2/Msc_Minghao/PaediatricsLog/autoseg_processing_log.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Section: Add RaystationUtils.py to sys.path
utils_path = r'/home/donal/data/server2/Msc_Minghao/RaystationUtils.py'
utils_dir = os.path.dirname(utils_path)
import numpy as np

if os.path.exists(utils_dir) and utils_dir not in sys.path:
    sys.path.append(utils_dir)
import RaystationUtils as utils

# Section: Utility function to round SimpleITK image voxel values
def sitk_round_to_decimals(image, decimals):
    """
    Rounds a SimpleITK image to a specified number of decimal places.

    Args:
        image (SimpleITK.Image): The input SimpleITK image.
        decimals (int): The number of decimal places to round to.

    Returns:
        SimpleITK.Image: The rounded SimpleITK image.
    """
    # Convert SimpleITK image to NumPy array
    array = sitk.GetArrayFromImage(image)

    # Round the NumPy array
    rounded_array = np.round(array, decimals=decimals)

    # Convert the NumPy array back to a SimpleITK image
    rounded_image = sitk.GetImageFromArray(rounded_array)

    # Copy the metadata from the original image
    rounded_image.CopyInformation(image)

    return rounded_image

# Section: Read DICOM series into SimpleITK image
def read_dicom_series_into_sitk_image(dicom_path):
    """
    Reads a DICOM series from a directory into a SimpleITK image.

    Args:
        dicom_path (str): The path to the directory containing the DICOM series.

    Returns:
        SimpleITK.Image: The resulting SimpleITK image.
    """
    # A file name that belongs to the series we want to read
    file_name = os.path.join(dicom_path, '000.dcm')

    # Read the file's meta-information without reading bulk pixel data
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)
    file_reader.ReadImageInformation()
    series_ID = file_reader.GetMetaData('0020|000e')

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path, series_ID)
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    return img

# Section: Main execution and batch processing
if __name__ == "__main__":
    ctScansPath = r'/home/donal/data/server2/Msc_Minghao/Paediatrics_withCardiacSubstructs'
    try:
        patient_ids = utils.txt_to_numpy_array(r'/home/donal/data/server2/Msc_Minghao/PaediatricsLog/Patients.txt')
    except Exception as e:
        logging.error(f"Failed to load patient IDs: {e}")
        sys.exit(1)

    for id in patient_ids:
        logging.info(f"Processing patient: {id}")
        try:
            # test_pat_path = os.path.join(ctScansPath, id, "dicomCT") # Option to read from DICOM series instead of MHD file
            test_pat_path = os.path.join(ctScansPath, id)
            logging.info(f"Processing patient: {id} at path: {test_pat_path}")
            print(f"Processing patient: {id} at path: {test_pat_path}")
            
            # test_image = read_dicom_series_into_sitk_image(test_pat_path) # Option to read from DICOM series instead of MHD file
            # Read NIfTI instead of MHD
            test_image = sitk.ReadImage(os.path.join(test_pat_path, 'ct.nii.gz'))

            print("image type: ", test_image.GetPixelID())
            logging.info("image type: " + str(test_image.GetPixelID()))
            logging.info("Successfully read in image")
            # test_image = sitk_round_to_decimals(test_image, 6)
            # logging.info("Successfully rounded image")
        except Exception as e:
            logging.error(f"Error reading image for patient {id}: {e}")
            continue

        # Section: Check existing outputs and prepare output directory
        output_directory = os.path.join(test_pat_path, "substructures")
        if os.path.exists(output_directory):
            if len(os.listdir(output_directory)) > 2:  # >2 because expecting at least 2 files (ct.mhd and ct.raw)
                logging.info(f"Segmentations already exist for patient {id}")
                print(f"Segmentations already exist for patient {id}")
                continue

        # Section: Run hybrid segmentation and save label images
        try:
            auto_structures, _ = run_hybrid_segmentation(test_image)
            
            makedirs(output_directory, exist_ok=True)
            
            for struct_name, struct_image in auto_structures.items():
                sitk.WriteImage(struct_image, os.path.join(output_directory, f"{struct_name}.nii.gz"))
            
            logging.info(f"Segmentations saved to: {output_directory}")
            print(f"Segmentations saved to: {output_directory}")
        except Exception as e:
            logging.error(f"Segmentation failed for patient {id}: {e}")
            continue

        # Section: Visualisation snapshot for QA
        try:
            vis = ImageVisualiser(test_image, cut=get_com(auto_structures["Heart"]))
            vis.add_contour({struct: auto_structures[struct] for struct in auto_structures.keys()})
            vis.set_limits_from_label(auto_structures["Heart"], expansion=20)
            vis.show()
            
            snapshot_path = os.path.join(output_directory, "snapshot.png")
            plt.savefig(snapshot_path)
            logging.info(f"Snapshot saved to: {snapshot_path}")
            print(f"Snapshot saved to: {snapshot_path}")
        except Exception as e:
            logging.error(f"Failed to generate visualization for patient {id}: {e}")
