"""
File Purpose: Quick inspection tool to examine labels within a NIfTI segmentation volume.
What this script does: Loads a NIfTI file, extracts voxel data, and prints unique label values and the raw data array.
"""

# Section: Imports
import numpy as np
import nibabel as nib

# Section: Input configuration
nifti_file = "/home/donal/data/server2/Msc_Minghao/REQUITE/RQ00048-0/PyCeRR_subseg/ct_heart.nii.gz"

# Section: Load NIfTI image and extract voxel data
nii_img = nib.load(nifti_file)
data = nii_img.get_fdata()

# Section: Compute and report unique labels and data
unique_labels = np.unique(data)
print("Unique labels found:", unique_labels)
print("Unique data found:", data)
