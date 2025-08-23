"""
File Purpose: Inspect a NIfTI file's header, spacing, and orientation for quick QA.
What this script does: Loads a NIfTI image, prints voxel spacing, units, q/sform codes, affine matrix, axis codes, and verifies z-spacing via the affine.
"""

# Section: Imports
import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes

# Section: Input path
nifti_file = "/home/donal/data/server2/Msc_Minghao/Lymphoma/122819449/ct.nii.gz"

# Section: Load image and header
nii = nib.load(nifti_file)
hdr = nii.header

# Section: Basic spacing information
spacing_xyz = hdr.get_zooms()[:3]
print("voxel spacing (x, y, z) [mm]:", spacing_xyz)
print("slice thickness [mm] (assumed z spacing):", spacing_xyz[2])

# Section: Additional header details
print("pixdim:", hdr["pixdim"])
print("xyzt_units:", hdr.get_xyzt_units())  # e.g., ('mm', 'sec')
print("qform_code:", hdr["qform_code"], "sform_code:", hdr["sform_code"])

# Section: Affine and orientation
aff = nii.affine
print("affine:\n", aff)
print("axis codes (orientation):", aff2axcodes(aff))

# Section: Cross-check z spacing from affine
z_spacing_from_affine = np.linalg.norm(aff[:3, 2])
print("z spacing from affine [mm]:", z_spacing_from_affine)
