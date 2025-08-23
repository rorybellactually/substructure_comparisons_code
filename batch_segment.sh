#!/bin/bash
# """
#  File Purpose: Batch-process heart substructure segmentation for all patients and extract binary masks.
#  What this script does: Activates a Conda environment, validates paths, loops through patient IDs, runs segmentation, extracts masks, and prints a summary.
# """

set -uo pipefail

# -------------------------------------------------------------------
# Section: Overview and runtime context
# -------------------------------------------------------------------
echo "Script started at $(date)"
echo "Current directory: $(pwd)"
echo "Script path: ${BASH_SOURCE[0]}"

# -------------------------------------------------------------------
# Section: Conda bootstrap and environment activation
# -------------------------------------------------------------------
CONDA_ROOT="$HOME/miniconda3"
echo "Conda root: $CONDA_ROOT"

if [ ! -d "$CONDA_ROOT" ]; then
    echo "ERROR: Conda root directory not found at $CONDA_ROOT"
    exit 1
fi

eval "$("$CONDA_ROOT/bin/conda" shell.bash hook)"

echo "Activating conda environment 'pycerr'..."
if ! conda activate pycerr; then
    echo "Failed to activate conda environment 'pycerr'. Please check if it exists."
    exit 1
fi
echo "Conda environment activated"

# -------------------------------------------------------------------
# Section: Core directories and inputs
# -------------------------------------------------------------------
BASE_DIR="/home/donal/data/server2/Msc_Minghao"
MODEL_DIR="${BASE_DIR}/model_installer/CT_cardiac_structures_deeplab/model_wrapper"
PATIENT_ROOT="${BASE_DIR}/REQUITE"
PATIENT_LIST="${BASE_DIR}/REQUITELog/Patients.txt"

echo "Base directory: $BASE_DIR"
echo "Model directory: $MODEL_DIR"
echo "Patient root: $PATIENT_ROOT"
echo "Patient list: $PATIENT_LIST"

# -------------------------------------------------------------------
# Section: Sanity checks for required paths
# -------------------------------------------------------------------
echo "Checking directories and files..."
if [ ! -d "$BASE_DIR" ]; then echo "Base directory missing"; exit 1; fi
echo "Base directory exists"

if [ ! -d "$MODEL_DIR" ]; then echo "Model directory missing"; exit 1; fi
echo "Model directory exists"

if [ ! -d "$PATIENT_ROOT" ]; then echo "Patient root missing"; exit 1; fi
echo "Patient root exists"

if [ ! -f "$PATIENT_LIST" ]; then echo "Patient list file missing"; exit 1; fi
echo "Patient list file exists"

# -------------------------------------------------------------------
# Section: Mask extractor script path
# -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASK_EXTRACTOR_SCRIPT="${SCRIPT_DIR}/Cerr_2BinaryMask.py"

echo "Script directory: $SCRIPT_DIR"
echo "Mask extractor script: $MASK_EXTRACTOR_SCRIPT"
if [ ! -f "$MASK_EXTRACTOR_SCRIPT" ]; then
    echo "Mask extractor script missing"
else
    echo "Mask extractor script exists"
fi

# -------------------------------------------------------------------
# Section: GPU selection
# -------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

# -------------------------------------------------------------------
# Section: Counters and state
# -------------------------------------------------------------------
total_patients=0
successful_segmentations=0
successful_extractions=0
failed_patients=()

echo "Starting patient processing loop..."

# -------------------------------------------------------------------
# Section: Main loop over patient IDs
# -------------------------------------------------------------------
while IFS= read -r patient_id; do
    echo "Read line: '$patient_id'"

    # Skip empty lines
    if [[ -z "$patient_id" ]]; then
        echo "Skipping empty line"
        continue
    fi

    ((total_patients++))
    echo ""
    echo "======== Processing patient ${patient_id} (${total_patients}) ========"

    # Create output directory
    OUTPUT_FOLDER="${PATIENT_ROOT}/${patient_id}/PyCeRR_subseg"
    echo "Output folder: $OUTPUT_FOLDER"

    if [ ! -d "${OUTPUT_FOLDER}" ]; then
        echo "Creating output directory: ${OUTPUT_FOLDER}"
        if ! mkdir -p "${OUTPUT_FOLDER}"; then
            echo "Failed to create directory ${OUTPUT_FOLDER}"
            failed_patients+=("${patient_id} - Failed to create output directory")
            continue
        fi
    fi

    # Validate patient directory
    echo "Looking for scan files in: ${PATIENT_ROOT}/${patient_id}"
    if [ ! -d "${PATIENT_ROOT}/${patient_id}" ]; then
        echo "Patient directory not found: ${PATIENT_ROOT}/${patient_id}"
        failed_patients+=("${patient_id} - Patient directory not found")
        continue
    fi

    # Locate scan file
    SCAN_FILE=$(find "${PATIENT_ROOT}/${patient_id}" -name "*.nii.gz" 2>/dev/null | head -1)

    if [ -z "$SCAN_FILE" ]; then
        echo "No .nii.gz file found for patient ${patient_id}"
        echo "Files in patient directory:"
        ls -la "${PATIENT_ROOT}/${patient_id}" 2>/dev/null || echo "Cannot list directory contents"
        failed_patients+=("${patient_id} - No scan file found")
        continue
    fi

    echo "Found scan file: $SCAN_FILE"

    # Run segmentation
    echo "Running segmentation for patient ${patient_id}..."
    echo "Changing to model directory: $MODEL_DIR"

    if ! cd "${MODEL_DIR}"; then
        echo "Failed to change to model directory: $MODEL_DIR"
        failed_patients+=("${patient_id} - Cannot access model directory")
        continue
    fi

    echo "Running: python run_segmentation.py ${PATIENT_ROOT}/${patient_id} ${OUTPUT_FOLDER}"

    if python run_segmentation.py "${PATIENT_ROOT}/${patient_id}" "${OUTPUT_FOLDER}" 2>&1; then
        echo "Segmentation completed for patient ${patient_id}"
        ((successful_segmentations++))
    else
        echo "Segmentation failed for patient ${patient_id}"
        failed_patients+=("${patient_id} - Segmentation failed")
        continue
    fi

    # Verify expected heart mask
    HEART_STRUCTURE_FILE="${OUTPUT_FOLDER}/ct_heart.nii.gz"
    if [ ! -f "$HEART_STRUCTURE_FILE" ]; then
        echo "Warning: ct_heart.nii.gz not found for patient ${patient_id}"
        echo "Listing sample files in ${OUTPUT_FOLDER}:"
        find "${OUTPUT_FOLDER}" -name "*.nii.gz" -type f 2>/dev/null | head -5
        failed_patients+=("${patient_id} - ct_heart.nii.gz not generated")
        continue
    fi

    echo "Found ct_heart.nii.gz for patient ${patient_id}"

    # Extract binary masks
    echo "Extracting binary masks for patient ${patient_id}..."

    INDIVIDUAL_MASKS_DIR="${OUTPUT_FOLDER}"

    echo "Running binary mask extraction script from: ${MASK_EXTRACTOR_SCRIPT}"

    if [ ! -f "${MASK_EXTRACTOR_SCRIPT}" ]; then
        echo "Error: Python script not found at ${MASK_EXTRACTOR_SCRIPT}"
        failed_patients+=("${patient_id} - Python script not found")
        continue
    fi

    if ! cd "${SCRIPT_DIR}"; then
        echo "Failed to change to script directory: $SCRIPT_DIR"
        failed_patients+=("${patient_id} - Cannot access script directory")
        continue
    fi

    echo "Running: python ${MASK_EXTRACTOR_SCRIPT} ${patient_id} ${OUTPUT_FOLDER}"

    if python "${MASK_EXTRACTOR_SCRIPT}" "${patient_id}" "${OUTPUT_FOLDER}" 2>&1; then
        echo "Binary mask extraction completed for patient ${patient_id}"
        ((successful_extractions++))
    else
        echo "Binary mask extraction failed for patient ${patient_id}"
        failed_patients+=("${patient_id} - Binary mask extraction failed")
        continue
    fi

    echo "Completed all processing for patient ${patient_id}"

done < "${PATIENT_LIST}"

echo "Finished reading patient list"

# -------------------------------------------------------------------
# Section: Summary
# -------------------------------------------------------------------
echo ""
echo "======== PROCESSING SUMMARY ========"
echo "Total patients processed: ${total_patients}"
echo "Successful segmentations: ${successful_segmentations}"
echo "Successful binary extractions: ${successful_extractions}"
echo "Failed patients: $((total_patients - successful_extractions))"

if [ ${#failed_patients[@]} -gt 0 ]; then
    echo ""
    echo "======== FAILED PATIENTS ========"
    for failure in "${failed_patients[@]}"; do
        echo "- $failure"
    done
fi

echo ""
echo "======== All patients processed. ========"
echo "Script finished at $(date)"
