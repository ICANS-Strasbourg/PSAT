#!/bin/bash
# segment_folder.sh
#
# This script loops over all nii.gz files in an input folder,
# runs TotalSegmentator (generating all classes) with multi-label output,
# and then remaps the resulting multi-label segmentation to a custom label mapping.
#
# The desired final mapping is:
#   background:      0
#   Spleen:          1
#   Kidney-Right:    2
#   Kidney-Left:     3
#   Gall-Bladder:    4
#   Liver:           5
#   Stomach:         6
#   Pancreas:        7
#   Esophagus:       8
#   Small-Intestine: 9
#   Duodenum:        10
#   Bladder:         11
#   Prostate:        12
#   Spinal-Canal:    13
#
# Usage:
#   ./segment_folder.sh /path/to/input_folder /path/to/output_folder

# Check for two arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Create a temporary directory for TotalSegmentator outputs
TMP_DIR=$(mktemp -d)

# Function to process a single file
process_file() {
    local file="$1"
    local filename=$(basename "$file")
    local base="${filename%.nii.gz}"
    echo "Processing $filename..."

    # Create a temporary folder for this file's segmentation
    local FILE_TMP_DIR="$TMP_DIR/$base"
    mkdir -p "$FILE_TMP_DIR"

    # Run TotalSegmentator generating all classes with multi-label output
    if ! TotalSegmentator -i "$file" -o "$FILE_TMP_DIR" --ml; then
        echo "TotalSegmentator failed for $file; skipping."
        return
    fi

    # Log the contents of the temporary directory
    echo "Contents of $FILE_TMP_DIR:"
    ls -l "$FILE_TMP_DIR"

    # Find the segmentation file produced by TotalSegmentator.
    local SEG_FILE=$(find "$TMP_DIR" -maxdepth 1 -name "$base.nii" | head -n 1)
    if [ -z "$SEG_FILE" ]; then
        echo "No segmentation file found for $file; skipping."
        return
    fi

    # Run the Python script to remap labels
    if ! python3 scripts/remap_labels.py "$SEG_FILE" "$OUTPUT_FOLDER/${base}.nii.gz"; then
        echo "Label remapping failed for $file; skipping."
        return
    fi

    echo "Saved remapped segmentation as ${base}.nii.gz in $OUTPUT_FOLDER"
}

export -f process_file
export TMP_DIR
export OUTPUT_FOLDER

# Loop over all nii.gz files in the input folder and process them in parallel
find "$INPUT_FOLDER" -name "*.nii.gz" -print0 | xargs -0 -n 1 -P 4 bash -c 'process_file "$@"' _

# Clean up temporary files
rm -rf "$TMP_DIR"
echo "Processing complete."
