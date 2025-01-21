#!/bin/bash

# Set your paths
INPUT_DIR="ICANS_Pediatric"
OUTPUT_DIR="ICANS_TotalSegmentatorFormat"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to process each patient folder
process_patient() {
    local patient_dir="$1"
    local patient_name="${patient_dir##*/}"
    local patient_output_dir="$OUTPUT_DIR/$patient_name"

    mkdir -p "$patient_output_dir"
    mkdir -p "$patient_output_dir/segmentations"

    # Convert CT scans
    echo "Converting CT scans for patient: $patient_name"
    dcm2niix -z y -f ct -o "$patient_output_dir" "$patient_dir/CT"/*.dcm
    # Rename to ct.nii.gz if it's not already named so
    mv "$patient_output_dir/ct_0000.nii.gz" "$patient_output_dir/ct.nii.gz" 2>/dev/null || true

    # Convert RTSTRUCT
    local rtstruct=$(find "$patient_dir/RTSTRUCT" -name "*.dcm")
    if [[ -n "$rtstruct" ]]; then
        echo "Converting RTSTRUCT for patient: $patient_name"
        local ct_series="$patient_dir/CT"
        dcmrtstruct2nii convert -r "$rtstruct" -d "$ct_series" -o "$patient_output_dir/segmentations"

        # Rename files to match TotalSegmentator format (assuming structures are known)
        for seg_file in "$patient_output_dir/segmentations"/*.nii.gz; do
            if [[ -f "$seg_file" ]]; then
                # Extract structure name from filename;
                structure_name=$(basename "$seg_file" .nii.gz | sed 's/^[^-]*-//')
                mv "$seg_file" "$patient_output_dir/segmentations/${structure_name}.nii.gz"
            fi
        done
    else
        echo "No RTSTRUCT file found for patient: $patient_name"
    fi
}

# Process all patient folders with progress
find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d |
    pv -l -s $(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) |
    while read -r patient_dir; do
        process_patient "$patient_dir"
    done

echo "Conversion completed!"
