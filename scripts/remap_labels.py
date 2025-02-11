import sys
import nibabel as nib
import numpy as np

if len(sys.argv) != 3:
    print("Usage: remap_labels.py <input_segmentation> <output_segmentation>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Load the segmentation image
img = nib.load(input_file)
data = img.get_fdata().astype(np.int16)

# Define the ROI list (in the order used for segmentation)
roi_list = [
    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach",
    "pancreas", "esophagus", "small_bowel", "duodenum", "urinary_bladder", "prostate",
    "spinal_cord"
]

# Define desired mapping: keys are ROI names, values are target label integers
desired_mapping = {
    "spleen": 1,
    "kidney_right": 2,
    "kidney_left": 3,
    "gallbladder": 4,
    "liver": 5,
    "stomach": 6,
    "pancreas": 7,
    "esophagus": 8,
    "small_bowel": 9,      # corresponds to "Small-Intestine"
    "duodenum": 10,
    "urinary_bladder": 11,  # corresponds to "Bladder"
    "prostate": 12,
    "spinal_cord": 13       # corresponds to "Spinal-Canal"
}

# Get the unique labels (ignoring background 0)
unique_vals = np.unique(data)
unique_vals = unique_vals[unique_vals != 0]
unique_vals = np.sort(unique_vals)

if len(unique_vals) != len(roi_list):
    print("Warning: Expected {} ROI labels, but found {} in the segmentation.".format(len(roi_list), len(unique_vals)))

# Build a remapping dictionary based on the assumed order.
remap_dict = {}
for i, orig_label in enumerate(unique_vals):
    if i < len(roi_list):
        roi = roi_list[i]
        remap_dict[orig_label] = desired_mapping[roi]
    else:
        remap_dict[orig_label] = 0  # fallback to background if extra

# Create a new data array with remapped labels
new_data = np.zeros_like(data, dtype=np.int16)
for orig, new in remap_dict.items():
    new_data[data == orig] = new

# Save the new segmentation with the same header and affine
new_img = nib.Nifti1Image(new_data, img.affine, img.header)
nib.save(new_img, output_file)