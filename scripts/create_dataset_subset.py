import os
import shutil
import pandas as pd

# Paths
base_path = "/mnt/DATA/DATA/nnUNet_raw_data_base/Dataset297_TotalSegmentator"  # Original dataset path
new_base_path = "/mnt/DATA/DATA/nnUNet_raw_data_base/Dataset797_TotalSegmentator_plus_TCIA"  # New subset dataset path

# Subdirectories to handle
subdirs = ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]

# Load the selected subset
selected_subset = pd.read_csv("selected_subset.csv")

# Extract the relevant image IDs
selected_image_ids = selected_subset["image_id"].tolist()

# Create new dataset directory structure
os.makedirs(new_base_path, exist_ok=True)
for subdir in subdirs:
    os.makedirs(os.path.join(new_base_path, subdir), exist_ok=True)

# Copy relevant files
for subdir in subdirs:
    original_dir = os.path.join(base_path, subdir)
    new_dir = os.path.join(new_base_path, subdir)

    # Check if directory exists in the original dataset
    if not os.path.exists(original_dir):
        print(f"Directory {original_dir} does not exist. Skipping...")
        continue

    for file_name in os.listdir(original_dir):
        # Match file names with selected image IDs
        for image_id in selected_image_ids:
            if file_name.startswith(image_id):
                src = os.path.join(original_dir, file_name)
                dst = os.path.join(new_dir, file_name)
                shutil.copy(src, dst)

print("Subset dataset created successfully at:", new_base_path)
