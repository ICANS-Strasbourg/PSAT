# Resources

## TCIA Dataset: Pediatric CT Segmentation

- **Source:** [TCIA Pediatric CT Segmentation Collection](https://www.cancerimagingarchive.net/collection/pediatric-ct-seg/)
- **Description:** A collection of pediatric CT scans with expert-annotated organ and tumor segmentations, suitable for training and evaluating medical image segmentation models.
- **Metadata:**
  - `TCIA/meta.csv`: Contains patient IDs, scan information, and basic demographic data for each case in the dataset.

## TotalSegmentator Dataset

- **Source:** [TotalSegmentator on Zenodo](https://zenodo.org/records/10047292)
- **Description:** A large-scale dataset of adult CT scans with comprehensive multi-organ segmentations, designed for general-purpose medical image segmentation tasks.
- **Metadata:**
  - `TotalSegmentator/meta.csv`: Contains scan identifiers, acquisition parameters, and summary statistics for each scan in the dataset.

## Using Pretrained nnU-Net Models

We provide two nnU-Net v2 model checkpoints:
- `mixed_model_continual_learning.zip`
- `pure_pediatric_model.zip`

You can use these as pretrained weights for inference or further fine-tuning with nnU-Net v2.

### 1. Download the Model Weights

Go to the [GitHub Releases](https://github.com/ICANS-Strasbourg/PSAT/releases) page and download the desired zip files. Place them in a directory of your choice (e.g., `resources/checkpoints/`).

### 2. Install the Pretrained Model

Use the nnU-Net v2 utility to install the model from the zip file:

```bash
# For the mixed model:
nnUNetv2_install_pretrained_model_from_zip resources/checkpoints/mixed_model_continual_learning.zip

# For the pure pediatric model:
nnUNetv2_install_pretrained_model_from_zip resources/checkpoints/pure_pediatric_model.zip
```

### 3. Run Inference with the Installed Model

After installation, you can run inference using nnU-Net v2. For example:

```bash
nnUNetv2_predict -d <DATASET_ID> -i <INPUT_FOLDER> -o <OUTPUT_FOLDER> -c <CONFIGURATION> -f <FOLD>
```
- `<DATASET_ID>`: The dataset number or name (e.g., `297` for TotalSegmentator, `797` for the mixed dataset).
- `<INPUT_FOLDER>`: Folder with your images (NIfTI format).
- `<OUTPUT_FOLDER>`: Where predictions will be saved.
- `<CONFIGURATION>`: Model configuration (e.g., `3d_fullres`, `2d`).
- `<FOLD>`: Fold number (usually `0`, or `all` for all folds).

**Example:**
```bash
nnUNetv2_predict -d 797 -i imagesTs/ -o predictions/ -c 3d_fullres -f all
```