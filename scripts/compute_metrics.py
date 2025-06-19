"""
Adapted from `evaluate.py` in the TotalSegmentator repository:
https://github.com/wasserth/TotalSegmentator

Key enhancements include:
- Integration of DeepMind's `surface-distance` package for computing Dice scores
  and Hausdorff distances (https://github.com/deepmind/surface-distance).
- Dynamic retrieval of pixel spacing to support varying image resolutions.
- Reorientation checks to ensure proper alignment and spacing of NIfTI images.

Usage:
    python scripts/compute_metrics.py <ground_truth_dir> <predictions_dir>

Arguments:
    ground_truth_dir: Directory containing ground truth NIfTI files (*.nii.gz)
    predictions_dir: Directory containing predicted NIfTI files (*.nii.gz)

Expected file format:
    - Each subject should have a file named <subject>.nii.gz in both directories.
    - Files must be 3D or 4D NIfTI images with integer labels.

Dependencies:
    - nibabel
    - numpy
    - pandas
    - p_tqdm
    - scipy
    - surface-distance
"""

import sys
from pathlib import Path
from functools import partial
import logging

import numpy as np
import nibabel as nib
import pandas as pd
from p_tqdm import p_map
from scipy.stats import sem, t

from surface_distance import (
    compute_surface_distances,
    compute_robust_hausdorff,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def dice_score(y_true, y_pred):
    """
    Binary Dice score. Same results as sklearn f1 binary.
    Args:
        y_true (np.ndarray): Binary ground truth mask.
        y_pred (np.ndarray): Binary predicted mask.
    Returns:
        float: Dice coefficient.
    """
    intersect = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    f1 = (2 * intersect) / (denominator + 1e-6)
    return f1


def reorient_to_ras(img):
    """Reorient the image to RAS (Right-Anterior-Superior) orientation using nibabel."""
    return nib.as_closest_canonical(img)


def calc_metrics(subject, gt_dir=None, pred_dir=None, class_map=None):
    # Load ground truth and prediction images for a subject
    try:
        gt_img = nib.load(gt_dir / f"{subject}.nii.gz")
        pred_img = nib.load(pred_dir / f"{subject}.nii.gz")

        # Reorient images to RAS
        gt_img = reorient_to_ras(gt_img)
        pred_img = reorient_to_ras(pred_img)

        gt_all = gt_img.get_fdata()
        pred_all = pred_img.get_fdata()

        # Ensure voxel spacing is taken into account
        voxel_spacing = gt_img.header.get_zooms()

    except Exception as e:
        logging.error(f"Error loading data for subject {subject}: {e}")
        return None

    r = {"subject": subject}
    for idx, roi_name in class_map.items():
        gt = gt_all == idx
        pred = pred_all == idx

        # Handle cases where ground truth or prediction is missing for a class
        if gt.max() > 0 and pred.max() == 0:
            r[f"dice-{roi_name}"] = 0
            r[f"hausdorff-{roi_name}"] = 0
        elif gt.max() > 0:
            r[f"dice-{roi_name}"] = dice_score(gt, pred)
            try:
                sd = compute_surface_distances(gt, pred, voxel_spacing)
                r[f"hausdorff-{roi_name}"] = compute_robust_hausdorff(sd, 95.0)
            except Exception as e:
                logging.error(
                    f"Error computing surface distances for {roi_name} in subject {subject}: {e}"
                )
                r[f"hausdorff-{roi_name}"] = np.NaN
        else:
            r[f"dice-{roi_name}"] = np.NaN
            r[f"hausdorff-{roi_name}"] = np.NaN
    return r


def calculate_confidence_interval(data, confidence=0.95):
    # Ensure data contains only numeric values
    data = [x for x in data if isinstance(x, (int, float))]
    n = len(data)
    mean = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, mean - h, mean + h


if __name__ == "__main__":
    """
    Calculate Dice score and Hausdorff distance for your nnU-Net predictions.

    Example usage:
        python scripts/compute_metrics.py <ground_truth_dir> <predictions_dir>

    See the top-level docstring for more details.
    """
    # Parse input arguments
    gt_dir = Path(sys.argv[1])
    pred_dir = Path(sys.argv[2])

    logging.info(f"Ground truth directory: {gt_dir}")
    logging.info(f"Predictions directory: {pred_dir}")

    class_map = {
        1: "Spleen",
        2: "Kidney-Right",
        3: "Kidney-Left",
        4: "Gall-Bladder",
        5: "Liver",
        6: "Stomach",
        7: "Pancreas",
        8: "Esophagus",
        9: "Small-Intestine",
        10: "Duodenum",
        11: "Bladder",
        12: "Prostate",
        13: "Spinal-Canal",
    }

    subjects = [x.stem.split(".")[0] for x in gt_dir.glob("*.nii.gz")]
    logging.info(f"Subjects found: {subjects}")

    if not subjects:
        logging.error(
            "No subjects found. Please check the ground truth directory and file naming convention."
        )
        sys.exit(1)

    # Use multiple threads to calculate the metrics
    res = p_map(
        partial(calc_metrics, gt_dir=gt_dir, pred_dir=pred_dir, class_map=class_map),
        subjects,
        num_cpus=8,
        disable=True,
    )
    res = [r for r in res if r is not None]  # Filter out None results
    res_df = pd.DataFrame(res)

    # Save patient-wise metrics
    res_df.to_csv(pred_dir / "patient_wise_metrics.csv", index=False)
    logging.info(
        f"Patient-wise metrics saved to {pred_dir / 'patient_wise_metrics.csv'}"
    )

    results = []
    for metric in ["dice", "hausdorff"]:
        for roi_name in class_map.values():
            row_wo_nan = res_df[f"{metric}-{roi_name}"].dropna()
            mean, lower, upper = calculate_confidence_interval(row_wo_nan)
            results.append(
                {
                    "ROI": roi_name,
                    "Metric": metric,
                    "Mean": mean,
                    "Lower CI": lower,
                    "Upper CI": upper,
                    "n_samples": len(row_wo_nan),
                }
            )
            logging.info(
                f"{roi_name} {metric}: Mean={mean:.3f}, Lower CI={lower:.3f}, Upper CI={upper:.3f}, n_samples={len(row_wo_nan)}"
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(pred_dir / "evaluation_results.csv", index=False)
    logging.info(f"Results saved to {pred_dir / 'evaluation_results.csv'}")
