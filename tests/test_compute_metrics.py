import os
import sys
import types
import numpy as np
import nibabel as nib

# Ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dummy modules to satisfy imports in compute_metrics
sys.modules.setdefault('p_tqdm', types.SimpleNamespace(p_map=lambda *a, **k: []))
sys.modules.setdefault('surface_distance', types.SimpleNamespace(
    compute_surface_distances=lambda *a, **k: None,
    compute_robust_hausdorff=lambda *a, **k: 0,
))

from scripts.compute_metrics import dice_score, calculate_confidence_interval, reorient_to_ras


def test_dice_score_perfect_match():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1]])
    assert np.isclose(dice_score(y_true, y_pred), 1)


def test_dice_score_partial():
    y_true = np.array([[1, 1], [0, 0]])
    y_pred = np.array([[1, 0], [1, 0]])
    assert np.isclose(dice_score(y_true, y_pred), 0.5)


def test_confidence_interval():
    data = [1, 1, 1, 1]
    mean, lower, upper = calculate_confidence_interval(data, confidence=0.95)
    assert mean == 1
    assert lower <= mean <= upper


def test_reorient_to_ras_identity():
    arr = np.zeros((2, 2, 2))
    affine = np.eye(4)
    img = nib.Nifti1Image(arr, affine)
    ras_img = reorient_to_ras(img)
    assert ras_img.get_fdata().shape == img.get_fdata().shape
