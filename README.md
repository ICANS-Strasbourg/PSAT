# PSAT

This repository contains the code and configuration files for PSAT (Pediatric Segmentation Approaches via Adult Augmentations and Transfer Learning).

## Overview

PSAT addresses pediatric segmentation challenges by combining:
- **Training Plans:** Derived from adult, pediatric, or mixed data ($P_a$, $P_p$, $P_m$).
- **Learning Sets:** Adult-only, pediatric-only, or mixed ($S_a$, $S_p$, $S_m$).
- **Augmentations:** Default ($A_d$) and contraction-based ($A_c$) strategies.
- **Transfer Learning:** Direct inference ($T_o$), fine-tuning ($T_p$), or continual learning ($T_m$).

<img src="resources/images/PSAT_overview.png" alt="PSAT Overview" style="width:80%; max-width:1000px; display:block; margin: 0 auto;">

## Citation
If you use this code, please cite our paper:

```
@article{kirscher2025psat,
  title={PSAT: Pediatric Segmentation Approaches via Adult Augmentations and Transfer Learning},
  author={T. Kirscher et al},
  journal={MICCAI},
  year={2025},
  note={arXiv:xxxx.xxxxx}
}
```

## Quickstart

Install dependencies:
```bash
pip install -r requirements.txt
```

Run metrics evaluation (example):
```bash
python scripts/compute_metrics.py <ground_truth_dir> <predictions_dir>
```
Replace `<ground_truth_dir>` and `<predictions_dir>` with your folder paths containing NIfTI files.

## Dependencies
- nibabel
- numpy
- pandas
- p_tqdm
- scipy
- surface-distance

(See `requirements.txt` for full list.)

## Documentation

- [nnUNet](nnUNet/nnUNet.md)
- [Dataloading](nnUNet/dataloading/dataloading.md)
- [Preprocessing](nnUNet/preprocessing/preprocessing.md)
- [Training](nnUNet/training/training.md)
- [Resources](resources/resources.md)
- [Scripts](scripts/scripts.md)

## Running Tests

Install dependencies listed in `requirements.txt` and run:

```bash
pytest -q
```
