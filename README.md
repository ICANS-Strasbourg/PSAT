# PSAT

This repository contains the code and configuration files for PSAT (Pediatric Segmentation Approaches via Adult Augmentations and Transfer).

## Overview

PSAT addresses pediatric segmentation challenges by combining:
- **Training Plans:** Derived from adult, pediatric, or mixed data ($P_a$, $P_p$, $P_m$).
- **Learning Sets:** Adult-only, pediatric-only, or mixed ($S_a$, $S_p$, $S_m$).
- **Augmentations:** Default ($A_d$) and contraction-based ($A_c$) strategies.
- **Transfer Learning:** Direct inference ($T_o$), fine-tuning ($T_p$), or continual learning ($T_m$).

<img src="resources/images/PSAT_overview.png" alt="PSAT Overview" style="width:80%; max-width:1000px; display:block; margin: 0 auto;">

## Documentation

- [nnUNet](nnUNet/nnUNet.md)
- [Dataloading](nnUNet/dataloading/dataloading.md)
- [Preprocessing](nnUNet/preprocessing/preprocessing.md)
- [Training](nnUNet/training/training.md)
- [Notebooks](notebooks/notebooks.md)
- [Resources](resources/resources.md)
- [Scripts](scripts/scripts.md)
