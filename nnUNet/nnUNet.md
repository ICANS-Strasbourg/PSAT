# nnUNetv2

## Summary of changes made to nnUNet base implementation and default pipeline

### Preprocessing

The preprocessing pipeline follows the official nnU-Net v2 workflow
(reorientation, resampling, intensity normalisation, etc.).

### Augmentations

PSAT introduces a contraction-based augmentation strategy to expose the network 
to stronger anatomical scaling.  This is achieved by modifying the
`SpatialTransform` in nnU-Net so that random scaling factors are sampled from a
wider range `(0.7, 2.5)` rather than the default `(0.7, 1.4)`.  Mirroring is disabled
for the experiments.

### Training

Several custom trainers are provided to support different experimental setups.
`nnUNetTrainer_400epochs_NoMirroring_Finetune_lr_1e4` and
`nnUNetTrainer_200epochs_NoMirroring_Finetune_lr_1e4` perform fine‑tuning with a
lower learning rate and without mirroring.  The
`nnUNetTrainer_1000epochs_NoMirroring_contraction` variants train from scratch
for 1000 epochs while applying the contraction augmentation described above.

All trainers inherit from `nnUNetTrainer` and thus retain full compatibility
with the nnU-Net v2 command line tools.


You have to place the scripts in your nnUNetv2 library corresponding folders.
