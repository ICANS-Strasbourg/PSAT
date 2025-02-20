# Scripts

This document outlines the utility scripts available in the repository. 

- `compute_metrics.py`  
  Evaluates segmentation results by computing Dice score, Hausdorff distance, and surface distances.

- `convert_TCIA_to_nnunet.py`  
  Converts the TCIA pediatric dataset into the nnU-Net compliant format.

- `create_totalseg_subset.py`  
  Creates a balanced subset of the TotalSegmentator dataset for fingerprinting (P_m) on an equal number of pediatric and adult cases.

- `remap_labels.py`  
  Remaps segmentation labels to adhere to our unified labeling scheme.

- `run_TotalSegmentator.sh`  
  Executes the TotalSegmentator pipeline for baseline inference on various test sets.

## Installation

To evaluate segmentation results using `compute_metrics.py`, you need to install the `surface-distance` package. You can install it via pip:

```sh
$ git clone https://github.com/deepmind/surface-distance.git
$ pip install surface-distance/
