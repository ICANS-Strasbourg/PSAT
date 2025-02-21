# Scripts

This document outlines the utility scripts available in the repository. 

- `compute_metrics.py`  
  Evaluates segmentation results by computing Dice score, Hausdorff distance, and surface distances.\
  Outputs `patient_wise_metrics.csv` and aggredated `evaluation_results.csv`.

- `convert_TCIA_to_nnunet.py`  
  Converts the TCIA pediatric dataset into the nnU-Net compliant format.

- `create_totalseg_subset.py`  
  Creates a balanced subset of the TotalSegmentator dataset for fingerprinting (P_m) on an equal number of pediatric and adult cases.

- `remap_labels.py`  
  Remaps segmentation labels to adhere to our unified labeling scheme.

- `run_TotalSegmentator.sh`  
  Executes the TotalSegmentator pipeline for baseline inference on various test sets.

- `get_results.py`  
  Processes segmentation metrics from multiple models across different datasets, performs statistical comparisons against baseline models, identifies the best-performing scores per region of interest, and outputs the results as a formatted LaTeX table.


## Installation

To evaluate segmentation results using `compute_metrics.py`, you need to install the `surface-distance` package. You can install it via pip:

```sh
$ git clone https://github.com/deepmind/surface-distance.git
$ pip install surface-distance/
