#!/usr/bin/env python3
import os
import argparse
import logging
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def setup_logging():
    """Configure logging output."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def escape_latex(s: str) -> str:
    """Escape underscores for LaTeX output."""
    return s.replace("_", r"\_")

def is_baseline(trainer: str) -> bool:
    """
    Determine if a trainer is considered a baseline model.
    Baseline trainers contain "TotalSegmentator" or "ARTPLAN" in their name.
    """
    return "TotalSegmentator" in trainer or "ARTPLAN" in trainer

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table for Dice comparison across datasets."
    )
    parser.add_argument(
        "--datasets", type=str, default="all",
        help="Comma-separated list of dataset numbers (e.g., '500,67,297') or 'all' to include all."
    )
    return parser.parse_args()

def get_dataset_folders(base_dir: str, selected_datasets: set, dataset_map: dict) -> list:
    """
    Retrieve and filter dataset folders based on selected datasets.
    
    Parameters:
        base_dir (str): The directory containing dataset folders.
        selected_datasets (set): Set of dataset numbers to include (or None for all).
        dataset_map (dict): Mapping of dataset numbers to folder names.
    
    Returns:
        list: List of dataset folder paths.
    """
    all_folders = [
        os.path.join(base_dir, d) 
        for d in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if selected_datasets:
        valid_folders = {dataset_map[ds] for ds in selected_datasets if ds in dataset_map}
        return [d for d in all_folders if os.path.basename(d) in valid_folders]
    return all_folders

def process_dataset_folder(dataset_path: str, results: dict) -> str:
    """
    Process a single dataset folder: read trainer subfolders and update the results dictionary.
    
    Parameters:
        dataset_path (str): Path to the dataset folder.
        results (dict): Nested dictionary to store dice values and later metrics.
    
    Returns:
        str: The dataset name.
    """
    dataset_name = os.path.basename(dataset_path)
    logging.info(f"Processing dataset: {dataset_name}")
    
    trainer_folders = [
        os.path.join(dataset_path, d) 
        for d in os.listdir(dataset_path) 
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    
    for trainer_path in trainer_folders:
        trainer_name = os.path.basename(trainer_path)
        metrics_csv = os.path.join(trainer_path, "patient_wise_metrics.csv")
        if not os.path.exists(metrics_csv):
            logging.warning(f"{metrics_csv} not found. Skipping trainer {trainer_name}.")
            continue
        
        df = pd.read_csv(metrics_csv)
        # Identify dice columns (columns starting with "dice-")
        dice_columns = [col for col in df.columns if col.startswith("dice-")]
        if not dice_columns:
            continue
        
        for dice_col in dice_columns:
            # Crop ROI name to the first 4 characters for brevity
            roi = dice_col.split("-")[1][:4]
            dice_values = df[dice_col].dropna().values
            
            # Initialize the nested dictionary structure as needed
            results.setdefault(trainer_name, {}).setdefault(roi, {})[dataset_name] = {
                "dice": dice_values
            }
    return dataset_name

def compute_statistical_tests(results: dict):
    """
    Compute p-values comparing each non-baseline trainer against the baseline for each ROI and dataset
    using the Mann–Whitney U test for non-normal data.
    Additionally, store the maximum baseline mean Dice score (i.e. the best performing baseline)
    among all baselines.
    Updates the results dictionary in place.
    """
    for trainer, rois in results.items():
        if is_baseline(trainer):
            continue
        for roi, dataset_metrics in rois.items():
            for dataset, metrics in dataset_metrics.items():
                baseline_dice = []
                baseline_means = []
                for baseline_trainer, baseline_rois in results.items():
                    if is_baseline(baseline_trainer) and roi in baseline_rois and dataset in baseline_rois[roi]:
                        b_dice = baseline_rois[roi][dataset]["dice"]
                        baseline_dice.extend(b_dice)
                        baseline_means.append(np.mean(b_dice))
                if baseline_dice and baseline_means:
                    # Compute p-value using the Mann–Whitney U test
                    stat, pvalue = mannwhitneyu(metrics["dice"], baseline_dice, alternative='two-sided')
                    # Get the best performing baseline mean Dice score
                    max_baseline = max(baseline_means)
                    results[trainer][roi][dataset]["pvalue"] = pvalue
                    results[trainer][roi][dataset]["max_baseline"] = max_baseline

def determine_best_scores(results: dict, dataset_names: list, trainers: list, all_rois: list) -> dict:
    """
    For each ROI and dataset, determine which trainer achieved the highest mean Dice score.
    
    Returns:
        dict: A nested dictionary mapping each ROI and dataset to the best trainer.
    """
    best_scores = {roi: {} for roi in all_rois}
    for roi in all_rois:
        for dataset in dataset_names:
            best_score = -1
            best_trainer = None
            for trainer in trainers:
                trainer_data = results.get(trainer, {})
                if roi in trainer_data and dataset in trainer_data[roi]:
                    score = np.mean(trainer_data[roi][dataset]["dice"])
                    if score > best_score:
                        best_score = score
                        best_trainer = trainer
            best_scores[roi][dataset] = best_trainer
    return best_scores

def format_cell(trainer: str, roi: str, dataset_names: list, results: dict, best_scores: dict) -> str:
    """
    Format a table cell for a given trainer and ROI.
    
    Displays mean Dice percentages per dataset, adds a dagger if:
      - The trainer is not a baseline,
      - The p-value is below 0.05, and
      - The mean Dice score (in percentage) exceeds the maximum baseline mean (also converted to percentage),
        i.e. the model improves over every baseline,
      - The dataset is one of ["Dataset067_Pediatric_Internal", "Dataset500_TCIA"].
    
    The best score is bolded.
    """
    cell_values = []
    for dname in dataset_names:
        trainer_data = results.get(trainer, {})
        if roi in trainer_data and dname in trainer_data[roi]:
            rec = trainer_data[roi][dname]
            # Calculate the mean dice and convert to percentage
            if len(rec["dice"]) > 0:
                dice_mean = np.nanmean(rec["dice"]) * 100
            else:
                dice_mean = float('nan')
            # Round the values to the same precision as displayed (nearest integer)
            if not np.isnan(dice_mean):
                rounded_model = round(dice_mean)
                rounded_baseline = round(rec.get("max_baseline", 0) * 100)
                dice_text = f"{rounded_model:.0f}"
                # Add dagger only if the rounded new score is strictly greater than the rounded baseline score.
                if (not is_baseline(trainer) and 
                    rec.get("pvalue") is not None and 
                    rec.get("pvalue") < 0.05 and 
                    rounded_model > rounded_baseline and 
                    dname in ["Dataset067_Pediatric_Internal", "Dataset500_TCIA"]):
                    dice_text += r"$^{\dagger}$"
                # Bold the best score for this ROI and dataset
                if trainer == best_scores.get(roi, {}).get(dname):
                    dice_text = r"\textbf{" + dice_text + "}"
                cell_values.append(dice_text)
            else:
                cell_values.append("-")
        else:
            # If the model does not segment this ROI, indicate with an asterisk
            if "ARTPLAN" in trainer and roi in {"Panc", "Gall"}:
                cell_values.append("*")
            else:
                cell_values.append("-")
    return "/".join(cell_values)

def build_latex_table(results: dict, dataset_names: list, all_rois: list,
                      trainers: list, best_scores: dict, latex_path: str):
    """
    Build the LaTeX table lines and write them to the specified file.
    The table groups trainers into categories and highlights best scores.
    """
    # Reorder datasets as 297, 500, 67 using the mapping folder names
    desired_order = ["Dataset297_TotalSegmentator", "Dataset500_TCIA", "Dataset067_pediatric"]
    dataset_names = [d for d in desired_order if d in dataset_names]

    # Remove the Liver ROI (abbreviated as "Live")
    all_rois = [roi for roi in all_rois if roi.lower() != "live"]

    # Set the caption as requested.
    latex_lines = [
        r"\begin{sidewaystable}[htbp]",
        r"\centering",
        r"\caption{Dice coefficient (\%) comparison across datasets (adult / pediatric / pediatric internal).}"
    ]
    
    # Total number of columns is assumed to be 12 as specified.
    col_spec = "l" + "c" * len(all_rois)
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"\toprule")
    
    # Insert an extra header row for ROI columns using the same formatting as for training types
    header_cells = [r"\rowcolor{gray!30} Trainer"] + [escape_latex(roi) for roi in all_rois]
    latex_lines.append(" & ".join(header_cells) + r" \\")
    latex_lines.append(r"\midrule")
    
    # Categorize trainers based on naming heuristics
    direct_learning = []
    hybrid_learning = []
    transfer_learning = []
    baseline = []
    for trainer in trainers:
        # The categorization logic below is heuristic based on name patterns.
        if trainer.endswith("T_o$") and len(trainer) > 6 and trainer[3] == trainer[6]:
            direct_learning.append(trainer)
        elif trainer.endswith("T_o$") and len(trainer) > 6 and trainer[3] != trainer[6]:
            hybrid_learning.append(trainer)
        elif is_baseline(trainer):
            baseline.append(trainer)
        else:
            transfer_learning.append(trainer)
    
    sorted_categories = [
        ("Baseline", baseline),
        ("Direct Learning", direct_learning),
        ("Hybrid Learning", hybrid_learning),
        ("Transfer Learning", transfer_learning)
    ]
    
    # For alternating row colors in trainer rows.
    row_counter = 0
    for category, trainer_list in sorted_categories:
        if trainer_list:
            # Insert a formatted category header row using the specified style.
            latex_lines.append(r"\midrule")
            latex_lines.append(r"\rowcolor{gray!30}")
            latex_lines.append(r"\multicolumn{12}{c}{\textbf{" + category + r"}} \\")
            latex_lines.append(r"\midrule")
            for trainer in trainer_list:
                row_prefix = ""
                if row_counter % 2 == 0:
                    row_prefix = r"\rowcolor{gray!10} "
                row_cells = [row_prefix + trainer]  # Trainer names are not escaped
                for roi in all_rois:
                    cell_text = format_cell(trainer, roi, dataset_names, results, best_scores)
                    row_cells.append(cell_text)
                latex_lines.append(" & ".join(row_cells) + r" \\")
                row_counter += 1
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\footnotesize",
        r"\footnotesize \textbf{Bold} marks the best DSC for each ROI/dataset. ROI names are abbreviated: Blad (Bladder), Duod (Duodenum), Esop (Esophagus), Gall (Gallbladder), Kidn (Kidney), Panc (Pancreas), Pros (Prostate), Smal (Small Intestine), Spin (Spinal Canal), Sple (Spleen), Stom (Stomach). $^{\dagger}$ indicates a statistically significant improvement over the best performing baseline ($p < 0.05$). “-” indicates that the physician reference is unavailable and “*” that the model does not segment this ROI.",
        r"\end{tablenotes}",
        r"\end{sidewaystable}"
    ])
    
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    logging.info(f"LaTeX dice table saved to {latex_path}")

def main():
    setup_logging()
    args = parse_arguments()
    
    # Determine which datasets to include
    selected_datasets = None if args.datasets.lower() == "all" else set(args.datasets.split(","))
    
    base_dir = "nnUNet_predict"
    output_folder = "analysis_output"
    latex_folder = os.path.join(output_folder, "latex_tables")
    os.makedirs(latex_folder, exist_ok=True)
    tex_filename = os.path.join(latex_folder, "dice_comparison.tex")
    
    # Mapping dataset numbers to folder names
    dataset_map = {
        "297": "Dataset297_TotalSegmentator",
        "500": "Dataset500_TCIA",
        "67": "Dataset067_Pediatric_Internal"
    }
    
    dataset_folders = get_dataset_folders(base_dir, selected_datasets, dataset_map)
    
    results = {}
    dataset_names = []
    for dataset_path in dataset_folders:
        dataset_name = process_dataset_folder(dataset_path, results)
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
    
    dataset_names = list(set(dataset_names))
    
    # Determine the union of all ROIs across trainers
    all_rois = sorted({roi for trainer_data in results.values() for roi in trainer_data.keys()})
    
    # Sorted list of trainer names
    trainers = sorted(list(results.keys()))
    
    # Compute statistical tests comparing each trainer to the baseline
    compute_statistical_tests(results)
    
    # Determine the best trainer (highest mean Dice) for each ROI and dataset
    best_scores = determine_best_scores(results, dataset_names, trainers, all_rois)
    
    # Build and save the LaTeX table
    build_latex_table(results, dataset_names, all_rois, trainers, best_scores, tex_filename)

if __name__ == "__main__":
    main()
