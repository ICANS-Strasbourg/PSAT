import logging
from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

def plot_and_save_distribution(df: pd.DataFrame, title: str, filename: str) -> None:
    """Plot age and gender distribution and save to file.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least ``age`` and ``gender`` columns.
    title : str
        Title for the plot.
    filename : str
        Output image file path.
    """
    fig, ax = plt.subplots()
    df.boxplot(column="age", by="gender", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("gender")
    ax.set_ylabel("age")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def create_directory_structure(base_path: Path, subdirs: List[str]) -> None:
    """Create the required directory structure for the new dataset."""
    base_path.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)

def copy_selected_files(
    original_base: Path, new_base: Path, subdirs: List[str], selected_image_ids: List[str]
) -> None:
    """Copy files matching selected image IDs from the original dataset to the new dataset."""
    for subdir in subdirs:
        original_dir = original_base / subdir
        new_dir = new_base / subdir

        if not original_dir.exists():
            logging.warning(f"Directory {original_dir} does not exist. Skipping...")
            continue

        for file_name in original_dir.iterdir():
            if any(file_name.name.startswith(image_id) for image_id in selected_image_ids):
                shutil.copy(file_name, new_dir / file_name.name)

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Paths
    base_path = Path("nnUNet_raw_data_base/Dataset297_TotalSegmentator")
    new_base_path = Path("nnUNet_raw_data_base/Dataset797_TotalSegmentator_plus_TCIA")
    subdirs = ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]

    # Load the selected subset
    subset_csv = Path("resources/TotalSegmentator/dataset_subset.csv")
    selected_subset = pd.read_csv(subset_csv)
    selected_image_ids = selected_subset["image_id"].tolist()

    # Create new dataset directory structure
    create_directory_structure(new_base_path, subdirs)

    # Copy relevant files
    copy_selected_files(base_path, new_base_path, subdirs, selected_image_ids)

    logging.info(f"Subset dataset created successfully at: {new_base_path}")

if __name__ == "__main__":
    main()
