# FILE: tests/test_create_totalseg_subset.py
import os
import sys
import pandas as pd
import pytest

# Ensure package import works when tests are run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.create_totalseg_subset import plot_and_save_distribution, create_directory_structure, copy_selected_files

class TestCreateTotalSegSubset:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, tmp_path):
        # Create a sample dataframe
        self.data = pd.DataFrame({
            'age': [25, 35, 45, 55, 65, 75],
            'gender': ['M', 'F', 'M', 'F', 'M', 'F']
        })
        self.filename = tmp_path / 'test_distribution.png'
        yield
        # Remove the file after test if it exists
        if self.filename.exists():
            self.filename.unlink()

    def test_plot_and_save_distribution(self):
        # Call the function to plot and save the distribution
        plot_and_save_distribution(self.data, "Test Title", str(self.filename))
        # Check if the file is created
        assert self.filename.exists()

    def test_create_directory_structure(self, tmp_path):
        subdirs = ["imagesTr", "labelsTr"]
        create_directory_structure(tmp_path, subdirs)
        for subdir in subdirs:
            assert (tmp_path / subdir).exists()
            assert (tmp_path / subdir).is_dir()

    def test_copy_selected_files(self, tmp_path):
        # Setup original and new base directories
        original_base = tmp_path / "original"
        new_base = tmp_path / "new"
        subdirs = ["imagesTr", "labelsTr"]
        image_ids = ["img001", "img002"]
        create_directory_structure(original_base, subdirs)
        create_directory_structure(new_base, subdirs)
        # Create files in original_base
        for subdir in subdirs:
            for img_id in image_ids:
                file_path = original_base / subdir / f"{img_id}_something.nii.gz"
                file_path.write_text("dummy data")
            # Add a file that should not be copied
            (original_base / subdir / "otherfile.nii.gz").write_text("not copied")
        # Copy selected files
        copy_selected_files(original_base, new_base, subdirs, image_ids)
        # Check that only the correct files are copied
        for subdir in subdirs:
            for img_id in image_ids:
                assert (new_base / subdir / f"{img_id}_something.nii.gz").exists()
            assert not (new_base / subdir / "otherfile.nii.gz").exists()