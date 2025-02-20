# FILE: tests/test_create_totalseg_subset.py
import unittest
import pandas as pd
import os
from scripts.create_totalseg_subset import plot_and_save_distribution

class TestCreateTotalSegSubset(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe
        self.data = pd.DataFrame({
            'age': [25, 35, 45, 55, 65, 75],
            'gender': ['M', 'F', 'M', 'F', 'M', 'F']
        })
        self.filename = 'test_distribution.png'

    def test_plot_and_save_distribution(self):
        # Call the function to plot and save the distribution
        plot_and_save_distribution(self.data, "Test Title", self.filename)
        
        # Check if the file is created
        self.assertTrue(os.path.exists(self.filename))

    def tearDown(self):
        # Remove the file after test
        if os.path.exists(self.filename):
            os.remove(self.filename)

if __name__ == '__main__':
    unittest.main()