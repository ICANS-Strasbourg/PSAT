import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.get_results import escape_latex, is_baseline, compute_statistical_tests, determine_best_scores
import numpy as np


def test_escape_latex():
    assert escape_latex('foo_bar') == 'foo\\_bar'


def test_is_baseline():
    assert is_baseline('Something_TotalSegmentator')
    assert not is_baseline('OtherModel')


def test_compute_statistical_tests_and_best_scores():
    results = {
        'Baseline_TotalSegmentator': {'ROI': {'dataset': {'dice': [0.8, 0.9]}}},
        'NewModel': {'ROI': {'dataset': {'dice': [0.95, 0.96]}}},
    }
    compute_statistical_tests(results)
    assert 'pvalue' in results['NewModel']['ROI']['dataset']
    assert 'max_baseline' in results['NewModel']['ROI']['dataset']

    best = determine_best_scores(results, ['dataset'], ['Baseline_TotalSegmentator', 'NewModel'], ['ROI'])
    assert best['ROI']['dataset'] == 'NewModel'
