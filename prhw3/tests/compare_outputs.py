import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import parse as parser
from utils import calculate_errors as calc

# ==========================================================
# Compare PA3/PA4 outputs (debug datasets only)
# ==========================================================

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
out_dir  = os.path.join(os.path.dirname(__file__), '..', 'out')
datasets = ['A', 'B', 'C', 'D', 'E', 'F']  # Debug sets only


def compare_output(letter):
    print(f"--- Comparing outputs for dataset {letter} ---")

    # "Ground truth" sample output and generated output
    gt_path = os.path.join(data_dir, f"PA3-{letter}-Debug-Output.txt")
    my_path = os.path.join(out_dir,  f"PA3-{letter}-Debug-Output.txt")

    # Check file existence
    if not os.path.exists(gt_path):
        print(f"Missing ground truth for {letter}: {gt_path}")
        return
    if not os.path.exists(my_path):
        print(f"Missing your output for {letter}: {my_path}")
        return

    # Parse dk, ck from both files
    gt_dk, gt_ck = parser.parse_output(gt_path)
    my_dk, my_ck = parser.parse_output(my_path)

    # Calculate RMSE
    dk_rmse = calc.calculate_error_stats(gt_dk, my_dk)['rms']
    ck_rmse = calc.calculate_error_stats(gt_ck, my_ck)['rms']

    print(f"Dataset {letter} Results")
    print(f"    RMSE dk: {dk_rmse:.6f}")
    print(f"    RMSE ck: {ck_rmse:.6f}\n")


if __name__ == "__main__":
    for letter in datasets:
        compare_output(letter)