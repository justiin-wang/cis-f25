import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import parse as parser
from utils import calculate_errors as calc

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
out_dir  = os.path.join(os.path.dirname(__file__), '..', 'out')
datasets = ['A', 'B', 'C', 'D', 'E', 'F']  # Debug sets only

# Yah sam e old same old stuff with compare
def compare_output(letter):
    print(f"--- Comparing outputs for dataset {letter} ---")

    gt_path = os.path.join(data_dir, f"PA4-{letter}-Debug-Output.txt")
    my_path = os.path.join(out_dir,  f"PA4-{letter}-Debug-Output.txt")

    if not os.path.exists(gt_path):
        print(f"Missing ground truth for {letter}: {gt_path}")
        return
    if not os.path.exists(my_path):
        print(f"Missing your output for {letter}: {my_path}")
        return

    # Parse s_k and c_k 
    gt_s, gt_ck = parser.parse_output(gt_path)
    my_s, my_ck = parser.parse_output(my_path)

    # Compare s_k and c_k
    sk_rmse = calc.calculate_error_stats(gt_s, my_s)['rms']
    ck_rmse = calc.calculate_error_stats(gt_ck, my_ck)['rms']

    print(f"Dataset {letter} Results")
    print(f"    RMSE sk: {sk_rmse:.6f}")
    print(f"    RMSE ck: {ck_rmse:.6f}\n")


if __name__ == "__main__":
    for letter in datasets:
        compare_output(letter)