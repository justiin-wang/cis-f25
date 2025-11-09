import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import parse as parser
from utils import calculate_errors as calc

# TODO: Add comemnts

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
out_dir = os.path.join(os.path.dirname(__file__), '..', 'out')
datasets = ['a','b','c','d','e','f']
#datasets = ['a']


def compare_output_1(letter):
    # Ground truth and generated output 
    gt_path = os.path.join(data_dir, f"pa2-debug-{letter}-output1.txt")
    my_path = os.path.join(out_dir,  f"pa2-debug-{letter}-output1.txt")

    # Debug filepaths   
    if not os.path.exists(gt_path):
        print(f"Missing ground truth for {letter}: {gt_path}")
        return
    if not os.path.exists(my_path):
        print(f"Missing my output for {letter}: {my_path}")
        return

    gt_C, gt_pem, gt_popt = parser.parse_output_1(gt_path)
    my_C, my_pem, my_popt = parser.parse_output_1(my_path)

    # Frame errors
    rms_per_frame = [calc.calculate_error_stats(gt_C[k], my_C[k])['rms'] for k in range(len(gt_C))]
    mean_rms = np.mean(rms_per_frame)

    em_err = np.linalg.norm(gt_pem - my_pem)
    opt_err = np.linalg.norm(gt_popt - my_popt)

    print(f"Pivot EM error: {em_err:.6f}")
    print(f"Pivot OPT error: {opt_err:.6f}")
    print(f"Mean RMS Registration error: {mean_rms:.6f}")

def compare_output_2(letter):
    # Ground truth and generated output 
    gt_path = os.path.join(data_dir, f"pa2-debug-{letter}-output2.txt")
    my_path = os.path.join(out_dir,  f"pa2-debug-{letter}-output2.txt")

    # Debug filepaths
    if not os.path.exists(gt_path):
        print(f"Missing ground truth for {letter}: {gt_path}")
        return
    if not os.path.exists(my_path):
        print(f"Missing my output for {letter}: {my_path}")
        return

    gt_tips = parser.parse_output_2(gt_path)
    my_tips = parser.parse_output_2(my_path)

    rms_per_frame = [calc.calculate_error_stats(gt_tips[k:k+1], my_tips[k:k+1])['rms'] for k in range(len(gt_tips))]
    mean_rms = np.mean(rms_per_frame)

    print(f"Mean RMS Tip Position error: {mean_rms:.6f}\n")

if __name__ == "__main__":
    for letter in datasets:
        print(f"Dataset {letter}:")
        compare_output_1(letter)
        compare_output_2(letter)
