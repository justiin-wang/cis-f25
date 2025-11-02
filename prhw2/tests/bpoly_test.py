import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils.bpoly import BPoly    
from utils import calculate_errors as calcerr

# Create a random polynomial, fit it with BPoly, and verify we can recover it
def random_bpoly_test(order):
    np.random.seed(42)
    N = 200

    measured = np.random.rand(N, 3)

    # Create a known polynomial mapping
    expected = np.empty_like(measured)
    expected[:, 0] = measured[:, 0] + 0.1 * np.sin(2 * np.pi * measured[:, 1])
    expected[:, 1] = measured[:, 1] + 0.1 * np.cos(2 * np.pi * measured[:, 2])
    expected[:, 2] = measured[:, 2] + 0.1 * np.sin(2 * np.pi * measured[:, 0])

    # Fit and apply BPoly
    bpoly = BPoly(order=order)
    bpoly.fit(expected, measured)
    predicted = bpoly.apply(measured)

    # Calculate errors
    stats = calcerr.calculate_error_stats(expected, predicted)
    return stats

if __name__ == "__main__":
    # Play around with order to underfitting -> good fit
    for i in range (1, 10):
        print(f"Testing BPoly order: {i}")
        error_stats = random_bpoly_test(order=i)
        print("BPoly Fit Error Stats:\n")
        calcerr.print_error_stats(error_stats)
        if (error_stats['rms'] >= 0.02):
            print("FAIL")
        else:
            print("PASS")
        

