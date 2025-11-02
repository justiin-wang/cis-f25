import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils.bpoly import BPoly    
from utils import calculate_errors as calcerr

# Verify that BPoly fitting and applying works for a non-distorted set
def identity_bpoly_test(order):
    # Generate random point cloud
    np.random.seed(67)
    N = 1000
    tol = 1e-8
    X = np.random.rand(N, 3)

    # Fit and apply BPoly to itself
    bpoly = BPoly(order=order)
    bpoly.fit(X, X)
    X_pred = bpoly.apply(X)

    # Calculate errors using calcerr
    stats = calcerr.calculate_error_stats(X, X_pred)
    return stats


# Create a random polynomial, fit it with BPoly, and verify we can recover it
def random_bpoly_test(order):
    np.random.seed(67)
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

# TODO: load a data set and test fit and errors independently
# def dataset_test(order):


if __name__ == "__main__":
    # Test identity property
    for order in range(1, 10):
        print(f"Testing BPoly order: {order}")
        error_stats = identity_bpoly_test(order=order)
        calcerr.print_error_stats(error_stats)
        if error_stats['rms'] < 1e-8 and error_stats['max'] < 1e-8:
            print("PASS\n")
        else:
            print("FAIL\n")

    
    # # Play around with order to underfitting -> good fit
    # for i in range (1, 10):
    #     print(f"Testing BPoly order: {i}")
    #     error_stats = random_bpoly_test(order=i)
    #     print("BPoly Fit Error Stats:\n")
    #     calcerr.print_error_stats(error_stats)
    #     if (error_stats['rms'] >= 0.02):
    #         print("FAIL")
    #     else:
    #         print("PASS")
        

