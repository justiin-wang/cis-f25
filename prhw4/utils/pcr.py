import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import calculate_errors as calcerr

# Implementation of Kabsch point cloud registration algorithm 
# Assumes rigid, no scale, and point correspondence
def point_cloud_registration(a, b):
   
    # Verify input dims and mininum points
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 3
    n = a.shape[0]
    assert n >= 3

    # Find centroids and center two point clouds
    ca = a.mean(axis=0) # n, row by row mean
    cb = b.mean(axis=0)
    A = a - ca
    B = b - cb

    # Cross-covariance matrix
    H = A.T @ B / n 

    # SVD to solve for rotation
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Comput rotation indep
    D = np.eye(3)
    # Caveat of Kabsch PCR: ensure rotation belongs to SO(3)
    if np.linalg.det(V @ U.T) < 0: 
       D[2, 2] = -1.0
    # PA 1 grading alluded to this edge case in case of degenerate data
    if np.linalg.det(V @ U.T) == 0:
        raise ValueError("Singular matrix encountered in SVD for point cloud registration.")
    R = V @ D @ U.T

    # Computer translation
    p = cb - (R @ ca)

    # Reconstruct HTM
    F = np.eye(4)
    F[:3, :3] = R  
    F[:3, 3] = p

    return F

def random_pcr_test():
    # Generate random PC
    num_points = 100
    seed = 67 
    rand = np.random.default_rng(seed)
    original_points = rand.random((num_points, 3)) * 100.0
    transformed_points = original_points.copy()

    # Create random HTM
    angle_deg = rand.uniform(-180.0, 180.0)
    translation = rand.uniform(-100.0, 100.0, 3)
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]])
    t = np.array(translation)
    known_HTM = np.eye(4)
    known_HTM[:3, :3] = R
    known_HTM[:3, 3] = t
    print(f"Known HTM: \n{known_HTM}")

    # Apply transform to the copy
    pts_h = np.column_stack([transformed_points, np.ones(num_points)])
    transformed_points = (pts_h @ known_HTM.T)[:, :3]

    # Estimate HTM mapping original -> transformed using the calibrator object
    estimated_HTM = point_cloud_registration(original_points, transformed_points)
    print(f"Estimated HTM: \n{estimated_HTM}")

    # Check results
    assert np.allclose(estimated_HTM, known_HTM, atol=1e-6)
    print("PCR test pass")

    return calcerr.calculate_error_transformation(known_HTM, estimated_HTM)

if __name__ == "__main__":
    # Run random PCR test
    diff_HTM, angle_error, translation_error = random_pcr_test()
