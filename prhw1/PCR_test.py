import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import plot as plotter
from utils import point_cloud_registration as pcr
from utils import convert as cvt
from utils import parse as parser

"""
Unit Test for PCR
1. Creates a random point cloud
2. Copies random point cloud
2. Applied a known transformation to one
3. Uses PCR to estimate the transformation
4. Compares the estimated transformation to the known transformation
"""
def random_pcr_test():
    # 1. Create a random point cloud
    num_points = 100
    np.random.seed(42)  
    original_points = np.random.rand(num_points, 3) * 100 

    # 2. Copy the random point cloud
    transformed_points = original_points.copy()

    # 3. Apply a known HTM transformation 
    angle = np.radians(30)  
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Create known HTM (4x4 homogeneous transformation matrix)
    known_HTM = np.eye(4)
    known_HTM[:3, :3] = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    known_HTM[:3, 3] = np.array([20, 30, 10])  # translation
    
    # Apply HTM transformation to points
    # Convert to homogeneous coordinates (add ones column)
    transformed_points_hom = np.column_stack([transformed_points, np.ones(num_points)])
    # Apply transformation: (N×4) @ (4×4).T = (N×4)
    transformed_points_hom = transformed_points_hom @ known_HTM.T
    # Extract 3D coordinates
    transformed_points = transformed_points_hom[:, :3]

    # 4. Use PCR to estimate the transformation (returns HTM)
    estimated_HTM = pcr.point_cloud_registration(original_points, transformed_points)

    # Print known and estimated HTMs
    print("Known HTM:\n", known_HTM)
    print("\nEstimated HTM:\n", estimated_HTM)
    
    # Compare the transformations
    print("\nHTM Difference (should be close to zero):\n", known_HTM - estimated_HTM)
    
    # Extract and compare individual components
    print("\nKnown Rotation Matrix:\n", known_HTM[:3, :3])
    print("Estimated Rotation Matrix:\n", estimated_HTM[:3, :3])
    
    print("\nKnown Translation Vector:\n", known_HTM[:3, 3])
    print("Estimated Translation Vector:\n", estimated_HTM[:3, 3])

    # 5. Re-apply the estimated HTM to validate PCR results
    original_points_hom = np.column_stack([original_points, np.ones(num_points)])
    retransformed_points_hom = original_points_hom @ estimated_HTM.T
    retransformed_points = retransformed_points_hom[:, :3]
    
    # Calculate reconstruction error
    reconstruction_error = np.linalg.norm(transformed_points - retransformed_points, axis=1)
    print(f"\nReconstruction Error Statistics:")
    print(f"Mean: {np.mean(reconstruction_error):.6f}")
    print(f"Max: {np.max(reconstruction_error):.6f}")
    print(f"RMS: {np.sqrt(np.mean(reconstruction_error**2)):.6f}")

    # 6. Visualize all three point clouds
    fig, ax = plotter.plot_data_2(original_points, transformed_points, "Original Points", "Known Transform", number_points=False)
    plt.show()
    
    fig, ax = plotter.plot_data_2(transformed_points, retransformed_points, "Known Transform", "PCR Re-transform", number_points=False)
    plt.show()

def register_em_opt_trackers():
    path = "prhw1/data/pa1-debug-a-calbody.txt"
    D_frames, A_frames, C_frames = parser.parse_calreadings(path)
    print("D_frames shape:", D_frames.shape)
    print("A_frames shape:", A_frames.shape)
    print("C_frames shape:", C_frames.shape)


if __name__ == "__main__":
    random_pcr_test()