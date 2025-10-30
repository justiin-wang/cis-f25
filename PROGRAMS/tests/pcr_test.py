import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from utils.calibrator import ProbeCalibration
from utils import calculate_errors as calcerr
from utils import parse as parser
from utils import plot as plotter

tool = ProbeCalibration("test")

def find_expected_calibration_object(calbody_path, calreadings_path):
    d_known, a_known, c_known = parser.parse_calbody(calbody_path)
    D_frames, A_frames, C_frames = parser.parse_calreadings(calreadings_path)
    C_expected_pc = [] # k C_expected point cloud

    for k in range(len(D_frames)):
        D_measured = D_frames[k]
        A_measured = A_frames[k]

        F_D = tool.point_cloud_registration(d_known, D_measured)
        F_A = tool.point_cloud_registration(a_known, A_measured)

        F_C = np.linalg.inv(F_D) @ F_A # F_C = F_D^-1 * F_A

        C_expected = (F_C[:3, :3] @ c_known.T + F_C[:3, 3:4]).T # Apply the transformation to vector c_known
        C_expected_pc.append(C_expected)

    C_expected_pc = np.array(C_expected_pc)
    C_frames = np.array(C_frames)

    return C_expected_pc, C_frames

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
    calib = ProbeCalibration(name="PCR_test")
    estimated_HTM = calib.point_cloud_registration(original_points, transformed_points)
    print(f"Estimated HTM: \n{estimated_HTM}")

    # Check results
    assert np.allclose(estimated_HTM, known_HTM, atol=1e-6)
    print("PCR test pass")

    return calcerr.calculate_error_transformation(known_HTM, estimated_HTM)


if __name__ == "__main__":
    # Run random PCR test
    diff_HTM, angle_error, translation_error = random_pcr_test()
    print("HTM Difference:\n", diff_HTM)
    print(f"Angle Error (deg): {angle_error}")
    print(f"Translation Error: {translation_error}")

    # Run expected calibration object test for perfect data set
    perfect_expected, perfect_measured = find_expected_calibration_object("./data/pa1-debug-a-calbody.txt", "./data/pa1-debug-a-calreadings.txt")
    print(f"Perfect data shapes: expected {perfect_expected.shape}, measured {perfect_measured.shape}")

    # Flatten for plotting
    perfect_expected_flat = perfect_expected.reshape(-1, 3)
    perfect_measured_flat = perfect_measured.reshape(-1, 3)

    fig, ax = plotter.plot_data_2(perfect_expected_flat, perfect_measured_flat, "Perfect Expected", "Perfect Measured", number_points=False)
    fig, ax = plotter.plot_data_error_vectors(perfect_expected_flat, perfect_measured_flat, "Perfect Expected", "Perfect Measured")
    
    perfect_stats = calcerr.calculate_error_stats(perfect_expected_flat, perfect_measured_flat)
    calcerr.print_error_stats(perfect_stats)

    # Run expected calibration object test for distorted data set
    distorted_expected, distorted_measured = find_expected_calibration_object("./data/pa1-debug-c-calbody.txt", "./data/pa1-debug-c-calreadings.txt")

    # Flatten for plotting 
    distorted_expected_flat = distorted_expected.reshape(-1, 3)
    distorted_measured_flat = distorted_measured.reshape(-1, 3)

    fig, ax = plotter.plot_data_2(distorted_expected_flat, distorted_measured_flat, "Distorted Expected", "Distorted Measured", number_points=False)
    fig, ax = plotter.plot_data_error_vectors(distorted_expected_flat, distorted_measured_flat, "Distorted Expected", "Distorted Measured")   
    
    distorted_stats = calcerr.calculate_error_stats(distorted_expected_flat, distorted_measured_flat)
    calcerr.print_error_stats(distorted_stats)
   
    plt.show()

