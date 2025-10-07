import numpy as np
from utils.calibrator import CalibrationTools
from utils import calculate_errors as calcerr

def random_pcr_test(num_points=100, seed=42, angle_deg=30.0, translation=(20.0, 30.0, 10.0)):
    # Generate random PC
    rng = np.random.default_rng()
    original_points = rng.random((num_points, 3)) * 100.0
    transformed_points = original_points.copy()

    # Create known HTM
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]])
    t = np.array(translation, dtype=float)
    known_HTM = np.eye(4)
    known_HTM[:3, :3] = Rz
    known_HTM[:3, 3] = t

    # Apply transform to the copy
    pts_h = np.column_stack([transformed_points, np.ones(num_points)])
    transformed_points = (pts_h @ known_HTM.T)[:, :3]

    # Estimate HTM mapping original -> transformed using the calibrator object
    calib = CalibrationTools(name="PCR_test")
    estimated_HTM = calib.point_cloud_registration(original_points, transformed_points)

    return calcerr.calculate_error_transformation(known_HTM, estimated_HTM)


if __name__ == "__main__":
    diff_HTM, angle_error, translation_error = random_pcr_test()
    print("HTM Difference:\n", diff_HTM)
    print(f"Angle Error (deg): {angle_error:.6f}")
    print(f"Translation Error: {translation_error:.6f}")
