# # prhw1/tests/PCR_test.py
# import os
# import sys
# import numpy as np

# # ---- Robust import: works with `python -m prhw1.tests.PCR_test` and with direct runs ----
# try:
#     from prhw1.utils.calibrator import CalibrationTools
# except ModuleNotFoundError:
#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#     from utils.calibrator import CalibrationTools
import numpy as np
from prhw1.utils.calibrator import CalibrationTools

def _pose_errors(R_est, t_est, R_true, t_true):
    """Return (rotation error in deg, translation L2 error)."""
    R_delta = R_est.T @ R_true
    cos_theta = (np.trace(R_delta) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rot_err_deg = np.degrees(np.arccos(cos_theta))
    trans_err = np.linalg.norm(t_est - t_true)
    return rot_err_deg, trans_err


def _rmse_after_transform(src_pts, dst_pts, HTM):
    """RMSE between (src_pts transformed by HTM) and dst_pts."""
    n = src_pts.shape[0]
    src_h = np.column_stack([src_pts, np.ones(n)])
    src_xform = (src_h @ HTM.T)[:, :3]
    diffs = src_xform - dst_pts
    return np.sqrt(np.mean(np.sum(diffs**2, axis=1)))


def random_pcr_test(num_points=100, seed=42, angle_deg=30.0, translation=(20.0, 30.0, 10.0), verbose=True):
    """
    Unit Test for PCR using CalibrationTools.point_cloud_registration
    1) Create random point cloud
    2) Copy it
    3) Apply known SE(3) (HTM) to the copy
    4) Estimate with PCR (calibrator object)
    5) Report errors (rotation angle, translation norm, RMSE)
    """
    rng = np.random.default_rng()
    original_points = rng.random((num_points, 3)) * 100.0
    transformed_points = original_points.copy()

    # Known HTM (Rz + translation)
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
    calib = CalibrationTools(name="PCR-Tester")
    estimated_HTM = calib.point_cloud_registration(original_points, transformed_points)

    # Errors
    R_est, t_est = estimated_HTM[:3, :3], estimated_HTM[:3, 3]
    R_true, t_true = known_HTM[:3, :3], known_HTM[:3, 3]
    rot_err_deg, trans_err = _pose_errors(R_est, t_est, R_true, t_true)
    rmse = _rmse_after_transform(original_points, transformed_points, estimated_HTM)

    if verbose:
        np.set_printoptions(precision=6, suppress=True)
        print("Known HTM:\n", known_HTM)
        print("\nEstimated HTM:\n", estimated_HTM)
        print("\nRegistration errors:")
        print(f"  Rotation error:     {rot_err_deg:.9f} deg")
        print(f"  Translation error:  {trans_err:.9e}")
        print(f"  RMSE (after HTM):   {rmse:.9e}")

    return {
        "known_HTM": known_HTM,
        "estimated_HTM": estimated_HTM,
        "rotation_error_deg": rot_err_deg,
        "translation_error": trans_err,
        "rmse": rmse,
    }


if __name__ == "__main__":
    # Quick run + tight sanity checks for noiseless data
    metrics = random_pcr_test()
    assert metrics["rotation_error_deg"] < 1e-8
    assert metrics["translation_error"] < 1e-8
    assert metrics["rmse"] < 1e-8
