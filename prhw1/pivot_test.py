import numpy as np
from utils.calibrator import CalibrationTools
from utils import calculate_errors as calcerr

def test_pivot_calibration():
    # np.random.seed(0)
    # Pick a coordinate for tip in tool
    p_tip_true = np.array([0.1, 0.2, 0.3])
    # Pick a coordinate for pivot in world
    p_pivot_true = np.array([1.0, 2.0, 3.0])

    T_all = []
    for _ in range(10):
        theta = np.random.rand() * np.pi
        axis = np.random.rand(3) - 0.5
        axis /= np.linalg.norm(axis)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c + axis[0]**2*(1-c), axis[0]*axis[1]*(1-c)-axis[2]*s, axis[0]*axis[2]*(1-c)+axis[1]*s],
            [axis[1]*axis[0]*(1-c)+axis[2]*s, c+axis[1]**2*(1-c), axis[1]*axis[2]*(1-c)-axis[0]*s],
            [axis[2]*axis[0]*(1-c)-axis[1]*s, axis[2]*axis[1]*(1-c)+axis[0]*s, c+axis[2]**2*(1-c)]
        ])
        # Rearranging the pivot calibration equation to solve for the
        # displacement vector normally given by PCR
        p = p_pivot_true - R @ p_tip_true
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = p
        T_all.append(T)

    tool = CalibrationTools("pivot test")
    p_tip_est, p_pivot_est = tool.pivot_calibration(T_all)

    # Check results
    assert np.allclose(p_tip_est, p_tip_true, atol=1e-6)
    assert np.allclose(p_pivot_est, p_pivot_true, atol=1e-6)
    print("Pivot test pass")

    return np.linalg.norm(p_tip_est - p_tip_true), np.linalg.norm(p_pivot_est - p_pivot_true)

if __name__ == "__main__":
    tip_translation_error, pivot_translation_error = test_pivot_calibration()
    print(f"Tip translation error: {tip_translation_error}")
    print(f"Pivot translation error: {pivot_translation_error}")