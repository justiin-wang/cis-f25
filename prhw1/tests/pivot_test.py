import numpy as np
from prhw1.utils.calibrator import CalibrationTools

def test_pivot_calibration():
    np.random.seed(0)
    p_tip_true = np.array([0.1, 0.2, 0.3])
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
        p = p_pivot_true - R @ p_tip_true
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = p
        T_all.append(T)

    # Create object
    tool = CalibrationTools("TestTool")
    p_tip_est, p_pivot_est = tool.pivot_calibration(T_all)

    print(p_tip_est)
    print(p_tip_true)

    # Check results
    assert np.allclose(p_tip_est, p_tip_true, atol=1e-6)
    assert np.allclose(p_pivot_est, p_pivot_true, atol=1e-6)
    print("Pivot calibration test passed!")

if __name__ == "__main__":
    test_pivot_calibration()
