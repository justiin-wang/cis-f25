import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from utils.calibrator import ProbeCalibration
from utils import calculate_errors as calcerr
from utils import plot as plotter
from utils import parse as parser

tool = ProbeCalibration("test")

def test_pivot_calibration():
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

    tool = ProbeCalibration("pivot test")
    p_tip_est, p_pivot_est = tool.pivot_calibration(T_all)

    # Check results
    assert np.allclose(p_tip_est, p_tip_true, atol=1e-6)
    assert np.allclose(p_pivot_est, p_pivot_true, atol=1e-6)
    print("Pivot test pass")

    return np.linalg.norm(p_tip_est - p_tip_true), np.linalg.norm(p_pivot_est - p_pivot_true)

def em_pivot_calibration_test():
    emprobe = ProbeCalibration("emprobe")
    G_all = parser.parse_empivot(".\data\pa2-debug-f-empivot.txt")
    # calculate tool frame
    tool_origin = G_all[0].mean(axis=0)
    emprobe.local_frame_points = G_all[0] - tool_origin
    # Calculate PCR for each reading
    T_all = np.zeros((len(G_all),4,4))
    emprobe_expected_all = []
    for k,frame in enumerate(G_all):
        F_G = emprobe.point_cloud_registration(emprobe.local_frame_points,frame)
        T_all[k] = F_G

        emprobe_expected = (F_G[:3, :3] @ emprobe.local_frame_points.T + F_G[:3, 3:4]).T
        emprobe_expected_all.append(emprobe_expected)
    # Perform pivot calibration
    p_tip_em,p_pivot_em = emprobe.pivot_calibration(T_all)

    emprobe_expected_all = np.array(emprobe_expected_all)
    G_all = np.array(G_all)

    # Flatten for plotting: (frames, points, 3) -> (frames*points, 3)
    emprobe_expected_flat = emprobe_expected_all.reshape(-1, 3)
    G_all_flat = G_all.reshape(-1, 3)

    # Plot EM probe point clouds
    fig, ax = plotter.plot_data_2(emprobe_expected_flat, G_all_flat, "EM Expected Trackers", "EM Measured Trackers", number_points=False)
    ax.scatter(p_pivot_em[0], p_pivot_em[1], p_pivot_em[2], c='r', marker='*', s=100, label='EM Pivot Point')
    ax.legend()

    # Plot EM error vectors
    fig, ax = plotter.plot_data_error_vectors(emprobe_expected_flat, G_all_flat, "EM Expected", "EM Measured")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot tool markers
    ax.scatter(emprobe.local_frame_points[:,0], emprobe.local_frame_points[:,1], emprobe.local_frame_points[:,2], c='b', marker='o', s=50, label='Tool markers')

    # Plot tip
    ax.scatter(p_tip_em[0], p_tip_em[1], p_tip_em[2], c='r', marker='*', s=150, label='Tool tip')

    # Labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Tool frame markers and tip")

    # Error stats between EM expected and G_all
    em_rmse = calcerr.calculate_rms_error(emprobe_expected_flat, G_all_flat)
    print(f"EM RMSE: {em_rmse}")
    em_stats = calcerr.calculate_error_stats(emprobe_expected_flat, G_all_flat)
    calcerr.print_error_stats(em_stats)


def opt_pivot_calibration_test():
    optprobe = ProbeCalibration("optprobe")
    D_all, H_all = parser.parse_optpivot(".\data\pa2-debug-f-optpivot.txt")
    d, _, _ = parser.parse_calbody("./data/pa2-debug-f-calbody.txt")

    H_all_em = np.zeros(np.shape(H_all))

    # calculate tool frame
    tool_origin = H_all[0].mean(axis=0)
    optprobe.local_frame_points = H_all[0] - tool_origin
    # Calculate PCR for each reading
    T_all = np.zeros((len(H_all_em),4,4))
    optprobe_expected_all = []
    for k,frame in enumerate(H_all):
        F_D = optprobe.point_cloud_registration(D_all[k], d)
        R_D = F_D[:3,:3]
        t_D = F_D[:3,3:4]
        H_em = (R_D @ frame.T + t_D).T
        H_all_em[k] = H_em

        F_G = optprobe.point_cloud_registration(optprobe.local_frame_points,H_em)
        T_all[k] = F_G

        optprobe_expected = (F_G[:3, :3] @ optprobe.local_frame_points.T + F_G[:3, 3:4]).T
        optprobe_expected_all.append(optprobe_expected)
    # Perform pivot calibration
    p_tip_opt,p_pivot_opt = optprobe.pivot_calibration(T_all)

    optprobe_expected_all = np.array(optprobe_expected_all)
    H_all_em = np.array(H_all_em)

    # Flatten for plotting: (frames, points, 3) -> (frames*points, 3)
    optprobe_expected_flat = optprobe_expected_all.reshape(-1, 3)
    H_all_em_flat = H_all_em.reshape(-1, 3)

    # Plot optical probe point clouds  
    fig, ax = plotter.plot_data_2(optprobe_expected_flat, H_all_em_flat, "Optical Expected Trackers", "Optical Measured Trackers", number_points=False)
    ax.scatter(p_pivot_opt[0], p_pivot_opt[1], p_pivot_opt[2], c='r', marker='*', s=100, label='Optical Pivot Point')
    ax.legend()

    # Plot optical error vectors
    fig, ax = plotter.plot_data_error_vectors(optprobe_expected_flat, H_all_em_flat, "Optical Expected", "Optical Measured")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot tool markers
    ax.scatter(optprobe.local_frame_points[:,0], optprobe.local_frame_points[:,1], optprobe.local_frame_points[:,2], c='b', marker='o', s=50, label='Tool markers')

    # Plot tip
    ax.scatter(p_tip_opt[0], p_tip_opt[1], p_tip_opt[2], c='r', marker='*', s=150, label='Tool tip')

    # Labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Tool frame markers and tip")

    # Error stats between optical expected and H_all_em
    opt_rmse = calcerr.calculate_rms_error(optprobe_expected_flat, H_all_em_flat)
    print(f"Optical RMSE: {opt_rmse}")
    print("Optical data error stats:")
    opt_stats = calcerr.calculate_error_stats(optprobe_expected_flat, H_all_em_flat)
    calcerr.print_error_stats(opt_stats)




if __name__ == "__main__":
    # Run pivot calibration test
    tip_translation_error, pivot_translation_error = test_pivot_calibration()
    print(f"Tip translation error: {tip_translation_error}")
    print(f"Pivot translation error: {pivot_translation_error}")

    # Run EM pivot calibration test
    em_pivot_calibration_test()

    # Run Optical pivot calibration test
    opt_pivot_calibration_test()

    # Plot
    plt.show()
