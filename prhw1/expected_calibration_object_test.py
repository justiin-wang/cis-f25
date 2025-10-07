import numpy as np
import matplotlib.pyplot as plt
from utils.calibrator import CalibrationTools
from utils import calculate_errors as calcerr
from utils import parse as parser
from utils import plot as plotter

tool = CalibrationTools("test")

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

perfect_expected, perfect_measured = find_expected_calibration_object("./data/pa1-debug-a-calbody.txt", "./data/pa1-debug-a-calreadings.txt")
fig, ax = plotter.plot_data_2(perfect_expected, perfect_measured, "Perfect Expected", "Perfect Measured", number_points=False)
perfect_rmse = calcerr.calculate_rms_error(perfect_expected, perfect_measured)
print(f"Perfect RMSE: {perfect_rmse}")

distorted_expected, distorted_measured = find_expected_calibration_object("./data/pa1-debug-c-calbody.txt", "./data/pa1-debug-c-calreadings.txt")
fig, ax = plotter.plot_data_2(distorted_expected, distorted_measured, "Distorted Expected", "Distorted Measured", number_points=False)
distorted_rmse = calcerr.calculate_rms_error(distorted_expected, distorted_measured)
print(f"Distorted RMSE: {distorted_rmse}")
plt.show()
