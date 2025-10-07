import numpy as np
import matplotlib.pyplot as plt
from utils.calibrator import CalibrationTools
from utils import parse as parser
from utils import plot as plotter

tool = CalibrationTools("test")

print(tool.name)

# do problem 4 here

d, a, c = parser.parse_calbody("./data/pa1-debug-c-calbody.txt")
D_frames, A_frames, C_frames = parser.parse_calreadings("./data/pa1-debug-c-calreadings.txt")
C_expected_frames = [] # k C_expected point cloud

for k in range(len(D_frames)):
    D_measured = D_frames[k]
    A_measured = A_frames[k]

    F_D = tool.point_cloud_registration(d, D_measured)
    F_A = tool.point_cloud_registration(a, A_measured)

    F_C = np.linalg.inv(F_D) @ F_A # F_C = F_D^-1 * F_A

    C_expected = (F_C[:3, :3] @ c.T + F_C[:3, 3:4]).T # Apply the transformation to vector c
    C_expected_frames.append(C_expected)

C_expected_frames = np.array(C_expected_frames)
C_frames = np.array(C_frames)

# Plot C_expected vs C_frames
fig, ax = plotter.plot_data_2(C_expected_frames, C_frames, "C_expected", "C_measured", number_points=False)
plt.show()

# TODO: write out to txt file

