import numpy as np
import matplotlib.pyplot as plt
from utils.calibrator import CalibrationTools
from utils import parse as parser
from utils import plot as plotter

tool = CalibrationTools("test")

print(tool.name)

# do problem 4 here

d_known, a_known, c_known = parser.parse_calbody("./data/pa1-debug-c-calbody.txt")
D_frames, A_frames, C_frames = parser.parse_calreadings("./data/pa1-debug-c-calreadings.txt")

num_frames = len(D_frames)
C_expected_pc = []  

for k in range(num_frames):
    D_measured = D_frames[k]
    A_measured = A_frames[k]

    F_D = tool.point_cloud_registration(d_known, D_measured)
    F_A = tool.point_cloud_registration(a_known, A_measured)

    F_C = np.linalg.inv(F_D) @ F_A

    C_expected = (F_C[:3, :3] @ c_known.T + F_C[:3, 3:4]).T
    C_expected_pc.append(C_expected)

C_expected_pc = np.array(C_expected_pc)
C_frames = np.array(C_frames)

print("C_expected_pc shape:", C_expected_pc.shape)
print("C_frames shape:", C_frames.shape)
print("Plotting shapes - C_expected:", C_expected_pc.shape, "C_frames:", C_frames.shape)

# Plot C_expected vs C_frames
fig, ax = plotter.plot_data_2(C_expected_pc, C_frames, "C_expected", "C_measured", number_points=False)
plt.show()