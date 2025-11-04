import numpy as np
import matplotlib.pyplot as plt
from utils.calibrator import ProbeCalibration
from utils import parse as parser
from utils import plot as plotter
from utils import write_out as writer

# Main script for PA1. Run from start to finish to produce output.txt

#----------Problem 4----------
tool = ProbeCalibration("pcr")
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

#----------Problem 5----------
emprobe = ProbeCalibration("emprobe")
G_all = parser.parse_empivot(".\data\pa1-debug-c-empivot.txt")
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

# --------------Problem 6---------------
optprobe = ProbeCalibration("optprobe")
D_all, H_all = parser.parse_optpivot(".\data\pa1-debug-c-optpivot.txt")
d, _, _ = parser.parse_calbody("./data/pa1-debug-c-calbody.txt")

H_all_em = np.zeros(np.shape(H_all))

# calculate tool frame
tool_origin = H_all[0].mean(axis=0)
optprobe.local_frame_points = H_all[0] - tool_origin
# Calculate PCR for each reading
T_all = np.zeros((len(H_all_em),4,4))
optprobe_expected_all = []
for k,frame in enumerate(H_all):
  F_D = optprobe.point_cloud_registration(d, D_all[k])
  R_D = np.linalg.inv(F_D)[:3,:3]
  t_D = np.linalg.inv(F_D)[:3,3:4]
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

# Write out to output
writer.write_output_pa1(C_expected_frames, p_pivot_em, p_pivot_opt)

