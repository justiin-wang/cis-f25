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
fig, ax = plotter.plot_data_2(C_expected_pc, C_frames, "C_expected", "C_measured", number_points=False)
plt.show()

# Problem 5
emprobe = CalibrationTools("emprobe")
G_all = parser.parse_empivot("prhw1\data\pa1-debug-f-empivot.txt")
# calculate tool frame
tool_origin = G_all[0].mean(axis=0)
emprobe.local_frame_points = G_all[0] - tool_origin
# Calculate PCR for each reading
T_all = np.zeros((len(G_all),4,4))
emprobe_expecteH_all = []
for k,frame in enumerate(G_all):
  F_G = emprobe.point_cloud_registration(emprobe.local_frame_points,frame)
  T_all[k] = F_G

  emprobe_expected = (F_G[:3, :3] @ emprobe.local_frame_points.T + F_G[:3, 3:4]).T
  emprobe_expecteH_all.append(emprobe_expected)
# Perform pivot calibration
p_tip_em,p_pivot_em = emprobe.pivot_calibration(T_all)

emprobe_expecteH_all = np.array(emprobe_expecteH_all)
G_all = np.array(G_all)
fig, ax = plotter.plot_data_2(emprobe_expecteH_all, G_all, "expected trackers", "measured trackers", number_points=False)

ax.scatter(p_pivot_em[0], p_pivot_em[1], p_pivot_em[2], c='r', marker='*', s=100, label='Pivot')
plt.show()

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
plt.show()

# Problem 6
optprobe = CalibrationTools("optprobe")
_, H_all = parser.parse_optpivot("prhw1\data\pa1-debug-f-optpivot.txt")

H_all_em = np.zeros(np.shape(H_all))
R_D = F_D[:3,:3]
t_D = F_D[:3,3:4]

for k,frame in enumerate(H_all):
   H_all_em[k] = (R_D @ frame.T + t_D).T
# calculate tool frame
tool_origin = H_all_em[0].mean(axis=0)
optprobe.local_frame_points = H_all_em[0] - tool_origin
# Calculate PCR for each reading
T_all = np.zeros((len(H_all_em),4,4))
emprobe_expecteH_all = []
for k,frame in enumerate(H_all_em):
  F_G = optprobe.point_cloud_registration(optprobe.local_frame_points,frame)
  T_all[k] = F_G

  emprobe_expected = (F_G[:3, :3] @ optprobe.local_frame_points.T + F_G[:3, 3:4]).T
  emprobe_expecteH_all.append(emprobe_expected)
# Perform pivot calibration
p_tip_opt,p_pivot_opt = optprobe.pivot_calibration(T_all)

emprobe_expecteH_all = np.array(emprobe_expecteH_all)
H_all_em = np.array(H_all_em)
fig, ax = plotter.plot_data_2(emprobe_expecteH_all, H_all_em, "expected trackers", "measured trackers", number_points=False)

ax.scatter(p_pivot_opt[0], p_pivot_opt[1], p_pivot_opt[2], c='r', marker='*', s=100, label='Pivot')
plt.show()

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
plt.show()

# TODO: write out to txt file

