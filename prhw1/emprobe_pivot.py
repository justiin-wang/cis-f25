import numpy as np
import matplotlib.pyplot as plt
from utils.calibrator import CalibrationTools
from utils import parse as parser
from utils import plot as plotter

<<<<<<< Updated upstream
emprobe = CalibrationTools("emprobe")
]
=======
optprobe = CalibrationTools("optprobe")
D_all = parser.parse_empivot("prhw1\data\pa1-debug-a-optpivot.txt")
# calculate tool frame
tool_origin = D_all[0].mean(axis=0)
optprobe.local_frame_points = G_all[0] - tool_origin
# Calculate PCR for each reading
T_all = np.zeros((len(G_all),4,4))
emprobe_expected_all = []
for k,frame in enumerate(G_all):
  F_G = emprobe.point_cloud_registration(emprobe.local_frame_points,frame)
  T_all[k] = F_G

  emprobe_expected = (F_G[:3, :3] @ emprobe.local_frame_points.T + F_G[:3, 3:4]).T
  emprobe_expected_all.append(emprobe_expected)
# Perform pivot calibration
p_tip,p_pivot = emprobe.pivot_calibration(T_all)






emprobe_expected_all = np.array(emprobe_expected_all)
G_all = np.array(G_all)
fig, ax = plotter.plot_data_2(emprobe_expected_all, G_all, "expected trackers", "measured trackers", number_points=False)

ax.scatter(p_pivot[0], p_pivot[1], p_pivot[2], c='r', marker='*', s=100, label='Pivot')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot tool markers
ax.scatter(emprobe.local_frame_points[:,0], emprobe.local_frame_points[:,1], emprobe.local_frame_points[:,2], c='b', marker='o', s=50, label='Tool markers')

# Plot tip
ax.scatter(p_tip[0], p_tip[1], p_tip[2], c='r', marker='*', s=150, label='Tool tip')

# Labels and legend
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.set_title("Tool frame markers and tip")
plt.show()
>>>>>>> Stashed changes
