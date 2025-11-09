import numpy as np
import matplotlib.pyplot as plt
from utils import parse as parser
from utils import plot as plotter
from utils import pcr as pcr 
from utils import icp as icp
from utils import calculate_errors as calcerrSS
# ICP
# We want to find a transformation T s.t. target = T(source)

# target is vertices of in the mesh frame of the bone, built by CT scane
# source is positions of tip of body A in frame of body B, built by sampling its position when it touches the bone

# General steps:
# 1. Input the two sets of marker coords in body frame for rigid bodies A and B (Problem3-BodyA.txt and Problem3-BodyB.txt)
# 2. Input all k sample frames of a_i,k and b_i,k which are the optical trackers in base frame, as well as A_tip ()
# 3. Perform point-cloud-to-point-cloud registration to find F_a,k and F_b,k, HTMs of rigid bodies A and B w.r.t tracker
# 4. Find d_k, which is the position of the tip of rigid body A in frame of rigid body B(F_b,k^-1 * F_A,k * A_tip)
# 5. Find s_k, which is F_reg * d_k. For PA3, assume F_reg is identity.
# 6. Find points c_k that are closes to s_k. Compute magnitude of error
# 7. (PA4) Iterate to find best F_reg

# Output for PA3:
# d_k, c_k, and magnitude of errors between them


# Note: suffix of arrays containing points is the frame it is defined for
# bA is body A, bB is bodyB, ct is CT scan, tr is tracker base
body_A_markers_bA, body_A_tip_bA, num_trackers_bA = parser.parse_rigid_bodies("data/Problem3-BodyA.txt")  
body_B_markers_bB, body_B_tip_bB, num_trackers_bB = parser.parse_rigid_bodies("data/Problem3-BodyB.txt")
vertices_ct, vertices_inds = parser.parse_mesh("data/Problem3Mesh.sur")
body_A_markers_tr, body_B_markers_tr, num_samples = parser.parse_readings("data/PA3-A-Debug-SampleReadingsTest.txt", num_trackers_bA, num_trackers_bB)
# body_A_markers_bA = np.array(body_A_markers_bA)
# body_B_markers_bB = np.array(body_B_markers_bB)
# plotter.plot_data_2(body_A_markers_bA, body_B_markers_bB, "Body A Markers (Body A Frame)", "Body B Markers (Body B Frame)")
# vertices_ct = np.array(vertices_ct)
# plotter.plot_data_1(vertices_ct, "CT Scan Mesh Vertices")
# body_A_markers_tr = np.array(body_A_markers_tr)
# body_B_markers_tr = np.array(body_B_markers_tr)
# fig, ax = plotter.plot_data_2(body_A_markers_tr, body_B_markers_tr, "Body A Markers (Tracker Frame)", "Body B Markers (Tracker Frame)")
# plt.show()
body_A_markers_bA = np.array(body_A_markers_bA)
body_B_markers_bB = np.array(body_B_markers_bB)
body_A_markers_tr = np.array(body_A_markers_tr)
body_B_markers_tr = np.array(body_B_markers_tr)

# Compute F_A,k and F_B,k for each sample frame
F_A = []
F_B = []

for k in range(num_samples):
    # body_A_markers_bA : (N_A,3)
    # body_A_markers_tr[k] : (N_A,3)
    F_Ak = pcr.point_cloud_registration(body_A_markers_bA, body_A_markers_tr[k*6:k*6+6])
    F_Bk = pcr.point_cloud_registration(body_B_markers_bB, body_B_markers_tr[k*6:k*6+6])
    F_A.append(F_Ak)
    F_B.append(F_Bk)

F_A = np.array(F_A)  
F_B = np.array(F_B)  

d = []  # d_k points
for k in range(num_samples):
    body_A_tip_bA_homog = np.hstack((body_A_tip_bA, [1]))
    d_k_homog = np.linalg.inv(F_B[k]) @ F_A[k] @ body_A_tip_bA_homog
    d_k = d_k_homog[:3]  # Convert back to Cartesian coordinates
    d.append(d_k)
print(d)
    
F_reg = np.eye(4)  # Identity for PA3
# Find C_k 

# TEST
d = np.asarray(d)

# Some .sur readers return 6 ints per triangle (i1,i2,i3,n1,n2,n3); we only need first 3
triangles = np.asarray(vertices_inds)[:, :3].astype(int)
vertices  = np.asarray(vertices_ct, dtype=float)

closest_points = []
errors = []
tri_idxs = []

for d_k in d:
    c_k, e_k, idx = icp.search_closest_points_on_mesh(d_k, vertices, triangles)
    closest_points.append(c_k)
    errors.append(e_k)
    tri_idxs.append(idx)

closest_points = np.asarray(closest_points)
errors = np.asarray(errors)

# Optional quick viz overlay (your plotter expects Nx3 clouds)
try:
    plotter.plot_data_2(d, closest_points,
                        "Pointer tip samples (B frame)",
                        "Closest points on CT surface")
    plt.show()
except Exception as _:
    pass  # plotting is optional; don’t fail PA3 if display isn’t available

# Write PA3 output file (spec: Nsamps + file name header, then d, c, |d-c|)
import os, re

Nsamps = d.shape[0]

# Derive dataset letter from the sample readings filename you used above
# e.g., "data/PA3-A-Debug-SampleReadingsTest.txt" -> "A"
readings_path = "data/PA3-A-Debug-SampleReadingsTest.txt"
m = re.search(r'PA3-([A-Z])', os.path.basename(readings_path))
dataset_letter = m.group(1) if m else "X"

out_dir = "OUTPUT"
os.makedirs(out_dir, exist_ok=True)
out_name = f"pa3-{dataset_letter}-Output.txt"
out_path = os.path.join(out_dir, out_name)

with open(out_path, "w") as f:
    f.write(f"{Nsamps}, \"{out_name}\"\n")
    for dk, ck, ek in zip(d, closest_points, errors):
        f.write(f"{dk[0]:.6f} {dk[1]:.6f} {dk[2]:.6f} "
                f"{ck[0]:.6f} {ck[1]:.6f} {ck[2]:.6f} {ek:.6f}\n")

plt.show()
print(f"[PA3] Wrote {Nsamps} rows to {out_path}")
# -------------------------------------------------------------------------------