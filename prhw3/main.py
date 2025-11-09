import numpy as np
import matplotlib.pyplot as plt
from utils import parse as parser
from utils import plot as plotter
from utils import pcr as pcr 
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
