import numpy as np
import matplotlib.pyplot as plt
from utils import parse as parser
from utils import plot as plotter
from utils import pcr as pcr 
from utils import icp as icp
from utils import write_out as writer
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

data_sets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J']

body_A_markers_bA, body_A_tip_bA, num_trackers_bA = parser.parse_rigid_bodies("data/Problem3-BodyA.txt")  
body_B_markers_bB, body_B_tip_bB, num_trackers_bB = parser.parse_rigid_bodies("data/Problem3-BodyB.txt")
vertices_ct, vertices_inds = parser.parse_mesh("data/Problem3Mesh.sur")

for letter in data_sets:
    print(f"----------Processing dataset {letter}----------")

    prefix = 'Debug' if letter <= 'F' else 'Unknown'
    sample_readings_path = f"data/PA3-{letter}-{prefix}-SampleReadingsTest.txt"
    body_A_markers_tr, body_B_markers_tr, num_samples = parser.parse_readings(sample_readings_path, num_trackers_bA, num_trackers_bB)

    body_A_markers_bA = np.array(body_A_markers_bA)
    body_B_markers_bB = np.array(body_B_markers_bB)
    body_A_markers_tr = np.array(body_A_markers_tr)
    body_B_markers_tr = np.array(body_B_markers_tr)

    vertices_inds = np.array(vertices_inds)
    triangles = np.array(vertices_inds[:, :3], dtype=int) # Only i1,i2,i3 needed
    vertices  = np.array(vertices_ct, dtype=float)

    # Compute F_A,k and F_B,k for each sample frame using PCR
    F_A = []
    F_B = []
    for k in range(num_samples):
        F_Ak = pcr.point_cloud_registration(body_A_markers_bA, body_A_markers_tr[k*6:k*6+6])
        F_Bk = pcr.point_cloud_registration(body_B_markers_bB, body_B_markers_tr[k*6:k*6+6])
        F_A.append(F_Ak)
        F_B.append(F_Bk)
    F_A = np.array(F_A)  
    F_B = np.array(F_B)  

    # Compute d,k for each sample frame
    d = []  
    for k in range(num_samples):
        H_body_A_tip_bA = np.hstack((body_A_tip_bA, [1])) # Make HTM
        H_d_k = np.linalg.inv(F_B[k]) @ F_A[k] @ H_body_A_tip_bA # F_B^-1 * F_A * A_tip
        d_k = H_d_k[:3]  # Extract translation part
        d.append(d_k)
    d = np.array(d)

    # Apply registration F_reg 
    F_reg = np.eye(4)  # For PA3, assume identity
    s = []
    for k in range(num_samples):
        H_d_k = np.hstack((d[k], [1])) # Make HTM
        H_s_k = F_reg @ H_d_k # s = F_reg * d
        s_k = H_s_k[:3]  # Extract translation part
        s.append(s_k)
    s = np.array(s)

    # Find closest points on mesh and compute errors
    closest_points = []
    errors = []
    tri_idxs = []
    for k in d:
        c_k, e_k, idx = icp.search_closest_points_on_mesh(k, vertices, triangles)
        closest_points.append(c_k)
        errors.append(e_k)
        tri_idxs.append(idx)
    closest_points = np.asarray(closest_points)
    errors = np.asarray(errors)

    # Write out 
    output_path = f"./out/PA3-{letter}-{prefix}-Output.txt"
    writer.write_p3_output(d, closest_points, errors, output_path)

    

