import numpy as np
from utils import parse as parser
from utils import pcr as pcr 
from utils import icp as icp
from utils import write_out as writer
from utils.kdtree import KDTreeTriangles as kdtree
import time 

data_sets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J']

body_A_markers_bA, body_A_tip_bA, num_trackers_bA = parser.parse_rigid_bodies("data/Problem3-BodyA.txt")  
body_B_markers_bB, body_B_tip_bB, num_trackers_bB = parser.parse_rigid_bodies("data/Problem3-BodyB.txt")
vertices_ct, vertices_inds = parser.parse_mesh("data/Problem3Mesh.sur")

vertices_inds = np.array(vertices_inds)
triangles = np.array(vertices_inds[:, :3], dtype=int) # Only i1,i2,i3 needed
vertices  = np.array(vertices_ct, dtype=float)

start = time.perf_counter()
for letter in data_sets:
    print(f"----------Processing dataset {letter}----------")

    prefix = 'Debug' if letter <= 'F' else 'Unknown'
    sample_readings_path = f"data/PA3-{letter}-{prefix}-SampleReadingsTest.txt"
    body_A_markers_tr, body_B_markers_tr, num_samples = parser.parse_readings(sample_readings_path, num_trackers_bA, num_trackers_bB)

    body_A_markers_bA = np.array(body_A_markers_bA)
    body_B_markers_bB = np.array(body_B_markers_bB)
    body_A_markers_tr = np.array(body_A_markers_tr)
    body_B_markers_tr = np.array(body_B_markers_tr)

    mesh_tree = kdtree(vertices, triangles)

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
    # A_tip in body B frame
    d = []  
    for k in range(num_samples):
        H_body_A_tip_bA = np.append(body_A_tip_bA, 1.0) # Make HTM
        H_d_k = np.linalg.inv(F_B[k]) @ F_A[k] @ H_body_A_tip_bA # F_B^-1 * F_A * A_tip
        d_k = H_d_k[:3]  # Extract translation part
        d.append(d_k)
    d = np.array(d)

    # Apply registration F_reg 
    F_reg = np.eye(4)  # For PA3, assume identity
    s = []
    for k in range(num_samples):
        H_d_k = np.append(d[k], 1.0)# Make HTM
        H_s_k = F_reg @ H_d_k # s = F_reg * d
        s_k = H_s_k[:3]  # Extract translation part
        s.append(s_k)
    s = np.array(s)

    # Find closest points on mesh and compute errors
    closest_points = []
    errors = []
    tri_idxs = []
    for k in d:
        c_k, e_k, idx = mesh_tree.closest_point(k)
        closest_points.append(c_k)
        errors.append(e_k)
        tri_idxs.append(idx)
    closest_points = np.asarray(closest_points)
    errors = np.asarray(errors)

    # Write out 
    output_path = f"./OUTPUT/PA3-{letter}-{prefix}-Output.txt"
    writer.write_p3_output(d, closest_points, errors, output_path)
end = time.perf_counter()
print(f"Execution time: {end - start:.6f} seconds")
    

