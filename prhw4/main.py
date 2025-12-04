import numpy as np
import matplotlib.pyplot as plt
from utils import parse as parser
from utils import plot as plotter
from utils import pcr as pcr 
from utils import icp as icp
from utils import write_out as writer
from utils.kdtree import KDTreeTriangles as kdtree

import time
# Iterative ICP HW4
# From HW3, we create our KD tree that contains mesh triangles for closest point search
# We now assume Freg is NOT identity, so we compute s_k = F_reg * d_k
# Iterate until convergence to find F_reg, lets do until tiny change in F_reg OR mean error < threshold

data_sets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J'] # IDK why no I

body_A_markers_bA, body_A_tip_bA, num_trackers_bA = parser.parse_rigid_bodies("data/Problem4-BodyA.txt")  
body_B_markers_bB, body_B_tip_bB, num_trackers_bB = parser.parse_rigid_bodies("data/Problem4-BodyB.txt")
vertices_ct, vertices_inds = parser.parse_mesh("data/Problem4MeshFile.sur")

vertices_inds = np.array(vertices_inds)
triangles = np.array(vertices_inds[:, :3], dtype=int) # Only i1,i2,i3 needed
vertices  = np.array(vertices_ct, dtype=float)

start = time.perf_counter()
for letter in data_sets:
    print(f"----------Processing dataset {letter}----------")

    prefix = 'Debug' if letter <= 'F' else 'Unknown'
    sample_readings_path = f"data/PA4-{letter}-{prefix}-SampleReadingsTest.txt"
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
    F_reg = np.eye(4)
    max_iters = 100
    tol_F = 1e-6

    for iter in range(max_iters):

        # Compute s_k = F_reg * d_k
        H_d = np.hstack([d, np.ones((num_samples, 1))])   
        H_s = (F_reg @ H_d.T).T                           
        s = H_s[:, :3]

        # Find closest points on mesh
        closest_points = []
        errors = []
        tri_idxs = []
        for sk in s:
            c_k, e_k, idx = mesh_tree.closest_point(sk)
            closest_points.append(c_k)
            errors.append(e_k)
            tri_idxs.append(idx)
        closest_points = np.asarray(closest_points)
        errors = np.asarray(errors)

        # Registration map d to closest points
        F_reg_new = pcr.point_cloud_registration(d, closest_points)

        # Convergence check. Ether small change in F_reg or small mean error terminates loop
        if np.linalg.norm(F_reg_new - F_reg) < tol_F or np.mean(errors) < 1e-5: # Played around with bounds we can look more later
            if np.mean(errors) > 1e-1:
                print(f"FAIL: High mean error {np.mean(errors):.6f} at convergence.")
            F_reg = F_reg_new
            print(f"Converged at iteration {iter+1}, mean error = {np.mean(errors):.6f}")
            break

        print(f"Iteration {iter+1}: mean error = {np.mean(errors):.6f}")
        F_reg = F_reg_new
        if iter == max_iters - 1:
            print("WARNING: Reached maximum iterations without convergence.")

    # Final closest points and errors with converged F_reg
    # TODO maybe we can prevent a copy here?
    s = []
    for k in range(num_samples):
        H_d = np.append(d[k], 1.0)
        s.append((F_reg @ H_d)[:3])
    s = np.asarray(s)

    # final closest points for output
    closest_points = []
    errors = []
    for sk in s:
        ck, e, _ = mesh_tree.closest_point(sk)
        closest_points.append(ck)
        errors.append(e)
    errors = np.asarray(errors)
    closest_points = np.asarray(closest_points)

    # Write out 
    output_path = f"./out/PA4-{letter}-{prefix}-Output.txt"
    writer.write_p4_output(s, closest_points, errors, output_path)
end = time.perf_counter() # Timer for comparison to linear search
print(f"Execution time: {end - start:.6f} seconds")
    

