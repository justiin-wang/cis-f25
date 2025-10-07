import numpy as np

class CalibrationTools:
  def __init__(self, name="Tool", local_frame_points = None):
    self.name=name
    self.local_frame_points = local_frame_points

  # Implementation of Kabsch point cloud registration algorithm 
  # Assumes rigid, no scale, and point correspondence
  def point_cloud_registration(self, a, b):
   
    # Verify input dims and mininum points
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 3
    n = a.shape[0]
    assert n >= 3

    # Find centroids and center two point clouds
    ca = a.mean()
    cb = b.mean()
    A = a - ca
    B = b - cb

    # Cross-covariance matrix
    H = A.T @ B / n 

    # SVD to solve for rotation
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Comput rotation indep
    D = np.eye(3)
    # Caveat of Kabsch PCR: ensure rotation belongs to SO(3)
    if np.linalg.det(V @ U.T) < 0: 
        D[2, 2] = -1.0
    R = V @ D @ U.T

    # Computer translation
    p = cb - (R @ ca)

    # Reconstruct HTM
    F = np.eye(4)
    F[:3, :3] = R  
    F[:3, 3] = p

    return F

  def pivot_calibration(self, T_all):

    j = len(T_all)
    A = np.zeros((3 * j, 6))
    b = np.zeros(3 * j)

    for i, T in enumerate(T_all):
          R_curr = T[:3, :3]
          t_curr = T[:3, 3]
          
          A[3*i:3*i+3, :3] = R_curr
          A[3*i:3*i+3, 3:] = -np.eye(3)
          b[3*i:3*i+3] = -t_curr
    x, _, _, _ = np.linalg.lstsq(A, b)
    p_tip = x[:3]
    p_pivot = x[3:]
    
    return p_tip, p_pivot