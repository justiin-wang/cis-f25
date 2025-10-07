import numpy as np

class CalibrationTools:
  def __init__(self, name="Tool"):
    self.name=name

  # Implementation of the point cloud registration algorithm 
  # Assumes rigid, no scale, and point correspondence
  def point_cloud_registration(self, a, b):
   
    # Verify input dims and mininum points
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 3
    n = a.shape[0]
    assert n >= 10

    # Find centroids and center two point clouds
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    A = a - ca
    B = b - cb

    # Cross-covariance matrix
    H = A.T @ B / n 

    # SVD to solve for 
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Comput rotation
    D = np.eye(3)
    if np.linalg.det(V @ U.T) < 0:
        D[2, 2] = -1.0
    R = V @ D @ U.T

    # Computer translation
    p = cb - (R @ ca)

    # Reconstruct HTM
    E = np.eye(4)
    E[:3, :3] = R  
    E[:3, 3] = p

    return E

  def pivot_calibration(self, T_all):
    """
    Perform pivot calibration

    Parameters:
      R_all: list of 3x3 numpy arrays.
        Measured rotation matrices of tool in world.
      p_all: list of 3x1 numpy arrays.
        Measured translation vectors of tool in world.

    Returns:
      p_tip: 3x1 numpy array.
        Calculated translation vector of tip in tool.
      p_pivot: 3x1 numpy array.
        Calculated translation vector of pivot point in world
    """
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