import numpy as np

class CalibrationTools:
  def __init__(self, name="Tool", local_frame_points = None):
    self.name=name
    self.local_frame_points = local_frame_points
  def point_cloud_registration(self, a, b, allow_scale=False):
    """
    Estimate rigid (or similarity) transform aligning points a -> b via SVD (Kabsch/Umeyama).

    Parameters
    ----------
    a : (n,3) array
        Source points (with known correspondences to b).
    b : (n,3) array
        Target points.
    allow_scale : bool
        If True, also estimate an isotropic scale (Umeyama). Otherwise, pure rigid.

    Returns
    -------
    R : (3,3) array
        Rotation matrix.
    t : (3,) array
        Translation vector.
    s : float
        Scale (1.0 if allow_scale=False).
    E : (4,4) array
        Homogeneous transform so that b ≈ (s*R) @ a + t.

    Notes
    -----
    - Uses proper rotation (det(R)=+1); if reflection occurs, the last singular vector is flipped.
    - Requires >= 3 non-collinear points for a well-conditioned solution.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 3
    n = a.shape[0]
    assert n >= 3, "Need at least 3 points."

    # 1) centroids
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)

    # 2) center
    A = a - ca
    B = b - cb

    # 3) covariance
    H = A.T @ B / n 

    # 4) SVD
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # 5) proper rotation
    D = np.eye(3)
    if np.linalg.det(V @ U.T) < 0:
        D[2, 2] = -1.0
    R = V @ D @ U.T

    # 6) translation
    t = cb - (R @ ca)

    # Homogeneous transform
    E = np.eye(4)
    E[:3, :3] = R  
    E[:3, 3] = t

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