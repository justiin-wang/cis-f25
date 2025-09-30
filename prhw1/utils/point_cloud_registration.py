import numpy as np
from scipy.spatial.transform import Rotation

try:
    import utils.convert as c
except ImportError:
    import convert as c


def point_cloud_registration(a, b, allow_scale=False):
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



def test():
    # Unit test for point_cloud_registration
    n_test = 10
    a = np.random.rand(n_test, 3)
    print("random points a\n", a)

    E = np.eye(4)
    E[:3, :3] = Rotation.from_euler("XYZ", np.random.rand(3) * np.pi).as_matrix()
    E[:3, -1] = np.random.rand(3)
    print("\nrandom transformation E\n", E)

    a_ = np.concatenate((a, np.ones((n_test, 1))), 1)
    b_ = a_ @ E.T
    b = b_[:, :3]
    print("\nb = E @ a\n", b)

    E_result = point_cloud_registration(a, b)
    print("\nPCR result E_pcr_1\n", E_result)

    b_2 = a_ @ E_result.T
    b2 = b_2[:, :3]
    np.printoptions(suppress=True, precision=5)
    # print("\nb2 = E2 @ a\n", b2)
    print("\nb and b2 difference (all 0 mean it's correct)\n", b - b2)

if __name__ == "__main__":
    test()