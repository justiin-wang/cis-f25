import numpy as np
from scipy.spatial.transform import Rotation

try:
    import utils.convert as c
except ImportError:
    import convert as c


def point_cloud_registration(a, b):
    # Point cloud registration
    # The function would give a transformation matrix between two frames given
    # two sets of coordinates of the same set of points from the two frames
    # INPUT : a and b points sould have same size (n,3)
    # OUTPUT: Homogeneous transformation matrix so that b = F_E * a
    #


    assert a.shape == b.shape
    assert len(a.shape) == 2 and a.shape[1] == 3
    a0 = a[0, :]
    b0 = b[0, :]

    a = a - a0
    b = b - b0

    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            H[i, j] = np.dot(a[:, i], b[:, j])

    Delta = (H[1, 2] - H[2, 1], H[2, 0] - H[0, 2], H[0, 1] - H[1, 0])
    G = np.zeros((4, 4))
    G[0, 0] = np.trace(H)
    G[0, 1:], G[1:, 0] = Delta, Delta
    G[1:, 1:] = H + H.T - np.trace(H) * np.eye(3)

    _, eig_vector = np.linalg.eig(G)
    wxyz = eig_vector[:, 0]
    xyzw = np.zeros(wxyz.shape)
    xyzw[:3], xyzw[3] = wxyz[1:], wxyz[0]
    E = np.eye(4)
    R = Rotation.from_quat(xyzw).as_matrix()
    E[:3, :3] = R
    E[:3, 3] = b0 - R @ a0

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
