import numpy as np
from scipy.spatial.transform import Rotation as R


def skew(x):
    # Convert a vecter to a skew symmetric matric
    # INPUT (3,)
    assert x.shape == (3,)
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def wxyz_xyzw(wxyz):
    # Switch quaternion element order
    # INPUT (4,) or (4,n) / OUTPUT (4,) or (4,n)
    assert wxyz.shape == (4,) or (wxyz.shape[1] == 4 and wxyz.ndim == 2)
    if wxyz.shape == (4,):
        xyzw = wxyz[[1, 2, 3, 0]]
    else:
        xyzw = wxyz[:, [1, 2, 3, 0]]
    return xyzw


def xyzw_wxyz(xyzw):
    # Switch quaternion element order
    # INPUT (4,) or (4,n) / OUTPUT (4,) or (4,n)
    assert xyzw.shape == (4,) or (xyzw.shape[1] == 4 and xyzw.ndim == 2)
    if xyzw.shape == (4,):
        wxyz = xyzw[[3, 0, 1, 2]]
    else:
        wxyz = xyzw[:, [3, 0, 1, 2]]
    return wxyz


def quat_mat(quat, quat_order="xyzw"):
    # Convert quaternion to rotation matrix
    # INPUT (4,) / OUTPUT (3,3)
    assert quat.shape == (4,)
    if quat_order == "xyzw":
        xyzw = quat
    elif quat_order == "wxyz":
        xyzw = wxyz_xyzw(quat)
    else:
        print("quat_order only support ether xyzw or wxyz")
    return R.from_quat(xyzw).as_matrix()


def mat_quat(mat, quat_order="xyzw"):
    # Convert rotation matrix to quaternion
    # INPUT (3,3) / OUTPUT (4,)
    assert mat.shape == (3, 3)
    xyzw = R.from_matrix(mat).as_quat()
    xyzw = xyzw * np.sign(xyzw[3])
    if quat_order == "xyzw":
        return xyzw
    elif quat_order == "wxyz":
        return xyzw_wxyz(xyzw)
    else:
        print("quat_order only support ether xyzw or wxyz")


def htm_vect7(M, quat_order="xyzw"):
    # Convert homogeneous transformation matrices to vectors with quaternion
    # INPUT (4,4) or (n,4,4) / OUTPUT (7,) or (n,7)
    assert M.shape == (4, 4) or M.shape[1:] == (4, 4)
    if len(M.shape) == 2:
        v = np.zeros((7,))
        v[:3] = M[0:3, 3]
        v[3:] = mat_quat(M[:3, :3], quat_order)
    elif len(M.shape) == 3:
        v = np.zeros((M.shape[0], 7))
        for i in range(M.shape[0]):
            v[i, :3] = M[i, 0:3, 3]
            v[i, 3:] = mat_quat(M[i, :3, :3], quat_order)
    return v


def vect7_htm(v, quat_order="xyzw"):
    # Convert vectors with quaternion to homogeneous transformation matrices
    # INPUT (7,) or (n,7) / OUTPUT (4,4) or (n,4,4)
    assert v.shape == (7,) or v.shape[1:] == (7,)
    if v.shape == (7,):
        M = np.eye(4)
        M[:3, :3] = quat_mat(v[3:], quat_order)
        M[:3, 3] = v[:3]
    else:
        M = np.zeros((v.shape[0], 4, 4))
        for i in range(v.shape[0]):
            M[i, :, :] = np.eye(4)
            M[i, :3, :3] = quat_mat(v[i, 3:], quat_order)
            M[i, :3, 3] = v[i, :3]
    return M


def htm_vect6(M):
    # Convert homogeneous transformation matrices to vectors with euler angle
    # INPUT (4,4) or (n,4,4) / OUTPUT (6,) or (n,6)
    assert M.shape == (4, 4) or M.shape[1:] == (4, 4)
    if len(M.shape) == 2:
        v = np.zeros((6,))
        v[:3] = M[0:3, 3]
        v[3:] = R.from_matrix(M[0:3, 0:3]).as_euler("XYZ")
    elif len(M.shape) == 3:
        v = np.zeros((M.shape[0], 6))
        for i in range(M.shape[0]):
            v[i, :3] = M[i, 0:3, 3]
            v[i, 3:] = R.from_matrix(M[i, 0:3, 0:3]).as_euler("XYZ")
    return v


def vect6_htm(v):
    # Convert vectors with euler angle to homogeneous transformation matrices
    # INPUT (6,) or (n,6) / OUTPUT (4,4) or (n,4,4)
    assert v.shape == (6,) or v.shape[1:] == (6,)
    if v.shape == (6,):
        M = np.eye(4)
        M[:3, :3] = R.from_euler("XYZ", v[3:]).as_matrix()
        M[:3, 3] = v[:3]
    else:
        M = np.zeros((v.shape[0], 4, 4))
        for i in range(v.shape[0]):
            M[i, :, :] = np.eye(4)
            M[i, :3, :3] = R.from_euler("XYZ", v[i, 3:]).as_matrix()
            M[i, :3, 3] = v[i, :3]
    return M


def vect7_vect6(v):
    # Convert vectors with quaternion to vectors with euler angle
    # INPUT (7,) or (n,7) / OUTPUT (6,) or (n,6)
    return htm_vect6(vect7_htm(v))


def vect6_vect7(v):
    # Convert vectors with euler angle to vectors with quaternion
    # INPUT (6,) or (n,6) / OUTPUT (7,) or (n,7)
    return htm_vect7(vect6_htm(v))


def inv(M):
    # Inverse of homogeneous transformation matrix
    assert M.shape == (4, 4)
    M_inv = np.eye(4)
    R = M[:3, :3]
    t = M[:3, 3]
    M_inv[:3, :3] = R.T
    M_inv[:3, 3] = -R.T @ t
    return M_inv
