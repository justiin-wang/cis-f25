import numpy as np
from math import comb

class BPoly:

    def __init__(self, order):
        self.order = order # Degree of polynomial, play around with order for best fit
        self.coeff_x = None 
        self.coeff_y = None
        self.coeff_z = None
        self.min = None # Define a normalization range
        self.max = None

    # Helper function to construct 1D Bernstein basis following formula: 
    # https://www2.math.upenn.edu/~kadison/bernstein.pdf
    def bernstein_1d(self, n, x):
        assert 0 <= x <= 1, "input must be normalized to [0,1]"

        B = np.zeros(n + 1)
        for i in range(n + 1):
            B[i] = comb(n, i) * (x ** i) * ((1 - x) ** (n - i))
        return np.array(B)

    # Helper function to construct 3D Bernstein basis
    def bernstein_3d_basis(self, n, xyz):
        # Compute 1D basis for each dim
        B_x = self.bernstein_1d(n, xyz[0])
        B_y = self.bernstein_1d(n, xyz[1])
        B_z = self.bernstein_1d(n, xyz[2]) 

        # Combine into 3D basis
        B_xyz = []
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    B_xyz.append(B_x[i] * B_y[j] * B_z[k])

        return np.array(B_xyz)

    # Given C_expected and C_measured point clouds (Nx3), fit a BPoly model
    def fit(self, expected, measured):
        assert measured.shape == expected.shape # Ensure we have correspondance
        assert measured.shape[1] == 3 # Both sets must be Nx3

        # Normalize for BPoly basis
        self.min_ = measured.min(axis=0)
        self.max_ = measured.max(axis=0)
        norm_points = (measured - self.min_) / (self.max_ - self.min_)

        n = self.order
        num_points = norm_points.shape[0]

        # Build coeff matrix A
        A = np.zeros((num_points, (n + 1) ** 3))
        for i in range(num_points):
            basis_vector = self.bernstein_3d_basis(n, norm_points[i]) 
            A[i, :] = basis_vector

        # Solve least squares independently for x, y, z
        self.coeff_x, *_ = np.linalg.lstsq(A, expected[:, 0], rcond=None)
        self.coeff_y, *_ = np.linalg.lstsq(A, expected[:, 1], rcond=None)
        self.coeff_z, *_ = np.linalg.lstsq(A, expected[:, 2], rcond=None)

    # Apply fitted BPoly correction to new PC
    def apply(self, points):
        assert self.coeff_x is not None, "Must fit() first."

        norm_points = (points - self.min_) / (self.max_ - self.min_)
        n = self.order
        num_points = norm_points.shape[0]

        # Build coeff matrix A
        A = np.zeros((num_points, (n + 1) ** 3))
        for i in range(num_points):
            basis_vector = self.bernstein_3d_basis(n, norm_points[i]) 
            A[i, :] = basis_vector

        x_corrected = A @ self.coeff_x
        y_corrected = A @ self.coeff_y
        z_corrected = A @ self.coeff_z
        return np.stack((x_corrected, y_corrected, z_corrected), axis=1)
    
    