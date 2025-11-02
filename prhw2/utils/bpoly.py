import numpy as np
from math import comb

class BPoly:

    def __init__(self, order):
        self.order = order
        self.coeff = None
        self.min_ = None
        self.max_ = None

    # Helper function to construct 1D Bernstein basis following formula: 
    # https://www2.math.upenn.edu/~kadison/bernstein.pdf
    def bernstein_1d(self, n, norm_points):

        N = np.arange(n + 1)        
        B = np.array([comb(n, i) * (norm_points ** i) * ((1 - norm_points) ** (n - i)) for i in N])
        return B  # shape: [(n+1), len(x)]

    # Helper function to construct 3D Bernstein basis
    def bernstein_3d(self, n, norm_point_cloud):
        N = norm_point_cloud.shape[0]
        x = norm_point_cloud[:, 0]
        y = norm_point_cloud[:, 1]
        z = norm_point_cloud[:, 2]

        # Compute 1D bases for all points using bernstein_1d
        Bx = self.bernstein_1d(n, x).T 
        By = self.bernstein_1d(n, y).T  
        Bz = self.bernstein_1d(n, z).T 

        # Construct full 3D basis 
        A = np.empty((N, (n+1)**3))
        idx = 0
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    A[:, idx] = Bx[:, i] * By[:, j] * Bz[:, k]
                    idx += 1
        return A


    # Given C_expected and C_measured point clouds (Nx3), fit a BPoly model
    def fit(self, measured, expected):
        assert measured.shape == expected.shape

        # Error target
        delta = expected - measured

        # Normalization (add 10% padding)
        self.min_ = measured.min(axis=0)
        self.max_ = measured.max(axis=0)
        pad = 0.10 * (self.max_ - self.min_ + 1e-9)
        self.min_ -= pad
        self.max_ += pad

        norm_point_cloud = (measured - self.min_) / (self.max_ - self.min_ + 1e-9)
        A = self.bernstein_3d(self.order, norm_point_cloud)

        # Least squares solution
        # AtA * coeff = Atb <=> argmin(||A*coeff - delta||^2)
        noise = 1e-6 # We add a noise term for numerical stability
        # See https://en.wikipedia.org/wiki/Tikhonov_regularization
        # I followed the paper and added noise and it worked so imma just keep it...
        # Else it blows up to >45mm on higher than 11 orders
        AtA = A.T @ A + noise * np.eye(A.shape[1])
        Atb = A.T @ delta
        self.coeff = np.linalg.solve(AtA, Atb)

    # Apply fitted BPoly correction to new PC
    def apply(self, point_cloud):
        assert self.coeff is not None
        norm_point_cloud = (point_cloud - self.min_) / (self.max_ - self.min_ + 1e-9)

        A = self.bernstein_3d(self.order, norm_point_cloud)
        correction = A @ self.coeff
        return point_cloud + correction
