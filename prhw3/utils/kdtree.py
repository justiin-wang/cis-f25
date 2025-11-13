import numpy as np
from utils import icp as icp

class KDTreeNode:
    def __init__(self, tri_indices, point=None, left=None, right=None, axis=0):
        self.tri_indices = tri_indices  # list of triangle indices in this node
        self.point = point              # centroid of median triangle
        self.left = left
        self.right = right
        self.axis = axis

class KDTreeTriangles:
    def __init__(self, vertices, triangles, leaf_size=1):
        self.vertices = vertices
        self.triangles = triangles
        self.leaf_size = leaf_size
        self.root = self.build_kdtree(np.arange(len(triangles)))

    def build_kdtree(self, tri_indices, depth=0):
        if len(tri_indices) == 0:
            return None
        axis = depth % 3
        # Compute centroids
        centroids = np.array([np.mean(self.vertices[self.triangles[i]], axis=0) for i in tri_indices])
        sorted_idx = np.argsort(centroids[:, axis])
        tri_indices = tri_indices[sorted_idx]
        centroids = centroids[sorted_idx]
        median_idx = len(tri_indices) // 2

        # Leaf node
        if len(tri_indices) <= self.leaf_size:
            return KDTreeNode(tri_indices, centroids[median_idx], None, None, axis)

        left = self.build_kdtree(tri_indices[:median_idx], depth + 1)
        right = self.build_kdtree(tri_indices[median_idx + 1:], depth + 1)
        return KDTreeNode([tri_indices[median_idx]], centroids[median_idx], left, right, axis)

    def closest_point(self, p):
        best = self._closest_point_recursive(p, self.root, best=None)
        dist, closest_point, tri_idx = best
        return closest_point, dist, tri_idx[0]

    def _closest_point_recursive(self, p, node, best=None):
        if node is None:
            return best

        for tri_idx in node.tri_indices:
            tri = self.triangles[tri_idx]
            q = icp.find_closest_point(p, self.vertices[tri[0]], self.vertices[tri[1]], self.vertices[tri[2]])
            dist = np.linalg.norm(p - q)
            if best is None or dist < best[0]:
                best = (dist, q, [tri_idx])

        axis = node.axis
        diff = p[axis] - node.point[axis]
        first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)

        best = self._closest_point_recursive(p, first, best)
        if abs(diff) < best[0]:
            best = self._closest_point_recursive(p, second, best)
        return best