import numpy as np
from utils import icp as icp

def point_aabb_distance(p, aabb_min, aabb_max):
    # distance from point to axis-aligned bounding box
    d = np.maximum(0.0, np.maximum(aabb_min - p, p - aabb_max))
    return np.linalg.norm(d)

class KDTreeNode:
    def __init__(self, tri_indices, point=None, left=None, right=None, axis=0, aabb_min=None, aabb_max=None):
        self.tri_indices = tri_indices
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis
        self.aabb_min = aabb_min
        self.aabb_max = aabb_max

class KDTreeTriangles:
    def __init__(self, vertices, triangles, leaf_size=1):
        self.vertices = vertices
        self.triangles = triangles
        self.leaf_size = leaf_size
        # precompute per-triangle AABBs and centroids
        self.tri_aabb_min = np.min(vertices[triangles], axis=1)   # shape (n_tri,3)
        self.tri_aabb_max = np.max(vertices[triangles], axis=1)
        self.centroids = np.mean(vertices[triangles], axis=1)
        self.root = self.build_kdtree(np.arange(len(triangles)))

    def build_kdtree(self, tri_indices, depth=0):
        if len(tri_indices) == 0:
            return None
        axis = depth % 3

        # sort by centroid along axis
        sorted_idx = np.argsort(self.centroids[tri_indices, axis])
        tri_indices = tri_indices[sorted_idx]
        median_idx = len(tri_indices) // 2
        median_tri = tri_indices[median_idx]
        centroid = self.centroids[median_tri]

        # compute aggregated AABB for this node (min over all triangles in node)
        aabb_min = np.min(self.tri_aabb_min[tri_indices], axis=0)
        aabb_max = np.max(self.tri_aabb_max[tri_indices], axis=0)

        # Leaf
        if len(tri_indices) <= self.leaf_size:
            return KDTreeNode(list(tri_indices), centroid, None, None, axis, aabb_min, aabb_max)

        left = self.build_kdtree(tri_indices[:median_idx], depth + 1)
        right = self.build_kdtree(tri_indices[median_idx + 1:], depth + 1)
        # store the median triangle in this node as well (optional)
        return KDTreeNode([median_tri], centroid, left, right, axis, aabb_min, aabb_max)

    def closest_point(self, p):
        best = self._closest_point_recursive(p, self.root, best=None)
        dist, closest_point, tri_idx = best
        return closest_point, dist, tri_idx[0]

    def _closest_point_recursive(self, p, node, best=None):
        if node is None:
            return best

        # conservative pruning: if distance from p to node's AABB >= current best, skip
        if best is not None:
            box_dist = point_aabb_distance(p, node.aabb_min, node.aabb_max)
            if box_dist >= best[0]:
                return best

        # check triangles stored in this node
        for tri_idx in node.tri_indices:
            v0, v1, v2 = self.vertices[self.triangles[tri_idx]]
            q = icp.find_closest_point(p, v0, v1, v2)
            dist = np.linalg.norm(p - q)
            if best is None or dist < best[0]:
                best = (dist, q, [tri_idx])

        # choose subtree order by centroid along axis (same as before)
        axis = node.axis
        diff = p[axis] - node.point[axis]
        first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)

        # recurse first subtree
        best = self._closest_point_recursive(p, first, best)

        # we should only visit the other subtree if it *might* have closer geometry
        if second is not None:
            # distance from point to that subtree's AABB
            box_dist = point_aabb_distance(p, second.aabb_min, second.aabb_max)
            if best is None or box_dist < best[0]:
                best = self._closest_point_recursive(p, second, best)
        return best
