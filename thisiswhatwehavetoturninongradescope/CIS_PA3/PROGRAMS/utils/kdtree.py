import numpy as np
from utils import icp as icp

def point_bbox_distance(p, bbox_min, bbox_max):
    # helper function to find distance from query point to a given bounding box
    # if distance to bounding box is negative, the point is inside bounding box
    # so definitely want to go there, set distance to 0
    d = np.maximum(0.0, np.maximum(bbox_min - p, p - bbox_max))
    return np.linalg.norm(d)

class KDTreeNode:
    def __init__(self, tri_idx, point=None, left=None, right=None, axis=0, bbox_min=None, bbox_max=None):
        self.tri_idx = tri_idx # index of the triangle stored at this node
        self.point = point # centroid of the triangle
        self.left = left # node that is on the left of the current node
        self.right = right # node that is on the right side of the current node
        self.axis = axis # the axis in which the current node's children are split by
        # we can define a triangle's bounding box with the two opposing corners of the triangle
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

class KDTreeTriangles:
    def __init__(self, vertices, triangles):
        # vertices and their indices will initially correspond
        self.vertices = vertices # [N_vertices,3] (x,y,z) coordinates of each vertex in the triangle mesh
        self.triangles = triangles # [N_triangles,3] indices of each vertex grouped by triangle
        # find bounding boxes for each triangle
        self.tri_bbox_min = np.min(vertices[triangles], axis=1) # component-wise min. vertex of each triangle
        self.tri_bbox_max = np.max(vertices[triangles], axis=1) # component-wise max vertex of each triangle
        self.centroids = np.mean(vertices[triangles], axis=1) # array of centroids for each triangle
        self.root = self.build_kdtree(np.arange(len(triangles)))

    def build_kdtree(self, tri_indices, depth=0):
        if len(tri_indices) == 0:
            return None
        axis = depth % 3

        # find the indices that would sort the triangles by the 
        # location of their centroids along the given axis
        sorted_idx = np.argsort(self.centroids[tri_indices, axis])
        tri_indices = tri_indices[sorted_idx] # tri_indices is now in sorted order
         # find the median triangle along the given axis and its centroid
        median_idx = len(tri_indices) // 2
        median_tri = tri_indices[median_idx]
        centroid = self.centroids[median_tri]

        # find the bounding box for this node (will cover itself and all of its children)
        # Note: the root node's bounding box will encompass the entire mesh.
        bbox_min = np.min(self.tri_bbox_min[tri_indices], axis=0)
        bbox_max = np.max(self.tri_bbox_max[tri_indices], axis=0)

        # make the leaf
        if len(tri_indices) == 1:
            return KDTreeNode(list(tri_indices), centroid, None, None, axis, bbox_min, bbox_max)

        # recursively build the tree by splitting the triangles along the median
        # choose axis to split along by round robin. found experimentally to
        # work well enough for the given data, likely because the points on
        # the meshes are evenly distributed w.r.t each axis
        left = self.build_kdtree(tri_indices[:median_idx], depth + 1)
        right = self.build_kdtree(tri_indices[median_idx + 1:], depth + 1)
        # the current node should store the median triangle
        return KDTreeNode([median_tri], centroid, left, right, axis, bbox_min, bbox_max)
    
    def closest_point(self, p):
        best = None
        stack = [self.root] # DFS

        while stack:
            node = stack.pop()
            if node is None:
                continue

            # skip if node's bounding box is farther than current best
            # not applicable on the first pass
            if best is not None:
                box_dist = point_bbox_distance(p, node.bbox_min, node.bbox_max)
                if box_dist >= best[0]:
                    continue

            # check distance to triangle stored in this node
            v0, v1, v2 = self.vertices[self.triangles[node.tri_idx[0]]]
            q = icp.find_closest_point(p, v0, v1, v2)
            dist = np.linalg.norm(p - q)
            if best is None or dist < best[0]:
                best = (dist, q, [node.tri_idx[0]])

            # decide which subtree to visit first, want to visit nearest one first
            axis = node.axis
            diff = p[axis] - node.point[axis] # want to find which half (left or right) is closer to the query point
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)

            # push the closer node last so we hit it first (DFS)
            stack.append(far)
            stack.append(near)

        dist, closest_point, tri_idx = best
        return closest_point, dist, tri_idx[0]

