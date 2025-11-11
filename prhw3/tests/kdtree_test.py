import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.kdtree import KDTreeTriangles


def test_triangle_kdtree():
    vertices = np.array([[0,0,0], [1,0,0], [0,1,0]])
    triangles = np.array([[0,1,2]])
    tree = KDTreeTriangles(vertices, triangles)
    
    # point inside triangle
    p = np.array([0.25,0.25,0])
    q, dist, tri_idx = tree.closest_point(p)
    assert np.allclose(q, p), f"Expected {p}, got {q}"

    # point outside triangle
    p2 = np.array([2,2,0])
    q2, dist2, tri_idx2 = tree.closest_point(p2)
    expected = np.array([0.5,0.5,0])  # closest vertex
    assert np.allclose(q2, expected), f"Expected {expected}, got {q2}"

    # multiple triangles
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    triangles = np.array([[0,1,2],[1,3,2]])
    tree = KDTreeTriangles(vertices, triangles)
    p3 = np.array([0.75,0.25,0])
    q3, dist3, tri_idx3 = tree.closest_point(p3)
    expected3 = np.array([0.75,0.25,0])
    assert np.allclose(q3, expected3), f"Expected {expected3}, got {q3}"

    print("kdtree tests passed")

if __name__ == "__main__":
    test_triangle_kdtree()
