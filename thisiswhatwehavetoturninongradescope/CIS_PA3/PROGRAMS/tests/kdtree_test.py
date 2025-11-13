import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.kdtree import KDTreeTriangles


def test_triangle_kdtree():
    # single triangle
    vertices = np.array([[0,0,0], [1,0,0], [0,1,0]])
    triangles = np.array([[0,1,2]])
    tree = KDTreeTriangles(vertices, triangles)
    
    # find point on triangle closest to a point inside triangle
    # should just return itself
    query1 = np.array([0.25,0.25,0])
    result1, _, _ = tree.closest_point(query1)
    assert np.allclose(query1, result1), f"Expected {query1}, got {result1}"

    # point outside triangle
    query2 = np.array([2,2,0])
    result2, _, _ = tree.closest_point(query2)
    expected = np.array([0.5,0.5,0])  # will be a point on an edge of the triangle
    assert np.allclose(result2, expected), f"Expected {expected}, got {result2}"

    # multiple triangles
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[1,1,1]])
    triangles = np.array([[0,1,2],[1,3,2],[1,3,4]])
    tree = KDTreeTriangles(vertices, triangles)
    # point inside a triangle
    query3 = np.array([0.75,0.25,0])
    result3, _, _ = tree.closest_point(query3)
    expected3 = np.array([0.75,0.25,0])
    assert np.allclose(result3, expected3), f"Expected {expected3}, got {result3}"

    # point outside all triangles
    query4 = np.array([1,1,2])
    result4, _, _ = tree.closest_point(query4)
    # closes point is one of the vertices
    expected4 = np.array([1,1,1])
    assert np.allclose(result4, expected4), f"Expected {expected4}, got {result4}"

    print("kdtree tests passed")

if __name__ == "__main__":
    test_triangle_kdtree()
