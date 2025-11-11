import numpy as np

def find_closest_point_on_triangle(p, a, b, c):
    # Compute edges and vector from a→p
    ab = b - a
    ac = c - a
    ap = p - a

    # Compute dot products
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a  # Closest to vertex A

    # Check if in vertex region around B
    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b  # Closest to vertex B

    # Check if on edge AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return a + v * ab

    # Check if in vertex region around C
    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c  # Closest to vertex C

    # Check if on edge AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return a + w * ac

    # Check if on edge BC
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    # Inside face region — compute barycentric coordinates
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w

def linear_search_closest_points_on_mesh(p, vertices, triangles):
    min_dist = np.inf
    q_closest = None
    tri_index = -1

    # Simple linear search over all triangles
    for i, tri in enumerate(triangles): 
        q = find_closest_point_on_triangle(p, vertices[tri[0]], vertices[tri[1]], vertices[tri[2]])
        dist = np.linalg.norm(p - q)
        if dist < min_dist:
            min_dist = dist
            q_closest = q
            tri_index = i

    return q_closest, min_dist, tri_index
def ktree_search_closest_points_on_mesh(p, vertices, triangles, tree, centroids, k=10):
    """
    Find closest point on mesh to p using KD-Tree over triangle centroids.
    """
    # 1. Query k nearest triangle centroids to point p
    dists, idxs = tree.query(p, k=k)  # can return single or array of indices
    if np.isscalar(idxs):
        idxs = [idxs]

    # 2. Search only these triangles
    min_dist = np.inf
    q_closest = None
    tri_index = -1

    for i in idxs:
        tri = triangles[i]
        q = find_closest_point_on_triangle(p, vertices[tri[0]], vertices[tri[1]], vertices[tri[2]])
        dist = np.linalg.norm(p - q)
        if dist < min_dist:
            min_dist = dist
            q_closest = q
            tri_index = i

    return q_closest, min_dist, tri_index
    

def test_closest_point_on_triangle():
    # Simple test: point near triangle in XY-plane
    a = np.array([0., 0., 0.])
    b = np.array([1., 0., 0.])
    c = np.array([0., 1., 0.])

    # point above center
    p = np.array([0.3, 0.3, 1.])
    q = find_closest_point_on_triangle(p, a, b, c)
    print(q)  # Expect ≈ [0.3, 0.3, 0.]
    assert np.allclose(q, np.array([0.3, 0.3, 0.])), "FAIL"

if __name__ == "__main__":
    test_closest_point_on_triangle()