import numpy as np
import numpy as np

def project_on_segment(c, p, q):
    pq = q - p
    t = np.dot(c - p, pq) / np.dot(pq, pq)
    t = max(0, min(1, t))
    return p + t * pq

def find_closest_point(a, p, q, r):

    # Edge vectors
    qp = q - p
    rp = r - p

    # Solve least-squares for barycentric coords (slide 11)
    A = np.column_stack((qp,     rp))        # 3×2 matrix
    b = a - p                            # 3×1

    # λ, μ = least-squares solution
    lam_mu, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    lam, mu = lam_mu

    # Interior test (slide 12/15)
    if lam >= 0 and mu >= 0 and lam + mu <= 1:
        return p + lam * qp + mu * rp

    # Otherwise determine which edge region (slide 12/15)
    # Region tests using barycentric coordinates:
    # λ < 0 → edge (r,p)
    # μ < 0 → edge (p,q)
    # λ + μ > 1 → edge (q,r)

    c = p + lam * qp + mu * rp   # unconstrained projection

    if lam < 0:          # region near edge r-p
        return project_on_segment(c, r, p)
    if mu < 0:           # region near edge p-q
        return project_on_segment(c, p, q)
    # else λ + μ > 1 → region near edge q-r
    return project_on_segment(c, q, r)


def linear_search_closest_points_on_mesh(p, vertices, triangles):
    min_dist = np.inf
    q_closest = None
    tri_index = -1

    # Simple linear search over all triangles
    for i, tri in enumerate(triangles): 
        q = find_closest_point(p, vertices[tri[0]], vertices[tri[1]], vertices[tri[2]])
        dist = np.linalg.norm(p - q)
        if dist < min_dist:
            min_dist = dist
            q_closest = q
            tri_index = i

    return q_closest, min_dist, tri_index

def ktree_search_closest_points_on_mesh(p, vertices, triangles, tree, centroids, k=10):
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
        q = find_closest_point(p, vertices[tri[0]], vertices[tri[1]], vertices[tri[2]])
        dist = np.linalg.norm(p - q)
        if dist < min_dist:
            min_dist = dist
            q_closest = q
            tri_index = i

    return q_closest, min_dist, tri_index
    

def test_closest_point_on_triangle():
    # Define simple triangle
    p = np.array([0., 0., 0.])
    q = np.array([1., 0., 0.])
    r = np.array([0., 1., 0.])

    # Point near vertex p
    a = np.array([-0.5, -0.4, 0.2])
    expected = p
    result = find_closest_point(a, p, q, r)
    assert np.allclose(result, expected), f"FAIL"

     # Point beyond q
    a = np.array([1.3, -0.2, 0.5])
    expected = q
    result = find_closest_point(a, p, q, r)
    assert np.allclose(result, expected), f"FAIL"

    # Point beyond r
    a = np.array([-0.2, 1.2, 0.5])
    expected = r
    result = find_closest_point(a, p, q, r)
    assert np.allclose(result, expected), f"FAIL"

    # Point pq
    a = np.array([0.5, -0.3, 0.1])
    expected = np.array([0.5, 0.0, 0.0])
    result = find_closest_point(a, p, q, r)
    assert np.allclose(result, expected), f"FAIL"

    # Point pr
    a = np.array([-0.2, 0.5, -0.1])
    expected = np.array([0.0, 0.5, 0.0])
    result = find_closest_point(a, p, q, r)
    assert np.allclose(result, expected), f"FAIL"

    # Point qr
    a = np.array([0.6, 0.6, 0.3])
    expected = np.array([0.5, 0.5, 0.0]) 
    result = find_closest_point(a, p, q, r)
    assert np.allclose(result, expected), f"FAIL"

    # Point on triangle face
    a = np.array([0.25, 0.25, 0.8])
    expected = np.array([0.25, 0.25, 0.0])
    result = find_closest_point(a, p, q, r)
    assert np.allclose(result, expected), f"FAIL"    

if __name__ == "__main__":
    test_closest_point_on_triangle() # If no assertion errors, all pass
    print("All tests PASS") 