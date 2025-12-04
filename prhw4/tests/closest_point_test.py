import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.icp import find_closest_point, project_on_segment


def assert_close(x, y, msg):
    if not np.allclose(x, y, atol=1e-6):
        raise AssertionError(f"{msg}\nExpected: {y}\Recieved: {x}")


def run_tests():
    p = np.array([0., 0., 0.])
    q = np.array([1., 0., 0.])
    r = np.array([0., 1., 0.])

    # Interior 
    a = np.array([0.2, 0.2, 0.7])   
    expected = np.array([0.2, 0.2, 0.0])
    c = find_closest_point(a, p, q, r)
    assert_close(c, expected, "FAIL")

    # Vertex p
    a = np.array([-0.3, -0.2, 0.4])
    expected = p
    c = find_closest_point(a, p, q, r)
    assert_close(c, expected, "FAIL")

    # Vertex q
    a = np.array([1.3, -0.1, 0.4])
    expected = q
    c = find_closest_point(a, p, q, r)
    assert_close(c, expected, "FAIL")

    # Vertex r
    a = np.array([-0.1, 1.4, -0.2])
    expected = r
    c = find_closest_point(a, p, q, r)
    assert_close(c, expected, "FAIL")

    # Edge pq
    a = np.array([0.5, -0.3, 0.2])
    expected = np.array([0.5, 0.0, 0.0])
    c = find_closest_point(a, p, q, r)
    assert_close(c, expected, "FAIL")

    # Edge pr
    a = np.array([-0.2, 0.5, 0.1])
    expected = np.array([0.0, 0.5, 0.0])
    c = find_closest_point(a, p, q, r)
    assert_close(c, expected, "FAIL")

    # Edge qr
    a = np.array([0.7, 0.7, -0.3])
    expected = np.array([0.5, 0.5, 0.0])
    c = find_closest_point(a, p, q, r)
    assert_close(c, expected, "FAIL")

    # Near edge pr
    a = np.array([-0.5, 0.2, 0.3])
    expected = project_on_segment(a, r, p)
    c = find_closest_point(a, p, q, r)
    assert_close(c, expected, "FAIL")

    # Near edge pq
    a = np.array([0.3, -0.5, 0.1])
    expected = project_on_segment(a, p, q)
    c = find_closest_point(a, p, q, r)
    assert_close(c, expected, "FAIL")

    # Near edge qr
    a = np.array([1.1, 1.1, -0.2])
    expected = project_on_segment(a, q, r)
    c = find_closest_point(a, p, q, r)
    assert_close(c, expected, "FAIL")

    print("All tests PASS")


if __name__ == "__main__":
    run_tests()
