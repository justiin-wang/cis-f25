import os
import numpy as np

def write_p4_output(s_points, c_points, errors, output_path):
    s_points = np.asarray(s_points, dtype=float)
    c_points = np.asarray(c_points, dtype=float)
    errors   = np.asarray(errors, dtype=float).reshape(-1)

    if s_points.shape != c_points.shape or s_points.shape[1] != 3:
        raise ValueError("s_points and c_points must be (N,3) with the same shape.")
    if errors.shape[0] != s_points.shape[0]:
        raise ValueError("errors must have length N matching s_points.")

    N = s_points.shape[0]

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"{N} {os.path.basename(output_path)}\n")
        for i in range(N):
            sx, sy, sz = s_points[i]
            cx, cy, cz = c_points[i]
            e = errors[i]
            f.write(
                f"{sx:10.3f} {sy:10.3f} {sz:10.3f}   "
                f"{cx:10.3f} {cy:10.3f} {cz:10.3f}   "
                f"{e:10.6f}\n"
            )

    print(f"Wrote PA4 output to {output_path}")
