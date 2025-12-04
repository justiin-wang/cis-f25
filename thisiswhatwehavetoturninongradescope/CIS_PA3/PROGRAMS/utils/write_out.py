import os
import numpy as np

def write_p3_output(d_points, c_points, errors, output_path):
    d_points = np.asarray(d_points, dtype=float)
    c_points = np.asarray(c_points, dtype=float)
    errors   = np.asarray(errors, dtype=float).reshape(-1)

    if d_points.shape != c_points.shape or d_points.shape[1] != 3:
        raise ValueError("d_points and c_points must be (N,3) with the same shape.")
    if errors.shape[0] != d_points.shape[0]:
        raise ValueError("errors must have length N matching d_points.")

    N = d_points.shape[0]

    # Ensure directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Write file
    with open(output_path, "w") as f:
        f.write(f"{N} {os.path.basename(output_path)}\n") # Match given header (space instead of comma)
        for i in range(N):
            dx, dy, dz = d_points[i]
            cx, cy, cz = c_points[i]
            e = errors[i]
            f.write(
                f"{dx:8.2f} {dy:8.2f} {dz:8.2f}     "
                f"{cx:8.2f} {cy:8.2f} {cz:8.2f}   "
                f"{e:8.3f}\n"
            )

    print(f"Wrote PA3 output to {output_path}")
