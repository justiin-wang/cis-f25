import numpy as np
from pathlib import Path

def write_output_pa1(C_expected_frames, p_post_em, p_post_opt):
    # Flatten frames
    C_expected_frames = np.asarray(C_expected_frames)
    p_post_em = np.asarray(p_post_em).flatten()
    p_post_opt = np.asarray(p_post_opt).flatten()

    # Retrieve header
    Nf, Nc, _ = C_expected_frames.shape

    # Output path
    out_dir = Path("./OUTPUT")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "pa1-output-1.txt"

    # Write out
    with open(output_path, "w") as f:
        # Header 
        f.write(f"{Nc}, {Nf}, pa1-output-1.txt\n")

        # 2-3 Pivot Cals
        f.write(f"{p_post_em[0]:.2f}, {p_post_em[1]:.2f}, {p_post_em[2]:.2f}\n")
        f.write(f"{p_post_opt[0]:.2f}, {p_post_opt[1]:.2f}, {p_post_opt[2]:.2f}\n")

        # Write out C expected frames
        for k in range(Nf):
            for c in range(Nc):
                x, y, z = C_expected_frames[k, c]
                f.write(f"{x:.2f}, {y:.2f}, {z:.2f}\n")

    print(f"Wrote output to {output_path}")
