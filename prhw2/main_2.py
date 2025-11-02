import numpy as np
import matplotlib.pyplot as plt
from utils.calibrator import ProbeCalibration
from utils import parse as parser
from utils import plot as plotter
from utils import write_out as writer

# Main script for PA. Run from start to finish to produce all output.txt's
# datasets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'] 
datasets = ['a']

for letter in datasets:
    prefix = 'debug' if letter <= 'g' else 'unknown'

    # File paths
    calbody_path = f"./data/pa2-{prefix}-{letter}-calbody.txt"
    calreadings_path = f"./data/pa2-{prefix}-{letter}-calreadings.txt"
    empivot_path = f"./data/pa2-{prefix}-{letter}-empivot.txt"
    optpivot_path = f"./data/pa2-{prefix}-{letter}-optpivot.txt"

    output_path = f"./out/pa2-{prefix}-{letter}-output-1.txt"

    #----------Step 1----------
    # Find C expected frames from D and A readings
    tool = ProbeCalibration("pcr")
    d, a, c = parser.parse_calbody(calbody_path)
    D_frames, A_frames, C_frames = parser.parse_calreadings(calreadings_path)
    C_expected_frames = [] # k C_expected poinno t cloud

    for k in range(len(D_frames)):
        D_measured = D_frames[k]
        A_measured = A_frames[k]

        F_D = tool.point_cloud_registration(d, D_measured)
        F_A = tool.point_cloud_registration(a, A_measured)

        F_C = np.linalg.inv(F_D) @ F_A # F_C = F_D^-1 * F_A

        C_expected = (F_C[:3, :3] @ c.T + F_C[:3, 3:4]).T # Apply the transformation to vector c
        C_expected_frames.append(C_expected)
    # We now have C_expected_frames and C_frames to create out error function
    C_expected_frames = np.array(C_expected_frames)
    C_frames = np.array(C_frames)
    print(C_expected_frames.shape, C_frames.shape)

    # Flatten into paired points
    C_measured_flat = C_frames.reshape(-1, 3)
    C_expected_flat = C_expected_frames.reshape(-1, 3)
    errors = C_expected_flat - C_measured_flat # Error function

    #----------Step 2----------
    # Fit error function to polynomial
    













    # Write out to output
    #writer.write_output_pa1(C_expected_frames, p_pivot_em, p_pivot_opt, output_path)
