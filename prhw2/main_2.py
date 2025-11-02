import numpy as np
import matplotlib.pyplot as plt
from utils.calibrator import ProbeCalibration
from utils.bpoly import BPoly
from utils import parse as parser
from utils import plot as plotter
from utils import write_out as writer

# Main script for PA2. Run from start to finish to produce all output.txt's
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

    # TODO: Add paths for EM/CT fiducials, EM Nav

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

    #----------Step 2----------
    # Fit error function to polynomial
    bpoly = BPoly(order=3)
    bpoly.fit(C_measured_flat, C_expected_flat)

    #----------Step 3----------
    # Interpolate and apply corrections
    C_predicted_flat = bpoly.apply(C_measured_flat)
    C_predicted_frames = C_predicted_flat.reshape(C_frames.shape)

    #----------Step 4----------
    # Repeat pivot calibration with corrected points
    emprobe = ProbeCalibration("emprobe")
    G_all = parser.parse_empivot(empivot_path)

    # Apply distortion correction to G_all
    G_all_corrected = np.zeros_like(G_all)
    for k in range(len(G_all)):
        G_all_corrected[k] = bpoly.apply(G_all[k])

    # calculate tool frame
    tool_origin = G_all_corrected[0].mean(axis=0)
    emprobe.local_frame_points = G_all_corrected[0] - tool_origin

    # Calculate PCR for each reading
    T_all = np.zeros((len(G_all_corrected),4,4))
    for k, frame in enumerate(G_all_corrected):
        T_all[k] = emprobe.point_cloud_registration(emprobe.local_frame_points, frame)

    p_tip_em, p_pivot_em = emprobe.pivot_calibration(T_all)

    #----------Step 5----------
    # TODO: Correct fiducial points 

    #----------Step 6----------
    # TODO: CT Registration

    #----------Step 7----------
    # TODO: Navigation

    # TODO: Write out to output

    