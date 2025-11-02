import numpy as np
import matplotlib.pyplot as plt
from utils.calibrator import ProbeCalibration
from utils.bpoly import BPoly
from utils import parse as parser
from utils import plot as plotter
from utils import write_out as writer
from utils import calculate_errors as calcerr

ERROR_DISTORTION_THRESHOLD = 2.0 
CONVERGENCE_THRESHOLD = 0.001 

# Main script for PA2. Run from start to finish to produce all output.txt's

#datasets = ['a']
#datasets = ['a', 'b', 'c', 'd', 'e', 'f'] # Only debug datasets, use for diffing output test
datasets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] # All datasets


for letter in datasets:
    # Special case when we transition from debug to unknown sets
    prefix = 'debug' if letter <= 'f' else 'unknown'

    # File paths
    calbody_path = f"./data/pa2-{prefix}-{letter}-calbody.txt"
    calreadings_path = f"./data/pa2-{prefix}-{letter}-calreadings.txt"
    empivot_path = f"./data/pa2-{prefix}-{letter}-empivot.txt"
    optpivot_path = f"./data/pa2-{prefix}-{letter}-optpivot.txt"
    ctfid_path = f"./data/pa2-{prefix}-{letter}-ct-fiducials.txt"
    emfid_path = f"./data/pa2-{prefix}-{letter}-em-fiducialss.txt"
    emnav_path = f"./data/pa2-{prefix}-{letter}-EM-nav.txt"

    output_path1 = f"./out/pa2-{prefix}-{letter}-output1.txt"
    output_path2 = f"./out/pa2-{prefix}-{letter}-output2.txt"

    print(f"----------Processing dataset {letter}----------")
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

    # Flatten into paired point clouds
    C_measured_flat = C_frames.reshape(-1, 3)
    C_expected_flat = C_expected_frames.reshape(-1, 3)

    #----------Step 2----------
    # Fit error function to polynomial
    # If RMS error is high, use BPoly to fit and apply corrections
    C_rms = calcerr.calculate_rms_error(C_measured_flat, C_expected_flat)
    if (C_rms > ERROR_DISTORTION_THRESHOLD):
        bpoly = None
        order = 0
        best_rms = np.inf
        prev_rms = np.inf

        print(f"Fitting distortion for {letter} with RMS error {C_rms:.4f} mm")
        for test_order in range(1, 100):
            test = BPoly(order=test_order)
            test.fit(C_measured_flat, C_expected_flat)
            test_corrected = test.apply(C_measured_flat)
            test_error_rms = calcerr.calculate_rms_error(test_corrected, C_expected_flat)
            print(f"    Order {test_order}: RMS = {test_error_rms:.4f} mm")

            if test_error_rms < best_rms:
                best_rms = test_error_rms
                order = test_order
                bpoly = test
            
            # Check convergence
            if test_order > 1 and prev_rms - test_error_rms < CONVERGENCE_THRESHOLD:
                print(f"    Converged at order {test_order}")
                break
            
            prev_rms = test_error_rms
        
        print(f"    Selected polynomial order {order} with RMS error {best_rms:.4f} mm")
        # Interpolate and apply corrections
        C_predicted_flat = bpoly.apply(C_measured_flat)
        C_predicted_frames = C_predicted_flat.reshape(C_frames.shape)
    else:
        C_predicted_flat = C_measured_flat.copy() 
        C_predicted_frames = C_frames.copy() 

    #----------Step 3----------
    # Repeat pivot calibration with corrected points
    emprobe = ProbeCalibration("emprobe")
    G_all = parser.parse_empivot(empivot_path)

    # Apply distortion correction to G_all
    G_all_corrected = np.zeros_like(G_all)
    for k in range(len(G_all)):
        if (C_rms > ERROR_DISTORTION_THRESHOLD):
            G_all_corrected[k] = bpoly.apply(G_all[k])
        else: 
            G_all_corrected[k] = G_all[k]  # rm bpoly

    # calculate tool frame
    tool_origin = G_all_corrected[0].mean(axis=0)
    emprobe.local_frame_points = G_all_corrected[0] - tool_origin

    # Calculate PCR for each reading
    T_all = np.zeros((len(G_all_corrected),4,4))
    em_probe_expected_all = []
    for k, frame in enumerate(G_all_corrected):
        F_G = emprobe.point_cloud_registration(emprobe.local_frame_points, frame)
        T_all[k] = F_G

    p_tip_em, p_pivot_em = emprobe.pivot_calibration(T_all)

    #----------Step 3b----------
    # Do Optical pivot calibration for output 1
    # (Not needed by rest of the code)
    optprobe = ProbeCalibration("optprobe")
    D_all, H_all = parser.parse_optpivot(optpivot_path)
    d, _, _ = parser.parse_calbody(calbody_path)

    H_all_em = np.zeros(np.shape(H_all))

    # calculate tool frame
    tool_origin = H_all[0].mean(axis=0)
    optprobe.local_frame_points = H_all[0] - tool_origin
    # Calculate PCR for each reading
    T_all = np.zeros((len(H_all_em),4,4))
    optprobe_expected_all = []
    for k,frame in enumerate(H_all):
        F_D = optprobe.point_cloud_registration(d, D_all[k])
        R_D = np.linalg.inv(F_D)[:3,:3]
        t_D = np.linalg.inv(F_D)[:3,3:4]
        H_em = (R_D @ frame.T + t_D).T
        H_all_em[k] = H_em

        F_G = optprobe.point_cloud_registration(optprobe.local_frame_points,H_em)
        T_all[k] = F_G

        optprobe_expected = (F_G[:3, :3] @ optprobe.local_frame_points.T + F_G[:3, 3:4]).T
        optprobe_expected_all.append(optprobe_expected)
    # Perform pivot calibration
    p_tip_opt,p_pivot_opt = optprobe.pivot_calibration(T_all)
    

     #---------- Step 4 ----------
    # Correct fiducials 
    b_ct = parser.parse_ctfiducials(ctfid_path)        
    G_fid_all = parser.parse_emfiducials(emfid_path)  

    G_fid_all_corrected = np.empty_like(G_fid_all)
    for i in range(G_fid_all.shape[0]):
        if (C_rms > ERROR_DISTORTION_THRESHOLD):
            G_fid_all_corrected[i] = bpoly.apply(G_fid_all[i])
        else: 
            G_fid_all_corrected[i] = G_fid_all[i]  # rm bpoly

    # Compute tip positions in EM base frame
    tip_positions_em = np.zeros((b_ct.shape[0], 3))
    for i in range(G_fid_all_corrected.shape[0]):
        F_G = emprobe.point_cloud_registration(emprobe.local_frame_points, G_fid_all_corrected[i])
        tip_positions_em[i] = F_G[:3, :3] @ p_tip_em + F_G[:3, 3]   # tip in EM base

    #---------- Step 5 ----------
    # Registration EM -> CT
    registration = ProbeCalibration("registration")
    F_reg = registration.point_cloud_registration(tip_positions_em, b_ct)  

    #---------- Step 6 ----------
    # Navigation frames -> CT coordinates
    G_nav_all = parser.parse_emnav(emnav_path)  
    G_nav_corrected = np.empty_like(G_nav_all)
    for k in range(G_nav_all.shape[0]):
        if (C_rms > ERROR_DISTORTION_THRESHOLD):
            G_nav_corrected[k] = bpoly.apply(G_nav_all[k])
        else: 
            G_nav_corrected[k] = G_nav_all[k]  # rm bpoly

    tip_positions_ct = np.zeros((G_nav_corrected.shape[0], 3))
    for k in range(G_nav_corrected.shape[0]):
        F_G = emprobe.point_cloud_registration(emprobe.local_frame_points, G_nav_corrected[k])
        p_tip_em_frame = F_G[:3, :3] @ p_tip_em + F_G[:3, 3]
        tip_positions_ct[k] = F_reg[:3, :3] @ p_tip_em_frame + F_reg[:3, 3]

    # Write out
    writer.write_output_pa1(C_expected_frames, p_pivot_em, p_pivot_opt, output_path1)
    writer.write_output_pa2(tip_positions_ct, output_path2)
    

    