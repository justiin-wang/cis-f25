import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import parse as parser

# TODO: TS FILE PATH DOESNT WORK MF

if __name__ == "__main__":
    # Verify parsing functions are good by checking shapes
    '''
    calbody_path = "prhw1/data/pa1-debug-a-calbody.txt"
    d, a, c = parse_calbody(calbody_path)
    print("D points shape:", d.shape)
    print("A points shape:", a.shape)
    print("C points shape:", c.shape)

    calreadings_path = "prhw1/data/pa1-debug-a-calreadings.txt"
    D_frames, A_frames, C_frames = parse_calreadings(calreadings_path)
    print("D_frames shape:", D_frames.shape)
    print("A_frames shape:", A_frames.shape)
    print("C_frames shape:", C_frames.shape)

    empivot_path = "prhw1/data/pa1-debug-a-empivot.txt"
    empivot_frames = parse_empivot(empivot_path)
    print("Empivot frames shape:", empivot_frames.shape)

    optpivot_path = "prhw1/data/pa1-debug-a-optpivot.txt"
    D_opt_frames, H_frames = parse_optpivot(optpivot_path)
    print("D_opt_frames shape:", D_opt_frames.shape)
    print("H_frames shape:", H_frames.shape)
    '''

    ctfiducials_path = "./data/pa2-debug-a-ct-fiducials.txt"
    ct_fiducials = parser.parse_ctfiducials(ctfiducials_path)
    print("CT fiducials shape:", ct_fiducials.shape)

    emfiducials_path = "./data/pa2-debug-a-em-fiducials.txt"
    em_fiducials = parser.parse_emfiducials(emfiducials_path)
    print("EM fiducials shape:", em_fiducials.shape)

    emnav_path = "./data/pa2-debug-a-EM-nav.txt"
    em_nav_frames = parser.parse_emnav(emnav_path)
    print("EM nav frames shape:", em_nav_frames.shape)



