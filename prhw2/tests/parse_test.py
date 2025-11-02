import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import parse as parser

# TODO: TS FILE PATH DOESNT WORK MF

if __name__ == "__main__":
    # Verify parsing functions are good by checking shapes

    ctfiducials_path = "./data/pa2-debug-a-ct-fiducials.txt"
    ct_fiducials = parser.parse_ctfiducials(ctfiducials_path)
    print("CT fiducials shape:", ct_fiducials.shape)

    emfiducials_path = "./data/pa2-debug-a-em-fiducials.txt"
    em_fiducials = parser.parse_emfiducials(emfiducials_path)
    print("EM fiducials shape:", em_fiducials.shape)

    emnav_path = "./data/pa2-debug-a-EM-nav.txt"
    em_nav_frames = parser.parse_emnav(emnav_path)
    print("EM nav frames shape:", em_nav_frames.shape)



