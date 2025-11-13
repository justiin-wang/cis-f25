Source files:

main.py: contains all code necessary for this assignment


tests/compare_outputs.py: contains script to test the accuracy of the matching phase of ICP (output of main.py)
tests/pivot_kdtree_test.py: contains script to test the kd-tree implementation

utils/calculate_errors.py: contains functions meant to calculate a variety of errors for difference use cases
utils/icp.py: contains helper functions needed for matching phase for ICP
utils/kdtree.py: contains implementation of kd-tree to store triangle mesh, including member function to find nearest point on triangle mesh to a given query point.
utils/parse.py: contains functions to parse the provided data based on their headers
utils/pcr.py: contains implementation of point cloud registration
utils/write_out.py: contains function to produce output file in the correct format


Instructions to run main.py: execute with python version and packages as specified in requirements.txt while in the PROGRAMS 
directory. Outputs for all files in the data folder (by default contains all files provided for the assignment) will be generated
in OUTPUT/ folder.