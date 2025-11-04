Source files:

main_2.py: contains all code necessary for this assignment

tests/pcr_test.py: contains script to test the point cloud registration function implementation
tests/pivot_test.py: contains script to test the pivot calibration function implementation
tests/bpoly_test.py: contains script to test the bpoly function implementation
tests/compare_outputs.py: contains script to allo easy comparison of calibration outputs

utils/parse.py: contains functions to parse the provided data based on their headers
utils/plot.py: contains functions to plot point clouds, offering a visual guide for debugging transformation chains
utils/calculate_errors.py: contains functions meant to calculate a variety of errors for difference use cases
utils/write_out.py: contains function to produce output file in the correct format
utils/calibrator.py: contains the ProbeCalibration object, which owns and allows intuitive use of the PCR and pivot calibration methods.
utils/bpoly.py: contains the BPoly object, which owns and allows intuitive use of Bernstein polynomial based correction

Instructions to run main_2.py: execute with python version and packages as specified in requirements.txt while in the PROGRAMS 
directory. Outputs for all files in the data folder (by default contains all files provided for the assignment) will be generated
in OUTPUT/ folder.