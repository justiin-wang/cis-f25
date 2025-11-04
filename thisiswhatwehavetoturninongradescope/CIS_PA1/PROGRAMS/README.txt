Source files:

main.py: contains all code necessary for directly answering Q4, Q5, and Q6

tests/pcr_test.py: contains script to test the point cloud registration function implementation
tests/pivot_test.py: contains script to test the pivot calibration function implementation

utils/parse.py: contains functions to parse the provided data based on their headers
utils/plot.py: contains functions to plot point clouds, offering a visual guide for debugging transformation chains
utils/calculate_errors.py: contains functions meant to calculate a variety of errors for difference use cases
utils/write.py: contains function to produce output file in the correct format
utils/calibrate.py: contains the ProbeCalibration object, which owns and allows intuitive use of the PCR and pivot calibration methods.

Instructions to run main.py: execute with python version and packages as specified in requirements.txt while in the PROGRAMS 
directory. Exact file names (choosing between "a" set vs "b" set of data, etc) can be changed manually for each question.
For Q4, this can be changed on lines 12 and 13
For Q5, this can be changed on line 33
For Q6, this cann be changed on lines 55 and 56
Final output text file will be generated in the OUTPUT directory after the file finishes running.