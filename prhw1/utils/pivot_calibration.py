import numpy as np

# Pivot calibration
# Known: 
#   - Tool origin (R, p) in world frame (from registration)
# Want:
#   - Tool tip offset in tool frame (p_tip)
#   - Pivot location in world frame (p_pivot)

# For a given tracker, let:
# p_world be p_tip in world frame.
# We take j measurements of tool frame (R_j, p_j)
# For a given measurement i,
#   p_world = R_i * p_tip + p_i
# But p_world = p_pivot, so
#   p_pivot = R_i * p_tip + p_i
#   R_i * p_tip - p_pivot = -p_i
# Repeat j times and stack
#   [R_1 -I;   [  p_t;     [-p_1;
#    R_2 -I; *  p_pivot] =  -p_2;
#    ..  ..                  ...
#    R_j -I]                -p_j]
# This is just an Ax = b problem.
# x = A^-1 * b which can be solved with SVD
# 
# Assume

def pivot_calibration(T_all):
  """
  Perform pivot calibration

  Parameters:
    R_all: list of 3x3 numpy arrays.
      Measured rotation matrices of tool in world.
    p_all: list of 3x1 numpy arrays.
      Measured translation vectors of tool in world.

  Returns:
    p_tip: 3x1 numpy array.
      Calculated translation vector of tip in tool.
    p_pivot: 3x1 numpy array.
      Calculated translation vector of pivot point in world
  """
  j = len(T)
  A = np.zeros(3 * j, 6)
  b = np.zeros(3 * j, 1)

  for i, T in enumerate(T_all):
        R_curr = T[:3, :3]
        t_curr = T[:3, 3]
        
        A[3*i:3*i+3, :3] = R_curr
        A[3*i:3*i+3, 3:] = -np.eye(3)
        b[3*i:3*i+3] = t_curr
  x = np.linalg.lstsq(A, b)
  p_tip = x[:3]
  p_pivot = x[3:]
  
  return p_tip, p_pivot
    
