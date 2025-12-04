import numpy as np

# Classic rotation matrices
def Rx(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])
def Ry(theta):
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
def Rz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

# Class skew symmetric matrix
def skew(v):
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])

# Build measurement Jacobian H from fiducial points a_list and nominal pose F_nominal
def compute_H(a_list, F_nominal):
    rows = []
    for a in a_list:
        b_nom = F_nominal[:3,:3] @ a + F_nominal[:3,3]
        W = skew(b_nom)        
        Hk = np.hstack([-W, np.eye(3)])
        rows.append(Hk)
    return np.vstack(rows)       

# Build covariance matrix from measurement Jacobian H and noise sigma
def compute_covariance(H, sigma):
    Cnoise = (sigma**2) * np.eye(3)
    Cinv = np.kron(np.eye(len(H)//3), np.linalg.inv(Cnoise))
    return np.linalg.inv(H.T @ Cinv @ H)   

# Build Jacobian Hq to propagate covariance to pointer tip
def compute_Hq(F_ptr, p_tip):
    p_cam = F_ptr[:3,:3] @ p_tip + F_ptr[:3,3]
    return np.hstack([-skew(p_cam), np.eye(3)])  

#---------- Givens from Problem 2 Setup ----------
# F_ptr
R_ptr = Rx(np.pi/4) @ Ry(-np.pi/6) @ Rz(-np.pi/6)
p_ptr = np.array([1500, 200, 300])
F_ptr = np.eye(4)
F_ptr[:3, :3] = R_ptr
F_ptr[:3, 3] = p_ptr

# F_A
R_A = Ry(np.pi/6) @ Rz(np.pi/6)
p_A = F_ptr[:3,:3] @ np.array([50, 50, 50]) + F_ptr[:3,3] # Bit change from piaza

F_A = np.eye(4)
F_A[:3, :3] = R_A
F_A[:3, 3]  = p_A

# F_AC
F_AC = np.eye(4) # Identity from pdf

# Inverse FA
FA_inv = np.linalg.inv(F_A)

# p/L_tip
ptip = np.array([0.0, 0.0, -100.0]) 

# Noise
sigma_b = 0.15 # Tracker inaccuracy
sigma_s = 0.10 # Segmentation inaccuracy

# Specs
sigma_nav = 2.00 # Max allowed navigation uncertainty
sigma_q = 0.5 # Max pointer tip uncertainty

#---------- Problem 2a ----------
# Construct pointer tracker target from a_ptr_k fiducials
a_ptr = [
    np.array([-20, 0.0, 20]), # a_ptr_0
    np.array([20, 0.0, 40]), # a_ptr_1
    np.array([40, 0, 10]), # a_ptr_2
    np.array([-40,0, -40])  # a_ptr_3
]
H_ptr = compute_H(a_ptr, F_ptr) # Jacobian for pointer tracker
C_ptr = compute_covariance(H_ptr, sigma_b) # Covariance of pointer tracker target
H_q = compute_Hq(F_ptr, ptip) # Jacobian for tip point
C_q = H_q @ C_ptr @ H_q.T # Cov propagation to tip
eig_C_ptr = np.linalg.eigvalsh(C_ptr)
eig_C_q = np.linalg.eigvalsh(C_q)
print("Problem 2a:")
print("Eigenvalues of C_ptr:")
print(eig_C_ptr)
print("\nEigenvalues of C_q:")
print(eig_C_q)
print("Max, worst-case eig_q:", np.sqrt(np.max(eig_C_q)))
assert np.sqrt(np.max(eig_C_q)) < sigma_q, "Pointer tip uncertainty exceeds max allowance"

#---------- Problem 2b ----------
a_A = [
    2*np.array([-40, 0, 40]),  # a_A_0
    2*np.array([40, 0, 20]),  # a_A_1
    2*np.array([20, -20, 0]),  # a_A_2
    2*np.array([-40, 0, -40])   # a_A_3
]

H_A = compute_H(a_A, F_A)
C_A = compute_covariance(H_A, sigma_b)
eig_C_A = np.linalg.eigvalsh(C_A)
print("Problem 2b:")
print("Eigenvalues of C_A:")
print(eig_C_A)

#---------- Problem 2c ----------
H_AC = compute_H(a_A, F_AC)
C_AC = compute_covariance(H_AC, sigma_s)
eig_C_AC = np.linalg.eigvalsh(C_AC)
print("Problem 2c:")
print("Eigenvalues of C_AC:")
print(eig_C_AC)

#---------- Problem 2d ----------
R_ptr = F_ptr[:3,:3]
p_ptr = F_ptr[:3,3]
R_A   = F_A[:3,:3]
p_A   = F_A[:3,3]
R_AC  = F_AC[:3,:3]
p_AC  = F_AC[:3,3]

p_tip = ptip

# Tip in camera frame
b_tip = R_ptr @ p_tip + p_ptr

# Combined rotation from camera to CT via A
R_G = R_AC @ R_A.T

# Tip in A coords
y = FA_inv[:3,:3] @ (b_tip - p_A)

# Navigation point in CT coords
p_nav = R_AC @ y + p_AC
M_ptr = np.hstack([
    -R_G @ skew(b_tip),   
     R_G                  
])

M_A = np.hstack([
     R_G @ skew(b_tip),   
    -R_G                  
])

M_AC = np.hstack([
    -skew(p_nav),         
     np.eye(3)            
])
# Covariance propagation
C_nav = (
    M_ptr @ C_ptr @ M_ptr.T +
    M_A   @ C_A   @ M_A.T   +
    M_AC  @ C_AC  @ M_AC.T
)

eig_C_nav = np.linalg.eigvalsh(C_nav)
sigma_max = np.sqrt(np.max(eig_C_nav))

print("Problem 2d:")
print("Eigenvalues of C_nav:")
print(eig_C_nav)
print("Max, worst-case sigma_nav:", sigma_max)

assert sigma_max < sigma_nav, "Navigation uncertainty exceeds max allowance"

# Generate output

with open("output.txt", "w") as f:
    # Anatomy markers
    for a in a_A:
        f.write(f"A,{a[0]},{a[1]},{a[2]}\n")

    # Pointer markers
    for a in a_ptr:
        f.write(f"P,{a[0]},{a[1]},{a[2]}\n")

    # Eigenvalues (pointer body, pointer tip, anatomy body, segmentation, navigation)
    f.write("\nEigenvalues\n")

    f.write("eig_C_ptr = " + np.array2string(eig_C_ptr, separator=",") + "\n")
    f.write("eig_C_q = "   + np.array2string(eig_C_q,   separator=",") + "\n")
    f.write("eig_C_A = "   + np.array2string(eig_C_A,   separator=",") + "\n")
    f.write("eig_C_AC = "   + np.array2string(eig_C_AC,  separator=",") + "\n")
    f.write("eig_C_nav = "   + np.array2string(eig_C_nav, separator=",") + "\n")




