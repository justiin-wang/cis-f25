import numpy as np

def calculate_rms_error(pc1, pc2):
    assert pc1.shape == pc2.shape 
    diff = pc1 - pc2
    squared_diff = np.sum(diff**2, axis=1)
    mean_squared_diff = np.mean(squared_diff)
    rms_error = np.sqrt(mean_squared_diff)

    return rms_error

def calculate_error_transformation(Fa, Fb):
    assert Fa.shape == Fb.shape and Fa.shape == (4, 4)
    F_diff = np.linalg.inv(Fa) @ Fb
    R_diff = F_diff[:3, :3]
    p_diff = F_diff[:3, 3]

    angle_error = np.arccos((np.trace(R_diff) - 1) / 2)
    translation_error = np.linalg.norm(p_diff)

    return F_diff, angle_error, translation_error

def calculate_error_stats(pc1, pc2):
    assert pc1.shape == pc2.shape and pc1.ndim == 2 and pc1.shape[1] == 3

    diff = pc1 - pc2
    error_magnitudes = np.linalg.norm(diff)

    mean_error = np.mean(error_magnitudes)
    max_error = np.max(error_magnitudes)
    min_error = np.min(error_magnitudes)

    return mean_error, max_error, min_error