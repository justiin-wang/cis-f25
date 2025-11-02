import numpy as np

# TODO: Add comments

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
    error_magnitudes = np.linalg.norm(diff, axis=1)  # Compute norm along each row 

    mean_error = np.mean(error_magnitudes)
    max_error = np.max(error_magnitudes)
    min_error = np.min(error_magnitudes)
    std_error = np.std(error_magnitudes)
    rms_error = np.sqrt(np.mean(error_magnitudes**2))

    return {
        'mean': mean_error,
        'max': max_error, 
        'min': min_error,
        'std': std_error,
        'rms': rms_error,
        'count': len(error_magnitudes)
    }

def print_error_stats(stats):
    print(f"Points analyzed: {stats['count']}")
    print(f"Mean error:      {stats['mean']:.6f}")
    print(f"RMS error:       {stats['rms']:.6f}")
    print(f"Std deviation:   {stats['std']:.6f}")
    print(f"Min error:       {stats['min']:.6f}")
    print(f"Max error:       {stats['max']:.6f}")