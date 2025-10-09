import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import colors

def plot_data_1(data, var_name, number_points=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_title(f"{var_name} ({data.shape[0]} points)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add point indices if requested
    if number_points:
        for i in range(len(data)):
            ax.text(data[i, 0], data[i, 1], data[i, 2], f"{i}", fontsize=8)
    
    print(f"{var_name} ({data.shape[0]} points)")
    return fig, ax

def plot_data_2(data1, data2, var_name1, var_name2, number_points=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c='r', alpha=0.2, label=var_name1) 
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c='b', alpha=0.5, label=var_name2)
    ax.set_title(f"{var_name1} and {var_name2} ({data1.shape[0]} and {data2.shape[0]} points)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Add point indices if requested
    if number_points:
        # Number the first dataset (red points)
        for i in range(len(data1)):
            ax.text(data1[i, 0], data1[i, 1], data1[i, 2], f"{i}", color='darkred', fontsize=8)
        
        # Number the second dataset (blue points)
        for i in range(len(data2)):
            ax.text(data2[i, 0], data2[i, 1], data2[i, 2], f"{i}", color='darkblue', fontsize=8)
    
    return fig, ax

def plot_data_error_vectors(data1, data2, var_name1, var_name2):
    """
    Plot two point clouds with error vectors between corresponding points.
    Colors are mapped using viridis colormap based on translational error magnitude.
    
    Args:
        data1, data2: numpy arrays of shape (N, 3) - corresponding 3D points
        var_name1, var_name2: strings for plot labels
    """
    # Ensure both datasets have the same number of points
    assert data1.shape == data2.shape, f"Point clouds must have same shape: {data1.shape} vs {data2.shape}"
    assert data1.shape[1] == 3, f"Expected 3D points, got shape: {data1.shape}"
    
    # Calculate error vectors and magnitudes
    error_vectors = data2 - data1
    error_magnitudes = np.linalg.norm(error_vectors, axis=1)
    
    # Create the plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot reference points (data1) as small gray dots
    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], 
              c='gray', alpha=0.7, s=10, label=f'{var_name1} (reference)')
    
    # Use absolute error magnitudes for consistent comparison across datasets
    scale_factor = 10.0  # Global scale factor - adjust to make arrows visible
    
    # Create colormap based on absolute error magnitudes (not normalized per plot)
    # Use a global scale that works well for typical calibration errors
    global_max_error = 5.0  # Adjust based on expected error range
    norm = colors.Normalize(vmin=0, vmax=global_max_error)
    cmap = cm.get_cmap('viridis')
    
    # Plot error vectors as 3D arrows with length = actual error magnitude
    for i in range(len(data1)):
        if error_magnitudes[i] > 0:  # Only draw arrows for non-zero errors
            # Get color based on absolute error magnitude
            color = cmap(norm(error_magnitudes[i]))
            
            # Scale the actual error vector by a constant factor for visibility
            scaled_error = error_vectors[i] * scale_factor
            
            # Draw arrow with length proportional to actual error magnitude
            ax.quiver(data1[i, 0], data1[i, 1], data1[i, 2],
                     scaled_error[0], scaled_error[1], scaled_error[2],
                     color=color, alpha=0.8, arrow_length_ratio=0.1,
                     linewidth=2)
    
    # Add a dummy scatter for colorbar with global scale
    scatter = ax.scatter([], [], [], c=[], cmap='viridis', 
                       vmin=0, vmax=global_max_error)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Error Magnitude', rotation=270, labelpad=15)
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Error: {var_name1} and {var_name2}')
    
    return fig, ax
