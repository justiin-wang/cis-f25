import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import colors
from utils import convert as cvt

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

def plot_raw_data(data, var_name):
    """Plot raw data from dataframe with position and orientation."""
    pose_cols = ['pose.position.x', 'pose.position.y', 'pose.position.z',
                'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w']
    
    pose_data = data[pose_cols].values
    
    # Convert each 7-vector to a 6-vector (e.g., [x, y, z, roll, pitch, yaw])
    euler_data = np.array([cvt.vect7_vect6(pose) for pose in pose_data])
    
    # Convert position from meters to millimeters
    positions_mm = euler_data[:, 0:3] * 1000
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions_mm[:, 0], positions_mm[:, 1], positions_mm[:, 2], color='b')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f"{var_name} ({data.shape[0]} points)")
    
    return fig, ax

def plot_error_vectors(base_points, error_vectors, title="Error Vectors", scale=10, 
                     color_data=None, color_label="Error Magnitude (m)"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create colormap
    cmap = cm.get_cmap('viridis')
    
    # If color data is not provided, use the magnitude of the error vectors
    if color_data is None:
        color_data = np.linalg.norm(error_vectors, axis=1)
    
    # Get max for normalization
    color_max = np.max(color_data)
    normalized_colors = color_data / color_max
    
    # Plot error vectors
    quiver = ax.quiver(base_points[:, 0], base_points[:, 1], base_points[:, 2],
                      error_vectors[:, 0], error_vectors[:, 1], error_vectors[:, 2],
                      length=scale,
                      norm=colors.Normalize(0, color_max),
                      color=cmap(normalized_colors))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title} (scale={scale})")
    
    # Add colorbar
    cbar = fig.colorbar(quiver, ax=ax, shrink=0.8, label=color_label)
    
    # Highlight first point
    ax.plot(base_points[0, 0], base_points[0, 1], base_points[0, 2], 'r*', markersize=10, label='Start Point')
    
    # Highlight point with maximum error
    max_idx = np.argmax(color_data)
    ax.plot(base_points[max_idx, 0], base_points[max_idx, 1], base_points[max_idx, 2], 
           'y*', markersize=10, label='Max Error Point')
    
    ax.legend()
    plt.tight_layout()
    
    return fig, ax


def plot_rotation_error(base_points, error_vectors, orientation_errors, title="Rotation Error Vectors", 
                    scale=1, normalize=True, color_label="Normalized orientation error (deg)"):
    """
    Plot 3D vectors showing positional error with color indicating magnitude of orientation error.
    
    Parameters:
    -----------
    base_points : ndarray
        Nx3 array of base point coordinates (x, y, z)
    error_vectors : ndarray
        Nx3 array of translation error vectors
    orientation_errors : ndarray
        Array of orientation error magnitudes used for coloring
    title : str
        Plot title
    scale : float
        Scaling factor for vector lengths
    normalize : bool
        Whether to normalize the orientation errors for coloring
    color_label : str
        Label for the colorbar
        
    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure and axis objects
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create colormap
    cmap = cm.get_cmap('viridis')
    
    # Normalize orientation errors if requested
    color_data = orientation_errors
    color_max = np.max(color_data)
    
    if normalize:
        normalized_colors = color_data / color_max
    else:
        normalized_colors = color_data
    
    # Plot error vectors
    quiver = ax.quiver(base_points[:, 0], base_points[:, 1], base_points[:, 2],
                      error_vectors[:, 0], error_vectors[:, 1], error_vectors[:, 2],
                      length=scale,
                      norm=colors.Normalize(0, color_max),
                      color=cmap(normalized_colors))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title} (scale={scale})")
    
    # Add colorbar
    cbar = fig.colorbar(quiver, ax=ax, shrink=0.8, label=color_label)
    
    # Highlight first point
    ax.plot(base_points[0, 0], base_points[0, 1], base_points[0, 2], 'r*', markersize=10, label='Start Point')
    
    # Highlight point with maximum orientation error
    max_idx = np.argmax(color_data)
    ax.plot(base_points[max_idx, 0], base_points[max_idx, 1], base_points[max_idx, 2], 
           'y*', markersize=10, label='Max Rotation Error Point')
    
    ax.legend()
    plt.tight_layout()
    
    return fig, ax
