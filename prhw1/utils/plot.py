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
