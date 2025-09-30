import numpy as np          
import scipy as sp
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D  
import pandas as pd
from utils import plot as plotter
from utils import point_cloud_registration as pcr
from utils import convert as cvt

em_pivot = pd.read_csv('./prhw1/data/pa1-debug-a-empivot.txt', skiprows=1, header=None).to_numpy()
opt_pivot = pd.read_csv('./prhw1/data/pa1-debug-a-optpivot-filtered.txt', skiprows=1, header=None).to_numpy()

print("em_pivot shape:", em_pivot.shape)
print("opt_pivot shape:", opt_pivot.shape)

fig, ax = plotter.plot_data_2(em_pivot, opt_pivot, "EM Pivot", "Optical Pivot", number_points=True)
#plt.show()

E = pcr.point_cloud_registration(em_pivot, opt_pivot)

em_pivot_transformed_homogeneous = E @ cvt.vect3_htm(em_pivot) 
em_pivot_transformed = cvt.htm_vect3(em_pivot_transformed_homogeneous)

fig, ax = plotter.plot_data_2(em_pivot_transformed, opt_pivot, "Transformed EM Pivot", "Optical Pivot", number_points=True)
plt.show()

print("Test")
