# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:50:10 2016

@author: eejvt
"""


import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import numpy as np
import Jesuslib as jl
import os
from scipy.io.idl import readsav
from glob import glob
from scipy.io import netcdf
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats.stats import pearsonr
from glob import glob
from scipy.io.idl import readsav
from scipy import stats
from scipy.optimize import curve_fit
import scipy

data_values=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')[20,].mean(axis=-1)
press=(readsav('/nfs/a173/earjbr/mode10_setup/GLOMAP_mode_pressure_mp_mode10_2001.sav').pl_m*1e-2).mean(axis=-1)

point=[12,43.3,560]

lats=jl.lat
lons=jl.lon

n = np.array([abs(i-point[0]) for i in jl.lat])
args=np.argsort(n)
lat1=args[0]
lat2=args[1]

n = np.array([abs(i-point[1]) for i in jl.lon])
args=np.argsort(n)
lon1=args[0]
lon2=args[1]


def two_closest(array, value):
    n = np.array([abs(i-point[1]) for i in array])
    args=np.argsort(n)
    return args[0],args[1]

lat_planes=two_closest(jl.lat,point[0])
lon_planes=two_closest(jl.lon,point[1])
press_one_dim=press[:,lat_planes[0],lon_planes[0]]
press_planes=two_closest(press_one_dim,point[2])

from scipy.interpolate import interpn

pres
grid_x=jl.lat
grid_y=jl.lon
grid_z=press_one_dim
point=[560,12,43.3]
point[1]=point[1]*(-1)
interpn((grid_z[:],(-1)*grid_x,grid_y),data_values,point)

'''
Interpolation works as long as latitudes are the other way around (positives are negative and bv)
for some reason the function wants the values to be ascending
it is working
'''









#%%
points_interpolate=[]
for i in lat_planes:
    for j in lon_planes:
        for k in press_planes:
            points_interpolate.append([i,j,k])
points_interpolate=np.array(points_interpolate)
norm_lat=jl.lat/max(jl.lat)
#%%


def interpolate_3_dim_spatial(array,point,lats,lons,pressures):

import iris

cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
#%%


import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
plt.figure()
# Note that the output interpolated coords will be the same dtype as your input
# data.  If we have an array of ints, and we want floating point precision in
# the output interpolated points, we need to cast the array as floats
data = np.arange(40).reshape((8,5)).astype(np.float)

# I'm writing these as row, column pairs for clarity...
coords = np.array([[1.2, 3.5], [6.7, 2.5], [7.9, 3.5], [3.5, 3.5],[6.4,4]])
# However, map_coordinates expects the transpose of this
coords = coords.T

# The "mode" kwarg here just controls how the boundaries are treated
# mode='nearest' is _not_ nearest neighbor interpolation, it just uses the
# value of the nearest cell if the point lies outside the grid.  The default is
# to treat the values outside the grid as zero, which can cause some edge
# effects if you're interpolating points near the edge
# The "order" kwarg controls the order of the splines used. The default is 
# cubic splines, order=3
#%%
import numpy as np
from scipy import ndimage

data = np.arange(3*5*9).reshape((3,5,9)).astype(np.float)
coords = np.array([[1.2, 3.5, 7.8], [0.5, 0.5, 6.8]])
zi = ndimage.map_coordinates(data, coords.T)
#zi = ndimage.map_coordinates(data, coords, order=1, mode='nearest')

row, column = coords
nrows, ncols = data.shape
im = plt.imshow(data, interpolation='nearest', extent=[0, ncols, nrows, 0])
plt.colorbar(im)
plt.scatter(column, row, c=zi, vmin=data.min(), vmax=data.max())
for r, c, z in zip(row, column, zi):
    plt.annotate('%0.3f' % z, (c,r), xytext=(-10,10), textcoords='offset points',
            arrowprops=dict(arrowstyle='->'), ha='right')
plt.show()
#%%



def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

grid_x, grid_y = np.meshgrid(jl.lat,jl.lon)

points = np.random.rand(1000, 2)
points[:,0]=(points[:,0]-0.5)*90
points[:,0]=(points[:,1])*360

values = func(points[:,0], points[:,1])

from scipy.interpolate import griddata

grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')






#%%


>>> import matplotlib.pyplot as plt
>>> plt.subplot(221)
>>> plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
>>> plt.plot(points[:,0], points[:,1], 'k.', ms=1)
>>> plt.title('Original')
>>> plt.subplot(222)
>>> plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
>>> plt.title('Nearest')
>>> plt.subplot(223)
>>> plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
>>> plt.title('Linear')
>>> plt.subplot(224)
>>> plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
>>> plt.title('Cubic')
>>> plt.gcf().set_size_inches(6, 6)
>>> plt.show()




#%%


from scipy.interpolate import interpn

pres
grid_x=jl.lat
grid_y=jl.lon
grid_z=press_one_dim
point=[560,12,43.3]
point[1]=point[1]*(-1)
interpn((grid_z[:],(-1)*grid_x,grid_y),data_values,point)


def interpolate_GLOMAP
'''
Interpolation works as long as latitudes are the other way around (positives are negative and bv)
for some reason the function wants the values to be ascending
it is working
'''












