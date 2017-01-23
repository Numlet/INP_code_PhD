# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:56:41 2016

@author: eejvt
"""
import iris.quickplot as qplt
import numpy as np
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import os
from glob import glob
import pylab
import matplotlib.pyplot as plt
import scipy as sc
from glob import glob
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
from scipy.io.idl import readsav
reload(jl)
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
import iris
data_dir='/nfs/a201/eejvt/CASSIM/first_runs/SO/673/'
lat0=-60
lon0=45
data_dir='/nfs/a201/eejvt/CASSIM/first_runs/robin/'
lat0=87.3
lon0=-6
#%%

runs=[array[len(data_dir):] for array in glob(data_dir+'*') if 'umnsaa_cb' in array]
print runs
a=glob(data_dir+'umnsaa_cb*')
a=glob(data_dir+'*')

times=[]
cubes=iris.load(data_dir+'umnsaa_pb000')
cubes=iris.load(a[0])
cubes=iris.load('/nfs/a201/eejvt/CASSIM/first_runs/SO/673/umnsaa_pa018')
#%%
for run in a:
    print run    
    if 'cb' in run:
        continue    
    cubes=iris.load(run)
    for cube in cubes:
        print  ukl.get_stash(cube),cube.var_name
        if ukl.get_stash(cube)=='m01s00i408':
            print run,'pressure'
#%%     
    cube=cubes[0]
    times.append(cube.coord('time').points)

#%%

cube=cubes[5]
#cube.coord('grid_latitude').points=cube.coord('grid_latitude').points-30
#cube.coord('grid_longitude').points=cube.coord('grid_longitude').points+50
lat = cube.coord('grid_latitude').points
lon = cube.coord('grid_longitude').points
print lat
print lon
#lon_un,lat_un=iris.analysis.cartography.unrotate_pole(lon,lat,lon0,lat0)

cube.coord('grid_latitude').points=cube.coord('grid_latitude').points+lat0
cube.coord('grid_longitude').points=cube.coord('grid_longitude').points+lon0-180
#jl.plot(cube[0,].data,lon=lon,lat=lat)
#%%
#qplt.contourf(cube[level,],color_levels,cmap=cmap,norm=matplotlib.colors.LogNorm())
qplt.contourf(cube[0,],9,cmap=cmap)
plt.gca().coastlines()
#Basemap.drawmeridians(lat)
#%%
data=cube[0,].data
lat = cube.coord('grid_latitude').points
lon = cube.coord('grid_longitude').points
cmap='viridis'
fig=plt.figure(figsize=(20, 12))
m = fig.add_subplot(1,1,1)
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
    llcrnrlon=-180,urcrnrlon=180,lat_ts=20)
m.drawcoastlines()

X,Y=np.meshgrid(lon,lat)



cs=m.contourf(X,Y,data,9,latlon=True,cmap=cmap)
plt.colorbar()











