# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 09:55:20 2016

@author: eejvt
"""

import numpy as np
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
sys.path.append('/nfs/see-fs-01_users/eejvt/UKCA_postproc')
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
import UKCA_lib as ukl
reload(ukl)

#%%

cube=iris.load_cube(ukl.Obtain_name('/nfs/a201/eejvt/UKCA_TEST_FILES/tebxd/2008jan','s38i472'))
new_coords=[]
data_size=1

for coor in cube.coords():
    print coor.standard_name
    if coor.standard_name != 'latitude' and coor.standard_name != 'longitude':
#    if coor.standard_name != 'latitude' and coor.standard_name != 'longitude':
        #data_size*=len(coor.points)
        #print coor
        print len(coor.points)
    #%%
        
#new_coords=[]
#data_size=1
##new_coords.append((cube.coord('model_level_number'),3))
#new_coords.append(cube.coord('model_level_number'))
#latitude=cube.coord('latitude')
#longitude=cube.coord('longitude')
#new_lat_points=np.arange(-90,90.01,2.5)
#new_lon_points=np.arange(0,360.01,3.75)
#new_lat=iris.coords.DimCoord(np.arange(-90., 90.1, 2.5),standard_name='latitude',var_name='latitude', units='degrees')
#new_lon=iris.coords.DimCoord(np.arange(0,360.01,3.75),standard_name='longitude',var_name='longitude', units='degrees')
##new_coords.append((new_lat,1))
##new_coords.append((new_lon,2))
#new_coords.append(new_lat)
#new_coords.append(new_lon)
#data_size*=len(new_lat.points)
#data_size*=len(new_lon.points)
#data_size*=85
#print [coor.standard_name for coor in new_coords]
cube=iris.load_cube(ukl.Obtain_name('/nfs/a201/eejvt/UKCA_TEST_FILES/tebxd/2008jan','s38i472'))
cube_n48=iris.load('/nfs/a68/amtgwm/GLOMAP_output/UKCA_UM_v7_3/xjuoa/xjuoaa_pmj1jan_aertracersMD_Jan13m.nc')[0]
#coords_names=['mode_level_number','latitude','longitude']
#for coord in cube.coords():
#    if not coord.standard_name in coords_names:
#        cube.remove_coord(coord)
#data = np.reshape(np.zeros(data_size),(85,len(new_lat.points), len(new_lon.points)))
#print data.shape
#return_cube = iris.cube.Cube(data, dim_coords_and_dims=zip(new_coords, range(data.ndim)))

new_cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)


import time
cube_n48.coord('latitude').coord_system = new_cs

cube_n48.coord('longitude').coord_system = new_cs
t1=time.time()
reg_cube = cube.regrid(cube_n48[0,:,:,:], iris.analysis.Linear())
t2=time.time()
print t2-t1
reg_cube = iris.analysis.interpolate.regrid(cube, cube_n48)#, **kwargs)

#cube_cs = cube.coord_system(iris.coord_systems.CoordSystem) 

#%%
print cube
cube.remove_coord(cube.coord('model_level_number'))

coord_height=cube.coord('atmosphere_hybrid_height_coordinate')
cube.remove_coord(cube.coord('atmosphere_hybrid_height_coordinate'))
cube.add_dim_coord(coord_height,0)
old_time = hist[2].coord('time')

new_height=iris.coords.DimCoord(coord_height.points,var_name='hybrid_ht', units=coord_height.units)
cube.add_dim_coord(new_height,0)
cube_cs = cube.coord_system(iris.coord_systems.CoordSystem) 

#%%
from mpl_toolkits.basemap import Basemap
import iris.quickplot as qplt
import iris
import numpy as np
import time
import iris
from glob import glob
import matplotlib.pyplot as plt
from iris.analysis.interpolate import linear
import cartopy.crs as ccrs 





files_directory='/nfs/a201/eejvt/CASIM/SO_KALLI/LARGE_DOMAIN/'


pp_files=glob(files_directory+'umnsaa_*')

cube=iris.load(pp_files[0])[2]

cube=iris.load('/nfs/a201/eejvt/CASIM/SO_KALLI/LARGE_DOMAIN/umnsaa_cb006')[0]#[2]

rot_pole = cube.coord('grid_latitude').coord_system.as_cartopy_crs() 
ll = ccrs.Geodetic() 

lon, lat = 0.01, -52
rotated_lon, rotated_lat = 0.01, -52
target_xy = rot_pole.transform_point(lon, lat, ll) 
target_xy = ll.transform_point(rotated_lon, rotated_lat, rot_pole) 
cube.coord('grid_longitude').points

#extracted_cube = linear(cube, [('grid_latitude', target_xy[1]), ('grid_longitude', target_xy[0])]) 

lons, lats =iris.analysis.cartography.unrotate_pole(cube.coord('grid_longitude').points,cube.coord('grid_latitude').points,0.01,90-52.0)
lons
cube.coord('grid_longitude').points=lons
cube.coord('grid_latitude').points=lats
#%%

#print np.amax(data),np.amin(data)
fig=plt.figure(figsize=(20, 12))
m = fig.add_subplot(1,1,1)
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='cyl',lon_0=0)
m.drawcoastlines()
qplt.contourf(cube[0,])
plt.gca().coastlines()
#m.drawmeridians(np.arange(0.,360.,60.))
#m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
#m.drawparallels(np.arange(-90.,120.,10.))
#plt.figtext(0.501,0.282,'60N',fontsize=3)
#plt.figtext(0.501,0.248,'30N',fontsize=3)
#plt.figtext(0.501,0.168,'30S',fontsize=3)
#plt.figtext(0.501,0.133,'60S',fontsize=3)
#plt.figtext(0.4371,0.209,'60W',fontsize=3)
#plt.figtext(0.385,0.209,'120W',fontsize=3)
#plt.figtext(0.55,0.209,'60E',fontsize=3) 
#plt.figtext(0.597,0.209,'120E',fontsize=3)
