# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:45:22 2016

@author: eejvt
"""

import numpy as np
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import os
from glob import glob
import matplotlib.pyplot as plt
import scipy as sc
from glob import glob
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
from scipy.io.idl import readsav
from scipy.io import netcdf

reload(jl)


#%%
directory='/nfs/a201/eejvt/HUMMEL/'
file_name='NFPTAERO.cam.h0.OM_AC.lev29.2000-YA.nc'
os.chdir(directory)

mb=netcdf.netcdf_file(directory+file_name,'r') 
from scipy.interpolate import interpn

air_density_IUPAC=1.2754#kg/m3 IUPAC

hummel = type('test', (), {})()
hummel.name='hummel'
hummel.wiom_mace=np.zeros(12)
hummel.wiom_ams=np.zeros(12)
hummel.wiom_reyes=np.zeros(12)

for month in range(12):
    data_values=mb.variables['OM_AC'][month,0,:,:]*air_density_IUPAC*1e9#ug/m3
    
    grid_x=mb.variables['lat'].data
    grid_y=mb.variables['lon'].data
    points=[jl.mace_head_latlon_values,jl.amsterdam_island_latlon_values,jl.point_reyes_latlon_values]
    hummel.wiom_mace[month],hummel.wiom_ams[month],hummel.wiom_reyes[month]=interpn((grid_x,grid_y),data_values,points)




mace_wiom=np.array([0.08,0.088,0.1,0.24,0.51,0.2,0.183,0.155,0.122,0.24,0.062,0.07])
ams_wioc=np.array([189,122,96,68,64,41,49,47,64,54,64,130])/1e3
ams_wiom=ams_wioc*1.9

point_reyes_wiom=np.array([np.nan,0.177429,np.nan,0.053704,0.098196,0.281652,0.01127,0.56441,0.220598,np.nan,np.nan,np.nan])

#%%
plt.figure()
plt.title('Mace Head WIOM')
plt.plot(hummel.wiom_mace,'k--',label=hummel.name)
plt.plot(mace_wiom,'bo',label='Observations')
plt.ylabel('$\mu g/ m^3$')
plt.xlabel('Month')
plt.xticks(np.arange(12),jl.months_str)
plt.legend()
plt.savefig('Mace Head WIOM hummel')
plt.show()

plt.figure()
plt.title('Amsterdam island WIOM')
plt.plot(hummel.wiom_ams,'k--',label=hummel.name)
plt.plot(ams_wiom,'bo',label='Observations')
plt.ylabel('$\mu g/ m^3$')
plt.xlabel('Month')
plt.xticks(np.arange(12),jl.months_str)
plt.legend()
plt.savefig('Amsterdam island WIOM hummel')
plt.show()

plt.figure()
plt.title('Point Reyes WIOM')
plt.plot(hummel.wiom_reyes,'k--',label=hummel.name)
plt.plot(point_reyes_wiom,'bo',label='Observations')
plt.ylabel('$\mu g/ m^3$')
plt.xlabel('Month')
plt.xticks(np.arange(12),jl.months_str)
plt.legend()
plt.savefig('Point Reyes WIOM hummel')
plt.show()
















