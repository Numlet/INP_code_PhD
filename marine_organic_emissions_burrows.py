# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:05:35 2015

@author: eejvt
"""

import numpy as np
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
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
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import csv
import random
import datetime
def read_ncdf(file_name):
    mb=netcdf.netcdf_file(file_name,'r') 
    return mb


archive_directory='/nfs/a201/eejvt/'
project='MARINE_ORGANIC_EMISSIONS_BURROWS'
os.chdir(archive_directory+project)
mb=read_ncdf('/nfs/a201/eejvt/MARINE_ORGANIC_EMISSIONS_BURROWS/marine_organics_bub_frac_v9.nc')
#%%
print mb.variables
nlat=(mb.variables['NLAT'].data)
grid_lat=mb.variables['TLAT'].data
nlon=mb.variables['NLON'].data
grid_lon=mb.variables['TLONG'].data
mass_frac=np.copy(mb.variables['MASS_FRAC_ORG_TOT_BUB'].data[:,0,:,:])
mass_frac[mass_frac==mb.variables['MASS_FRAC_ORG_TOT_BUB'].missing_value]=0
#jl.plot(mass_frac[2,:,:],lat=grid_lat,lon=grid_lon,show=1)
#%%
CF=plt.contourf(grid_lon,grid_lat,mass_frac[1,:,:],interpolate=None)
X,Y=np.meshgrid(nlon,nlat)
CF=plt.contourf(X,Y,lon)
#CB=plt.colorbar(CF,ticks=levelsmo,drawedges=1,label='$m^{-3}$')
CB=plt.colorbar(CF)
plt.gca().invert_yaxis()
#plt.colorbar()
plt.show()

#%%
lon_glo=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
lat_glo=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
grid_index_lat=np.zeros(grid_lat.shape)
grid_index_lon=np.zeros(grid_lat.shape)
for ilon in range(len(nlon)):
    print ilon
    for ilat in range(len(nlat)):
        lat=grid_lat[ilat,ilon]
        lon=grid_lon[ilat,ilon]
        grid_index_lat[ilat,ilon]=jl.find_nearest_vector_index(lat_glo.glat,lat)
        grid_index_lon[ilat,ilon]=jl.find_nearest_vector_index(lon_glo.glon,lon)

np.save('lat_idex_from_burrows_to_GLOMAP',grid_index_lat)
np.save('lon_idex_from_burrows_to_GLOMAP',grid_index_lon)
#%%
lon_glo=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
lat_glo=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
grid_index_lat=np.load('lat_idex_from_burrows_to_GLOMAP.npy')
grid_index_lon=np.load('lon_idex_from_burrows_to_GLOMAP.npy')
#%%
omf_glomap=np.zeros((len(lat_glo.glat),len(lon_glo.glon)))
for ilon in range(len(lon_glo.glon)):
    print ilon
    for ilat in range(len(lat_glo.glat)):
        print ilat
        omf_glomap[ilat,ilon]=mass_frac[0,np.argwhere([grid_index_lon==ilon] and [grid_index_lat==ilat] and [mass_frac!=0])].mean()


#%%
lon_glo=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
lat_glo=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
omf_glomap=np.zeros((12,len(lat_glo.glat),len(lon_glo.glon)))
grid_glomap=np.array(np.meshgrid(lat_glo.glat,lon_glo.glon))
for imon in range(12):
    print imon,imon,imon
    for ilon in range(128):
        print ilon
        for ilat in range(64):
            #print ilat
            dist=np.sqrt((grid_lon-lon_glo.glon[ilon])**2+(grid_lat-lat_glo.glat[ilat])**2)
            omf_glomap[imon,ilat,ilon]=mass_frac[imon,][np.unravel_index(dist.argmin(), dist.shape)]

np.save('omf_glomap',omf_glomap)
omf_glomap=np.load('/nfs/a201/eejvt/MARINE_ORGANIC_EMISSIONS_BURROWS/omf_glomap.npy')

jl.plot(omf_glomap[:,:,:].mean(axis=0))

#%%
months_str=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
shifted_month_str=['apr','may','jun','jul','aug','sep','oct','nov','dec','jan','feb','mar']
for imon in range(12):
    omf_glomap_month=omf_glomap[imon,:,:].reshape(8192)
    np.savetxt('omf_'+shifted_month_str[imon]+'.dat',omf_glomap_month,fmt='%10.5f')

