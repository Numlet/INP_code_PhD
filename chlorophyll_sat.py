# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 10:25:31 2015

@author: eejvt
"""

import numpy as np
import Jesuslib as jl
import os
from scipy.io.idl import readsav
from glob import glob
from scipy.io import netcdf
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats.stats import pearsonr
from mpl_toolkits.basemap import Basemap
from scipy.io import netcdf
from netCDF4 import Dataset
from glob import glob

#%%
def dist(lonpoint, latpoint, lon_grid,lat_grid):
    X_lons,Y_lats=np.meshgrid(lon_grid,lat_grid)
    
    d=np.sqrt((lonpoint-X_lons)**2+(latpoint-Y_lats)**2)
    index=[0,0]#lon,lat
    index[0]=d.argmin(axis=(0))[0]
    index[1]=d.argmin(axis=(1))[0]
    #print lon_grid[index],lat_grid[index]
    return d,index



#%%
archive_directory='/nfs/a107/eejvt/'
project='SATELLITE/MERGED_GlobColour/CHL1_MONTHLY/ftp.hermes.acri.fr/981874235/'
output_file='/nfs/a107/eejvt/SATELLITE/MERGED_GlobColour/CHL1_MONTHLY/T42/'
output_file_congrid='/nfs/a107/eejvt/SATELLITE/MERGED_GlobColour/CHL1_MONTHLY/T42_congrid/'
os.chdir(archive_directory+project)
#%%
a=glob('*AV*')
data = Dataset(a[1], 'r')
name='prueba'
#%%
for name in a:
    
    sat_file=name
    
    data = Dataset(sat_file, 'r')
    
    
    sat_lat=data.variables['lat'][:]
    sat_lon=data.variables['lon'][:]
    #sat_lon[sat_lon<0]=360+sat_lon[sat_lon<0]    CHL1_prop=data.variables['CHL1_mean']
    CHL1=data.variables['CHL1_mean'][:,:].data
    print data.variables['CHL1_mean']
    CHL1[CHL1==data.variables['CHL1_mean']._FillValue]=0
    lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
    X_lons,Y_lats=np.meshgrid(lon.glon,lat.glat)
    glo_lon=lon.glon
    glo_lat=lat.glat

    #CHL1_glo=jl.interpolate_grid(CHL1,sat_lon,sat_lat,glo_lon,glo_lat)
    CHL1_glo=jl.congrid(CHL1,(64,128))
    CHL1_glo=np.roll(CHL1_glo,128/2,axis=1)
    
    CHL1_glo=CHL1_glo.reshape(8192)
    np.savetxt(output_file_congrid+name+'.dat',CHL1_glo,fmt='%10.5f')
    #np.savetxt('prueba.dat',CHL1_glo)
    print name

#m.scatter([250],[-30],s=30)
#%%
np.reshape()

lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
X_lons,Y_lats=np.meshgrid(lon.glon,lat.glat)
glo_lon=lon.glon
glo_lat=lat.glat



#%%


#CHL1_glo=jl.interpolate_grid(CHL1,sat_lon,sat_lat,glo_lon,glo_lat)





#%%

#m=jl.plot(CHL1_glo[:,:],show=1,lon=glo_lon,lat=glo_lat,clevs=[0,0.01,0.05,0.1,0.5,1,5,10,16,17],cblabel=CHL1_prop.units,return_fig=1)





#%%
output_file='/nfs/a107/eejvt/SATELLITE/MERGED_GlobColour/CHL1_MONTHLY/T42/'
os.chdir(output_file)
a=glob('*2001*')
chl=np.zeros((64,128,12))
for i in range(len(a)):
    array=np.genfromtxt(a[i])
    chl[:,:,i]=np.reshape(array,(64,128))
        
        
        
jl.grid_earth_map(chl,levels=np.logspace(-2,1,10).tolist(),cmap=plt.cm.Greens)
jl.grid_earth_map((75.9*chl-3.99)*0.01,cmap=plt.cm.jet,levels=[0,0.1,0.2,0.3,0.4,0.5,0.78,1,10])
        
#%%

os.chdir('/nfs/a201/eejvt/MARINE_ORGANIC_EMISSIONS_BURROWS/EMISSION_FILES_T42')
a=glob('omf*')
omf=np.zeros((64,128,12))
months_str=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
for i in range(len(months_str)):
    print i
    array=np.genfromtxt('omf_%s.dat'%months_str[i])
    omf[:,:,i]=np.reshape(array,(64,128))

jl.grid_earth_map(omf,levels=[0,0.05,0.1,0.2,0.3,0.5,0.7,1,3],cmap=plt.cm.rainbow)
        
#%%
'''
chlor_file='MY1DMM_CHLORA_2014-11-01_rgb_360x180.CSV'
chlor_data = np.genfromtxt(chlor_file, delimiter=',')
lats=np.linspace(90,-90,180)
lons=np.linspace(-180,180,360)
m=jl.plot(chlor_data,lon=lons,lat=lats,show=1,clevs=np.logspace(-4,2,15).tolist(),cmap=plt.cm.Reds,return_fig=1)
#m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
#    llcrnrlon=-180,urcrnrlon=180,lat_ts=20)
m.fillcontinents(color='green',lake_color='aqua')
'''



'''

sat_lon=data.variables['lon'][:]
sat_lon[sat_lon<0]=360+sat_lon[sat_lon<0]#doing lon positive
lon_idx=np.zeros(sat_lon.shape)
lat_idx=np.zeros(sat_lat.shape)

for ilon in range(len(sat_lon)):
    lon_idx[ilon]=jl.find_nearest_vector_index(glo_lon,sat_lon[ilon])

for ilat in range(len(sat_lat)):
    lat_idx[ilat]=jl.find_nearest_vector_index(glo_lat,sat_lat[ilat])

CHL1_glo=np.zeros((len(glo_lat),len(glo_lon)))
ilon=0
ilat=0
for ilon in range(len(glo_lon)):
    for ilat in range(len(glo_lat)):
        
            
        lons=np.array(lon_idx==ilon)        

        lats=np.array(lat_idx==ilat)        
        total=0
        for intlons in range(len(lons)):
            for intlats in range(len(lats)):
                if lons[intlons]:
                    if lats[intlats]:
                       total=total+CHL1[intlats,intlons]
        
        mean=total/(np.array(lats).sum()*np.array(lons).sum())
        
        CHL1_glo[ilat,ilon]=mean

'''