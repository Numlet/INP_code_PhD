# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:44:32 2016

@author: eejvt
"""
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



INP_marine_ambient_constantpress_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_ambient_constant_press.npy')*1e-3 #l
INP_feldext_ambient_constantpress_daily=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_feld_extamb_constant_press.npy')*1e3#l
temperatures=np.load('/nfs/a107/eejvt/temperatures_daily.npy')
temperatures_monthly=jl.from_daily_to_monthly(temperatures)

#%%
plt.figure()
ilat=30
ilon=5
for i in range(365):
    plt.plot(INP_feldext_ambient_constantpress_daily[:,ilat,ilon,i]+INP_marine_ambient_constantpress_daily[:,ilat,ilon,i],jl.pressure_constant_levels,'o')

plt.xscale('log')
#plt.plot(jl.pressure_constant_levels,INP_feldext_ambient_constantpress_daily[:,60,0,:].max(axis=-1))




#%%
title=0
lat_point=-52
lon_point=0
INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')*1e-3#l
INP_feldspar_alltemps=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e3#l

lev=20
ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
ilon=jl.find_nearest_vector_index(jl.lon,lon_point)
column_feldspar=INP_feldspar_alltemps[:,:,ilat,ilon,:]
column_marine=INP_marine_alltemps[:,:,ilat,ilon,:]
temps=np.arange(-37,1,1)
temps=temps[::-1]
plt.figure()
plt.fill_between(temps[25:],column_feldspar[25:,lev,:].min(axis=-1),column_feldspar[25:,lev,:].max(axis=-1),color='r',alpha=0.3)
plt.fill_between(temps[5:26],column_feldspar[5:26,lev,:].min(axis=-1),column_feldspar[5:26,lev,:].max(axis=-1),color='r',label='K-feldspar')
plt.fill_between(temps[7:],column_marine[7:,lev,:].min(axis=-1),column_marine[7:,lev,:].max(axis=-1),color='g',label='Marine Organics')
plt.fill_between(temps[:6],column_feldspar[:6,lev,:].min(axis=-1),column_feldspar[:6,lev,:].max(axis=-1),color='r',alpha=0.3)
plt.fill_between(temps[:8],column_marine[:8,lev,:].min(axis=-1),column_marine[:8,lev,:].max(axis=-1),color='g',alpha=0.3)
plt.plot(temps,column_feldspar[:,lev,:].max(axis=-1)+column_marine[:,lev,:].max(axis=-1),c='k',ls='--')    
#for i in range(len(column_marine[0,:])):
#    if i <22:
#        continue
#    
#    plt.plot(temps,column_marine[:,i],'g--',label='Marine organics')
#    plt.plot(temps,column_feldspar[:,i],'r--',label='K-feldspar')
if not title:
    title='latitude: %1.2f longitude: %1.2f level:%i'%(lat_point,lon_point,lev)
plt.title(title)
plt.xlim(-27)
plt.yscale('log')
plt.grid()
plt.ylabel('$[INP]/L$')
plt.xlabel('Temperature $^oC$')
plt.legend()

t_max=temperatures_monthly[lev,ilat,ilon,:].max()
t_min=temperatures_monthly[lev,ilat,ilon,:].min()
plt.axvline(t_max)
plt.axvline(t_min)
print t_max
print t_min
plt.show()

