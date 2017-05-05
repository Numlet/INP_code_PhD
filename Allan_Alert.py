# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:28:37 2016

@author: eejvt
"""

import numpy as np
import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
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
import datetime
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
from scipy.io import netcdf
import scipy as sc


INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')*1e-3#l
INP_feldspar_alltemps=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e3#l


title=' Alert, Canada (Artic lat=82.20,lon=-62.21) '
lat_point=82.20#alert
lon_point=-62.21#alert
obs_temps=[-15,-20,-25,-30]
obs_values=[1e-2,8*1e-2,2*1e-1,1]#alert

title=' Leeds'
lat_point=53.47
lon_point=-1.38




title=' Ucluelet '
lat_point=48.92#Ucluelet 
lon_point=-125.54#Ucluelet 
obs_temps=[-15,-20,-25,-30]
obs_values=[0.015,0.248,0.946,5.738]
obs_values_uc=[0.015,0.248,0.946,5.738]

title=' Amundsen (Labrador sea)'
lat_point=54.50#Amundsen
lon_point=-55.37#Amundsen
obs_temps=[-15,-20,-25,-30]
obs_values=[0.383,1.317,2.786,4.823]
obs_values_ad=[0.383,1.317,2.786,4.823]


ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
ilon=jl.find_nearest_vector_index(jl.lon,lon_point)
column_feldspar=INP_feldspar_alltemps[:,:,ilat,ilon,:]
column_marine=INP_marine_alltemps[:,:,ilat,ilon,:]
temps=np.arange(-37,1,1)
temps=temps[::-1]
column_total_ad=column_feldspar+column_marine



ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
ilon=jl.find_nearest_vector_index(jl.lon,lon_point)
column_feldspar=INP_feldspar_alltemps[:,:,ilat,ilon,:]
column_marine=INP_marine_alltemps[:,:,ilat,ilon,:]
temps=np.arange(-37,1,1)
temps=temps[::-1]
column_total_uc=column_feldspar+column_marine


ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
ilon=jl.find_nearest_vector_index(jl.lon,lon_point)
column_feldspar=INP_feldspar_alltemps[:,:,ilat,ilon,:]
column_marine=INP_marine_alltemps[:,:,ilat,ilon,:]
temps=np.arange(-37,1,1)
temps=temps[::-1]
column_total=column_feldspar+column_marine
#%%
plt.figure()
plt.fill_between(temps[25:],column_feldspar[25:,30,:].min(axis=-1),column_feldspar[25:,30,:].max(axis=-1),color='r',alpha=0.3)
plt.fill_between(temps[5:26],column_feldspar[5:26,30,:].min(axis=-1),column_feldspar[5:26,30,:].max(axis=-1),color='r',label='K-feldspar')
plt.fill_between(temps[27:],column_marine[27:,30,:].min(axis=-1),column_marine[27:,30,:].max(axis=-1),color='g',alpha=0.3)
plt.fill_between(temps[7:28],column_marine[7:28,30,:].min(axis=-1),column_marine[7:28,30,:].max(axis=-1),color='g',label='Marine Organics')
plt.fill_between(temps[:6],column_feldspar[:6,30,:].min(axis=-1),column_feldspar[:6,30,:].max(axis=-1),color='r',alpha=0.3)
plt.fill_between(temps[:8],column_marine[:8,30,:].min(axis=-1),column_marine[:8,30,:].max(axis=-1),color='g',alpha=0.3)
#plt.plot(temps,column_feldspar[:,30,2]+column_marine[:,30,2],c='b',ls='-',lw=2,label='March')    
#plt.plot(temps,column_feldspar[:,30,:].max(axis=-1)+column_marine[:,30,:].max(axis=-1),c='k',ls='-',lw=3)    
#plt.plot(temps,column_feldspar[:,30,:].min(axis=-1)+column_marine[:,30,:].min(axis=-1),c='k',ls='-',lw=3)    
plt.scatter(obs_temps,obs_values,label='Observations')
#for i in range(len(column_marine[0,:])):
#    if i <22:
#        continue
#    
#    plt.plot(temps,column_marine[:,i],'g--',label='Marine organics')
#    plt.plot(temps,column_feldspar[:,i],'r--',label='K-feldspar')
#plt.plot(temps,column_total[:,30,8],'k-',lw=4, label='September mean')
if not title:
    title='latitude: %1.2f longitude: %1.2f'%(lat_point,lon_point)
plt.title(title)
plt.xlim(-32,0)
plt.yscale('log')
plt.grid()
plt.ylabel('$[INP]/L$')
plt.xlabel('Temperature $^oC$')
plt.legend()
plt.show()
#%%
plt.figure()
#plt.title(title)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Observed values $L^{-1}$')
plt.ylabel('Simulated values $L^{-1}$')
line=np.linspace(1e-8,1e4,100)
plt.plot(line,line,'k-')
plt.plot(line,line*10,'k--')
plt.plot(line,line*10**(1.5),'k-.')
plt.plot(line,line*0.1,'k--')
plt.plot(line,line/10**(1.5),'k-.')
lim_bot=1e-3
lim_up=1e3
plt.xlim(lim_bot,lim_up)
plt.ylim(lim_bot,lim_up)
sim_values=np.array([column_total_ad[int(-temp),30,:].mean() for temp in obs_temps])
sim_values_max=np.array([column_total_ad[int(-temp),30,:].max() for temp in obs_temps])
sim_values_min=np.array([column_total_ad[int(-temp),30,:].min() for temp in obs_temps])
plt.errorbar(obs_values_ad,sim_values,
                 yerr=[sim_values-sim_values_min,sim_values_max-sim_values],
                 linestyle="None",c='k',zorder=0)
plt.plot(obs_values_ad,sim_values,'bo', label=' Amundsen (Labrador sea)')
sim_values=np.array([column_total_uc[int(-temp),30,:].mean() for temp in obs_temps])
sim_values_max=np.array([column_total_uc[int(-temp),30,:].max() for temp in obs_temps])
sim_values_min=np.array([column_total_uc[int(-temp),30,:].min() for temp in obs_temps])
plt.errorbar(obs_values_uc,sim_values,
                 yerr=[sim_values-sim_values_min,sim_values_max-sim_values],
                 linestyle="None",c='k',zorder=0)
plt.plot(obs_values_uc,sim_values,'yo', label='Ucluelet')
plt.legend(loc='best')
