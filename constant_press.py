# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:12:01 2015

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
archive_directory='/nfs/a107/eejvt/'
project='BC_INP/'
os.chdir(archive_directory+project)
#%%
levels=21
step=1000./levels
ps=np.linspace(0,levels,levels)*step
print ps
#%%


step=1000./levels
ps=np.linspace(0,levels,levels)*step
press_constant_index=np.zeros(press_daily.shape)
for i in range(len(ps)):
    press_constant_index[i,]=ps[i]
    
#%%
press_daily=np.load('press_daily.npy')
pres=(press_daily[:,32,64,:]).mean(axis=-1)#[:]
press_mean=press_daily.mean(axis=-1)
#%%

INP_feldext_ambient=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feldext_ambient.npy').mean(axis=-1)
ar,ps,press_index=constant_pressure_level_array(INP_feldext_ambient,press_mean)
#%%


#%%
def find_nearest_vector_index(array, value):
    n = np.array([abs(i-value) for i in array])
    nindex=np.apply_along_axis(np.argmin,0,n) 

    return nindex
    
    

def constant_pressure_level_array(array,pressures,levels=21):
    step=1000./levels
    ps=np.linspace(0,levels,levels)*step
    press_constant=np.zeros(pressures.shape)
    for i in range(len(ps)):
        press_constant[i,]=ps[i]
    press_constant=press_constant[:levels,]
    array_constant_index=np.zeros(array.shape)
    #array_constant=np.zeros(array.shape)
    array_constant_index=find_nearest_vector_index(ps,pressures)
    array_constant=np.zeros(press_constant.shape)
        
    for itime in range (len(array_constant[0,0,0,:])):
        for ilev in range (len(array_constant[:,0,0])):
            for ilat in range (len(array_constant[0,:,0])):
                for ilon in range (len(array_constant[0,0,:])):
                    if np.array([array_constant_index[:,ilat,ilon,itime]==ilev]).any():
                        
                        array_constant[ilev,ilat,ilon,itime]=np.mean(array[array_constant_index[:,ilat,ilon,itime]==ilev,ilat,ilon,itime])
                    else:
                        array_constant[ilev,ilat,ilon,itime]=0
    return array_constant,press_constant,array_constant_index



