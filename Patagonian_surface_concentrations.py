#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:43:31 2017

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
from scipy.interpolate import interpn
directory='/nfs/see-fs-02_users/amtgwm/tex/obsdata_various/UMiami/WoodwardData/'

#file_means='dust_UMiami_Stephanie_monthlymean.dat'
#file_std='dust_UMiami_Stephanie_monthlystdev.dat'
archive_directory='/nfs/a201/eejvt/'
project='PATAGONIAN'
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
import glob as glob
os.chdir(archive_directory+project)

#%%
rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3

def read_data(simulation):
    s={}
    a=glob(simulation+'/tot_mc*.sav')

    print a

    for i in range (len(a)):

        s=readsav(a[i],idict=s)

        print i, len(a)
        #np.save(a[i][:-4]+'python',s[keys[i]])
        print a[i]
    keys=s.keys()
    for j in range(len(keys)):
        print keys[j]
        print s[keys[j]].shape, s[keys[j]].ndim
    #variable_list=s.keys()
    #s=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav',idict=s)
    #s=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav',idict=s)
    return s



x1=jl.read_data('/nfs/a201/eejvt/BASE_RUN/2001/')
x0=jl.read_data('PATX0')
x2=jl.read_data('PATX2')
x4=jl.read_data('PATX4')

#%%
ds_x1=x1.tot_mc_feldspar_mm_mode[:,:,:,:,:].sum(axis=0)+x1.tot_mc_dust_mm_mode[:,:,:,:,:].sum(axis=0)
ds_x0=x0.tot_mc_feldspar_mm_mode[:,:,:,:,:].sum(axis=0)+x0.tot_mc_dust_mm_mode[:,:,:,:,:].sum(axis=0)
ds_x2=x2.tot_mc_feldspar_mm_mode[:,:,:,:,:].sum(axis=0)+x2.tot_mc_dust_mm_mode[:,:,:,:,:].sum(axis=0)
ds_x4=x4.tot_mc_feldspar_mm_mode[:,:,:,:,:].sum(axis=0)+x4.tot_mc_dust_mm_mode[:,:,:,:,:].sum(axis=0)
reload(jl)
lat=-64.10
lon=-57.45
ilat=jl.find_nearest_vector_index(jl.lat,lat)
ilon=jl.find_nearest_vector_index(jl.lon180,lon)


plt.figure()
plt.plot(ds_x1[30,ilat,ilon,:],label='x1',c='b')
plt.axhline(ds_x1[30,ilat,ilon,:].mean(),ls='--',label='x1',c='b')
#plt.plot(ds_x1[30,ilat,ilon,:],label='x1')
plt.plot(ds_x0[30,ilat,ilon,:],label='x0',c='r')
plt.axhline(ds_x0[30,ilat,ilon,:].mean(),ls='--',label='x0',c='r')
plt.plot(ds_x2[30,ilat,ilon,:],label='x2',c='g')
plt.axhline(ds_x2[30,ilat,ilon,:].mean(),ls='--',label='x2',c='g')
plt.plot(ds_x4[30,ilat,ilon,:],label='x4',c='y')
plt.axhline(ds_x4[30,ilat,ilon,:].mean(),ls='--',label='x4',c='y')
plt.legend()
plt.yscale('log')
plt.xlabel('months')
plt.ylabel('dust concentration (ug/m-3)')
plt.title(' James Ross Island ')

plt.savefig(jl.home_dir+'PATAGONIAN/surface_concentration_log.png')



jl.antartic_plot((ds_x4/ds_x1)[20,:,:,:].mean(axis=-1),title='600hpa x4/x1',cblabel='ratio')
plt.savefig(jl.home_dir+'PATAGONIAN/ratio_x4-x1_600hpa.png')
jl.antartic_plot((ds_x4/ds_x1)[30,:,:,:].mean(axis=-1),title='Surface concentrations x4/x1',cblabel='ratio')
plt.savefig(jl.home_dir+'PATAGONIAN/ratio_x4-x1_surface.png')



#dust_concentration_total_surface_mm=dust_concentration.sum(axis=0)[30,]