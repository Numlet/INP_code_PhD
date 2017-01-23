# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:51:32 2016

@author: eejvt
"""

import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
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
from scipy.io import netcdf
from scipy.optimize import curve_fit
import scipy
import os
import psutil
process = psutil.Process(os.getpid())
print process.memory_info().rss

path='/nfs/a201/eejvt/CLIMATOLOGY/'
#class structured_year():
#    def __init__(self,data_values):
#        self.data_values=data_values

os.chdir(path)
years=['2001','2002','2003']#,'2004']
s={}
year=years[1]
#%%
print 'asdftrydf'

def create_netcdf_feldspar(year,s):

    f = netcdf.netcdf_file('INP_feldext_'+year+'.nc', 'w')
    f.createDimension('temperature',38)
    f.createDimension('days',365)
    f.createDimension('months',12)
    f.createDimension('levels',31)
    f.createDimension('lat',64)
    f.createDimension('lon',128)

    INP_feldext = f.createVariable('INP_feldspar', 'float', ('temperature','levels','lat','lon','days'))
    INP_feldext_montly = f.createVariable('INP_feldspar_monthly', 'float', ('temperature','levels','lat','lon','months'))
    INP_feldext[:,:,:,:,:]=s['INP_feldspar_365_'+year]*1e6
    INP_feldext_montly[:,:,:,:,:]=jl.from_daily_to_monthly(s['INP_feldspar_365_'+year]*1e6)
    INP_feldext.units='m3'
    INP_feldext_montly.units='m3'
    glon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    glat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
    lat=f.createVariable('lat','float',('lat',))
    lon=f.createVariable('lon','float',('lon',))
    
    lat[:]=glat.glat[:]
    lon[:]=glon.glon[:]
    f.close()

def create_netcdf_marine(year,s):

    f = netcdf.netcdf_file('INP_marine_'+year+'.nc', 'w')
    f.createDimension('temperature',38)
    f.createDimension('days',365)
    f.createDimension('months',12)
    f.createDimension('levels',31)
    f.createDimension('lat',64)
    f.createDimension('lon',128)

    INP_marineorganics=f.createVariable('INP_marine','float',('temperature','levels','lat','lon','days'))
    INP_marineorganics_monthly=f.createVariable('INP_marine_monthly','float',('temperature','levels','lat','lon','months'))
    INP_marineorganics[:,:,:,:,:]=s['INP_marine_365_'+year]
    INP_marineorganics_monthly[:,:,:,:,:]=jl.from_daily_to_monthly(s['INP_marine_365_'+year])
    INP_marineorganics.units='m3'
    INP_marineorganics_monthly.units='m3'
    glon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    glat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
    lat=f.createVariable('lat','float',('lat',))
    lon=f.createVariable('lon','float',('lon',))
    lat[:]=glat.glat[:]
    lon[:]=glon.glon[:]
    f.close()




for year in years:
    print year
    s['INP_feldspar_365_'+year]=np.load(path+year+'/INP_feldext_alltemps_'+year+'.npy')
    create_netcdf_feldspar(year,s)
    s.clear()
    s['INP_marine_365_'+year]=np.load(path+year+'/INP_marine_alltemps_'+year+'.npy')
    create_netcdf_marine(year,s)
    s.clear()
    #plt.plot(s['INP_feldspar_365_'+year][20,30,40,40,:],label=year)

    
    
    

print 'climatology 2 finished'


'''
INP_feldspar_climatology=np.zeros((38, 31, 64, 128, 12))
INP_feldspar_climatology_std=np.zeros((38, 31, 64, 128, 12))
for imon in range(12):
    print imon
    secuence=[s['INP_feldspar_365_'+year][:,:,:,:,jl.days_end_month[imon]:jl.days_end_month[imon+1]] for year in years]
    INP_monthly=np.concatenate(secuence,axis=-1)
    print INP_monthly.shape
    INP_feldspar_climatology[:,:,:,:,imon]=np.mean(INP_monthly,axis=-1)
    INP_feldspar_climatology_std[:,:,:,:,imon]=np.std(INP_monthly,axis=-1)

np.save(path+'INP_feldspar_climatology',INP_feldspar_climatology)
np.save(path+'INP_feldspar_climatology_std',INP_feldspar_climatology_std)
'''
#%%

#INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')


#plt.plot(s['INP_feldspar_365_'+year][20,30,45,55,:])

#plt.plot(jl.mid_month_day,INP_feldext[20,30,45,55,:],'o')
#plt.yscale('log')


