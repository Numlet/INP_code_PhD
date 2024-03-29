#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:30:10 2017

@author: eejvt
"""

import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib
import sys
import glob
dir_scripts='/nfs/see-fs-01_users/eejvt/UKCA_postproc'#Change this to the downloaded folder
sys.path.append(dir_scripts)
import UKCA_lib as ukl
import iris.quickplot as qp
import numpy as np
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import Jesuslib as jl
import matplotlib.pyplot as plt
import matplotlib.animation as animationlt
from scipy.io import netcdf
sys.path.append('/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/code')
#from amsre_daily_v7 import AMSREdaily
from pyhdf import SD
import datetime
import scipy as sc
from scipy.io import netcdf
import time
import scipy
import iris
from collections import OrderedDict




def PDF(data,nbins=100):
    min_val=data.min()
    max_val=data.max()
    if isinstance(nbins,np.ndarray):
        bins=nbins
        data=data.flatten()
        data=data[data<bins[0]]
        data=data[data>bins[-1]]
    else:
        bins=np.linspace(min_val,max_val,nbins)
    size_bin=bins[1:]-bins[:-1]
    bins_midpoint=(bins[1:]-bins[:-1])/2.+bins[1:]
    number_ocurrencies=np.zeros_like(bins_midpoint)
    for ibin in range(len(number_ocurrencies)):
        larger=[data>bins[ibin]]
        smaller=[data<bins[ibin+1]]
        
        number_ocurrencies[ibin]=np.sum(np.logical_and(larger,smaller))
        
    normalized_pdf=number_ocurrencies/float(len(data))/size_bin
    return bins_midpoint,normalized_pdf




def unrotated_grid(cube):
    rotated_cube=isinstance(cube.coord('grid_longitude').coord_system,iris.coord_systems.RotatedGeogCS)
    if rotated_cube:
        pole_lat=cube.coord('grid_longitude').coord_system.grid_north_pole_latitude
        pole_lon=cube.coord('grid_longitude').coord_system.grid_north_pole_longitude
        lons, lats =iris.analysis.cartography.unrotate_pole(cube.coord('grid_longitude').points,cube.coord('grid_latitude').points,pole_lon,pole_lat)
    else:
        lons=cube.coord('grid_longitude').points
        lats=cube.coord('grid_latitude').points
    return lons,lats



def plot_map(maps_dict,levels,lat,lon,variable_name='test',means=0):
    plt.figure(figsize=(18,15))
    if len(lat.shape)==1:
        X,Y=np.meshgrid(lat,lon)
    else:
        X,Y=lat,lon
    number_of_plots=len(maps_dict)
#    number_of_plots=7
    #make plot square
    if np.sqrt(number_of_plots)%1==0: 
        a=int(np.sqrt(number_of_plots))
        b=int(np.sqrt(number_of_plots))

    else:
        a=int(np.sqrt(number_of_plots))+1
        if a*(a-1)<number_of_plots:    
            b=a
        else:
            b=a-1
    i2=str(a)
    i1=str(b)
#    plt.figure(figsize=(15,13))
    for i in range(number_of_plots):    
        print i 
        print i1+i2+str(i+1)
        plt.subplot(i1+i2+str(i+1))
        name_run=maps_dict.keys()[i]
        print name_run
        plt.contourf(X,Y,maps_dict[name_run],levels, origin='lower',cmap=plt.cm.RdBu_r)
        if means:
            plt.title(name_run+' mean=%f'%np.nanmean(maps_dict[name_run]))
        else:
            plt.title(name_run)
        cb=plt.colorbar()
        print int(i2)%(i+1)
        if not (i+1)%int(i2):
            cb.set_label(variable_name)

    plt.savefig('/nfs/see-fs-01_users/eejvt/CASIM/'+variable_name+'.png')
    plt.show()



def clean_cube(cube,value=100):
    if len(cube.shape)==4:
        new_cube=cube[:,:,value:,value:]
        new_cube=new_cube[:,:,:-value,:-value]
    elif len(cube.shape)==3:
        new_cube=cube[:,value:,value:]
        new_cube=new_cube[:,:-value,:-value]
    else:
        new_cube=cube[value:,value:]
        new_cube=new_cube[:-value,:-value]
    return new_cube

#%%
#posibilities=np.linspace(1,9,9)
#print posibilities
#for pos in posibilities:
#    
#    print pos,':'
#    if np.sqrt(pos)%1==0: 
#        a=int(np.sqrt(pos))
#        b=int(np.sqrt(pos))
#
#    else:
#        a=int(np.sqrt(pos))+1
#        if a*(a-1)<pos:    
#            b=a
#        else:
#            b=a-1
#    print a,b
#    print 'multiplied'
#    print a*b, pos
#    if a*b<pos:
#        print 'NOT VALID!!!!!!!'
#    print '---------'
#%%
def plot_PDF(maps_dict,same_bins,variable_name='test'):
#    same_bins=np.linspace(250,273,50)
    plt.figure(figsize=(15,13))
    number_of_runs=len(maps_dict)

    for i in range(number_of_runs):
            
        name_run=maps_dict.keys()[i]
        data=maps_dict[name_run].flatten()
        data_not_nan=data[np.logical_not(np.isnan(data))]
        bins,pdf=PDF(data_not_nan,same_bins)
        if i==0:
            plt.plot(bins, pdf,'k--',label=name_run,lw=3)
            sat_pdf=np.copy(pdf)
        else:
            plt.plot(bins, pdf,label=name_run+' R=%1.2f'%np.corrcoef(pdf[:],sat_pdf[:])[0,1])
            
    plt.legend(loc='best')
    plt.title(variable_name)
    plt.ylabel('Normalized PDF')
    plt.xlabel(variable_name)
    plt.savefig('/nfs/see-fs-01_users/eejvt/CASIM/'+variable_name+'_PDF.png')
    plt.show()


def coarse_grain(data,factor=5):
    if data.ndim==2:
        a=[0,0]
        a[0]=data.shape[0]/factor
        a[1]=data.shape[1]/factor
        new_data=jl.congrid(data,a)
    else:
        raise NameError('Need data with 2 horizontal dimensions')
    return new_data





