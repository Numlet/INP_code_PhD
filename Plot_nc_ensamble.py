# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:15:23 2016

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
from scipy import stats
from scipy.optimize import curve_fit
import scipy
import iris


data_path='/nfs/a107/earkpr/DataVisualisation/Jesus/UKCA_FILES/CCN/'
folder_plots='/nfs/see-fs-01_users/eejvt/svn_test/plots/'
cmap=plt.cm.OrRd
folders=['Hires_N96/','Lores_N48/']
a=glob(data_path+folders[0]+'*')
for nc_file in a :
    nc_file=nc_file[len(data_path+folders[0]):]
    print nc_file
    file_name=data_path+folders[0]+'CCN0pt1_tebaa-tebgz_pm2008jul.nc'
    mb=netcdf.netcdf_file(file_name,'r') 
    
    cube=iris.load(file_name)[0]
    lon=cube.coord('longitude').points[:]
    lat=cube.coord('latitude').points[:]
    
    cube_data=cube.data[:,0,:,:,:]
    
    mean_values=cube_data.mean(axis=0)
    std_values=cube_data.std(axis=0)
    
    file_name_plot=folder_plots+nc_file+'_plot_surface_mean'
    jl.plot(mean_values[0,:,:],title='Mean '+cube.long_name,lat=lat,lon=lon,cblabel=cube.units.origin,file_name=file_name_plot,saving_format='png',cmap=cmap)
    plt.close()    
    jl.plot(mean_values[0,:,:],title='Mean '+cube.long_name,lat=lat,lon=lon,cblabel=cube.units.origin,file_name=file_name_plot,saving_format='ps',cmap=cmap)
    plt.close()    
    jl.plot(mean_values[0,:,:],title='Mean '+cube.long_name,lat=lat,lon=lon,cblabel=cube.units.origin,file_name=file_name_plot,saving_format='pdf',cmap=cmap)
    plt.close()    
    jl.plot(mean_values[0,:,:],title='Mean '+cube.long_name,lat=lat,lon=lon,cblabel=cube.units.origin,file_name=file_name_plot,cmap=cmap)
    plt.close()    
    
    file_name_plot=folder_plots+nc_file+'_plot_surface_std'
    jl.plot(std_values[0,:,:],title='Std '+cube.long_name,lat=lat,lon=lon,cblabel=cube.units.origin,file_name=file_name_plot,saving_format='png',cmap=cmap)
    plt.close()    
    jl.plot(std_values[0,:,:],title='Std '+cube.long_name,lat=lat,lon=lon,cblabel=cube.units.origin,file_name=file_name_plot,saving_format='ps',cmap=cmap)
    plt.close()    
    jl.plot(std_values[0,:,:],title='Std '+cube.long_name,lat=lat,lon=lon,cblabel=cube.units.origin,file_name=file_name_plot,saving_format='pdf',cmap=cmap)
    plt.close()    
    jl.plot(std_values[0,:,:],title='Std '+cube.long_name,lat=lat,lon=lon,cblabel=cube.units.origin,file_name=file_name_plot,cmap=cmap)
    plt.close()    

























