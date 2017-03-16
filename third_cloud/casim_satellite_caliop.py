#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:39:16 2017

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
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
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

sys.path.append('/nfs/a107/eejvt/PYTHON_CODE/Satellite_Comparison')
import satellite_comparison_suite as stc

from collections import OrderedDict


path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/'

hdf  =SD.SD(path+'caliop/'+'CAL_LID_L2_05kmCLay-Prov-V3-30.2014-12-09T13-12-45ZD_Subset.hdf')
nc_file  =iris.load(path+'caliop/'+'CAL_LID_L2_05kmCLay-Prov-V3-30.2014-12-09T13-12-45ZD_Subset.nc')

mb=netcdf.netcdf_file(path+'caliop/'+'CAL_LID_L2_05kmCLay-Prov-V3-30.2014-12-09T13-12-45ZD_Subset.nc')
import netCDF4
mb=netCDF4.Dataset(path+'caliop/'+'CAL_LID_L2_05kmCLay-Prov-V3-30.2014-12-09T13-12-45ZD_Subset.nc','r')

#print hdf.datasets().keys()
for k in hdf.datasets().keys():
    if 'ati' in k:
        print k

lat = hdf.select('Latitude')
latitude = lat[:,:]
lon = hdf.select('Longitude')
longitude = lon[:,:]


column_opt_dept=hdf.select('Column_Optical_Depth_Cloud_532').get()
sds=hdf.select('Layer_Top_Altitude')
data=sds.get()
data=hdf.select('Layer_Top_Temperature').get()
data[data==data.min()]=np.nan
data=-data
#%%
for i in range(10):
    plt.plot(lat.get()[:,1],data[:,i])

#%%
