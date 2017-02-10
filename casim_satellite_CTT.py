#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:12:18 2017

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
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
from collections import OrderedDict

runs_dict=OrderedDict()

matplotlib.rc('font', **font)

path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/'
SDS_NAME  = 'Cloud_Top_Temperature'
hdf  =SD.SD(path+'modis/'+'MYD06_L2.A2014343.1325.006.2014344210847.hdf')
#print hdf.datasets().keys()
sds = hdf.select(SDS_NAME)
data = sds.get()
data=(data+15000)*0.009999999776482582
mask_value=-9999
mask_value=-32767
data[data==mask_value]=np.float64('Nan')
print data.shape
lat = hdf.select('Latitude')
latitude = lat[:,:]
lon = hdf.select('Longitude')
longitude = lon[:,:]


sat_lon=longitude.flatten()
sat_lat=latitude.flatten()
sat_data=data.flatten()
#for att in sds.attributes():
#    print att
#%%

#cube_3ord = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/L1/','CTT'))[0]
#cube_2m = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_MORE/L1/','CTT'))[0]
cube = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/L1/','CTT'))[0]
#cube_con = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/BASE_CONTACT_242/L1/','CTT'))[0]
#cube_oldm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/OLD_MICRO/All_time_steps/','m01s09i223'))[0]
#cube_nh = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/L1/','CTT'))[0]
#cube_csb = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/CLOUD_SQUEME/BASE/L1/','CTT'))[0]
cube_csbm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/CLOUD_SQUEME/MEYERS/L1/','CTT'))[0]
cube_m = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/MEYERS/L1/','CTT'))[0]

#%%
coord=np.zeros([len(sat_lon),2])
coord[:,0]=sat_lon
coord[:,1]=sat_lat
cm=plt.cm.RdBu_r
model_lons,model_lats=stc.unrotated_grid(cube)
X,Y=np.meshgrid(model_lons, model_lats)
#%%
reload(stc)
grid_z1 = sc.interpolate.griddata(coord, sat_data, (X,Y), method='linear')
grid_z1[np.isnan(grid_z1)]=0

runs_dict=OrderedDict()
runs_dict['Satellite']=grid_z1
runs_dict['ALL_ICE_PROC']=cube[12].data
#runs_dict['BASE (CS)']=cube_csb[13].data
runs_dict['MEYERS (CS)']=cube_csbm[13].data
runs_dict['MEYERS']=cube_m[13].data
#runs_dict['3_ORD_LESS']=cube_3ord[13].data
#runs_dict['2_ORD_MORE']=cube_2m[13].data
variable='CCT'
levels=np.linspace(260,273,15)
stc.plot_map(runs_dict,levels,lat=X,lon=Y,variable_name=variable)
bins=np.linspace(250,273,100)
stc.plot_PDF(runs_dict,bins,variable_name=variable)


