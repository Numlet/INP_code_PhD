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

path='/nfs/a201/eejvt/CASIM/SECOND_CLOUD/SATELLITE/'
SDS_NAME  = 'Cloud_Top_Temperature'
hdf  =SD.SD(path+'MODIS/'+'MOD06_L2.A2015060.1050.006.2015061055531.hdf')
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



sim_path='/nfs/a201/eejvt/CASIM/SECOND_CLOUD/'
sub_folder='L1/'
code='CTT'

cube_DM10 = iris.load(ukl.Obtain_name(sim_path+'/DM10/'+sub_folder,code))[0]
cube_GLO_HIGH = iris.load(ukl.Obtain_name(sim_path+'/GLO_HIGH/'+sub_folder,code))[0]
cube_GLO_MEAN = iris.load(ukl.Obtain_name(sim_path+'/GLO_MEAN/'+sub_folder,code))[0]
cube_GLO_MIN = iris.load(ukl.Obtain_name(sim_path+'/GLO_MIN/'+sub_folder,code))[0]
cube_GP_HAM_DMDUST = iris.load(ukl.Obtain_name(sim_path+'/GP_HAM_DMDUST/'+sub_folder,code))[0]
cube_MEYERS = iris.load(ukl.Obtain_name(sim_path+'/MEYERS/'+sub_folder,code))[0]



#%%
coord=np.zeros([len(sat_lon),2])
coord[:,0]=sat_lon
coord[:,1]=sat_lat
cm=plt.cm.RdBu_r
model_lons,model_lats=stc.unrotated_grid(cube_DM10)
X,Y=np.meshgrid(model_lons, model_lats)
#%%
reload(stc)
grid_z1 = sc.interpolate.griddata(coord, sat_data, (X,Y), method='linear')
grid_z1[np.isnan(grid_z1)]=0

it=16
runs_dict=OrderedDict()
runs_dict['Satellite']=grid_z1
runs_dict['DM10']=cube_DM10[it].data
runs_dict['GLO_HIGH']=cube_GLO_HIGH[16].data
runs_dict['GLO_MEAN']=cube_GLO_MEAN[16].data
runs_dict['GLO_MIN']=cube_GLO_MIN[16].data
runs_dict['GP_HAM_DMDUST']=cube_GP_HAM_DMDUST[16].data
runs_dict['MEYERS']=cube_MEYERS[16].data
variable='2nd CTT'
levels=np.linspace(250,273,15)
stc.plot_map(runs_dict,levels,lat=X,lon=Y,variable_name=variable)
bins=np.linspace(250,273,100)
stc.plot_PDF(runs_dict,bins,variable_name=variable)
