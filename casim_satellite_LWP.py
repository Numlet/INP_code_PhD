# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:23:55 2016

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
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE/Satellite_Comparison')
import satellite_comparison_suite as stc

from collections import OrderedDict

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/'
lon_offset=7


def show_dimensions(ds):
    print
    print 'Dimensions'
    for dim in ds.dimensions:
        print ' '*4, dim, ':', ds.dimensions[dim]

def show_variables(ds):
    print
    print 'Variables:'
    for var in ds.variables:
        print ' '*4, var, ':', ds.variables[var].long_name

def show_validrange(ds):
    print
    print 'Valid min and max and units:'
    for var in ds.variables:
        print ' '*4, var, ':', \
              ds.variables[var].valid_min, 'to', \
              ds.variables[var].valid_max,\
              '(',ds.variables[var].units,')'

def show_somedata(ds):
    print
    print 'Show some data for:',avar
    print 'index range: (' + str(iasc) + ', ' + \
          str(ilat[0]) + ':' + str(ilat[1]) + ' ,' + \
          str(ilon[0]) + ':' + str(ilon[1]) + ')'
    print ds.variables[avar][iasc, ilat[0]:ilat[1]+1, ilon[0]:ilon[1]+1]


sys.path.append('/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/code')
from amsr2_daily_v7 import AMSR2daily
amsr_data='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/code/data.remss.com/amsr2/bmaps_v07.2/y2014/m12/'
glob.glob(amsr_data+'*')
def read_data(filename=amsr_data+'f34_20141209v7.2.gz'):
    dataset = AMSR2daily(filename, missing=missing)
    if not dataset.variables: sys.exit('problem reading file')
    return dataset

ilon = (169,174)
ilat = (273,277)
iasc = 0
avar = 'vapor'
missing = -999.

dataset = read_data()
show_dimensions(dataset)
show_variables(dataset)
show_validrange(dataset)
show_somedata(dataset)


#%%

cube_3ord = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/L1/','LWP'))[0]
cube_single = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/SINGLE_MOMENT/L1/','LWP'))[0]
cube_SM_100_COOPER = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/SM_100_COOPER/L1/','LWP'))[0]
cube_SM_T40 = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/SM_T40/L1/','LWP'))[0]
cube_SM_LCOND_FALSE = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/SM_LCOND_FALSE/L1/','LWP'))[0]
cube_noice = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NOICE/L1/','LWP'))[0]
cube_2l = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_LESS/L1/','LWP'))[0]
cube_2m = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_MORE/L1/','LWP'))[0]
cube = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/L1/','LWP'))[0]
cube_con = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/BASE_CONTACT_242/L1/','LWP'))[0]
#cube_oldm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/OLD_MICRO/L1/','LWP'))[0]
#cube_nh = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/L1/','LWP'))[0]
cube_ni = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NOICE/L1/','LWP'))[0]

cube_csb = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/CLOUD_SQUEME/BASE/L1/','LWP'))[0]
cube_csbm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/CLOUD_SQUEME/MEYERS/L1/','LWP'))[0]
cube_m = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/MEYERS/L1/','LWP'))[0]
cube_gloprof= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/GLOMAP_PROFILE_DM/L1/','LWP'))[0]
cube_gl_csed=  iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/GP_HIGH_CSED/L1/','LWP'))[0]
cube_gl_low_csed=  iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/GP_LOW_CSED/L1/','LWP'))[0]
cube_SM_NOBIGG= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/SM_NOBIGG_T40/L1/','LWP'))[0]

cube_gpham= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/GP_HAMISH/L1/','LWP'))[0]
#cube_global= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/GLOBAL/L1/','LWP'))[0]



times=dataset.variables['time'][0,]
#times[times==missing]=0
LWP=dataset.variables['cloud'][0,]
LWP[LWP==missing]=np.nan
LWP[times<12]=np.nan
LWP[times>15]=np.nan
lon=dataset.variables['longitude']
lat=dataset.variables['latitude']


Xsat,Ysat=np.meshgrid(lon,lat)
Xsat[np.isnan(LWP)]=np.nan
Ysat[np.isnan(LWP)]=np.nan

Xsat_flat=Xsat.flatten()
Ysat_flat=Ysat.flatten()
LWP_flat=LWP.flatten()
Xsat_flat=Xsat_flat[np.logical_not(np.isnan(Xsat_flat))]
Ysat_flat=Ysat_flat[np.logical_not(np.isnan(Ysat_flat))]
LWP_flat=LWP_flat[np.logical_not(np.isnan(LWP_flat))]





#model_lons,model_lats=stc.unrotated_grid(stc.clean_cube(cube.copy()))
model_lons,model_lats=stc.unrotated_grid(cube)

coord=np.zeros([len(Xsat_flat),2])
coord[:,0]=Xsat_flat
coord[:,1]=Ysat_flat
cm=plt.cm.RdBu_r
#model_lons=np.linspace(-5,20,500)
X,Y=np.meshgrid(model_lons, model_lats)
grid_z1 = sc.interpolate.griddata(coord, LWP_flat, (X,Y), method='linear')


runs_dict=OrderedDict()
runs_dict['Satellite (AMSR2)']=grid_z1
runs_dict['DEMOTT2010']=cube[12].data
#runs_dict['BASE (CS)']=cube_csb[13].data
#runs_dict['MEYERS (CS)']=cube_csbm[13].data
#runs_dict['MEYERS']=cube_m[13].data
runs_dict['SINGLE_MOMENT']=cube_single[13].data
#runs_dict['SM_100_COOPER']=cube_SM_100_COOPER[13].data
#runs_dict['SM_T40']=cube_SM_T40[13].data
runs_dict['SM_LCOND_FALSE']=cube_SM_LCOND_FALSE[13].data
runs_dict['SM_NOBIGG']=cube_SM_NOBIGG[13].data

runs_dict['NOICE']=cube_noice[13].data
#runs_dict['2_ORD_LESS']=cube_2l[13].data
runs_dict['2_ORD_MORE']=cube_2m[13].data

         
for run in runs_dict:
    runs_dict[run]=stc.coarse_grain(stc.clean_cube(runs_dict[run]))
#    runs_dict[run]=stc.clean_cube(runs_dict[run])
model_lons,model_lats=stc.unrotated_grid(stc.clean_cube(cube))
X,Y=np.meshgrid(model_lons, model_lats)
X=stc.coarse_grain(X)
Y=stc.coarse_grain(Y)
#runs_dict['GLOPROF']=cube_gloprof[13].data
#runs_dict['GP_HIGH_CSED']=cube_gl_csed[13].data
#runs_dict['GP_LOW_CSED']=cube_gl_low_csed[13].data
#runs_dict['GP_HAM']=cube_gpham[13].data

         
levels=np.arange(0,0.45,0.05).tolist()
same_bins=np.linspace(0,0.5,100)

#levels=np.linspace(runs_dict['Satellite (AMSR2)'].min(),runs_dict['Satellite (AMSR2)'].max(),15)
stc.plot_map(runs_dict,levels,lat=X,lon=Y,variable_name='(first) LWP mm')
stc.plot_PDF(runs_dict,same_bins,
             variable_name=' (first) LWP mm')











#%%
for i in range(10):
    plt.close()

















