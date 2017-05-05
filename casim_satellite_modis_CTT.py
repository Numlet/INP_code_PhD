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
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE/Satellite_Comparison')
import satellite_comparison_suite as stc
from collections import OrderedDict

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

def PDF(data,nbins=100):
    min_val=data.min()
    max_val=data.max()
    if isinstance(nbins,np.ndarray):
        bins=nbins
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

import pprint
path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/'
SDS_NAME  = 'Cloud_Top_Height_Nadir_Night'
SDS_NAME  = 'cloud_top_height_1km'
#hdf = SD.SD(FILE_NAME)
SDS_NAME  = 'Cloud_Water_Path'
SDS_NAME  = 'Cloud_Water_Path_16'
SDS_NAME  = 'Cloud_Top_Height'
SDS_NAME  = 'Cloud_Top_Temperature'
hdf  =SD.SD(path+'modis/'+'MYD06_L2.A2014343.1325.006.2014344210847.hdf')
#print hdf.datasets().keys()
for k in hdf.datasets().keys():
    if 'Tem' in k:
        print k

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

sim_path='/nfs/a201/eejvt/CASIM/SO_KALLI/'
sub_folder='L1/'
code='top_temp'




cloud_top= iris.load(ukl.Obtain_name(sim_path+'TRY2/ALL_ICE_PROC/'+sub_folder,code))[0]
#cube_3ord = iris.load(ukl.Obtain_name(sim_path+'TRY2/3_ORD_LESS_762/'+sub_folder,code))[0]
cube_2m = iris.load(ukl.Obtain_name(sim_path+'TRY2/2_ORD_MORE/'+sub_folder,code))[0]
cube = iris.load(ukl.Obtain_name(sim_path+'TRY2/ALL_ICE_PROC/'+sub_folder,code))[0]
#cube_con = iris.load(ukl.Obtain_name(sim_path+'TRY2/BASE_CONTACT_242/'+sub_folder,code))[0]
#cube_oldm = iris.load(ukl.Obtain_name(sim_path+'OLD_MICRO/'+sub_folder,code))[0]
#cube_nh = iris.load(ukl.Obtain_name(sim_path+'TRY2/NO_HALLET/'+sub_folder,code))[0]
cube_single = iris.load(ukl.Obtain_name(sim_path+'SINGLE_MOMENT/'+sub_folder,code))[0]

cube_SM_100_COOPER = iris.load(ukl.Obtain_name(sim_path+'SM_100_COOPER/'+sub_folder,code))[0]
cube_SM_T40 = iris.load(ukl.Obtain_name(sim_path+'SM_T40/'+sub_folder,code))[0]
cube_SM_LCOND_FALSE = iris.load(ukl.Obtain_name(sim_path+'SM_LCOND_FALSE/'+sub_folder,code))[0]
cube_noice = iris.load(ukl.Obtain_name(sim_path+'SECOND_DOMAIN/NOICE/'+sub_folder,code))[0]
cube_m = iris.load(ukl.Obtain_name(sim_path+'NO_CLOUD_SQUEME/MEYERS/'+sub_folder,code))[0]

#%%
coord=np.zeros([len(sat_lon),2])
coord[:,0]=sat_lon
coord[:,1]=sat_lat
cm=plt.cm.RdBu_r
model_lons,model_lats=stc.unrotated_grid(cloud_top)
X,Y=np.meshgrid(model_lons, model_lats)
reload(stc)
grid_z1 = sc.interpolate.griddata(coord, sat_data, (X,Y), method='linear')
grid_z1[np.isnan(grid_z1)]=0

#%%


it=16

runs_dict=OrderedDict()


runs_dict['Satellite']=grid_z1
#runs_dict['DEMOTT2010']=cube[12].data
##runs_dict['BASE (CS)']=cube_csb[13].data
##runs_dict['MEYERS (CS)']=cube_csbm[13].data
#runs_dict['MEYERS']=cube_m[13].data
##runs_dict['3_ORD_LESS']=cube_3ord[13].data
#runs_dict['SINGLE_MOMENT']=cube_single[13].data
#runs_dict['2_ORD_LESS']=cube_2l[13].data
#runs_dict['2_ORD_MORE']=cube_2m[13].data
#runs_dict['OLD_MICRO']=cube_oldm[13].data


runs_dict['DEMOTT2010']=cube[12].data
#runs_dict['BASE (CS)']=cube_csb[13].data
#runs_dict['MEYERS (CS)']=cube_csbm[13].data
runs_dict['MEYERS']=cube_m[13].data
runs_dict['SINGLE_MOMENT']=cube_single[13].data
#runs_dict['SM_100_COOPER']=cube_SM_100_COOPER[13].data
#runs_dict['SM_T40']=cube_SM_T40[13].data
#runs_dict['SM_LCOND_FALSE']=cube_SM_LCOND_FALSE[13].data
#runs_dict['NOICE']=cube_noice[13].data
         
         
         
#runs_dict['2_ORD_LESS']=cube_2l[13].data
runs_dict['2_ORD_MORE']=cube_2m[13].data

#runs_dict['GLOPROF']=cube_gloprof[13].data
#runs_dict['GP_HIGH_CSED']=cube_gl_csed[13].data
#runs_dict['GP_LOW_CSED']=cube_gl_low_csed[13].data
#runs_dict['GP_HAM']=cube_gpham[13].data

#runs_dict['GLOMAP_PROFILE']=cube_gloprof[13].data
#runs_dict['LARGE_DOM']=cube_large_dom[13].data
#grid_z2 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='cubic')
#grid_z2[grid_z2<0]=0
#grid_z2[grid_z2==np.nan]=0
variable='CTT'
levels=np.linspace(0,runs_dict['Satellite'].max(),15)#runs_dict['Satellite'].min()
s=250
e=290
levels=np.linspace(s,e,15)
bins=np.linspace(s,e,100)
stc.plot_map(runs_dict,levels,lat=X,lon=Y,variable_name=variable)
stc.plot_PDF(runs_dict,bins,variable_name=variable)








#%%








