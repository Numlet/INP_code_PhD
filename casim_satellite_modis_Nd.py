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

from collections import OrderedDict
my_dictionary=OrderedDict()
my_dictionary['foo']=3
my_dictionary['aol']=1




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
import iris
#import pprint
Nd=iris.load('/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/modis/dan/Nd2_MYD06_L2.A2014343.1325.006.2014344210847.hdf.nc')[0]


path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/'
hdf  =SD.SD(path+'modis/'+'MYD03.A2014343.1325.006.2014344162340.hdf')
for k in hdf.datasets().keys():
    if '' in k:
        print k

SDS_NAME  = 'Latitude'
sds = hdf.select(SDS_NAME)
lat = sds.get()
SDS_NAME  = 'Longitude'
sds = hdf.select(SDS_NAME)
lon = sds.get()
#plt.contourf(lon,lat,Nd.data)

#%%

sat_lon=lon.flatten()
sat_lat=lat.flatten()
sat_data=Nd.data.flatten()
#for att in sds.attributes():
#    print att
#%%
cube_cdnc= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/L1/','Cloud_droplet_concentratio_at_maximum_cloud_water_content'))[0]
cube_gloprof= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/GLOMAP_PROFILE_DM/L1/','Cloud_droplet_concentratio_at_maximum_cloud_water_content'))[0]
cube_gl_csed= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/GP_HIGH_CSED/L1/','Cloud_droplet_concentratio_at_maximum_cloud_water_content'))[0]
cube_gl_low_csed= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/GP_LOW_CSED/L1/','Cloud_droplet_concentratio_at_maximum_cloud_water_content'))[0]
cube_csb = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/CLOUD_SQUEME/BASE/L1/','Cloud_droplet_concentratio_at_maximum_cloud_water_content'))[0]
#cube_cdnc= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/L1/','Cloud_droplet_concentratio_at_maximum_cloud_water_content'))[0]
#cube_cdnc= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/L1/','Cloud_droplet_concentratio_at_maximum_cloud_water_content'))[0]
#cube_cdnc= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/L1/','loud droplet number concentration'))[0]
#cube_cdnc=cube_cdnc.collapsed(['model_level_number'],iris.analysis.MAX)
#cube_cdnc=cube_cdnc.collapsed(['model_level_number'],iris.analysis.MAX)

cube_3ord = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/L1/','LWP'))[0]
cube_2m = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_MORE/L1/','LWP'))[0]
cube = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/L1/','LWP'))[0]
cube_con = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/BASE_CONTACT_242/L1/','LWP'))[0]
#cube_oldm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/OLD_MICRO/L1/','LWP'))[0]
#cube_nh = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/L1/','LWP'))[0]
cube_ni = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NOICE/L1/','LWP'))[0]

cube_csbm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/CLOUD_SQUEME/MEYERS/L1/','LWP'))[0]
cube_m = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/MEYERS/L1/','LWP'))[0]



#%%


coord=np.zeros([len(sat_lon),2])
coord[:,0]=sat_lon
coord[:,1]=sat_lat
cm=plt.cm.RdBu_r
model_lons,model_lats=unrotated_grid(cube_cdnc)
#model_lons=np.linspace(-5,20,500)
X,Y=np.meshgrid(model_lons, model_lats)
#%%
#lon_old,lat_old=unrotated_grid(cube_oldm)
#Xo,Yo=np.meshgrid(lon_old,lat_old)
#coord_model=np.zeros((len(Xo.flatten()),2))
#coord_model[:,0]=Xo.flatten()
#coord_model[:,1]=Yo.flatten()
#data_old= sc.interpolate.griddata(coord_model, cube_oldm.data.flatten(), (X,Y), method='linear')
#grid_z0 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='nearest')
grid_z1 = sc.interpolate.griddata(coord, sat_data, (X,Y), method='linear')
#grid_z2 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='cubic')
#grid_z2[grid_z2<0]=0

'''
grid_z1[np.isnan(grid_z1)]=0
'''

        
runs_dict=OrderedDict()
runs_dict['Satellite']=grid_z1
runs_dict['ALL_ICE_PROC']=cube_cdnc[12].data*1e-6
runs_dict['BASE (CS)']=cube_csb[13].data*1e-6
runs_dict['GLOMAP_PROFILE']=cube_gloprof[13].data*1e-6
runs_dict['GP_HIGH_CSED']=cube_gl_csed[13].data*1e-6
runs_dict['GP_LOW_CSED']=cube_gl_low_csed[13].data*1e-6

         
#runs_dict['BASE (CS)']=cube_csb[13].data
#runs_dict['MEYERS (CS)']=cube_csbm[13].data
#runs_dict['MEYERS']=cube_m[13].data
#runs_dict['3_ORD_LESS']=cube_3ord[13].data
#runs_dict['2_ORD_MORE']=cube_2m[13].data
#runs_dict['OLD_MICRO']=cube_oldm[13].data
#grid_z2 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='cubic')
#grid_z2[grid_z2<0]=0
#grid_z2[grid_z2==np.nan]=0
#%%
reload(stc)
def min_max(dictionary):
    minval=99999
    maxval=0
    for key in dictionary.keys():
        new_min=np.min(dictionary[key][np.logical_not(np.isnan(dictionary[key]))])
        new_max=np.max(dictionary[key][np.logical_not(np.isnan(dictionary[key]))])
        if new_min<minval:
            minval=np.copy(new_min)
        if new_max>maxval:
            maxval=np.copy(new_max)
    return minval,maxval

name='CDNC'
minval,maxval = min_max(runs_dict)

#levels=np.linspace(runs_dict['ALL_ICE_PROC'].min(),runs_dict['ALL_ICE_PROC'].max(),15)
levels=np.linspace(20,200,15)
stc.plot_map(runs_dict,levels,lat=X,lon=Y,variable_name=name)


bins=np.linspace(20,200,150)
stc.plot_PDF(runs_dict,bins,variable_name=name)






        
        
#%%

        
cube_cdnc.coord('model_level_number')
CDNC_max_cloud_water=cube_cdnc.copy()

CDNC_max_cloud_water=CDNC_max_cloud_water.collapsed(['model_level_number'],iris.analysis.MAX)

cdnc=cube_cdnc.data
cube_l = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/','m01s00i254'))[0]
lw=cube_l.data
args=np.argmax(lw,axis=1)
data=np.zeros(args.shape)
for it in range(data.shape[0]):
    print it
    for ilat in range(data.shape[1]):
        for ilon in range(data.shape[2]):
            data[it,ilat,ilon]=cdnc[it,args[it,ilat,ilon],ilat,ilon]

CDNC_max_cloud_water.data=data
        
        
    
#
