#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:24:07 2017

@author: eejvt
"""

import iris
import iris.plot as iplt
import sys
dir_scripts='/nfs/see-fs-01_users/eejvt/UKCA_postproc'#Change this to the downloaded folder
sys.path.append(dir_scripts)
import UKCA_lib as ukl
import numpy as np
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import matplotlib.pyplot as plt
from scipy.io import netcdf
sys.path.append('/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/code')
from scipy.io import netcdf
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE/Satellite_Comparison')
import satellite_comparison_suite as stc
import datetime
import scipy as sc
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

from collections import OrderedDict

runs_dict=OrderedDict()


matplotlib.rc('font', **font)


path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/'
from scipy.io import netcdf
cubes  =iris.load(path+'ceres/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc')
mb=netcdf.netcdf_file(path+'ceres/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc','r') 
mb=netcdf.netcdf_file(path+'ceres_all_SO/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc','r') 

#SW=cubes[1]
#model_lons=np.arange(-0.02*250,250*0.02,0.02)[0:]
#model_lats=np.arange(-0.02*250-52,250*0.02-52,0.02)

#model_lons=np.linspace(-7,17)
#model_lats=np.linspace(-47.5,-58)
times_ceres=mb.variables['time'].data*24*60*60

#model_lons=model_lons+lon_offset

LW=np.copy(mb.variables['CERES_SW_TOA_flux___upwards'].data)
#SW[SW>1400]=0
lon=mb.variables['lon'].data
lat=mb.variables['lat'].data

ti=13#h
te=23#h

tdi=(datetime.datetime(2014,12,9,ti)-datetime.datetime(1970,1,1)).total_seconds()
tde=(datetime.datetime(2014,12,9,te)-datetime.datetime(1970,1,1)).total_seconds()

t13=(datetime.datetime(2014,12,9,14)-datetime.datetime(1970,1,1)).total_seconds()/3600.
    
cube_3ord = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/All_time_steps/','m01s01i208'))[0]
cube_2m = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_MORE/All_time_steps/','m01s01i208'))[0]
cube_2l = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_LESS/All_time_steps/','m01s01i208'))[0]
cube = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/','m01s01i208'))[0]
#cube_con = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/BASE_CONTACT_242/All_time_steps/','m01s01i208'))[0]
cube_nh = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/All_time_steps/','m01s01i208'))[0]
cube_ni = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NOICE/All_time_steps/','m01s01i208'))[0]
cube_csb = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/CLOUD_SQUEME/BASE/All_time_steps/','m01s01i208'))[0]
cube_csbm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/CLOUD_SQUEME/MEYERS/All_time_steps/','m01s01i208'))[0]
cube_m = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/MEYERS/All_time_steps/','m01s01i208'))[0]
cube_gloprof= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/GLOMAP_PROFILE_DM/All_time_steps/','m01s01i208'))[0]

cube_gl_csed=  iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/GP_HIGH_CSED/All_time_steps/','m01s01i208'))[0]
cube_gl_low_csed=  iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/GP_LOW_CSED/All_time_steps/','m01s01i208'))[0]



cube_large_dom= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/LARGE_DOMAIN/All_time_steps/','m01s01i208'))[0]

cube_oldm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/OLD_MICRO/All_time_steps/','m01s01i208'))[0]
cube_oldm = cube_oldm.regrid(cube_m, iris.analysis.Linear())


#cube=cube.extract(iris.Constraint(time=jl.find_nearest_vector(cube.coord('time').points,t13)))
#cube_csb=cube_csb[13,:,:]
#cube=cube[12,:,:]
#cube_nh=cube_nh[13,:,:]
#cube_3ord=cube_3ord[13,:,:]
#cube_2m=cube_2m[13,:,:]
##cube_con=cube_con[13,:,:]
#cube_oldm=cube_oldm[13,:,:]

cube_large_dom = cube_large_dom.regrid(cube_m, iris.analysis.Linear())


#lmodel_lons,lmodel_lats=stc.unrotated_grid(cube_large_dom)

#%%
reload(stc)
model_lons,model_lats=stc.unrotated_grid(cube)
#times_range=np.argwhere((times_ceres >= tdi) & (times_ceres <=tde))
times_range=np.logical_and([times_ceres >= tdi],[times_ceres <=tde])[0]
sat_lon=lon[times_range]
sat_lat=lat[times_range]
sat_LW=LW[times_range]
coord=np.zeros([len(sat_lon),2])
coord[:,0]=sat_lon
coord[:,1]=sat_lat
cm=plt.cm.RdBu_r
#model_lons=np.linspace(-5,20,500)
X,Y=np.meshgrid(model_lons, model_lats)
#Xo,Yo=np.meshgrid(lon_old,lat_old)
#data_old= sc.interpolate.griddata(coord_model, cube_oldm.data.flatten(), (X,Y), method='linear')
#grid_z0 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='nearest')
grid_z1 = sc.interpolate.griddata(coord, sat_LW, (X,Y), method='linear')
#%%


runs_dict=OrderedDict()
runs_dict['Satellite']=grid_z1
runs_dict['ALL_ICE_PROC']=cube[12].data
runs_dict['BASE (CS)']=cube_csb[13].data
#runs_dict['MEYERS (CS)']=cube_csbm[13].data
#runs_dict['MEYERS']=cube_m[13].data
#runs_dict['3_ORD_LESS']=cube_3ord[13].data
runs_dict['2_ORD_LESS']=cube_2l[13].data
runs_dict['2_ORD_MORE']=cube_2m[13].data
runs_dict['OLD_MICRO']=cube_oldm[13].data
runs_dict['GLOPROF']=cube_gloprof[13].data
runs_dict['GP_HIGH_CSED']=cube_gl_csed[13].data
runs_dict['GP_LOW_CSED']=cube_gl_low_csed[13].data

#runs_dict['GLOMAP_PROFILE']=cube_gloprof[13].data
#runs_dict['LARGE_DOM']=cube_large_dom[13].data
#grid_z2 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='cubic')
#grid_z2[grid_z2<0]=0
#grid_z2[grid_z2==np.nan]=0
levels=np.linspace(150,800,15)#runs_dict['Satellite'].min()
levels=np.linspace(150,runs_dict['Satellite'].max(),15)#runs_dict['Satellite'].min()
levels_bin=np.linspace(150,800,150)#runs_dict['Satellite'].min()
levels_bin=np.linspace(150,runs_dict['Satellite'].max(),30)#runs_dict['Satellite'].min()
stc.plot_map(runs_dict,levels,lat=X,lon=Y,variable_name='TOA shortwave Wm-2')
stc.plot_PDF(runs_dict,levels_bin,variable_name='TOA shortwave Wm-2')






