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
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import Jesuslib as jl
import matplotlib.pyplot as plt
from scipy.io import netcdf
sys.path.append('/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/code')
from scipy.io import netcdf
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE/Satellite_Comparison')
import satellite_comparison_suite as stc
import datetime
import scipy as sc
#font = {'family' : 'normal',
#        'weight' : 'normal',
#        'size'   : 16}

from collections import OrderedDict

runs_dict=OrderedDict()


#matplotlib.rc('font', **font)


path='/nfs/a201/eejvt/CASIM/SECOND_CLOUD/SATELLITE/'
from scipy.io import netcdf
#cubes  =iris.load(path+'ceres/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc')
#mb=netcdf.netcdf_file(path+'ceres_all_SO/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc','r') 

mb=netcdf.netcdf_file(path+'CERES/'+'CERES_SSF_NPP-XTRK_Edition1A_Subset_2015030100-2015030223.nc','r') 

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

ti=15#h
te=16#h

tdi=(datetime.datetime(2015,03,1,ti)-datetime.datetime(1970,1,1)).total_seconds()
tde=(datetime.datetime(2015,03,1,te)-datetime.datetime(1970,1,1)).total_seconds()

t16=(datetime.datetime(2015,03,1,16)-datetime.datetime(1970,1,1)).total_seconds()/3600.
    
sim_path='/nfs/a201/eejvt/CASIM/SECOND_CLOUD/'

cube_DM10 =  stc.clean_cube(iris.load(ukl.Obtain_name(sim_path+'/DM10/All_time_steps/','m01s01i208'))[0])
cube_GLO_HIGH =  stc.clean_cube(iris.load(ukl.Obtain_name(sim_path+'/GLO_HIGH/All_time_steps/','m01s01i208'))[0])
#cube_GLO_MEAN=  stc.clean_cube(iris.load(ukl.Obtain_name(sim_path+'/GLO_MEAN/All_time_steps/','m01s01i208'))[0])
cube_GLO_MIN=  stc.clean_cube(iris.load(ukl.Obtain_name(sim_path+'/GLO_MIN/All_time_steps/','m01s01i208'))[0])
cube_GP_HAM_DMDUST=  stc.clean_cube(iris.load(ukl.Obtain_name(sim_path+'/GP_HAM_DMDUST/All_time_steps/','m01s01i208'))[0])
cube_MEYERS=  stc.clean_cube(iris.load(ukl.Obtain_name(sim_path+'/MEYERS/All_time_steps/','m01s01i208'))[0])
cube_global=  stc.clean_cube(iris.load(ukl.Obtain_name(sim_path+'/GLOBAL/All_time_steps/','m01s01i208'))[0])
#cube_DM10 = stc.clean_cube(cube_DM10)
#cube_GLO_HIGH = stc.clean_cube(cube_GLO_HIGH)
#cube_GLO_MEAN = stc.clean_cube(cube_GLO_MEAN)
#cube_GLO_MIN = stc.clean_cube(cube_GLO_MIN)
#cube_GP_HAM_DMDUST = stc.clean_cube(cube_GP_HAM_DMDUST)
#cube_MEYERS = stc.clean_cube(cube_MEYERS)
#cube_GLO_MIN = stc.clean_cube(cube_GLO_MIN)


#cube=cube.extract(iris.Constraint(time=jl.find_nearest_vector(cube.coord('time').points,t13)))
#cube_csb=cube_csb[13,:,:]
#cube=cube[12,:,:]
#cube_nh=cube_nh[13,:,:]
#cube_3ord=cube_3ord[13,:,:]
#cube_2m=cube_2m[13,:,:]
##cube_con=cube_con[13,:,:]
#cube_oldm=cube_oldm[13,:,:]

#cube_large_dom = cube_large_dom.regrid(cube_m, iris.analysis.Linear())


#lmodel_lons,lmodel_lats=stc.unrotated_grid(cube_large_dom)
cube_global = cube_global.regrid(cube_DM10, iris.analysis.Linear())

#%%
reload(stc)
model_lons,model_lats=stc.unrotated_grid(cube_DM10)
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
plt.imshow(grid_z1)
#%%

it=16
runs_dict=OrderedDict()
runs_dict['Satellite']=grid_z1
runs_dict['GLOBAL']=cube_global[it].data
runs_dict['MEYERS']=cube_MEYERS[it].data

runs_dict['DM10']=cube_DM10[it].data
runs_dict['GLO_HIGH']=cube_GLO_HIGH[it].data
#runs_dict['MEYERS (CS)']=cube_csbm[13].data
#runs_dict['MEYERS']=cube_m[13].data
#runs_dict['3_ORD_LESS']=cube_3ord[13].data
#runs_dict['2_ORD_LESS']=cube_2l[13].data
#runs_dict['2_ORD_MORE']=cube_2m[13].data
#runs_dict['OLD_MICRO']=cube_oldm[13].data
#runs_dict['GLOPROF']=cube_gloprof[13].data
runs_dict['GLO_MEAN']=cube_GLO_MEAN[it].data
runs_dict['GLO_MIN']=cube_GLO_MIN[it].data
runs_dict['GP_HAM_DMDUST']=cube_GP_HAM_DMDUST[it].data

#runs_dict['GLOMAP_PROFILE']=cube_gloprof[13].data
#runs_dict['LARGE_DOM']=cube_large_dom[13].data
#grid_z2 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='cubic')
#grid_z2[grid_z2<0]=0
#grid_z2[grid_z2==np.nan]=0
levels=np.linspace(0,runs_dict['Satellite'].max(),15)#runs_dict['Satellite'].min()
levels=np.linspace(0,680,15)#runs_dict['Satellite'].min()
levels_bin=np.linspace(0,runs_dict['Satellite'].max(),30)#runs_dict['Satellite'].min()
levels_bin=np.linspace(120,680,150)#runs_dict['Satellite'].min()
stc.plot_map(runs_dict,levels,lat=X,lon=Y,variable_name='(SC) TOA shortwave Wm-2')
stc.plot_PDF(runs_dict,levels_bin,variable_name='(SC) TOA shortwave Wm-2')






