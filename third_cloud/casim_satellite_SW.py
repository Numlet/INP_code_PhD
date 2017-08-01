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
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

from collections import OrderedDict

runs_dict=OrderedDict()


matplotlib.rc('font', **font)


path='/nfs/a201/eejvt/CASIM/THIRD_CLOUD/SATELLITE/'
from scipy.io import netcdf
#cubes  =iris.load(path+'ceres/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc')
#mb=netcdf.netcdf_file(path+'ceres_all_SO/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc','r') 

mb=netcdf.netcdf_file(path+'CERES/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2015011000-2015011123.nc','r') 

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
te=18#h

tdi=(datetime.datetime(2015,01,10,ti)-datetime.datetime(1970,1,1)).total_seconds()
tde=(datetime.datetime(2015,01,10,te)-datetime.datetime(1970,1,1)).total_seconds()

t16=(datetime.datetime(2015,01,10,16)-datetime.datetime(1970,1,1)).total_seconds()/3600.
    
sim_path='/nfs/a201/eejvt/CASIM/THIRD_CLOUD/'

cube_DM10 = iris.load(ukl.Obtain_name(sim_path+'/DM10/All_time_steps/','m01s01i208'))[0]
cube_GLO_HIGH = iris.load(ukl.Obtain_name(sim_path+'/GLO_HIGH/All_time_steps/','m01s01i208'))[0]
cube_GLO_MEAN= iris.load(ukl.Obtain_name(sim_path+'/GLO_MEAN/All_time_steps/','m01s01i208'))[0]
cube_GLO_MIN= iris.load(ukl.Obtain_name(sim_path+'/GLO_MIN/All_time_steps/','m01s01i208'))[0]
cube_GP_HAM_DMDUST= iris.load(ukl.Obtain_name(sim_path+'/DM_DUST/All_time_steps/','m01s01i208'))[0]
cube_MEYERS= iris.load(ukl.Obtain_name(sim_path+'/MEYERS/All_time_steps/','m01s01i208'))[0]
#cube_GLO_MIN= iris.load(ukl.Obtain_name(sim_path+'/GLO_MIN/All_time_steps/','m01s01i208'))[0]


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
#%%

it=16
runs_dict=OrderedDict()
congrid=False
if congrid:
    size_grid=(50,50)

    X=jl.congrid(X,size_grid)
    Y=jl.congrid(Y,size_grid)
    
    
    runs_dict['Satellite']=jl.congrid(grid_z1,size_grid)
    runs_dict['DM10']=jl.congrid(cube_DM10[it].data,size_grid)
    runs_dict['GLO_HIGH']=jl.congrid(cube_GLO_HIGH[it].data,size_grid)
    #runs_dict['MEYERS (CS)']=cube_csbm[13].data
    #runs_dict['MEYERS']=cube_m[13].data
    #runs_dict['3_ORD_LESS']=cube_3ord[13].data
    #runs_dict['2_ORD_LESS']=cube_2l[13].data
    #runs_dict['2_ORD_MORE']=cube_2m[13].data
    #runs_dict['OLD_MICRO']=cube_oldm[13].data
    #runs_dict['GLOPROF']=cube_gloprof[13].data
    runs_dict['GLO_MEAN']=jl.congrid(cube_GLO_MEAN[it].data,size_grid)
    runs_dict['GLO_MIN']=jl.congrid(cube_GLO_MIN[it].data,size_grid)
    runs_dict['GP_HAM_DMDUST']=jl.congrid(cube_GP_HAM_DMDUST[it].data,size_grid)
    runs_dict['MEYERS']=jl.congrid(cube_MEYERS[it].data,size_grid)
    
    #runs_dict['GLOMAP_PROFILE']=cube_gloprof[13].data
    #runs_dict['LARGE_DOM']=cube_large_dom[13].data
    #grid_z2 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='cubic')
    #grid_z2[grid_z2<0]=0
    #grid_z2[grid_z2==np.nan]=0
    levels=np.linspace(0,runs_dict['Satellite'].max(),15)#runs_dict['Satellite'].min()
    levels=np.linspace(130,800,15)#runs_dict['Satellite'].min()
    levels_bin=np.linspace(0,runs_dict['Satellite'].max(),30)#runs_dict['Satellite'].min()
    levels_bin=np.linspace(130,800,50)#runs_dict['Satellite'].min()
    stc.plot_map(runs_dict,levels,lat=X,lon=Y,variable_name='(THIRD) TOA shortwave reduced Wm-2')
    stc.plot_PDF(runs_dict,levels_bin,variable_name='(THIRD) TOA shortwave reduced  Wm-2')
else:
    size_grid=(50,50)

#    X=jl.congrid(X,size_grid)
#    Y=jl.congrid(Y,size_grid)
    
    
    runs_dict['Satellite']=grid_z1
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
    runs_dict['MEYERS']=cube_MEYERS[it].data
    
    #runs_dict['GLOMAP_PROFILE']=cube_gloprof[13].data
    #runs_dict['LARGE_DOM']=cube_large_dom[13].data
    #grid_z2 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='cubic')
    #grid_z2[grid_z2<0]=0
    #grid_z2[grid_z2==np.nan]=0
    
    for run in runs_dict:
        runs_dict[run]=stc.coarse_grain(stc.clean_cube(runs_dict[run]),factor=10)
    #    runs_dict[run]=stc.clean_cube(runs_dict[run])
    model_lons,model_lats=stc.unrotated_grid(stc.clean_cube(cube_DM10))
    X,Y=np.meshgrid(model_lons, model_lats)
    X=stc.coarse_grain(X,factor=10)
    Y=stc.coarse_grain(Y,factor=10)

    levels=np.linspace(0,runs_dict['Satellite'].max(),15)#runs_dict['Satellite'].min()
    levels=np.linspace(100,700,15)#runs_dict['Satellite'].min()
    levels_bin=np.linspace(0,runs_dict['Satellite'].max(),30)#runs_dict['Satellite'].min()
    levels_bin=np.linspace(100,700,15)#runs_dict['Satellite'].min()
    stc.plot_map(runs_dict,levels,lat=X,lon=Y,variable_name='(THIRD) TOA shortwave Wm-2')
    stc.plot_PDF(runs_dict,levels_bin,variable_name='(THIRD) TOA shortwave Wm-2')




#%%



from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
 
map = Basemap(projection='cea')
 
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color = 'coral')
map.drawmapboundary()
 
lons = [-135.3318, -134.8331, -134.6572]
lats = [57.0799, 57.0894, 56.2399]

plons=[]
plats=[]
for i in range(len(sat_lat)):
#    if i<10:
    print sat_lon[i], sat_lat[i]
    plons.append(sat_lon[i])
    plats.append(sat_lat[i])
#    if plon<0:
#        plon

x,y = map(plons, plats)
map.plot(x, y, 'bo', markersize=1)
 
plt.show()
#%%

#X,Y=np.meshgrid(lat,lon)
plt.figure()
plt.contourf(X,Y,grid_z1, origin='lower',cmap=plt.cm.RdBu_r)
plt.plot(plons,plats,'ko',markersize=1)
cb=plt.colorbar()
#%%
plt.figure()
plt.contourf(X,Y,cube_DM10[it].data, origin='lower',cmap=plt.cm.RdBu_r)
plt.plot(X.flatten(),Y.flatten(),'ko',markersize=0.1)
#%%
plt.figure()
X_con=jl.congrid(X,size_grid)
Y_con=jl.congrid(Y,size_grid)
plt.contourf(X_con,Y_con,jl.congrid(cube_DM10[it].data,size_grid), origin='lower',cmap=plt.cm.RdBu_r)
plt.plot(X_con.flatten(),Y_con.flatten(),'ko',markersize=1)


