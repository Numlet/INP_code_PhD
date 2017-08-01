# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:33:56 2016

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
import matplotlib.pyplot as plt
import matplotlib.animation as animationlt
#from mayavi import mlab
import variable_dict as vd
#from cube_browser import Contour, Browser, Contourf, Pcolormesh
#%%
word1='SW'
word2='shortwave'
word1='LW'
word2='longwave'
word1='CLOUD'
word2='cloud'
word2='TEMP'
word1='ICE'
word2='ice'
word=''
#path='/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/DEMOTT3ORD/All_time_steps/'
path='/nfs/a201/eejvt/CASIM/SO_KALLI/NO_INITIAL_ICE/BASE_RUN/All_time_steps/'

nc_files=glob.glob(path+'*.nc')
for inc in range(len(nc_files)):
    if word in nc_files[inc][len(path):] or word.upper() in nc_files[inc][len(path):]:
        print inc,nc_files[inc][len(path):]
stash_code='m01s09i216'#_TOTAL_CLOUD_AMOUNT_-_RANDOM_OVERLAP_.
stash_code='m01s00i075'
stash_code='m01s00i268'
stash_code='m01s00i271'#CLOUD_ICE_(CRYSTALS)_AFTER_TIMESTEP_
stash_code='m01s04i241'#HETEROGENOUS_NUCLEATION_RATE_kgdivkgdivs
stash_code='m01s00i272'#RAIN_AFTER_TIMESTEP_
stash_code='m01s00i150'#upward_air_velocity
stash_code='m01s15i102'#height

stash_code='00i078'#ICE_NUMBER_AFTER_TIMESTEP
stash_code='m01s00i254'#mass_fraction_of_cloud_liquid_water_in_air
stash_code='m01s00i012'#mass_fraction_of_cloud_ice_in_air
#%%


class Experiment():
    def __init__(self,path,name):
        self.path=path
        self.name=name
    def Read_cube(self,string):
        self.cube = iris.load(ukl.Obtain_name(self.path,string))[0][:,:30,:,:]
        
        
#ordered_list=['BASE_RUN','3_ORD_LESS','NO_ICE']
#ordered_list=['BASE_RUN','3_ORD_LESS']
#ordered_list=['CONTACT_RUN','NO_HALLET']
#ordered_list=['ALL_ICE_PROC','NO_HALLET']
#ordered_list=['BASE_RUN','CONTACT_RUN','NO_HALLET','3_ORD_LESS']
#ordered_list=['CONTACT_RUN','NO_HALLET','3_ORD_LESS']
#ordered_list=['CONTACT_RUN','ALL_ICE_PROC','NO_HALLET','3_ORD_LESS','NO_ICE']
#ordered_list=['ALL_ICE_PROC','3_ORD_LESS']#,'NO_ICE']
#ordered_list=['CONTACT_RUN','ALL_ICE_PROC','NO_ICE']
#ordered_list=['ALL_ICE_PROC','NO_HALLET','3_ORD_LESS']#,'NO_ICE']
#ordered_list=['ALL_ICE_PROC','3_ORD_LESS','2_ORD_MORE']
from collections import OrderedDict

run_dict=OrderedDict()

#'/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/BASE_RUN2'
#run_dict['BASE_RUN']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/BASE_RUN2/All_time_steps/','BASE_RUN')
#run_dict['DEMOTT']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/DEMOTT3ORD/All_time_steps/','DEMOTT')
#run_dict['BASE_RUN']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_INITIAL_ICE/BASE_RUN/All_time_steps/','BASE_RUN')
#run_dict['CONTACT_RUN']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/BASE_CONTACT_242/All_time_steps/','CONTACT_RUN')
#run_dict['ALL_ICE_PROC']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/','ALL_ICE_PROC')
##run_dict['NO_HALLET']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/All_time_steps/','NO_HALLET')
#run_dict['3_ORD_LESS']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/All_time_steps/','3_ORD_LESS')
#run_dict['2_ORD_MORE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_MORE/All_time_steps/','2_ORD_MORE')
##run_dict['NO_ICE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NOICE/All_time_steps/','NO_ICE')


run_dict['BASE_DM']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/GP_HAMISH_DMDUST/All_time_steps/','BASE_DM')
#run_dict['NO_HALLET']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/All_time_steps/','NO_HALLET')
#run_dict['OLD_BASE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/BASE_RUN2/All_time_steps/','OLD_BASE')
#run_dict['3_ORD_LESS']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/All_time_steps/','3_ORD_LESS')
#run_dict['SM_100_COOPER']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SM_NOBIGG_T40/All_time_steps/','SM_100_COOPER')
#run_dict['SM_100_COOPER']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SM_100_COOPER/All_time_steps/','SM_100_COOPER')
run_dict['SM_LCOND_FALSE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SM_T40/All_time_steps/','SM_LCOND_FALSE')
#run_dict['2_ORD_MORE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_MORE/All_time_steps/','2_ORD_MORE')
run_dict['NO_ICE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NOICE/All_time_steps/','NO_ICE')
#run_dict['2_ORD_MORE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_MORE/All_time_steps/','2_ORD_MORE')

ordered_list=run_dict.keys()
for run in ordered_list:
    run_dict[run].Read_cube(stash_code)
    print run_dict[run].name,run_dict[run].cube.long_name, run_dict[run].cube.shape
#%%
path='/nfs/a201/eejvt/CASIM/SO_KALLI/NO_INITIAL_ICE/3_ORD_LESS/All_time_steps/'
potential_temperature=iris.load(ukl.Obtain_name(path,'m01s00i004'))[0][:,:30,:,:]
air_pressure=iris.load(ukl.Obtain_name(path,'m01s00i408'))[0][:,:30,:,:]
p0 = iris.coords.AuxCoord(1000.0,
                          long_name='reference_pressure',
                          units='hPa')
p0.convert_units(air_pressure.units)

Rd=287.05 # J/kg/K
cp=1005.46 # J/kg/K
Rd_cp=Rd/cp

temperature=potential_temperature*(air_pressure/p0)**(Rd_cp)
#temperature=iris.load('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_CLOUD_SQUEME/GP_HAMISH_DMDUST/L1/L1_temperature_Temperature.nc')[0]
#print temperature.data[0,0,0,0]
#temperature._var_name='temperature'
#R_specific=iris.coords.AuxCoord(287.058,
#                          long_name='R_specific',
#                          units='J-kilogram^-1-kelvin^-1')#J/(kg·K)
#
#air_density=(air_pressure/(temperature*R_specific))

#%%
# This example uses a MovieWriter directly to grab individual frames and
# write them to a file. This avoids any event loop integration, but has
# the advantage of working with even the Agg backend. This is not recommended
# for use in an interactive setting.
# -*- noplot -*-
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import matplotlib.animation as manimation
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=2, metadata=metadata)
log_scale=True
log_scale=False
fig = plt.figure(figsize=(len(ordered_list)*5, 12))

lat_sum=[]
lon_sum=[]
vertical_sum=[]
n_runs=len(ordered_list)
sample_cube=run_dict[ordered_list[0]].cube
for run in ordered_list:
    cube=run_dict[run].cube
    try:
        cube.remove_coord('surface_altitude')
    except:
        a=8754784
    lat_sum.append(cube.collapsed(['grid_latitude'],iris.analysis.SUM))
    lon_sum.append(cube.collapsed(['grid_longitude'],iris.analysis.SUM))
    vertical_sum.append(cube.collapsed(['model_level_number'],iris.analysis.SUM))

#defining levels

lat_temperatures=temperature.collapsed(['grid_latitude'],iris.analysis.MEAN)
lon_temperatures=temperature.collapsed(['grid_longitude'],iris.analysis.MEAN)
vertical_temperatures=temperature.collapsed(['model_level_number'],iris.analysis.MEAN)

temperature_levels=np.arange(-40,10,5).tolist()

max_val=lat_sum[0][1:,:,:].data.max()
min_val=lat_sum[0][1:,:,:].data.min()

min_val = np.min(lat_sum[0][:,:,:].data[np.nonzero(lat_sum[0][:,:,:].data)])
print min_val
lat_levels=np.linspace(min_val,max_val,9).tolist()
if log_scale:
    lat_levels=np.logspace(np.log10(min_val),np.log10(max_val),9).tolist()

max_val=lon_sum[0][1:,:,:].data.max()
min_val=lon_sum[0][1:,:,:].data.min()
min_val = np.min(lon_sum[0][:,:,:].data[np.nonzero(lon_sum[0][:,:,:].data)])
lon_levels=np.linspace(min_val,max_val,9).tolist()
if log_scale:
    lon_levels=np.logspace(np.log10(min_val),np.log10(max_val),9).tolist()
print min_val

max_val=vertical_sum[0][1:,:,:].data.max()
min_val=vertical_sum[0][1:,:,:].data.min()
min_val = np.min(vertical_sum[0][:,:,:].data[np.nonzero(vertical_sum[0][:,:,:].data)])
vertical_levels=np.linspace(min_val,max_val,9).tolist()
if log_scale:
    vertical_levels=np.logspace(np.log10(min_val),np.log10(max_val),9).tolist()
#vertical_levels=np.logspace(np.log10(min_val),np.log10(max_val),9).tolist()
#%%
#vertical_levels=np.linspace(0,0.8,15).tolist()
#vertical_levels[0]=0.0001
#vertical_levels[-1]=1
print min_val

lats=sample_cube.coord('grid_latitude').points-52
lons=sample_cube.coord('grid_longitude').points-180
try:
    vertical=sample_cube.coord('level_height').points*1e-3
    vertical_temps=temperature.coord('level_height').points*1e-3
except:
    vertical=sample_cube.coord('altitude').points*1e-3
    vertical_temps=temperature.coord('level_height').points*1e-3

lin_log='linear'
if log_scale:
    lin_log='logscale'

sub_plots=[]
sample_cube=lat_sum[0]
time_coord=sample_cube.coord('time')
time_coord.convert_units('seconds since 1970-01-01 00:00:0.0')
names='runs'
for name in ordered_list:
    names=names+'-'+name

with writer.saving(fig,'/nfs/a201/eejvt/SO_VIDEOS/Triple_plot_01116_'+lin_log+'_'+sample_cube.var_name+'_'+names+".mp4", 200):

    for it in range(len(sample_cube.coord('time').points)-1):
        big_title=sample_cube.var_name.replace('_',' ')+' '+datetime.datetime.fromtimestamp(time_coord.points[it]).strftime('%D %H:%M:%S')
        print big_title        
        try:
            txt.remove()
        except:
            adsf=2
        txt=plt.figtext(0.2,0.95,big_title,fontsize=15)
        if it!=0:
            for x in sub_plots:
                x.cla()
        for ir in range(n_runs):            
            try:
                vertical= lat_sum[ir].coord('level_height').points*1e-3
            except:
                vertical= lat_sum[ir].coord('altitude').points*1e-3
            latx=plt.subplot(3,n_runs,ir+1)
            sub_plots.append(latx)
            latx.set_title(ordered_list[ir],fontsize=15)
            mapablelat=latx.contourf(lons,vertical, lat_sum[ir][it,:,:].data,lat_levels, cmap='viridis',norm= colors.BoundaryNorm(lat_levels, 256))
            if ir+1==n_runs:
                if it==0:
                    cb=plt.colorbar(mapablelat,label=sample_cube.units.origin,format='%.2e',ticks=lat_levels)
            mapable2=latx.contour(lons,vertical_temps, lat_temperatures[it,:,:].data-273.15,temperature_levels, cmap='RdBu_r')
            plt.clabel(mapable2, inline=1, fontsize=10,fmt='%1.1f')
            plt.xlabel('Longitude')
            if ir==0:            
                plt.ylabel('Vertical (km)')
            plt.ylim(0,8)

            lonx=plt.subplot(3,n_runs,ir+1+n_runs)
            sub_plots.append(lonx)
            mapablelon=lonx.contourf(lats,vertical, lon_sum[ir][it,:,:].data,lon_levels, cmap='viridis',norm= colors.BoundaryNorm(lon_levels, 256))
            if ir+1==n_runs:
                if it==0:
                    cb=plt.colorbar(mapablelon,label=sample_cube.units.origin,format='%.2e',ticks=lon_levels)

            mapable2=lonx.contour(lats,vertical_temps, lon_temperatures[it,:,:].data-273.15,temperature_levels, cmap='RdBu_r')
            plt.clabel(mapable2, inline=1, fontsize=10,fmt='%1.1f')

            plt.xlabel('Latitude')
            if ir==0:            
                plt.ylabel('Vertical (km)')
            plt.ylim(0,8)

            verx=plt.subplot(3,n_runs,ir+1+2*n_runs)
            sub_plots.append(verx)
            mapablever=verx.contourf(lons,lats, vertical_sum[ir][it,:,:].data,vertical_levels, cmap='viridis',norm= colors.BoundaryNorm(vertical_levels, 256))
            if ir+1==n_runs:
                if it==0:
                    cb=plt.colorbar(mapablever,label=sample_cube.units.origin,format='%.2e',ticks=vertical_levels)

            if ir==0:            
                plt.ylabel('Latitude')
            plt.xlabel('Longitude')
            
            
            
        writer.grab_frame()








#%%
