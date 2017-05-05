# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:41:52 2015

@author: eejvt
"""


import numpy as np
import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import Jesuslib as jl

import os
from scipy.io.idl import readsav
from glob import glob
from scipy.io import netcdf
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats.stats import pearsonr
from glob import glob
from scipy.io.idl import readsav
from mpl_toolkits.basemap import Basemap
import datetime
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
from scipy.io import netcdf
import scipy as sc
archive_directory='/nfs/a201/eejvt/'
project='MARINE_PARAMETERIZATION/FOURTH_TRY'
project='MARINE_PARAMETERIZATION/DAILY'
os.chdir(archive_directory+project)
amsterdam_island_latlon_index=[45,27]
mace_head_latlon_index=[13,124]#tambien [13,124]
point_reyes_latlon_index=[18,84]

#%%




#%%
def marine_org_parameterization(T):
    a=11.2186
    b=-0.4459
    INP=np.exp(a+b*T)#[per gram]
    return INP


names=[#'tot_mc_oc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'tot_mc_bc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'tot_mc_su_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
'tot_mc_ss_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'tot_mc_feldspar_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'GLOMAP_mode_mol_no_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'mean_dry_r_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'sigmachained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
'hindcast3_temp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
'GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
]
names=[
'tot_mc_ss_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
'hindcast3_temp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
'GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
]
s={}
a=glob('*.sav')
for name in names:
    print name
    s=readsav(name,idict=s)
#%%
total_marine_mass=s.tot_mc_ss_mm_mode[2,]#+s.tot_mc_ss_mm_mode[3,]#ug/m3
total_marine_mass_year_mean=total_marine_mass.mean(axis=-1)
max_MH=[]
max_AI=[]
min_MH=[]
min_AI=[]
for i in range(len(jl.days_end_month)-1):
    max_MH.append(total_marine_mass[30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],jl.days_end_month[i]:jl.days_end_month[i+1]].max())
    min_MH.append(total_marine_mass[30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],jl.days_end_month[i]:jl.days_end_month[i+1]].min())
    max_AI.append(total_marine_mass[30,jl.amsterdam_island_latlon_index[0],jl.amsterdam_island_latlon_index[1],jl.days_end_month[i]:jl.days_end_month[i+1]].max())
    min_AI.append(total_marine_mass[30,jl.amsterdam_island_latlon_index[0],jl.amsterdam_island_latlon_index[1],jl.days_end_month[i]:jl.days_end_month[i+1]].min())
#%%
total_marine_mass_monthly_mean=jl.from_daily_to_monthly(total_marine_mass)
winter_months=[9,10,11,0,1,2]
summer_months=[3,4,5,6,7,8]
jl.grid_earth_map(total_marine_mass_monthly_mean[30,],cblabel='$\mu g/m^3$',levels=np.logspace(-1.5,np.log10(0.6),15).tolist(),cmap=plt.cm.Greens)
jl.plot(total_marine_mass_monthly_mean[30,:,:,summer_months].mean(axis=0),cblabel='$\mu g/m^3$',clevs=np.logspace(-1.5,np.log10(0.6),10).tolist(),cmap=plt.cm.Greens)
jl.plot(total_marine_mass_monthly_mean[30,:,:,winter_months].mean(axis=0),cblabel='$\mu g/m^3$',clevs=np.logspace(-1.5,np.log10(0.6),10).tolist(),cmap=plt.cm.Greens)
jl.plot(total_marine_mass_year_mean[30,:,:],title='Year mean concentration',cblabel='$\mu g/m^3$',clevs=np.logspace(-1.5,np.log10(0.6),10).tolist(),cmap=plt.cm.Greens)




total_marine_mass_grams_OC=total_marine_mass*1e-6/1.9#g/m3
temperatures=s.t3d_mm
temperatures=temperatures-273.15
temperatures[temperatures<-37]=1000#menores que -37
temperatures[temperatures<-27]=-27#menor que -25 =-25
temperatures[temperatures>-6]=1000#mayor que -15 = 0

'''
INP_marine_ambient=total_marine_mass_grams_OC*marine_org_parameterization(temperatures)#m3
INP_marine_ambient_constant_press,new_pressures,idexes=jl.constant_pressure_level_array(INP_marine_ambient,s.pl_m*1e-2)
np.save('INP_marine_ambient_constant_press',INP_marine_ambient_constant_press)
'''

INP_marine_ambient_constant_press=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_ambient_constant_press.npy')
INP_marine_ambient_constant_press_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_ambient_constant_press.npy')
INP_feld_extamb_constant_press_m3=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_feld_extamb_constant_press.npy')*1e6
#%%
INP_marine_alltemps=np.zeros((38,31,64,128,12))
for i in range (38):
    INP_marine_alltemps[i,]=total_marine_mass_grams_OC*marine_org_parameterization(-i)
np.save('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy',INP_marine_alltemps)
INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')

#%%
X,Y=np.meshgrid(jl.lat,jl.pressure_constant_levels)
levels=[1,2,5,10,20,50,100,200,500]
plt.contourf(X,Y,INP_marine_ambient_constant_press[:,:,:,[5,6,7]].mean(axis=(2,3)),levels=levels,norm= colors.BoundaryNorm(levels, 256),cmap=plt.cm.Reds)
plt.gca().invert_yaxis()
plt.colorbar(ticks=levels)
plt.show()









#%%
def from_daily_to_monthly(array):
    shape=np.array(array.shape)
    day_index=(shape==365).argmax()
    shape[day_index]=12
    array_new=np.zeros(shape)
    print array_new.shape
    mdays=np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])
    for i in range(len(mdays)-1):
        array_new[:,:,:,i]=array[:,:,:,mdays[i]:mdays[i+1]].mean(axis=-1)
    return array_new
#%%
point_reyes_OM=np.array([ 1.93709412,  1.42364333,  0.85588794,  0.73728426,  0.73297048,
        0.59337717,  0.62721217,  0.80146062,  0.934555  ,  1.70635694,
        2.25701474,  2.13727307])
point_reyes_wiom=np.array([np.nan,0.177429,np.nan,0.053704,0.098196,0.281652,0.01127,0.56441,0.220598,np.nan,np.nan,np.nan])
ams_wioc=np.array([189,122,96,68,64,41,49,47,64,54,64,130])/1e3
mace_wiom=np.array([0.08,0.088,0.1,0.24,0.51,0.2,0.183,0.155,0.122,0.24,0.062,0.07])#using data from rinaldi thanks to susannah. December interpolated
plt.figure(1)
plt.plot(total_marine_mass[30,13,124,:],label='GLOMAP')
mdays=np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])
mid_month_day=(mdays[1:]-mdays[:-1])/2+mdays[:-1]
plt.plot(mid_month_day,mace_wiom,'ro',label='Observations')
plt.xlabel('Day')
plt.ylabel('WIOM')
plt.title('Mace head')
plt.legend()
plt.show()

plt.figure(3)
plt.plot(total_marine_mass[30,18,84,:],label='GLOMAP')
mdays=np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])
mid_month_day=(mdays[1:]-mdays[:-1])/2+mdays[:-1]
plt.plot(mid_month_day,point_reyes_wiom,'ro',label='Observations')
plt.xlabel('Day')
plt.ylabel('WIOM')
plt.title('Point Reyes')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(total_marine_mass[30,45,27,:]/1.9,label='GLOMAP')
mdays=np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])
mid_month_day=(mdays[1:]-mdays[:-1])/2+mdays[:-1]
plt.plot(mid_month_day,ams_wioc,'ro',label='Observations')
plt.xlabel('Day')
plt.ylabel('WIOC')
plt.title('Amsterdam Island')
plt.legend()
plt.show()

#%%
total_marine_mass_monthly=from_daily_to_monthly(total_marine_mass)
plt.plot(total_marine_mass_monthly[30,13,124,:])
plt.plot(mace_wiom)







#%%
'''

Something went wrong with the distributions of INP marine alltemps

this is the code to do them
'''


def marine_org_parameterization(T):
    a=11.2186
    b=-0.4459
    INP=np.exp(a+b*T)#[per gram]
    return INP



archive_directory='/nfs/a201/eejvt/'
project='MARINE_PARAMETERIZATION/FOURTH_TRY'
project='MARINE_PARAMETERIZATION/DAILY'
os.chdir(archive_directory+project)

names=[#'tot_mc_oc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'tot_mc_bc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'tot_mc_su_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
'tot_mc_ss_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'tot_mc_feldspar_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'GLOMAP_mode_mol_no_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'mean_dry_r_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'sigmachained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
'hindcast3_temp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
'GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
]
names=[
'tot_mc_ss_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
'hindcast3_temp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
'GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
]
s={}
a=glob('*.sav')
for name in names:
    print name
    s=readsav(name,idict=s)
total_marine_mass=s.tot_mc_ss_mm_mode[2,]#+s.tot_mc_ss_mm_mode[3,]#ug/m3
total_marine_mass_year_mean=total_marine_mass.mean(axis=-1)
total_marine_mass_monthly_mean=jl.from_daily_to_monthly(total_marine_mass)
total_marine_mass_grams_OC=total_marine_mass*1e-6/1.9
total_marine_mass_grams_OC=total_marine_mass_monthly_mean*1e-6/1.9

INP_marine_alltemps=np.zeros((38,31,64,128,12))
for i in range (38):
    INP_marine_alltemps[i,]=total_marine_mass_grams_OC*marine_org_parameterization(-i)
np.save('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy',INP_marine_alltemps)
