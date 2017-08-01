# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:40:32 2016

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



lat_point=53.5
lon_point=1.32
title='Leeds'
lat_point=jl.cape_verde_latlon_values[0]
lon_point=jl.cape_verde_latlon_values[1]
title='Cape Verde\'s marine boundary layer'


lat_point=13.05
lon_point=-59.36
title='Barbados'
INP_marine_alltemps_monthly=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy', mmap_mode='r')#m3
INP_feldspar_alltemps_monthly=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy', mmap_mode='r')*1e6#m3
INP_feldspar_alltemps_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_feldext_alltemps_daily.npy', mmap_mode='r')*1e6#m3
INP_marine_alltemps_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_alltemps_daily.npy', mmap_mode='r')#m3
#INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')#m3
color_dict={}
color_dict['b919_4.csv']='red'
color_dict['b920_2.csv']='yellow'
color_dict['b920_4.csv']='saddlebrown'
color_dict['b920_6.csv']='indianred'
color_dict['b921_2.csv']='pink'
color_dict['b921_4.csv']='k'
color_dict['b921_6.csv']='darkviolet'
color_dict['b922_2.csv']='orange'
color_dict['b922_4.csv']='lime'
color_dict['b924_4.csv']='brown'
color_dict['b924_6.csv']='fuchsia'
color_dict['b924_8.csv']='gray'
color_dict['b925_2.csv']='lightgreen'
color_dict['b926_2.csv']='cyan'
color_dict['b927_2.csv']='crimson'
color_dict['b928_2.csv']='goldenrod'
color_dict['b928_4.csv']='indigo'
color_dict['b928_6.csv']='orangered'
color_dict['b928_9.csv']='olive'
color_dict['b929_1.csv']='green'
color_dict['b929_2.csv']='blue'
color_dict['b931_2.csv']='dodgerblue'
color_dict['b932_2.csv']='mediumslateblue'
color_dict['b932_4.csv']='deeppink'
color_dict['b932_6.csv']='y'


#%%

#INP_marine_alltemps_montly=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')#m3
#INP_feldspar_alltemps_montly=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e6#m3
INP_feldspar_alltemps=INP_feldspar_alltemps_daily
INP_marine_alltemps=INP_marine_alltemps_daily

INP_feldspar_alltemps=INP_feldspar_alltemps_monthly
INP_marine_alltemps=INP_marine_alltemps_monthly
#%%
sd=243#sSeptember
ed=304#eNovember
sd=0#September
ed=-1#November
sd=212#sAugust
sd=181#sJuly
ed=243#sSeptember
ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
ilon=jl.find_nearest_vector_index(jl.lon180,lon_point)
column_feldspar=INP_feldspar_alltemps[:,:,ilat,ilon,:]
column_marine=INP_marine_alltemps[:,:,ilat,ilon,:]
temps=np.arange(-37,1,1)
temps=temps[::-1]
top_lev=30

plt.figure()

plt.fill_between(temps[25:],column_feldspar[25:,top_lev,sd:ed].min(axis=-1),column_feldspar[25:,30,sd:ed].max(axis=-1),color='r',alpha=0.3)
plt.fill_between(temps[5:26],column_feldspar[5:26,top_lev,sd:ed].min(axis=-1),column_feldspar[5:26,30,sd:ed].max(axis=-1),color='r',label='K-feldspar')
plt.fill_between(temps[7:],column_marine[7:,top_lev,sd:ed].min(axis=-1),column_marine[7:,30,sd:ed].max(axis=-1),color='g',label='Marine Organics')
plt.fill_between(temps[:6],column_feldspar[:6,top_lev,sd:ed].min(axis=-1),column_feldspar[:6,30,sd:ed].max(axis=-1),color='r',alpha=0.3)
plt.fill_between(temps[:8],column_marine[:8,top_lev,sd:ed].min(axis=-1),column_marine[:8,30,sd:ed].max(axis=-1),color='g',alpha=0.3)

feld_max=[]
feld_min=[]


#plt.fill_between(temps[25:],column_feldspar[25:,22,6],column_feldspar[25:,30,6],color='r',alpha=0.3)
#plt.fill_between(temps[5:26],column_feldspar[5:26,22,6],column_feldspar[5:26,30,6],color='r',label='K-feldspar')
#plt.fill_between(temps[7:],column_marine[7:,22,6],column_marine[7:,30,6],color='g',label='Marine Organics')
#plt.fill_between(temps[:6],column_feldspar[:6,22,6],column_feldspar[:6,30,6],color='r',alpha=0.3)
#plt.fill_between(temps[:8],column_marine[:8,22,6],column_marine[:8,30,6],color='g',alpha=0.3)
#for i in range(len(column_marine[0,:])):
#    if i <22:
#        continue
#    
#    plt.plot(temps,column_marine[:,i],'g--',label='Marine organics')
#    plt.plot(temps,column_feldspar[:,i],'r--',label='K-feldspar')
#data=np.genfromtxt(jl.home_dir+'Cape_verde_obs.txt')
#data=np.genfromtxt(jl.home_dir+'Cape_verde_obs.txt')
temps=[]
INP=[]
err_low=[]
err_up=[]

runs=[run[38:-4] for run in glob(jl.home_dir+'Cape_Verde/*') if 'b9' in run]

#runs=['b920_2','b928_2','b931_2','b932_2']
skip=['b919_4','b928_9','b932_4','b922_4']
for run in runs:
    if run in skip:
#        print run
        continue
    data=np.genfromtxt(jl.home_dir+'Cape_Verde/'+run+'.csv',delimiter=',',skip_header=7)
    temps=data[:,0].tolist()
    INP=data[:,3].tolist()
    err_low=data[:,9].tolist()
    err_up=data[:,10].tolist()
    print run,'///////////'
    if any(np.array(INP[:-1])-np.array(err_up[:-1])<0):
        err_up=np.array(err_up)
        err_up[np.array(INP[:])-np.array(err_up[:])<0]=0
        err_up=err_up.tolist()
        print run
        print err_low[0],INP[0],err_up[0]
        print INP[0]-err_low[0],err_up[0]-INP[0]
    print run,'///////////'
#    plt.errorbar(temps,INP,yerr=[err_up,err_low],ecolor='dimgrey', linestyle="None")
#    plt.plot(temps,INP,'o',color=color_dict[run+'.csv'])#,label=run)
INP=np.array(INP)
err_up=np.array(err_up)
err_low=np.array(err_low)
temps=np.array(temps)
#print temps
#print INP
#print len(INP)
#data= np.genfromtxt("/nfs/see-fs-01_users/eejvt/Documents/dereje_data.dat",delimiter="\t")  
if not title:
    title='latitude: %1.2f longitude: %1.2f'%(lat_point,lon_point)
#err=data[:,10]-data[:,9]
#data=data[err>0]
plt.title(title)
plt.xlim(-27,0)
plt.yscale('log')
plt.grid()
plt.ylabel('$[INP]/m^{3}$')
plt.xlabel('Temperature /$^oC$')
plt.legend()
plt.show()












#%%


plt.figure()


plt.title('Barbados $[INP]_{-20}$ July-August daily variation')
plt.plot(np.arange(len(column_feldspar[20,top_lev,sd:ed]))+1,column_feldspar[20,top_lev,sd:ed]+column_marine[20,top_lev,sd:ed])
plt.xlabel('days')
plt.ylabel('$[INP]/m^{3}$')

plt.yscale('log')





