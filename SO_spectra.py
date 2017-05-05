# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:16:10 2016

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


lat_point=13.05
lon_point=-59.36
title='Barbados'


lon_point=jl.mace_head_latlon_values[1]
lat_point=jl.mace_head_latlon_values[0]
title='Mace Head'


lon_point=1.32
lat_point=53.5
title='Leeds'

lon_point=8
lat_point=-52

title='Southern Ocean grid'
INP_feldspar_alltemps_monthly_paper=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e6#m3
INP_feldspar_alltemps_monthly_other=np.load('/nfs/a107/eejvt/JB_TRAINING/Feldspar_distributions/INP_feld_ext_alltemps.npy')*1e6#m3

#%%
jl.plot


                                           
#%%


INP_niemand_daily=np.load('/nfs/a201/eejvt/INP_niemand_ext_alltemps.npy')*1e6#m3
#%%
INP_marine_alltemps_monthly=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')#m3
INP_feldspar_alltemps_monthly=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e6#m3
INP_feldspar_alltemps_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_feldext_alltemps_daily.npy')*1e6#m3
INP_marine_alltemps_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_alltemps_daily.npy')#m3
#INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')#m3

#%%
INP_feldspar_alltemps=INP_feldspar_alltemps_daily
INP_marine_alltemps=INP_marine_alltemps_daily
sd=0#September
ed=-1#November
sd=212#sAugust
sd=181#sJuly
ed=243#sSeptember
ed=304#eNovember
sd=0#sjanuary
ed=364#edicember

sd=243#sSeptember
ed=304#e2ndNovember
sd=0#s
ed=360#e

ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
ilon=jl.find_nearest_vector_index(jl.lon180,lon_point)
column_feldspar=INP_feldspar_alltemps[:,:,ilat,ilon,:]
column_marine=INP_marine_alltemps[:,:,ilat,ilon,:]

column_dust=INP_niemand_daily[:,:,ilat,ilon,:]
temps=np.arange(-37,1,1)
temps=temps[::-1]
top_lev=20
column_total=column_feldspar+column_marine

#%%

plt.figure()

column_total=column_feldspar+column_marine

plt.fill_between(temps[25:],column_feldspar[25:,top_lev,sd:ed].min(axis=-1),
                 column_feldspar[25:,30,sd:ed].max(axis=-1),color='r',alpha=0.3)

plt.fill_between(temps[5:26],column_feldspar[5:26,top_lev,sd:ed].min(axis=-1),
                 column_feldspar[5:26,30,sd:ed].max(axis=-1),color='r',label='K-feldspar')

plt.fill_between(temps[7:],column_marine[7:,top_lev,sd:ed].min(axis=-1),
                 column_marine[7:,30,sd:ed].max(axis=-1),color='g',label='Marine Organics')

plt.fill_between(temps[:6],column_feldspar[:6,top_lev,sd:ed].min(axis=-1),
                 column_feldspar[:6,30,sd:ed].max(axis=-1),color='r',alpha=0.3)

plt.fill_between(temps[:8],column_marine[:8,top_lev,sd:ed].min(axis=-1),
                 column_marine[:8,30,sd:ed].max(axis=-1),color='g',alpha=0.3)


plt.fill_between(temps,column_total[:,top_lev,sd:ed].min(axis=-1),
column_total[:,30,sd:ed].max(axis=-1),color='k',alpha=0.4,label='Total range')
plt.plot(temps,column_total[:,top_lev,sd:ed].mean(axis=-1),color='k',lw=3,label='Total mean')


plt.title(title)
plt.xlim(-27,0)
plt.yscale('log')
plt.grid()
plt.ylabel('$[INP]/m^{3}$')
plt.xlabel('Temperature /$^oC$')
plt.legend()
plt.show()




#%%
sd=0#s
ed=360#e

#column_total=column_dust
INP_max=column_total[:,30,sd:ed].max(axis=-1)
INP_mean=column_total[:,top_lev,sd:ed].mean(axis=-1)
INP_min=column_total[:,top_lev,sd:ed].min(axis=-1)
INP_niemad_max=column_dust[:,30,sd:ed].max(axis=-1)
INP_niemad_min=column_dust[:,top_lev,sd:ed].min(axis=-1)
#%%


plt.figure()
plt.figure(figsize=(15,7))
ax = plt.subplot(121)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
#plt.plot(temps,INP_max,'o')
#plt.plot(temps,INP_mean,'o')
#plt.plot(temps,INP_min,'o')
plt.yscale('log')

def demott(T,N):
    a_demott=5.94e-5
    b_demott=3.33
    c_demott=0.0264
    d_demott=0.0033
    Tp01=0.01-T
    dN_imm=1e3*a_demott*(Tp01)**b_demott*(N)**(c_demott*Tp01+d_demott)
    return dN_imm
def meyers_param(T,units_cm=0):
    a=-0.639
    b=0.1296
    return np.exp(a+b*(100*(jl.saturation_ratio_C(T)-1)))#L-1
def meyers_CASIM(T):
    return np.exp(0.4108-0.262*T)
def func(x,a,b,c,d,e,f):
    return a-b*x+c*x**2+d*x**3+e*x**4+f*x**5
    
def func_log(x,a,b):
    return np.exp(a)*np.exp(-b * x)

from scipy import stats


from scipy.optimize import curve_fit

def fit_to_INP(INP,function,temps=temps):
    
    T=temps[:]

    popt,pcov = curve_fit(func, T, INP)
    perr = np.sqrt(np.diag(pcov))
    ci = 0.95
    pp = (1. + ci) / 2.
    nstd = stats.norm.ppf(pp)
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr

    INP_fitted=function(temps,*popt)
    INP_low=function(T,*popt_dw)
    INP_high=function(T,*popt_up)
    return popt,pcov,INP_fitted,INP_low,INP_high

popt,pcov,INP_fitted,_,_=fit_to_INP(np.log(INP_max),func)
print INP_fitted
print popt



#plt.plot(temps,np.exp(INP_fitted),'k')
plt.grid()
#plt.plot(temps,meyers_param(temps)*1e3,'b',label='Meyers')
n05=56000*1e-6
n05_dust=56000*1e-6
n05_GLOMAP=21.26#cm-3 surface SO 
n05_bug_meters=56000
#plt.plot(temps[1:],demott(temps,n05)[1:],'r',label='DeMott 56000*1e-6 (cm-3)')
#plt.plot(temps[1:],demott(temps,n05_GLOMAP)[1:],'g',label='DeMott GLOMAP (21.26 cm-3)')
#plt.plot(temps[1:],demott(temps,n05_bug_meters)[1:],'y',label='DeMott CASIM bug metres (56000cm-3)')
#plt.plot(temps[1:],1e2*demott(temps,n05)[1:],'g--',label='DeMott 2ord')
#plt.plot(temps[1:],1e1*demott(temps,n05)[1:],'g--',label='DeMott 1ord')
#plt.plot(temps[1:],1e-1*demott(temps,n05)[1:],'y--',label='DeMott -1ord')
#plt.plot(temps[1:],1e-2*demott(temps,n05)[1:],'y--',label='DeMott -2ord')
#plt.plot(temps[1:],1e-3*demott(temps,n05)[1:],'y--',label='DeMott -3ord')
lw=3

plt.plot(temps,meyers_param(temps),'b',lw=lw,label='Meyers')
plt.plot(temps,INP_max*1e-3,'r--',lw=lw,label='GLOMAP-max')
plt.plot(temps,INP_mean*1e-3,'g--',lw=lw,label='GLOMAP-mean')
plt.plot(temps,INP_min*1e-3,'b--',lw=lw,label='GLOMAP-min')
#plt.plot(temps,INP_niemad_max*1e-6,'--',c='brown',lw=lw,label='GLOMAP_niemand-max')
#plt.plot(temps,INP_mean*1e-6,'g--',lw=lw,label='GLOMAP-mean')
#plt.plot(temps,INP_niemad_min*1e-6,'--',c='brown',lw=lw,label='GLOMAP_niemand-min')
#plt.yscale('log')

plt.plot(temps[1:],demott_dust(temps,n05_dust)[1:]*1e-3,'brown',lw=lw,label='DeMott 2015 dust')
plt.plot(temps[1:],demott(temps,n05_GLOMAP)[1:]*1e-3,'m',lw=lw,label='DeMott GLOMAP (21.26 cm-3)')
plt.plot(temps[1:],demott(temps,n05_bug_meters)[1:]*1e-3,'y',lw=lw,label='DeMott CASIM artificialy high')
plt.plot(temps[1:10],demott(temps,n05_bug_meters)[1:10]*1e-3*1e2,'y-.',lw=lw,label='DeMott CASIM artificialy high 2ord_more')
plt.plot(temps[1:],demott(temps,n05_bug_meters)[1:]*1e-3*1e-2,'y-.',lw=lw,label='DeMott CASIM artificialy high 2ord_less')

#plt.plot(temps,meyers_param(temps)*1e3,'r',label='mine')
#),
plt.axvline(-5,c='k')
plt.axvline(-15,c='k')
plt.axvline(-17,c='k')
plt.axhline(10,c='k')
plt.axhline(1,c='k')
plt.axhline(0.1,c='k')
ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))

plt.ylabel('L-1')
#plt.plot(temps,meyers_CASIM(temps),'g',label='casim')
#plt.legend(loc='lower left')
plt.savefig(jl.home_dir+'different_params.png')
#%%

#plt.figure()
def demott_dust(T,N):
    a_demott = 0.0
    b_demott = 1.25
    c_demott = 0.46
    d_demott = -11.6
    cf = 3.0
    Tp01=0.01-T

    dN_imm = 1.0e3*cf*(N)**(a_demott*(Tp01)+b_demott)*np.exp(c_demott*(Tp01)+d_demott)
    return dN_imm


plt.figure()
#plt.figure(figsize=(15,7))
#ax = plt.subplot(121)
ax = plt.subplot(111)
box = ax.get_position()
#ax.set_position([box.x0+0.2, box.y0, box.width, box.height])
lw=3
temps=np.arange(-37,1,1)

#plt.plot(temps[1:],demott_dust(temps,n05_dust)[1:]*1e-6,'k',lw=lw,label='DeMott 2015 dust')
#plt.plot(temps[1:],demott(temps,n05_GLOMAP)[1:]*1e-6,'m',lw=lw,label='DeMott GLOMAP (21.26 cm-3)')
#plt.plot(temps[1:],demott(temps,n05_bug_meters)[1:]*1e-6,'y',lw=lw,label='DeMott CASIM artificialy high')
temps=temps[::-1]
#plt.plot(temps,INP_max*1e-6,'r--',lw=lw,label='GLOMAP-max')
#plt.plot(temps,INP_mean*1e-6,'g--',lw=lw,label='GLOMAP-mean')
#plt.plot(temps,INP_min*1e-6,'b--',lw=lw,label='GLOMAP-min')
#plt.plot(temps,INP_niemad_max*1e-6,'--',c='brown',lw=lw,label='GLOMAP_niemand-max')
##plt.plot(temps,INP_mean*1e-6,'g--',lw=lw,label='GLOMAP-mean')
#plt.plot(temps,INP_niemad_min*1e-6,'--',c='brown',lw=lw,label='GLOMAP_niemand-min')
#plt.yscale('log')

INP_obs_mason=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",header=1)
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/TERRESTIAL_INFLUENCED.dat",header=1)
marker='o'
marker_mason='^'
marker_size=30
marker_size_mason=50
INP_obs_total=np.concatenate((INP_obs,INP_obs_mason))
temps_obs=INP_obs[:,1]
concentrations=INP_obs[:,2]
lats=INP_obs[:,3]
plot=plt.scatter(temps_obs,concentrations,c=lats,cmap=plt.cm.RdBu,marker=marker,s=marker_size)
temps_obs=INP_obs_mason[:,1]
concentrations=INP_obs_mason[:,2]
lats=INP_obs_mason[:,3]
plot=plt.scatter(temps_obs,concentrations,c=lats,cmap=plt.cm.RdBu,marker='^',s=marker_size_mason)
plt.ylim(INP_obs_total[:,2].min()*.5,INP_obs_total[:,2].max()*1.5)
plt.colorbar(label='Latitude')
plt.xlabel('T $^o$C')
plt.ylabel('INP concentration cm$^{-3}$')
plt.yscale('log')



def meyers_param(T,units_cm=0):
    a=-0.639
    b=0.1296
    return np.exp(a+b*(100*(jl.saturation_ratio_C(T)-1)))#L-1
    
def fletcher_param(T,units_cm=0):

    return 0.01*np.exp(-0.6*T)#m-3
def cooper_param(T,units_cm=0):

    return 5.0*np.exp(-0.304*T)#m-3
    

temps=np.arange(-37,1,1)

plt.scatter([],[],c='k',marker='^',label='Marine points')
plt.scatter([],[],c='k',marker='o',label='Terrestrial points')
plt.plot(temps,meyers_param(temps)*1e-3,'r',lw=lw,label='Meyers1992')
plt.plot(temps,fletcher_param(temps)*1e-6,'b',lw=lw,label='Flecher1962')
plt.plot(temps,cooper_param(temps)*1e-6,'g',lw=lw,label='Cooper1986')
ax.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
plt.xlim(-40,0)
plt.legend(loc='lower left')
#plt.legend(loc='best')
plt.title('Dataset used in Vergara-Temprado 2017')
#plt.savefig(jl.home_dir+'INP_observations_with_SO_range.png')
