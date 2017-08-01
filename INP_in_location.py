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







sd=0#September
ed=-1#November
sd=212#sAugust
ed=243#sSeptember
ed=304#eNovember
sd=0#sjanuary
ed=364#edicember

sd=0#sEnero
ed=90#eMarzo

ed=273#eSep
sd=181#sJuly
sd=243#sSeptember
ed=304#e2ndNovember







lon_point=8
lat_point=-52

title='Southern Ocean grid'




lat_point=50.473392
lon_point=-1.531623
title='English_Channel'
#50.473392 -1.531623
lon_point=jl.mace_head_latlon_values[1]
lat_point=jl.mace_head_latlon_values[0]


title='Mace Head'

lon_point=1.32
lat_point=53.5
title='Leeds'

lat_point=13.05
lon_point=-59.36

sd=196#15July
ed=258#15Sep
title='Barbados'
#


lat_point=82.3
lon_point=-62.21

sd=31+28
ed=31+28+31
title='Alert'



lat_point=51.509865
lon_point=-0.1180
sd=181#15July
ed=213#15Sep
title='London'



#%%

#INP_niemand_daily=np.load('/nfs/a201/eejvt/INP_niemand_ext_alltemps.npy')*1e6#m3
INP_niemand_daily=np.load('/nfs/a201/eejvt/INP_niemand_ext_alltemps.npy', mmap_mode='r')#cm-3
#%%
#INP_marine_alltemps_monthly=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')#m3
#INP_feldspar_alltemps_monthly=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e6#m3
#%%
#INP_feldspar_alltemps_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_feldext_alltemps_daily.npy')*1e6#m3
#INP_marine_alltemps_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_alltemps_daily.npy')#m3
INP_feldspar_alltemps_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_feldext_alltemps_daily.npy', mmap_mode='r')#cm3
INP_marine_alltemps_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_alltemps_daily.npy', mmap_mode='r')#m3
#INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')#m3

#%%
INP_feldspar_alltemps=INP_feldspar_alltemps_daily
INP_marine_alltemps=INP_marine_alltemps_daily
# sd=0#September
# ed=-1#November
# sd=212#sAugust
# ed=243#sSeptember
# ed=304#eNovember
# sd=0#sjanuary
# ed=364#edicember
#
# sd=0#sEnero
# ed=90#eMarzo
#
# ed=273#eSep
# sd=181#sJuly
# sd=243#sSeptember
# ed=304#e2ndNovember
#
#
#
#
# sd=196#15July
# ed=258#15Sep

ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
ilon=jl.find_nearest_vector_index(jl.lon180,lon_point)
column_feldspar=INP_feldspar_alltemps[:,:,ilat,ilon,:]*1e6#m3
column_marine=INP_marine_alltemps[:,:,ilat,ilon,:]

column_dust=INP_niemand_daily[:,:,ilat,ilon,:]*1e6#m3
temps=np.arange(-37,1,1)
temps=temps[::-1]
top_lev=30


plt.fill_between(jl.pressure,column_feldspar[20,:,sd:ed].min(axis=-1),
                 column_feldspar[20,:,sd:ed].max(axis=-1),color='r',alpha=0.3)
#%%

days=column_feldspar.shape[-1]
data=np.zeros((len(temps),days+1))
species=['feldspar','marine','total','Niemand']
for name in species:

    data[:,0]=np.arange(0,column_feldspar.shape[0],1)
    for i in range(days):
        data[:,i+1]=column_feldspar[:,30,i]
        data[:,i+1]=column_marine[:,30,i]
        data[:,i+1]=column_dust[:,30,i]
        data[:,i+1]=column_feldspar[:,30,i]+column_marine[:,30,i]

    np.savetxt('/nfs/see-fs-01_users/eejvt/For_danny/INP_spectra_danny_m3_'+name+'-'+title+'.csv',
               data,delimiter=',',header='Temps, days....')




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

#plt.plot(temps,column_total[:,top_lev,sd:ed].mean(axis=-1),color='k',lw=3,label='Total mean')





plt.title(title+' from '+jl.date(sd)+' to '+jl.date(ed)+' lat=%1.2f'%lat_point+' lon=%1.2f'%lon_point)
plt.xlim(-37,0)
plt.yscale('log')
plt.grid()
plt.ylabel('$[INP]/m^{3}$')
plt.xlabel('Temperature /$^oC$')
plt.legend(loc='best')
plt.show()

plt.savefig('/nfs/see-fs-01_users/eejvt/For_danny/'+title+'.png')
#%%

temps[:],column_marine[:,top_lev,sd:ed].min(axis=-1)
data=np.zeros((len(temps),5))

data[:,0]=temps[:]
data[:,1]=column_feldspar[:,top_lev,sd:ed].min(axis=-1)
data[:,2]=column_feldspar[:,top_lev,sd:ed].max(axis=-1)
data[:,3]=column_marine[:,top_lev,sd:ed].min(axis=-1)
data[:,4]=column_marine[:,top_lev,sd:ed].max(axis=-1)
np.savetxt('/nfs/see-fs-01_users/eejvt/For_danny/INP_spectra_Alberto_'+title+'.csv',
           data,delimiter=',',header='Temps, Feldspar_min,Feldspar_max, Marine_min,Marine_max')




INP_max=column_total[:,30,sd:ed].max(axis=-1)
INP_mean=column_total[:,top_lev,sd:ed].mean(axis=-1)
INP_min=column_total[:,top_lev,sd:ed].min(axis=-1)
#np.savetxt('/nfs/see-fs-01_users/eejvt/For_danny/INP_variability_'+title+'.csv',data,delimiter=',',header='Days since 1st of January,T=-15,T=-20,T=-25')
#%%
#
#
#plt.figure()
#plt.plot(temps,INP_max,'o')
#plt.plot(temps,INP_mean,'o')
#plt.plot(temps,INP_min,'o')
#plt.yscale('log')
#
#def demott(T,N):
#    a_demott=5.94e-5
#    b_demott=3.33
#    c_demott=0.0264
#    d_demott=0.0033
#    Tp01=0.01-T
#    dN_imm=1e3*a_demott*(Tp01)**b_demott*(N)**(c_demott*Tp01+d_demott)
#    return dN_imm
#def meyers_param(T,units_cm=0):
#    a=-0.639
#    b=0.1296
#    return np.exp(a+b*(100*(jl.saturation_ratio_C(T)-1)))#L-1
#def meyers_CASIM(T):
#    return np.exp(0.4108-0.262*T)
#def func(x,a,b,c,d,e,f):
#    return a-b*x+c*x**2+d*x**3+e*x**4+f*x**5
#
#def func_log(x,a,b):
#    return np.exp(a)*np.exp(-b * x)
#
#from scipy import stats
#
#
#from scipy.optimize import curve_fit
#
#def fit_to_INP(INP,function,temps=temps):
#
#    T=temps[:]
#
#    popt,pcov = curve_fit(func, T, INP)
#    perr = np.sqrt(np.diag(pcov))
#    ci = 0.95
#    pp = (1. + ci) / 2.
#    nstd = stats.norm.ppf(pp)
#    popt_up = popt + nstd * perr
#    popt_dw = popt - nstd * perr
#
#    INP_fitted=function(temps,*popt)
#    INP_low=function(T,*popt_dw)
#    INP_high=function(T,*popt_up)
#    return popt,pcov,INP_fitted,INP_low,INP_high
#
#popt,pcov,INP_fitted,_,_=fit_to_INP(np.log(INP_max),func)
#print INP_fitted
#print popt
#
#
#
#plt.plot(temps,np.exp(INP_fitted),'k')
#plt.grid()
#plt.plot(temps,meyers_param(temps)*1e3,'b',label='Meyers')
#n05=56000*1e-6
#n05_GLOMAP=21.26#cm-3 surface SO
#n05_bug_meters=56000
#plt.plot(temps[1:],demott(temps,n05)[1:],'r',label='DeMott 56000*1e-6 (cm-3)')
#plt.plot(temps[1:],demott(temps,n05_GLOMAP)[1:],'g',label='DeMott GLOMAP (21.26 cm-3)')
#plt.plot(temps[1:],demott(temps,n05_bug_meters)[1:],'y',label='DeMott CASIM bug metres (56000cm-3)')
##plt.plot(temps[1:],1e2*demott(temps,n05)[1:],'g--',label='DeMott 2ord')
##plt.plot(temps[1:],1e1*demott(temps,n05)[1:],'g--',label='DeMott 1ord')
##plt.plot(temps[1:],1e-1*demott(temps,n05)[1:],'y--',label='DeMott -1ord')
##plt.plot(temps[1:],1e-2*demott(temps,n05)[1:],'y--',label='DeMott -2ord')
##plt.plot(temps[1:],1e-3*demott(temps,n05)[1:],'y--',label='DeMott -3ord')
##plt.plot(temps,meyers_param(temps)*1e3,'r',label='mine')
##),
#plt.ylabel('m-3')
##plt.plot(temps,meyers_CASIM(temps),'g',label='casim')
#plt.legend(loc='lower left')
#plt.savefig(jl.home_dir+'different_params.png')
#%%

'''
plt.figure()

title='Farm'
plt.title(title+' [INP] 1 January to 31 March daily variation')

days=np.arange(len(column_feldspar[0,top_lev,sd:ed]))+1
#days=np.arange(len(column_dust[0,top_lev,sd:ed]))+1
data=np.zeros((len(days),4))
data[:,0]=days

T=15

values=column_feldspar[T,top_lev,sd:ed]
values=column_marine[T,top_lev,sd:ed]#+column_feldspar[T,top_lev,sd:ed]
#values=column_dust[T,top_lev,sd:ed]
plt.plot(days,values,label='T='+str(-T)+'$^{o}C$')
data[:,1]=values

T=20
values=column_feldspar[T,top_lev,sd:ed]#+column_marine[T,top_lev,sd:ed]
values=column_feldspar[T,top_lev,sd:ed]
values=column_marine[T,top_lev,sd:ed]#+column_feldspar[T,top_lev,sd:ed]
#values=column_dust[T,top_lev,sd:ed]
plt.plot(days,values,label='T='+str(-T)+'$^{o}C$')
data[:,2]=values

T=25
values=column_feldspar[T,top_lev,sd:ed]#+column_marine[T,top_lev,sd:ed]
values=column_feldspar[T,top_lev,sd:ed]
values=column_marine[T,top_lev,sd:ed]#+column_feldspar[T,top_lev,sd:ed]
#values=column_dust[T,top_lev,sd:ed]
plt.plot(days,values,label='T='+str(-T)+'$^{o}C$')
data[:,3]=values

plt.xlabel('Days')
plt.ylabel('$[INP]/m^{3}$')
plt.legend(loc='best')
plt.yscale('log')

np.savetxt('/nfs/see-fs-01_users/eejvt/For_danny/INP_variability_marine_'+title+'.csv',data,delimiter=',',header='Days ,T=-15,T=-20,T=-25')
#np.savetxt('/nfs/see-fs-01_users/eejvt/For_danny/INP_variability_Leeds_niemand.csv',data,delimiter=',',header='Days since 1st of September,T=-15,T=-20,T=-25')


#%%

fig=plt.figure(figsize=(20, 12))

m = fig.add_subplot(1,1,1)


projection='cyl'

m = Basemap(projection=projection,lon_0=0)
m.drawcoastlines()
lons=jl.lon[:]


lons[lons>=180.1]=lons[lons>=180.1]-360
X,Y =np.meshgrid(lons,jl.lat)
x,y=m(X,Y)

plt.plot(x,y,'ko',lw=5)

'''

#%%

#plt.figure()
#data=np.genfromtxt('INP_spectra_danny_m3_total.csv',delimiter=',')
#data=np.genfromtxt('INP_spectra_danny_feldspar.csv',delimiter=',')
#data=np.genfromtxt('INP_spectra_danny_marine.csv',delimiter=',')
#data=np.genfromtxt('INP_spectra_danny_m3_Niemand.csv',delimiter=',')
#data=data[:,1:]
#for iday in range(data.shape[1]):
#    plt.plot(data[:,iday])
#plt.yscale('log')
