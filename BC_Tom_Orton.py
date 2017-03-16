# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:43:37 2016

@author: eejvt
"""

import os
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
from glob import glob
archive_directory='/nfs/a201/eejvt/'
project='BC_INP'
os.chdir(archive_directory+project)
from multiprocessing import Pool
from scipy.integrate import quad
#pool = Pool()
rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3

def BC_parametrization_tom1(T):
    #A=-20.27
    #B=1.2
    return 10**(-2.949-0.1829*T)
    
Ts=np.linspace(-25,-10,100)
plt.plot(Ts,BC_parametrization_tom1(Ts))
plt.yscale('log')
def BC_parametrization_tom(T):
    #A=-20.27
    #B=1.2
    return 10**(-2.87-0.182*T)
    
Ts=np.linspace(-25,-10,100)
plt.plot(Ts,BC_parametrization_tom(Ts))
plt.yscale('log')


def volumes_of_modes(s):
    modes_vol=(s.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]
    +s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2]
    +s.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
    +s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]
    +s.tot_mc_oc_mm_mode[:,:,:,:,:]/rhocomp[5]
    +s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])
    return modes_vol
#%%
def feld_parametrization(T):
    ns=np.exp((-1.03802178356815*T)+275.263379304105)
    return ns
    
Ts=np.linspace(-25,-10,100)
plt.plot(Ts,feld_parametrization(Ts+273.15))
plt.yscale('log')
    

path='/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2'


s=jl.read_data(path)
#%%






s.tot_mc_bc_mm_mode.shape

jl.plot(s.tot_mc_bc_mm_mode.mean(axis=-1)[1,30,:,:])



jl.plot(s.tot_mc_bc_mm_mode[1,30,:,:,9])
jl.grid_earth_map(s.tot_mc_bc_mm_mode[1,30,:,:,:])

s.st_nd
#%%
temperatures=np.load('/nfs/a107/eejvt/temperatures_daily.npy')
temperatures=temperatures+273.15
temperatures[temperatures<236]=100000
temperatures[temperatures<248]=248
temperatures[temperatures>268]=100000
temperatures=temperatures-273.15
temperatures_monthly=jl.from_daily_to_monthly(temperatures)

#%%
INP_BC_ext=np.zeros((38,31,64,128,12))
modes_vol=volumes_of_modes(s)
BC_volfrac=(s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
Nd=s.st_nd[:,:,:,:,:]*BC_volfrac
rmean=s.rbardry[:,:,:,:,:]*1e2
for itemp in range (0,38):
    print itemp
    ns=BC_parametrization_tom(-itemp)
    for imode in range(7):
        INP_BC_ext[itemp,:,:,:,:]=INP_BC_ext[itemp,:,:,:,:]+Nd[imode,:,:,:,:]*(1-np.exp(-ns*4*np.pi*rmean[imode,:,:,:,:]**2))
        
#%%
INP_BC_ext_ambient=np.zeros((31,64,128,12))
modes_vol=volumes_of_modes(s)
BC_volfrac=(s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
Nd=s.st_nd[:,:,:,:,:]*BC_volfrac
rmean=s.rbardry[:,:,:,:,:]*1e2
ns=BC_parametrization_tom(temperatures_monthly)
for imode in range(7):
    print imode
    INP_BC_ext_ambient[:,:,:,:]=INP_BC_ext_ambient[:,:,:,:]+Nd[imode,:,:,:,:]*(1-np.exp(-ns*4*np.pi*rmean[imode,:,:,:,:]**2))
#%%
INP_feld_ext=np.zeros((38,31,64,128,12))
modes_vol=volumes_of_modes(s)
feld_volfrac=(s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
Nd=s.st_nd[:,:,:,:,:]*feld_volfrac
rmean=s.rbardry[:,:,:,:,:]*1e2
for itemp in range (0,38):
    print itemp
    ns=feld_parametrization(-itemp+273.15)
    for imode in range(7):
        INP_feld_ext[itemp,:,:,:,:]=INP_feld_ext[itemp,:,:,:,:]+Nd[imode,:,:,:,:]*(1-np.exp(-ns*4*np.pi*rmean[imode,:,:,:,:]**2))
        
#%%
INP_feld_ext_ambient=np.zeros((31,64,128,12))
modes_vol=volumes_of_modes(s)
feld_volfrac=(s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
Nd=s.st_nd[:,:,:,:,:]*feld_volfrac
rmean=s.rbardry[:,:,:,:,:]*1e2
ns=feld_parametrization(temperatures_monthly+273.15)
for imode in range(7):
    print imode
    INP_feld_ext_ambient[:,:,:,:]=INP_feld_ext_ambient[:,:,:,:]+Nd[imode,:,:,:,:]*(1-np.exp(-ns*4*np.pi*rmean[imode,:,:,:,:]**2))


#%%
INP_BC_ext_m3=INP_BC_ext*1e6
INP_feld_ext_m3=INP_feld_ext*1e6
INP_feld_ext_ambient_m3=INP_feld_ext_ambient*1e6
INP_BC_ext_ambient_m3=INP_BC_ext_ambient*1e6



#%%
levels=

#jl.plot(INP_BC_ext_m3[30,15,:,:,:].mean(axis=-1)*1e-6,title='INP BC 600hpa T=-20C', cblabel='$m^{-3}$',cmap=plt.cm.OrRd)
jl.plot(INP_BC_ext_m3[20,20,:,:,:].mean(axis=-1),title='INP BC 600hpa T=-20C', cblabel='$m^{-3}$',cmap=plt.cm.OrRd)
jl.plot(INP_feld_ext_m3[20,20,:,:,:].mean(axis=-1),title='INP feld 600hpa T=-20C', cblabel='$m^{-3}$',cmap=plt.cm.OrRd)


jl.plot(INP_BC_ext_ambient_m3[15,:,:,:].mean(axis=-1))
jl.plot(INP_feld_ext_ambient_m3[15,:,:,:].mean(axis=-1))

jl.grid_earth_map(INP_BC_ext_ambient[15,:,:,:])
jl.grid_earth_map(INP_feld_ext_ambient[15,:,:,:])
#%%
fig=plt.figure()
cx=plt.subplot(1,1,1)
CF=cx.contourf(Xmo,Ymo,INP_BC_ext_ambient_m3[:,:,:,:].mean(axis=(-1,-2)),cmap=plt.cm.YlOrRd)
CB=plt.colorbar(CF,drawedges=1,label='$m^{-3}$')

cx.invert_yaxis()
cx.set_ylim(ymax=200)
plt.show()
#%%



#[0.1,0.5,1,10,50,100,500,1000,5000,10000]
levelsbc=np.logspace(-8,8,15).tolist()
levelsbc=[0.1,0.5,1,10,50,100,500,1000,5000,10000]
levelsfel=[100,1000]
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
fs=10
glolevs=jl.pressure
fig=plt.figure()
cx=plt.subplot(1,1,1)
Xmo,Ymo= np.meshgrid(lat.glat, glolevs)
Xfel,Yfel=np.meshgrid(lat.glat,glolevs)
cx.set_title('Anual and longitudinal mean')
CS=cx.contour(Xfel,Yfel,INP_feld_ext_ambient_m3[:,:,:,:].mean(axis=(-1,-2)),levelsfel,colors='k',hold='on')#,linewidths=np.linspace(2, 6, 4))#,

plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
plt.setp(CS.collections )
CF=cx.contourf(Xmo,Ymo,INP_BC_ext_ambient_m3[:,:,:,:].mean(axis=(-1,-2)),levelsbc,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsbc, 256))
CB=plt.colorbar(CF,ticks=levelsbc,drawedges=1,label='$m^{-3}$')

cx.invert_yaxis()
cx.set_ylim(ymax=200)
cx.tick_params(axis='both', which='major', labelsize=fs)
cx.set_ylabel('Pressure level $(hPa)$')
cx.set_xlabel('Latitude')
cx.xaxis.set_ticks(np.arange(-90,100,20))
#plt.savefig('jja_mean_lat_mean_NA.svg',dpi=300,format='svg')
#plt.savefig('jja_mean_lat_mean_NA.png',dpi=600,format='png')

plt.show()


#%%
def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


plt.figure()
INP_feld_ext_ambient_m3=INP_feld_ext_ambient*1e6
INP_BC_ext_ambient_m3=INP_BC_ext_ambient*1e6

INP_BC_ext_ambient_m3=INP_BC_ext_ambient_m3.mean(axis=(-1,-2))
INP_feld_ext_ambient_m3=INP_feld_ext_ambient_m3.mean(axis=(-1,-2))
INP_BC_ext_ambient_m3[INP_BC_ext_ambient_m3<1e-2]=0
INP_feld_ext_ambient_m3[INP_feld_ext_ambient_m3<1e-2]=0
INP_BC_ext_ambient_m3[np.isnan(INP_BC_ext_ambient_m3)]=0
ratiobcfel1=INP_BC_ext_ambient_m3/INP_feld_ext_ambient_m3
ratiobcfel=INP_feld_ext_ambient_m3/INP_BC_ext_ambient_m3
ratiobcfel[np.isnan(ratiobcfel)]=0
ratiobcfel1[np.isnan(ratiobcfel1)]=0
levelsratio=np.logspace(-5,5,11).tolist()
#levelsratio=np.linspace(-100,100,11).tolist()
cx=plt.subplot(1,2,1)
plt.title('BC/Feld')
CF=cx.contourf(Xfel,Yfel,ratiobcfel1[:,:],levelsratio,cmap=plt.cm.RdBu_r,norm= colors.BoundaryNorm(levelsratio, 256))
CB=plt.colorbar(CF,ticks=levelsratio,drawedges=1,label='$ratio$',format=ticker.FuncFormatter(fmt))
cx.tick_params(axis='both', which='major', labelsize=10)
cx.invert_yaxis()    
cx.set_ylim(ymax=200)
cx=plt.subplot(1,2,2)
plt.title('Feld/BC')
CF=cx.contourf(Xfel,Yfel,ratiobcfel[:,:],levelsratio,cmap=plt.cm.RdBu_r,norm= colors.BoundaryNorm(levelsratio, 256))
CB=plt.colorbar(CF,ticks=levelsratio,drawedges=1,label='$ratio$',format=ticker.FuncFormatter(fmt))
cx.tick_params(axis='both', which='major', labelsize=10)
cx.invert_yaxis()    
cx.set_ylim(ymax=200)
#array1=ratiobcfel[:,:,:,5]*ratiobcfel1[:,:,:,5]
#array1[np.isnan(array1)]=1
plt.savefig('ratio')
'''
cx=plt.subplot(1,3,3)
CF=cx.contourf(Xfel,Yfel,ratiobcfel[:,:,:,5].mean(axis=(-1))/ratiobcfel1[:,:,:,5].mean(axis=(-1)),[0.9,1.1],cmap=plt.cm.jet,norm= colors.BoundaryNorm([0.9,1.1], 256))
CB=plt.colorbar(CF,ticks=[0.9,1.1],drawedges=1,label='$m^{-3}$',format=ticker.FuncFormatter(fmt))
cx.tick_params(axis='both', which='major', labelsize=10)
cx.invert_yaxis()    
cx.set_ylim(ymax=200)
'''
plt.show()


#%%
INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')*1e-3#l
INP_feldspar_alltemps=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e3#l



column_feldspar=INP_feldspar_alltemps[:,:,jl.cape_verde_latlon_index[0],jl.cape_verde_latlon_index[1],7]
column_marine=INP_marine_alltemps[:,:,jl.cape_verde_latlon_index[0],jl.cape_verde_latlon_index[1],7]
temps=np.arange(-37,1,1)
temps=temps[::-1]
#%%
plt.figure()
for i in range(len(column_marine[0,:])):
    if i <22:
        continue
    plt.plot(temps,column_marine[:,i],'g--',label='Marine organics')
    plt.plot(temps,column_feldspar[:,i],'r--',label='K-feldspar')
plt.yscale('log')
plt.ylabel('$[INP]/L$')
#%%
level=20
plt.plot(temps,column_marine[:,level],'g--')
plt.plot(temps,column_feldspar[:,level],'r--')
plt.yscale('log')
table=np.zeros((39,32))
ps=[(i+1)*1/31.*1000 for i in range(31)]
table[1:,0]=temps
table[0,1:]=ps
table[1:,1:]=column_feldspar
np.savetxt('marine_cape_verde.csv',table,delimiter=',')
np.savetxt('feldspar_cape_verde.csv',table,delimiter=',')
for i in range(len(column_marine[0,:])):


#plt.plot(ps)
#plt.plot(jl.pressure)
