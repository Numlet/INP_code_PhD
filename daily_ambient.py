# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:27:11 2015

@author: eejvt
"""

import os
import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import Jesuslib as jl
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
archive_directory='/nfs/a201/eejvt/'
project='DAILY_RUN/'
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
import glob as glob
os.chdir(archive_directory+project)

#%%


INP_BC_extamb_constant_press_m3=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_BC_extamb_constant_press.npy')*1e6
INP_feld_extamb_constant_press_m3=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_feld_extamb_constant_press.npy')*1e6

INP_BC_extamb_constant_press_m3[INP_BC_extamb_constant_press_m3<1]=0
INP_feld_extamb_constant_press_m3[INP_feld_extamb_constant_press_m3<1]=0
INP_ambient_total=INP_BC_extamb_constant_press_m3+INP_feld_extamb_constant_press_m3
ratio=INP_BC_extamb_constant_press_m3/INP_feld_extamb_constant_press_m3
ratio[np.isnan(ratio)]=0
print np.array([ratio==1]).sum()
ratio[ratio>1]=1
ratio[ratio<1]=0
ratio=ratio.sum(axis=-1)
for i in range (20):
    jl.plot2(ratio[i,:,:],contour=INP_ambient_total[i,:,:,:].mean(axis=-1),contourlevs=[1,10,100,1000],title='%i hpa'%((i+1)*50),cmap=plt.cm.OrRd,file_name='days_BC-Feld_%i'%((i+1)*50),saving_format='png')
    jl.plot2(ratio[i,:,:],contour=INP_ambient_total[i,:,:,:].mean(axis=-1),contourlevs=[1,10,100,1000],title='%i hpa'%((i+1)*50),cmap=plt.cm.OrRd,file_name='days_BC-Feld_%i'%((i+1)*50),saving_format='svg')
#%%
rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3

names=['tot_mc_oc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_daily_2001.sav',
'tot_mc_bc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_daily_2001.sav',
'tot_mc_su_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_daily_2001.sav',
'tot_mc_ss_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_daily_2001.sav',
'tot_mc_dust_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_daily_2001.sav',
'tot_mc_feldspar_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_daily_2001.sav',
'GLOMAP_mode_mol_no_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_daily_2001.sav',
'mean_dry_r_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_daily_2001.sav',
'sigmachained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_daily_2001.sav',
'hindcast3_temp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_daily_2001.sav',
'GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_daily_2001.sav']
s={}
a=glob.glob('*.sav')
for name in names:
    s=readsav(name,idict=s)

marine_mass=s.tot_mc_ss_mm_mode.sum(axis=0)
#jl.latplot(marine_mass.mean(axis=-1))
#s=readsav('GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_daily_2001.sav',idict=s)

#%%
temperatures=s.t3d_mm
#temperatures=temperatures+273.15
temperatures[temperatures<236]=1000#menores que -37
temperatures[temperatures<248]=248#menor que -25 =-25
temperatures[temperatures>268]=1000#mayor que -15 = 0

def area_lognormal(rbar,sigma,Nd):
    #print isinstance(sigma,float)
    if isinstance(sigma,float):
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        S=Nd*(2*rbar)**2*y
    else:
        S=np.zeros(rbar.shape)
        #y=np.zeros(len(rbar))
        #global S,y
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        for i in range (len(rbar)):
            S[i,]=Nd[i,]*(2*rbar[i,])**2*y[i]
    return S

def BC_parametrization(T):#T en C!!! rango de -16 a -24 podemos intentar -15, -25
    #A=-20.27
    #B=1.2
    return np.exp(-20.27-1.2*T)

def feld_parametrization(T):
    ns=np.exp((-1.03802178356815*T)+275.263379304105)
    return ns

def volumes_of_modes(s):
    modes_vol=(s.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]
    +s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2]
    +s.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
    +s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]
    +s.tot_mc_oc_mm_mode[:,:,:,:,:]/rhocomp[5]
    +s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])
    return modes_vol


def lognormal_PDF(rmean,r,std):
   X=(1/(r*np.log(std)*np.sqrt(2*np.pi)))*np.exp(-(np.log(r)-np.log(rmean))**2/(2*np.log(std)**2))
   return X


def calculate_INP_feld_ext_mean_area_fitted(T,fel_modes=[2,3]):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*kfeld_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2#factor 1e2 because of cm in feldspar parameterization
    ns=feld_parametrization(T)
    INP=np.zeros(Nd.shape)
    for i in fel_modes:
        print 'mode',i
        area_particle=area_lognormal_per_particle(rmean,std[i])
        ff=1-np.exp(-ns*area_particle)
        ff_fitted=jl.correct_ff(ff,std[i])
        INP=Nd*ff_fitted
    return INP
def calculate_INP_BC_ext_mean_area_fitted(T,BC_modes=[1,2,3]):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    BC_volfrac=(s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*BC_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
    if (T<0):
        ns=BC_parametrization(T)
    else:
        ns=BC_parametrization(T-273)
    INP=np.zeros(Nd.shape)

    for i in BC_modes:
        print 'mode',i
        area_particle=area_lognormal_per_particle(rmean,std[i])
        ff=1-np.exp(-ns*area_particle)
        ff_fitted=jl.correct_ff(ff,std[i])
        INP=Nd*ff_fitted
    return INP
def calculate_INP_BC_extamb_mean_area_fitted(Ts,BC_modes=[1,2,3]):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    BC_volfrac=(s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*BC_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
    if (Ts<0).any():
        ns=BC_parametrization(Ts)
        print 'Temperatures in C'
    else:
        ns=BC_parametrization(Ts-273)
        print 'Temperatures in K, converted to C inside function'
    INP=np.zeros(Nd.shape)

    for i in BC_modes:
        print 'mode',i
        area_particle=area_lognormal_per_particle(rmean,std[i])
        ff=1-np.exp(-ns*area_particle)
        ff_fitted=jl.correct_ff(ff,std[i])
        INP=Nd*ff_fitted
    return INP

def calculate_INP_feld_extamb_mean_area_fitted(Ts,fel_modes=[2,3]):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*kfeld_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
    ns=feld_parametrization(Ts)
    INP=np.zeros(Nd.shape)
    for i in fel_modes:
        print 'mode',i
        area_particle=area_lognormal_per_particle(rmean,std[i])
        ff=1-np.exp(-ns*area_particle)
        ff_fitted=jl.correct_ff(ff,std[i])
        INP=Nd*ff_fitted
    return INP

def calculate_INP_feld_ext(T):
    fel_modes=[2,3]#,5,6]
    #def INP_feldspar_ext(s,T):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*kfeld_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
    ns=feld_parametrization(T)
    INP=np.zeros(Nd.shape)
    for i in fel_modes:
        print 'Mode',i
        rmin=rmean[i,]/4/std[i]
        rmax=rmean[i,]*4*std[i]
        step=rmin
        for ilev in range(len(step[:,0,0,0])):
            print 'ilev',ilev
            for ilat in range(len(step[0,:,0,0])):
                for ilon in range(len(step[0,0,:,0])):
                    for imon in range(len(step[0,0,0,:])):
                        rs=np.arange(rmin[ilev,ilat,ilon,imon],rmax[ilev,ilat,ilon,imon],step[ilev,ilat,ilon,imon])
                        A=4*np.pi*rs**2
                        ff=1-np.exp(-ns*A)
                        PDF=lognormal_PDF(rmean[i,ilev,ilat,ilon,imon],rs,std[i])
                        #print PDF.sum()*step[ilev,ilat,ilon,imon]
                        dINP=PDF*Nd[i,ilev,ilat,ilon,imon]*ff
                        INP[i,ilev,ilat,ilon,imon]=INP[i,ilev,ilat,ilon,imon]+(dINP.sum()*step[ilev,ilat,ilon,imon])
                        #print 'INP',INP[i,ilev,ilat,ilon,imon]
    return INP
'''
INP_feld_extamb=np.zeros((31, 64, 128, 365))
INP_feld_extamb[:,:,:,:]=calculate_INP_feld_extamb_mean_area_fitted(temperatures).sum(axis=0)

np.save('INP_feld_extamb_daily',INP_feld_extamb)
'''
'''
INP_BC_extamb=np.zeros((31, 64, 128, 365))
INP_BC_extamb[:,:,:,:]=calculate_INP_BC_extamb_mean_area_fitted(temperatures).sum(axis=0)

np.save('INP_BC_extamb_daily',INP_BC_extamb)

'''
#%%
INP_feld_extamb=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_feld_extamb_daily.npy')
'''
INP_feld_extamb_constant_press,new_pressures,idexes=jl.constant_pressure_level_array(INP_feld_extamb,s.pl_m*1e-2)
np.save('INP_feld_extamb_constant_press',INP_feld_extamb_constant_press)
'''

INP_feld_extamb_constant_press=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_feld_extamb_constant_press.npy')

INP_BC_extamb=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_BC_extamb_daily.npy')
'''
INP_BC_extamb_constant_press,new_pressures,idexes=jl.constant_pressure_level_array(INP_BC_extamb,s.pl_m*1e-2)
np.save('INP_BC_extamb_constant_press',INP_BC_extamb_constant_press)
'''

INP_BC_extamb_constant_press=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_BC_extamb_constant_press.npy')


#%%

INP_feld_extamb_coarse=np.zeros((31, 64, 128, 365))
INP_feld_extamb_coarse[:,:,:,:]=calculate_INP_feld_extamb_mean_area_fitted(temperatures,fel_modes=[3]).sum(axis=0)
INP_feld_extamb_constant_press_coarse,new_pressures,idexes=jl.constant_pressure_level_array(INP_feld_extamb_coarse,s.pl_m*1e-2)
np.save('INP_feld_extamb_constant_press_coarse',INP_feld_extamb_constant_press_coarse)



#%%
jl.plot(INP_feld_extamb_constant_press[10,:,:,34]*1e3,cblabel='l-1',clevs=[0,0.1,0.5,1,2,5,10,20,50,100,200,500,1000],cmap=plt.cm.OrRd)
