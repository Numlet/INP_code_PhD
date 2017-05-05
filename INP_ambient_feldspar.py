# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:18:10 2015

@author: eejvt
"""

import os
import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import Jesuslib as jl
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
archive_directory='/nfs/a107/eejvt/'
project='JB_TRAINING/'
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
os.chdir(archive_directory+project)

rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3

temperatures=np.load('/nfs/a107/eejvt/temperatures_daily.npy')
temperatures=temperatures+273.15
temperatures[temperatures<236]=1000
temperatures[temperatures<248]=248
temperatures[temperatures>268]=1000


def area_lognormal(rbar,sigma,Nd):
    #print isinstance(sigma,np.float32)
    if isinstance(sigma,np.float32):
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
    

#%%

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
    
feld_paper_clevs=[0,0.0001,0.0002,0.0005,0.0010,0.002,0.005,0.01,0.02,0.05,0.1]

def lognormal_PDF(rmean,r,std):
   X=(1/(r*np.log(std)*np.sqrt(2*np.pi)))*np.exp(-(np.log(r)-np.log(rmean))**2/(2*np.log(std)**2))
   return X

s=jl.read_data('WITH_ICE_SCAV2')
sigma=s.sigma[:]


def calculate_INP_feld_ext_ambient(T):
    fel_modes=[2,3]#,5,6]
    #def INP_feldspar_ext(s,T):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*kfeld_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
    month_days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    INP=np.zeros(Nd.shape)
    for i in fel_modes:
        print 'Mode',i    
        rmin=rmean[i,]/3/std[i]
        rmax=rmean[i,]*3*std[i]
        step=rmin
        for ilev in range(len(step[:,0,0,0])):
            print 'ilev',ilev
            for ilat in range(len(step[0,:,0,0])):
                #print 'ilat',ilat
                for ilon in range(len(step[0,0,:,0])):
                    iday_cum=0                  
                    for imon in range(len(step[0,0,0,:])):
                        INP_days=0                        
                        for iday in range(month_days[imon]):
                            ns=feld_parametrization(T[ilev,ilat,ilon,iday_cum+iday])
                            Amax=4*np.pi*rmax[ilev,ilat,ilon,imon]**2#*kfeld_volfrac[i,ilev,ilat,ilon,imon]
                            exponent=ns*Amax
                            if exponent<0.05:
                                dINP=ns*area_lognormal(rmean[i,ilev,ilat,ilon,imon],std[i],Nd[i,ilev,ilat,ilon,imon])
                            else:
                                rs=np.arange(rmin[ilev,ilat,ilon,imon],rmax[ilev,ilat,ilon,imon],step[ilev,ilat,ilon,imon])
                                A=4*np.pi*rs**2
                                ff=1-np.exp(-ns*A)
                                PDF=lognormal_PDF(rmean[i,ilev,ilat,ilon,imon],rs,std[i])
                                #print PDF.sum()*step[ilev,ilat,ilon,imon]
                                dINP=PDF*Nd[i,ilev,ilat,ilon,imon]*ff
                                dINP=dINP.sum()*step[ilev,ilat,ilon,imon]
                            INP_days=INP_days+dINP
                        INP[i,ilev,ilat,ilon,imon]=INP[i,ilev,ilat,ilon,imon]+INP_days/month_days[imon]
                        iday_cum=iday_cum+month_days[imon]
                        #print 'INP',INP[i,ilev,ilat,ilon,imon]
    return INP

INP_ambient_feldext=calculate_INP_feld_ext_ambient(temperatures)

np.save('INP_ambient_feld_ext',INP_ambient_feldext)

#%%
def calculate_INP_feld_int_ambient(T):
    fel_modes=[2,3]#,5,6]
    #def INP_feldspar_ext(s,T):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]
    rmean=s.rbardry[:,:,:,:,:]*1e2
    month_days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    INP=np.zeros(Nd.shape)
    for i in fel_modes:
        print 'Mode',i    
        rmin=rmean[i,]/4/std[i]
        rmax=rmean[i,]*4*std[i]
        step=rmin
        for ilev in range(len(step[:,0,0,0])):
            print 'ilev',ilev
            for ilat in range(len(step[0,:,0,0])):
                print 'ilat',ilat
                for ilon in range(len(step[0,0,:,0])):
                    iday_cum=0                  
                    for imon in range(len(step[0,0,0,:])):
                        INP_days=0                        
                        for iday in range(month_days[imon]):
                            ns=feld_parametrization(T[ilev,ilat,ilon,iday_cum+iday])
                             
                            
                            rs=np.arange(rmin[ilev,ilat,ilon,imon],rmax[ilev,ilat,ilon,imon],step[ilev,ilat,ilon,imon])
                            A=4*np.pi*rs**2
                            ff=1-np.exp(-ns*A*kfeld_volfrac[i,ilev,ilat,ilon,imon])
                            PDF=lognormal_PDF(rmean[i,ilev,ilat,ilon,imon],rs,std[i])
                            #print PDF.sum()*step[ilev,ilat,ilon,imon]
                            dINP=PDF*Nd[i,ilev,ilat,ilon,imon]*ff
                            INP_days=INP_days+dINP.sum()*step[ilev,ilat,ilon,imon]
                        INP[i,ilev,ilat,ilon,imon]=INP[i,ilev,ilat,ilon,imon]+INP_days/month_days[imon]
                        iday_cum=iday_cum+month_days[imon]
                        #print 'INP',INP[i,ilev,ilat,ilon,imon]
    return INP



INP_ambient_feldint=calculate_INP_feld_int_ambient(temperatures)

np.save('INP_ambient_feld_int',INP_ambient_feldint)
#%%

'''

PRUEBA DE QUE EL NO ICE SCAV NO AFECTA TANTO

'''

def calculate_INP_feld_ext_ambient_AT(T):
    fel_modes=[2,3]#,5,6]
    #def INP_feldspar_ext(s,T):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*kfeld_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
    month_days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    INP=np.zeros((7, 31, 64, 1, 12))
    for i in fel_modes:
        print 'Mode',i    
        rmin=rmean[i,]/3/std[i]
        rmax=rmean[i,]*3*std[i]
        step=rmin
        for ilev in range(len(step[:,0,0,0])):
            print 'ilev',ilev
            for ilat in range(len(step[0,:,0,0])):
                #print 'ilat',ilat
                for ilon in [118]:
                    iday_cum=0                  
                    for imon in range(len(step[0,0,0,:])):
                        INP_days=0                        
                        for iday in range(month_days[imon]):
                            ns=feld_parametrization(T[ilev,ilat,ilon,iday_cum+iday])
                            Amax=4*np.pi*rmax[ilev,ilat,ilon,imon]**2#*kfeld_volfrac[i,ilev,ilat,ilon,imon]
                            exponent=ns*Amax
                            if exponent<0.05:
                                dINP=ns*area_lognormal(rmean[i,ilev,ilat,ilon,imon],std[i],Nd[i,ilev,ilat,ilon,imon])
                            else:
                                rs=np.arange(rmin[ilev,ilat,ilon,imon],rmax[ilev,ilat,ilon,imon],step[ilev,ilat,ilon,imon])
                                A=4*np.pi*rs**2
                                ff=1-np.exp(-ns*A)
                                PDF=lognormal_PDF(rmean[i,ilev,ilat,ilon,imon],rs,std[i])
                                #print PDF.sum()*step[ilev,ilat,ilon,imon]
                                dINP=PDF*Nd[i,ilev,ilat,ilon,imon]*ff
                                dINP=dINP.sum()*step[ilev,ilat,ilon,imon]
                            INP_days=INP_days+dINP
                        INP[i,ilev,ilat,0,imon]=INP[i,ilev,ilat,0,imon]+INP_days/month_days[imon]
                        iday_cum=iday_cum+month_days[imon]
                        #print 'INP',INP[i,ilev,ilat,ilon,imon]
    return INP
   
   
feldext_AT_noICE=calculate_INP_feld_ext_ambient_AT(temperatures)
INP_feldext_ambient=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_ambient_feld_ext.npy').sum(axis=0)
feldext_AT_noICE=feldext_AT_noICE.sum(axis=0)
feldext_AT_noICE=feldext_AT_noICE*1e6
AT_feldext=INP_feldext_ambient[:,:,118,:]*1e6

feldext_AT_noICE_clean=feldext_AT_noICE[:,:,0,:]
levelsmo=[10,100,1000,10000]
levelsfel=[10,100,1000,10000]
ps=np.array([   4.47640944,   28.3048172 ,   48.96452713,   69.30890656,
 89.56233978,  110.04908752,  131.62251282,  154.64620972,
179.33183289,  205.97129822,  234.46916199,  264.84896851,
297.05499268,  330.97183228,  366.49978638,  403.52679443,
441.94363403,  481.63827515,  522.48620605,  564.35626221,
607.08886719,  650.46594238,  694.17602539,  737.8137207 ,
780.80426025,  822.40307617,  861.61694336,  897.16723633,
927.43457031,  950.37841797,  963.48803711])

lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
#[:,:,0:30]
fs=13
#[:,:,180:211]
DJF=np.arange(-31,59,1)
DJF_months=np.array([11,0,1])
feb=np.arange(31,59,1)
fig=plt.figure()
cx=plt.subplot(1,1,1)
Xmo,Ymo= np.meshgrid(lat.glat, ps)
Xfel,Yfel=np.meshgrid(lat.glat,ps)
cx.set_title('Dec-Jan-Feb')
CS=cx.contour(Xfel,Yfel,AT_feldext[:,:,DJF_months].mean(axis=-1),levelsfel,colors='k',hold='on',linewidths=[2,2,2])#linewidths=np.linspace(2, 6, 3)
plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
plt.setp(CS.collections)
CF=cx.contourf(Xmo,Ymo,feldext_AT_noICE_clean[:,:,DJF_months].mean(axis=-1),levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
CB=plt.colorbar(CF,ticks=levelsmo,drawedges=1,label='$m^{-3}$')

cx.invert_yaxis()
cx.set_ylim(ymax=200)
cx.tick_params(axis='both', which='major', labelsize=fs)
cx.set_ylabel('Pressure level /hPa')
cx.set_xlabel('Latitude')
cx.xaxis.set_ticks(np.arange(-90,100,20))