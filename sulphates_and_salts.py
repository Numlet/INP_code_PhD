# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:41:42 2015

@author: eejvt
"""

from numba import autojit
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
from multiprocessing import Pool
from scipy.integrate import quad
archive_directory='/nfs/a201/eejvt/'
project='SULPHATES_AND_SALTS'
os.chdir(archive_directory+project)

cloud_drop_radii=10**-7#m**3 10microns
cloud_drop_radii=10**-6#m**3 10microns
vol_cloud_drop=4./3*np.pi*cloud_drop_radii**3

#%%


rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3

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
    

def feld_parametrization(T):
    ns=np.exp((-1.03802178356815*T)+275.263379304105)
    return ns


def feld_parametrization_with_ss(T,css=0):
    ns=np.zeros(css.shape)
    ns[[css>=0.00003]]=np.exp((-1.0387*T)+((275.26+(-0.7492*np.log(css[css>=0.00003]))-7.605)))
    ns[css<0.00003]=np.exp((-1.0387*T)+275.26)
    return ns

def feld_parametrization_with_sulphate(T,csul=0):
    ns=np.zeros(csul.shape)
    ns[csul>=0.00000001]=np.exp((-1.0387*T)+((275.26+(0.2063*np.log(csul[csul>=0.00000001]))+3.8937)))
    ns[csul<0.00000001]=np.exp((-1.0387*T)+275.26)
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
'''   
def calculate_INP_feld_ext_with_ss(T):
    fel_modes=[2,3]#,5,6]
    #def INP_feldspar_ext(s,T):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    salt_to_feld=s.tot_mc_ss_mm_mode/s.tot_mc_feldspar_mm_mode
    Nd=s.st_nd[:,:,:,:,:]*kfeld_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
    #ns=feld_parametrization(T)
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
                        Vp=4./3*np.pi*rs**3
                        Mfel=Vp*rhocomp[6]
                        Mss=Mfel*salt_to_feld[i,ilev,ilat,ilon,imon]
                        molss=Mss/58.9e6
                        Css=molss/vol_cloud_drop#mol/m**3
                        Css=Css*1e-3#mol/l
                        ns=feld_parametrization_with_ss(T,Css)
                        ff=1-np.exp(-ns*A)
                        PDF=lognormal_PDF(rmean[i,ilev,ilat,ilon,imon],rs,std[i])
                        #print PDF.sum()*step[ilev,ilat,ilon,imon]
                        dINP=PDF*Nd[i,ilev,ilat,ilon,imon]*ff
                        INP[i,ilev,ilat,ilon,imon]=INP[i,ilev,ilat,ilon,imon]+(dINP.sum()*step[ilev,ilat,ilon,imon])
                        #print 'INP',INP[i,ilev,ilat,ilon,imon]
    return INP
'''
def calculate_INP_feld_ext_with_ss(T):
    fel_modes=[2,3]#,5,6]
    modes_vol=volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    salt_to_feld=s.tot_mc_ss_mm_mode/s.tot_mc_feldspar_mm_mode
    Nd=s.st_nd[:,:,:,:,:]*kfeld_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
    INP=np.zeros(Nd.shape)
    A=4*np.pi*rmean**2
    Vp=4./3*np.pi*rmean**3
    Mfel=Vp*rhocomp[6]
    Mss=Mfel*salt_to_feld
    molss=Mss/58.9e6
    Css=molss/vol_cloud_drop#mol/m**3
    Css=Css*1e-3#mol/l
    Css[np.isnan(Css)]=0
    Css[np.isinf(Css)]=0
    ns=feld_parametrization_with_ss(T,Css)
    ff=1-np.exp(-ns*A)
    #print PDF.sum()*step[ilev,ilat,ilon,imon]
    dINP=Nd*ff
    INP=INP+dINP
    #print 'INP',INP[i,ilev,ilat,ilon,imon]
    return INP


def calculate_INP_feld_ext_with_sulphate(T):
    fel_modes=[2,3]#,5,6]
    modes_vol=volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    sulph_to_feld=s.tot_mc_su_mm_mode/s.tot_mc_feldspar_mm_mode
    Nd=s.st_nd[:,:,:,:,:]*kfeld_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
    INP=np.zeros(Nd.shape)
    A=4*np.pi*rmean**2
    Vp=4./3*np.pi*rmean**3
    Mfel=Vp*rhocomp[6]
    Msu=Mfel*sulph_to_feld
    molsu=Msu/98.0e6
    Csu=molsu/vol_cloud_drop#mol/m**3
    Csu=Csu*1e-3#mol/l
    Csu[np.isnan(Csu)]=0
    Csu[np.isinf(Csu)]=0
    ns=feld_parametrization_with_sulphate(T,Csu)
    ff=1-np.exp(-ns*A)
    #print PDF.sum()*step[ilev,ilat,ilon,imon]
    dINP=Nd*ff
    INP=INP+dINP
    #print 'INP',INP[i,ilev,ilat,ilon,imon]
    return INP
#%%
s=jl.read_data('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2')
#%%
'''
i=2
std=s.sigma[:]
rmean=s.rbardry[:,:,:,:,:]*1e2
rmin=rmean[i,]/4/std[i]
rmax=rmean[i,]*4*std[i]
step=rmin/10
Nd_other=np.zeros(step.shape)
vol_2=np.zeros(step.shape)
for ilev in range(len(step[:,0,0,0])):
    print 'ilev',ilev
    for ilat in range(len(step[0,:,0,0])):
        for ilon in range(len(step[0,0,:,0])):
            for imon in range(len(step[0,0,0,:])):
                rs=np.arange(rmin[ilev,ilat,ilon,imon],rmax[ilev,ilat,ilon,imon],step[ilev,ilat,ilon,imon])
                V=4/3*np.pi*rs**3
                PDF=lognormal_PDF(rmean[i,ilev,ilat,ilon,imon],rs,std[i])
                #print PDF.sum()*step[ilev,ilat,ilon,imon]
                Nd_other[ilev,ilat,ilon,imon]=s.tot_mc_feldspar_mm_mode[i,ilev,ilat,ilon,imon]/((PDF*V*step[ilev,ilat,ilon,imon]).sum()*rhocomp[6])
                vol_2[ilev,ilat,ilon,imon]=(PDF*V*step[ilev,ilat,ilon,imon]).sum()
'''
#%%


INP_ss_20=calculate_INP_feld_ext_with_ss(-20+273.15)

INP_ss_20=INP_ss_20.sum(axis=0)
feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/Feldspar_distributions/INP_feld_ext_alltemps.npy')
#%%

fel_old=feldext[20,].mean(axis=-1)

ratio=INP_ss_20.mean(axis=-1)/fel_old
levels=np.logspace(-4,4,9).tolist()
jl.plot(ratio[20,:,:],show=1,clevs=levels,title='-20 600hpa feldspar')

#%%
INP_su_20=calculate_INP_feld_ext_with_sulphate(-20+273.15)

INP_su_20=INP_su_20.sum(axis=0)
feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/Feldspar_distributions/INP_feld_ext_alltemps.npy')
#%%

fel_old=feldext[20,].mean(axis=-1)

ratio=INP_su_20.mean(axis=-1)/fel_old
levels=np.logspace(-4,4,9).tolist()
jl.plot(ratio[20,:,:],show=1,clevs=levels,title='-20 600hpa feldspar')


#%%
css=0.000000
Ts=np.linspace(248,268,21)
ns=np.exp((-1.0387*Ts)+275.26)
plt.plot(Ts,ns,label=css)
css=0.002
Ts=np.linspace(248,268,21)
ns=np.exp((-1.0387*Ts)+((275.26+(-0.7492*np.log(css))-7.605)))
plt.plot(Ts,ns,label=css)
css=0.0002
Ts=np.linspace(248,268,21)
ns=np.exp((-1.0387*Ts)+((275.26+(-0.7492*np.log(css))-7.605)))
plt.plot(Ts,ns,label=css)
plt.legend()
plt.yscale('log')
plt.show()

