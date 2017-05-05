# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:15:47 2015

@author: eejvt
"""


import os
import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import Jesuslib as jl
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
import glob as glob

#%%
#archive_directory='/nfs/a107/eejvt/'
#project='JB_TRAINING'

rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3

archive_directory='/nfs/a201/eejvt/'

project='MARINE_PARAMETERIZATION/'
os.chdir(archive_directory+project)
s=jl.read_data('DAILY')
#%%
def area_lognormal_per_particle(rbar,sigma):
    #print isinstance(sigma,float)
    if isinstance(sigma,np.float32):
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        S=(2*rbar)**2*y
    else:
        S=np.zeros(rbar.shape)
        #y=np.zeros(len(rbar))
        #global S,y
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        for i in range (len(rbar)):
            S[i,]=(2*rbar[i,])**2*y[i]
    return S

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
    

def niemand_parametrization(T):
    #T in K 
    return np.exp(-0.517*(T-273.15)+8.834)*1e-4#cm**2


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


def calculate_INP_niemand_ext_mean_area_fitted(T,dust_modes=[2,3]):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    dust_volfrac=((s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])+(s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]))/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*dust_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
    ns=niemand_parametrization(T)
    INP=np.zeros(Nd.shape)
    for i in dust_modes:
        print 'mode',i
        area_particle=area_lognormal_per_particle(rmean,std[i])
        ff=1-np.exp(-ns*area_particle)
        ff_fitted=jl.correct_ff(ff,std[i])
        INP=Nd*ff_fitted
    return INP

'''
def calculate_INP_niemand_ext(T):
    dust_modes=[2,3]#,5,6]
    #def INP_feldspar_ext(s,T):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*kfeld_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
    ns=niemand_parametrization(T)
    INP=np.zeros(Nd.shape)
    for i in dust_modes:
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
INP_niemand_ext=np.zeros((38, 31, 64, 128, 365))
for t in range(38):
    print t
    INP_niemand_ext[t,:,:,:,:]=calculate_INP_niemand_ext_mean_area_fitted(-t+273.15).sum(axis=0)

np.save('/nfs/a201/eejvt/INP_niemand_ext_alltemps',INP_niemand_ext)

#%%

def calculate_INP_hoose_ext(T,dust_modes=[2,3]):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    dust_volfrac=((s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])+(s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]))/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*dust_volfrac
    INP=np.zeros(Nd.shape)
    for i in dust_modes:
        print 'mode',i
        #ff=
        INP=Nd*ff_fitted
    return INP

#%%
#ts=np.arange(248,267)
#def osullivan_parametrization(T):
#    #return ((2.974e-3)*T*T-2.16*T+366.3)#cm**2
#    return np.exp(2.974e-3*T**2-2.160*T+366)
#print osullivan_parametrization(ts)
##2.974e-3*T**2-2.160*T+366
#
#def feld_parametrization(T):
#    ns=np.exp((-1.03802178356815*T)+275.263379304105)
#    return ns
#    
#
#def niemand_parametrization(T):
#    return np.exp(-0.517*(T-273.15)+8.834)*1e-4#cm**2
#
#plt.plot(ts,osullivan_parametrization(ts),'k-')
#plt.plot(ts,niemand_parametrization(ts))
#plt.plot(ts,feld_parametrization(ts))
#plt.yscale('log')