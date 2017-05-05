# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 09:57:03 2015

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
archive_directory='/nfs/a201/eejvt/'
project='NUC_SCAV'
os.chdir(archive_directory+project)

t25=jl.read_data('T25')

s=t25


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
INP_feld_ext=np.zeros((38,7, 31, 64, 128, 12))
for t in range(38):
    INP_feld_ext[t,:,:,:,:,:]=calculate_INP_feld_ext(-t+273.15)

INP_total_feld_ext=INP_feld_ext.sum(axis=1)
np.save('INP_feld_ext_alltemps_tice25',INP_total_feld_ext)
