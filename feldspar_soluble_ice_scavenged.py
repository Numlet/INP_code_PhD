# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 09:57:03 2015

@author: eejvt
"""

import numpy as np
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
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
project='FELDSPAR_SOLUBLE_REMOVED'
os.chdir(archive_directory+project)

#%%
s=jl.read_data('/nfs/a201/eejvt/FELDSPAR_SOLUBLE_REMOVED')


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



def calculate_INP_feld_ext(T,fel_modes=[2,3,5,6]):
    std=s.sigma[:]
    #T=258
    modes_vol=jl.volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/jl.rhocomp[6])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*kfeld_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2#factor 1e2 because of cm in feldspar parameterization
    ns=jl.feld_parametrization(T)
    INP=np.zeros(Nd.shape)
    for i in fel_modes:
        print 'mode',i
        area_particle=jl.area_lognormal_per_particle(rmean,std[i])
        ff=1-np.exp(-ns*area_particle)
        ff_fitted=jl.correct_ff(ff,std[i])
        INP=Nd*ff_fitted
    return INP

INP_feld_ext=np.zeros((38,7, 31, 64, 128, 12))
for t in range(38):
    INP_feld_ext[t,:,:,:,:,:]=calculate_INP_feld_ext(-t+273.15)

INP_total_feld_ext=INP_feld_ext
np.save(archive_directory+project+'/INP_feld_ext.npy',INP_total_feld_ext)
#%%

INP_feld_solrem=np.load(archive_directory+project+'/INP_feld_ext.npy')
#%%
rs=jl.logaritmic_steps(-10,-4,10000)
std=2
s.rbardry[6,30,:,0,0].max()
r=np.logspace(-10,-5,1000)
rmean=1.72e-6
rmean=0.72e-6
rmean=6e-6
rmean=2e-6
tot_vol=(4./3*rs.mid_points**3*lognormal_PDF(rmean,rs.mid_points,std)*rs.grid_steps_width).sum()
plt.plot(rs.mid_points,4./3*rs.mid_points**3*lognormal_PDF(rmean,rs.mid_points,std))
rmeanac=9.1597133e-07
stdac=1.4
#plt.plot(rs.mid_points,lognormal_PDF(rmeanac,rs.mid_points,stdac))
#plt.xscale('log')
print (lognormal_PDF(rmean,rs.mid_points,std)*rs.grid_steps_width).sum()
print (4/3*rs.mid_points**3*lognormal_PDF(rmean,rs.mid_points,std)).sum()
print (4/3*rs.mid_points**3*lognormal_PDF(rmean,rs.mid_points,std)*rs.grid_steps_width/tot_vol).sum()
limits=[4e-6,8e-6,16e-6]
#plt.yscale('log')
for limit in limits:
    plt.axvline(limit,c='k',ls='--')
    print 'proportions'
probs=[]
limits=[-10,-2]
limits=[-10,np.log10(4e-6),np.log10(8e-6),np.log10(16e-6),-2]
for i in range (4):
    print limits[i],limits[i+1]
    rs2=jl.logaritmic_steps(limits[i],limits[i+1],1000)
    probs.append((4./3*rs2.mid_points**3*lognormal_PDF(rmean,rs2.mid_points,std)*rs2.grid_steps_width/tot_vol).sum())
    print probs
print probs[0]+probs[1]+probs[2]+probs[3]
