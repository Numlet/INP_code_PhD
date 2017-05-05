# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 09:14:16 2016

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
import multiprocessing


rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3




folder='/nfs/a201/eejvt/FELDSPAR_SOLUBLE_REMOVED/ACCCOR/'
folder='/nfs/a201/eejvt/mineral_fractions_perlwitz/RUN/'
folder='/nfs/a201/eejvt/mineral_fractions_perlwitz/RUN_DUSTFRAC/'


archive_directory='/nfs/a201/eejvt/'
project='MARINE_PARAMETERIZATION/FOURTH_TRY'
project='MARINE_PARAMETERIZATION/DAILY'
folder='/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/'
#os.chdir(archive_directory+project)
#%%
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


def marine_org_parameterization(T):
    a=11.2186
    b=-0.4459
    INP=np.exp(a+b*T)#[per gram]
    return INP
#%%
#s={}
#a=glob(folder+'*.sav')
#
#s=jl.read_data(folder)
#
##readsav(a,idict=s)
#total_marine_mass=s.tot_mc_ss_mm_mode[2,]#+s.tot_mc_ss_mm_mode[3,]#ug/m3
#total_marine_mass_year_mean=total_marine_mass.mean(axis=-1)
##total_marine_mass_monthly_mean=jl.from_daily_to_monthly(total_marine_mass)
#
#
#total_marine_mass_grams_OC=total_marine_mass*1e-6/1.9#g/m3
#temperatures=s.t3d_mm
#temperatures=temperatures-273.15
#temperatures[temperatures<-37]=1000#menores que -37
#temperatures[temperatures<-27]=-27#menor que -25 =-25
#temperatures[temperatures>-5]=1000#mayor que -15 = 0
#
#INP_marine_ambient=total_marine_mass_grams_OC*marine_org_parameterization(temperatures)#m3
#INP_marine_ambient_constant_press,new_pressures,idexes=jl.constant_pressure_level_array(INP_marine_ambient,s.pl_m*1e-2)
#np.save(folder+'INP_marine_ambient_constant_press',INP_marine_ambient_constant_press)
#
#INP_marine_alltemps=np.zeros((38,31,64,128,12))
#for i in range (38):
#    INP_marine_alltemps[i,]=total_marine_mass_grams_OC*marine_org_parameterization(-i)
#np.save(folder+'INP_marine_alltemps.npy',INP_marine_alltemps)
#%%
s={}
a=glob(folder+'*.sav')

s=jl.read_data(folder)


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
        area_particle=area_lognormal_per_particle(rmean[i,],std[i])
        exponent=ns*area_particle
        ff=np.zeros_like(exponent)
        ff[exponent>=1e-5]=1-np.exp(-exponent[exponent>=1e-5])
        ff[exponent<1e-5]=exponent[exponent<1e-5]
        
        ff_fitted=jl.correct_ff(ff,std[i])
        INP[i,]=INP[i,]+Nd*ff_fitted
    return INP

INP_feldext_alltemps=np.zeros((38,31,64,128,12))
#INP_feldext_alltemps_modes=np.zeros((38,7,31,64,128,12))
INP_feldext_alltemps=np.zeros((38,31,64,128,365))

for i in range (38):
    print i
    INP_feldext_alltemps[i,]=calculate_INP_feld_ext_mean_area_fitted(-i+273.15).sum(axis=0)
#    INP_feldext_alltemps_modes[i,]=calculate_INP_feld_ext_mean_area_fitted(-i+273.15)

np.save(folder+'INP_feldext_alltemps_daily_with_coarse.npy',INP_feldext_alltemps)

jl.send_email()
##%%
#np.save(folder+'INP_feldext_alltemps_modes.npy',INP_feldext_alltemps_modes)
#
#
#temperatures=temperatures+273.15
#INP_ambient_feldext_modes=calculate_INP_feld_ext_mean_area_fitted(temperatures)
#INP_ambient_feldext=calculate_INP_feld_ext_mean_area_fitted(temperatures).sum(axis=0)
#
#INP_ambient_feldext_constant_press,new_pressures,idexes=jl.constant_pressure_level_array(INP_ambient_feldext,s.pl_m*1e-2)
#INP_ambient_feldext_modes_constant_press=np.zeros((7,21,64,128,12))
#for i in range(7):
#    INP_ambient_feldext_modes_constant_press[i,],new_pressures,idexes=jl.constant_pressure_level_array(INP_ambient_feldext_modes[i,],s.pl_m*1e-2)
#np.save(folder+'INP_ambient_feldext_constant_press',INP_ambient_feldext_constant_press)
#np.save(folder+'INP_ambient_feldext_modes_constant_press',INP_ambient_feldext_modes_constant_press)
#
