# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:52:30 2015

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


def young(T,units_cm=0):
    if units_cm:
         return 2e2*(270.16-T)**0.333*1e-3
    else:
         return 2e2*(270.16-T)**0.333

def meyers(T,units_cm=0):
    if units_cm:
        return np.exp(-2.8+0.262*(273.15-T))*1e-3
    else:
        return np.exp(-2.8+0.262*(273.15-T))#L-1
        

INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MURRAY.dat",header=1)
temps=INP_obs[:,1]
conc=INP_obs[:,2]#cm3
meyers_conc=meyers(temps+273.15, units_cm=1)
young_conc=young(temps+273.15, units_cm=1)
plt.scatter(conc,meyers_conc,c=temps)
#plt.scatter(conc,young_conc,c=temps)
x=np.logspace(-6,3)
plt.plot(x,x,'k-')
plt.plot(x,10*x,'k--')
plt.plot(x,0.1*x,'k--')
plt.xlim(1e-6,2)
plt.ylim(1e-6,2)
plt.xscale('log')
plt.yscale('log')
plt.colorbar(label='Temperature $C$')
plt.ylabel('Simulated ($cm^{-3}$)')
plt.xlabel('Observed ($cm^{-3}$)')
plt.title('Meyers')
plt.show()