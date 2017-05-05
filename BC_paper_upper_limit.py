#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:52:22 2017

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



def BC_parametrization_tom(T):
    #A=-20.27
    #B=1.2
    return 10**(-2.87-0.182*T)
rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3


Ts=np.linspace(-25,-10,100)
plt.plot(Ts,BC_parametrization_tom(Ts),label='BC ns')
plt.yscale('log')



def volumes_of_modes(s):
    modes_vol=(s.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]
    +s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2]
    +s.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
    +s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]
    +s.tot_mc_oc_mm_mode[:,:,:,:,:]/rhocomp[5]
    +s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])
    return modes_vol
def feld_parametrization(T):
    ns=np.exp((-1.03802178356815*T)+275.263379304105)
    return ns
    
Ts=np.linspace(-25,-10,100)
plt.plot(Ts,feld_parametrization(Ts+273.15),label='Feldspar ns')
plt.yscale('log')
plt.legend()
#%%

path='/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2'


s=jl.read_data(path)



