# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:59:47 2016

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


R=8.3
p_w=1
T=273
V_w=0.9
M_w=1
a_w=0.9
sigma_w=75.64
Dd=0.5
k=0.9

def sw_approx(D):
    return aw(D)*np.exp((4*sigma_w*M_w)/(R*T*p_w*D))
def sw(D):
    return np.exp(1/D)
def aw(D):
    return (D**3-Dd**3)/(D**3-Dd**3*(1-k))
def sw(D):
    return aw(D)*np.exp((4*sigma_w*V_w)/(R*T*D))
def k_exp(D):
    return np.exp((4*sigma_w*M_w)/(R*T*p_w*D))
    
Ds=np.logspace(-2,0,10000)
Ds=np.linspace(0.5,6,10000)
plt.plot(Ds,aw(Ds))
plt.plot(Ds,sw(Ds))
plt.plot(Ds,k_exp(Ds))
plt.axhline(1,c='k',ls='--')
plt.axvline(Ds[np.argmax(sw(Ds))],c='k',ls='-')




#plt.xscale('log')