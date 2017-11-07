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
import matplotlib as mpl

mpl.rcParams['font.size'] = 15
mpl.rcParams['legend.fontsize']= 15
mpl.rcParams['legend.frameon'] = 'False'

# matplotlib.RcParams['font.size']=15
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
Ds=np.linspace(0.35,200,100000)
#Ds=np.logspace(-1,1,1000)
plt.close()
plt.plot(Ds,aw(Ds)*100)
plt.plot(Ds,sw(Ds)*100)
plt.plot(Ds,k_exp(Ds)*100)
plt.ylim(50,120)
plt.axhline(100,c='k',ls='-',lw=0.4)
plt.axhline(np.max(sw(Ds)*100),xmin=0,xmax=0.3,c='k',ls='--')#np.max(sw(Ds))*100

plt.axvline(Ds[np.argmax(sw(Ds))],ymin=0,ymax=0.8,c='k',ls='--')#np.max(sw(Ds))*100
plt.text(Ds[np.argmax(sw(Ds))]-0.3,45,'$R_c$')
plt.text(0.17,np.max(sw(Ds)*100)-1,'$S_c$')
plt.xscale('log')
plt.ylabel('Relative humidity (%)')
plt.xlabel('Diameter $(\mu m)$')
plt.show()


#plt.xscale('log')
