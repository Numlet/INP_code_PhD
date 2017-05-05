# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:05:07 2016

@author: eejvt
"""


import matplotlib.pyplot as plt
from scipy.io.idl import readsav

from scipy import stats
from scipy.optimize import curve_fit#
import os
import scipy as sc
import numpy as np
import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import Jesuslib as jl



#%%


#def lognormal_PDF(rmean,r,std):
#   X=(1/(r*np.log(std)*np.sqrt(2*np.pi)))*np.exp(-(np.log(r)-np.log(rmean))**2/(2*np.log(std)**2))
#   return X

def I(rm,sigma,log_lim=[-9,-3]):
    rs=jl.logaritmic_steps(log_lim[0],log_lim[1],100000)
    result=np.array(rs.grid_steps_width*rs.mid_points**2*np.exp(-(np.log(rs.mid_points)-np.log(rm))**2/(2*np.log(sigma)**2))).sum()
    return result


def second_term(rm,sigma):
    result=np.sqrt(2*np.pi)*np.log(sigma)*rm**3*np.exp(4.5*np.log(sigma)**2)
    return result


array_I=[]
array_2nd=[]
sigma=1.5
rm_range=np.logspace(-7,-4)
for irm in rm_range:
    array_I.append(I(irm,sigma))
    array_2nd.append(second_term(irm,sigma))

plt.plot(rm_range,array_I)
plt.plot(rm_range,array_2nd)
plt.xscale('log')
plt.yscale('log')
#%%
def vol_u(rm,sigma):
    return ((2*rm)**3/6.)*np.pi*np.exp(4.5*np.log(sigma)**2)


def vol_i(rm,sigma):
    return (4.*np.pi)/(3*np.sqrt(2*np.pi)*np.log(sigma))*I(rm,sigma)

array_u=[]
array_i=[]
sigma=1.7
rm_range=np.logspace(-7,-4)
for irm in rm_range:
    array_u.append(vol_u(irm,sigma))
    array_i.append(vol_i(irm,sigma))
    print 1/(vol_u(irm,sigma)/vol_i(irm,sigma))


plt.plot(rm_range,array_u,label='usual')
plt.plot(rm_range,array_i,label='integrated')
plt.xscale('log')
plt.yscale('log')
plt.legend()



#%%

def I_analitic(rm,sigma):
    b=2*np.log(sigma)**2
    return np.sqrt(np.pi*b)*rm**3*np.exp(9./4.*b)




array_IA=[]
array_I=[]
sigma=1.7
rm_range=np.logspace(-7,-4)
for irm in rm_range:
    array_IA.append(I_analitic(irm,sigma))
    array_I.append(I(irm,sigma))
    print 1/(I_analitic(irm,sigma)/I(irm,sigma))


plt.plot(rm_range,array_IA,label='usual')
plt.plot(rm_range,array_I,label='integrated')
plt.xscale('log')
plt.yscale('log')
plt.legend()



   