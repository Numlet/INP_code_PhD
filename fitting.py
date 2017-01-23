# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:39:34 2015

@author: eejvt
"""

import numpy as np
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
from pylab import *

from scipy import stats
from scipy.optimize import curve_fit

points=[2e10,2e10,9e9,7e9,3e11,7e11,1e12,4e11]
Ts=[-18.3,-18.7,-18.1,-18.3,-25,-25,-27,-27.5]


def func(x,a,b):
    return a-b*x    

    
def func_log(x,a,b):
    return np.exp(a) * np.exp(-b * x)


def feld_parametrization(T):
    ns=np.exp((-1.03802178356815*T)+275.263379304105)
    return ns

    
def niemand_parametrization(T):
    #T in K 
    return np.exp(-0.517*(T-273.15)+8.834)#cm**2
    

popt,pcov = curve_fit(func_log, Ts, points)
popt,pcov = curve_fit(func, Ts, np.log(points))
#popt=np.array([-18.59897567,   1.10249526])
#pcov=np.array([[ 0.50795402, -0.02496729],[-0.02496729,  0.00123657]])
perr = np.sqrt(np.diag(pcov))
ci = 0.95
pp = (1. + ci) / 2.
nstd = stats.norm.ppf(pp)
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr
a=feld_parametrization(-25+273.15)*1e4
b=feld_parametrization(-15+273.15)*1e4
plt.figure()
ax=plt.subplot(121)
Tx=np.linspace(-29,-10,100)
ns_fitted=func_log(Tx,popt[0],popt[1])
ns_low=func_log(Tx,*popt_dw)
ns_high=func_log(Tx,*popt_up)
#plt.plot(-25,a,'ro')

plt.plot(Ts,points,'bo')
plt.plot(Tx,ns_fitted,'k-',lw=3,label='emersic data fit')
plt.plot(Tx,feld_parametrization(Tx+273.15)*1e4,'b-',lw=3,label='Feldspar parameterization')
plt.plot(Tx,niemand_parametrization(Tx+273.15),'g-',lw=3,label='Niemand parameterization')
#plt.plot(Tx,ns_high,'k--')
#plt.plot(Tx,ns_low,'k--')
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel('ns $(m^{-2})$')
plt.xlabel('$T(^oC)$')

bx=plt.subplot(122)
plt.plot(Tx,feld_parametrization(Tx+273.15)*1e4/ns_fitted,'r')
plt.plot(Tx,feld_parametrization(Tx+273.15)*1e4/niemand_parametrization(Tx+273.15),'g')
plt.title('ratio feldspar / emersic fit')
plt.axvline(-25,c='k',ls='--')
plt.axhline(a/func_log(-25,popt[0],popt[1]),c='k',ls='--')
plt.ylabel('ratio ns_feldspar / ns_emersic_fit')
plt.xlabel('$T(^oC)$')
plt.yscale('log')
plt.axhline(b/func_log(-15,popt[0],popt[1]),c='k',ls='--')
plt.axvline(-15,c='k',ls='--')
plt.grid()
plt.show()






