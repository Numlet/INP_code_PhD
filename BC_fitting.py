# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:14:41 2015

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
#%%
archive_directory='/nfs/a107/eejvt/'
project='BC_INP'
os.chdir(archive_directory+project)

data_ndec=np.genfromtxt('ndecane_selected.txt',delimiter='\t')

data_eug=np.genfromtxt('eugenol_selected.txt',delimiter='\t')

data=np.concatenate((data_eug,data_ndec),axis=0)
#%%

T=data[:,0]
F=data[:,1]
ns=data[:,2]
#ns=np.log(ns)
dns=data[:,3]
#dns=0
#def func(x, a, b):
#    return a * np.exp(-b * x)
def func(x,a,b):
    return a-b*x    
    
def func_log(x,a,b):
    return np.exp(a) * np.exp(-b * x)


popt,pcov = curve_fit(func, T, np.log(ns))
#popt=np.array([-18.59897567,   1.10249526])
#pcov=np.array([[ 0.50795402, -0.02496729],[-0.02496729,  0.00123657]])
perr = np.sqrt(np.diag(pcov))
ci = 0.95
pp = (1. + ci) / 2.
nstd = stats.norm.ppf(pp)
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

#ns[5]=1000000

'''
Function Ns=e^(A-B*T)
A=-18.60
B=1.10
errA=1.396
errB=0.068
95% confidance
'''
Tx=np.linspace(-24,-13,100)
ns_fitted=func_log(Tx,popt[0],popt[1])
ns_low=func_log(Tx,*popt_dw)
ns_high=func_log(Tx,*popt_up)

f = figure()
ax = f.add_subplot(111)

plt.errorbar(T,ns,yerr=dns,fmt='ro')#, capthick=2,ecolor='black')
plt.errorbar(data_eug[:,0],data_eug[:,2],yerr=data_eug[:,3],fmt='bo',label='Eugenol')#, capthick=2,ecolor='black')
plt.errorbar(data_ndec[:,0],data_ndec[:,2],yerr=data_ndec[:,3],fmt='ro',label='Ndecane')#, capthick=2,ecolor='black')
plt.title('Black Carbon Parameterization')

plt.plot(Tx,ns_fitted,'k-',lw=3)
plt.plot(Tx,ns_high,'k--')
plt.plot(Tx,ns_low,'k--')
plt.xlabel('$T(^oC)$')
plt.yscale('log')
plt.xlim(-24,-13)
plt.text(0.8, 0.95,'Function $n_s=e^{(A-B*T)}$', ha='center', va='center', transform=ax.transAxes)
plt.text(0.8, 0.9,'$A=%.6f\pm%.6f$'%(popt[0],nstd*perr[0]), ha='center', va='center', transform=ax.transAxes)
plt.text(0.8, 0.85,'$B=%.6f\pm%.6f$'%(popt[1],nstd*perr[1]), ha='center', va='center', transform=ax.transAxes)
plt.text(0.8, 0.8,'95% confidence interval', ha='center', va='center', transform=ax.transAxes)
plt.ylabel('$n_s(cm^{-2})$')
plt.legend(loc='lower left')
plt.savefig('Black_Carbon_param',format='png')
#plt.text(0.5, 0.5,' A=-18.60,B=1.10,errA=1.396,errB=0.068')

plt.show()

