'''
plots INP parameterizations
'''





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
def demott(T,N):
    a_demott=5.94e-5
    b_demott=3.33
    c_demott=0.0264
    d_demott=0.0033
    Tp01=0.01-T
    dN_imm=1e3*a_demott*(Tp01)**b_demott*(N)**(c_demott*Tp01+d_demott)
    return dN_imm
def demott15(T,N):
    a_demott = 0.0
    b_demott = 1.25
    c_demott = 0.46
    d_demott = -11.6
    cf = 3.0

    dN_imm = 1.0e3/rho(k)*cf*(rho(k)*N*1e-6)**(a_demott*(273.16+T)+b_demott)*exp(c_demott*(273.16+T)+d_demott)
    return dN_imm


def meyers_param(T,units_cm=0):
    a=-0.639
    b=0.1296
    return np.exp(a+b*(100*(jl.saturation_ratio_C(T)-1)))#L-1
def meyers_CASIM(T):
    return np.exp(0.4108-0.262*T)
def func(x,a,b,c,d,e,f):
    return a-b*x+c*x**2+d*x**3+e*x**4+f*x**5

def func_log(x,a,b):
    return np.exp(a)*np.exp(-b * x)

from scipy import stats


from scipy.optimize import curve_fit

# def fit_to_INP(INP,function,temps=temps):
#
#     T=temps[:]
#
#     popt,pcov = curve_fit(func, T, INP)
#     perr = np.sqrt(np.diag(pcov))
#     ci = 0.95
#     pp = (1. + ci) / 2.
#     nstd = stats.norm.ppf(pp)
#     popt_up = popt + nstd * perr
#     popt_dw = popt - nstd * perr
#
#     INP_fitted=function(temps,*popt)
#     INP_low=function(T,*popt_dw)
#     INP_high=function(T,*popt_up)
#     return popt,pcov,INP_fitted,INP_low,INP_high

# popt,pcov,INP_fitted,_,_=fit_to_INP(np.log(INP_max),func)
# print INP_fitted
# print popt
temps=np.arange(-37,1,1)
temps=temps[::-1]

plt.figure()
# plt.plot(temps,INP_max,'o')
# plt.plot(temps,INP_mean,'o')
# plt.plot(temps,INP_min,'o')
plt.yscale('log')

# plt.plot(temps,np.exp(INP_fitted),'k')
plt.grid()
plt.plot(temps,meyers_param(temps)*1e3,'b',label='Meyers')
n05=56000*1e-6
n05_GLOMAP=21.26#cm-3 surface SO
n05_bug_meters=56000

n05_dust_high=0.16
n05_dust_low=0.03
plt.plot(temps[1:],demott(temps,n05)[1:],'r',label='DeMott 56000*1e-6 (cm-3)')
plt.plot(temps[1:],demott(temps,n05_dust_high)[1:],'k',label='DeMott2015 high')
plt.plot(temps[1:],demott(temps,n05_dust_low)[1:],'grey',label='DeMott2015 low')
plt.plot(temps[1:],demott(temps,n05_GLOMAP)[1:],'g',label='DeMott GLOMAP (21.26 cm-3)')
plt.plot(temps[1:],demott(temps,n05_bug_meters)[1:],'y',label='DeMott CASIM bug metres (56000cm-3)')
plt.plot(temps[1:],demott(temps,n05_bug_meters)[1:],'y',label='DeMott CASIM bug metres (56000cm-3)')

plt.legend()
