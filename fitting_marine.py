# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:58:29 2015

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
from scipy.optimize import curve_fit
from scipy import stats

x=np.linspace(-27,-6,1000)
plt.plot(x,np.exp(11.2186+(-0.4459*x)),label='susannah')

'''
def func(x, a, b):
        return a * np.exp(-b * x)


# Read data.
data = np.genfromtxt('/nfs/a107/eejvt/INP_DATA/measured/marine_measured.dat',delimiter="\t")
#data=np.sort(data,axis=0)
x=data[:,0]
y=data[:,1]
# Define confidence interval.
ci = 0.95
# Convert to percentile point of the normal distribution.
# See: https://en.wikipedia.org/wiki/Standard_score
pp = (1. + ci) / 2.
# Convert to number of standard deviations.
nstd = stats.norm.ppf(pp)
print nstd

# Find best fit.
popt, pcov = curve_fit(func, x, y)
# Standard deviation errors on the parameters.
perr = np.sqrt(np.diag(pcov))
# Add nstd standard deviations to parameters to obtain the upper confidence
# interval.
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

# Plot data and best fit curve.
scatter(x, y)
#x = linspace(11, 23, 100)
plot(x, func(x, *popt), c='g', lw=2.)
#xscale('log')
#yscale('log')
plot(x, func(x, *popt_up), c='r', lw=2.)
plot(x, func(x, *popt_dw), c='r', lw=2.)
text(12, 0.5, '{}% confidence interval'.format(ci * 100.))    

show()
'''

