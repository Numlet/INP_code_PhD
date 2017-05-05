# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:15:33 2016

@author: eejvt
"""
import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
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
from scipy import stats
from scipy.io import netcdf
from scipy.optimize import curve_fit
import scipy
import os
import psutil
process = psutil.Process(os.getpid())
print process.memory_info().rss

path='/nfs/a201/eejvt/CLIMATOLOGY/'
#class structured_year():
#    def __init__(self,data_values):
#        self.data_values=data_values

os.chdir(path)
years=['2001','2002','2003']#,'2004']
s={}
year=years[0]
for year in years:
        
    file_name='INP_feldext_'+year+'.nc'
    
    mb=netcdf.netcdf_file(path+file_name,'r')
    
    values_month=mb.variables['INP_feldspar'].data[15,30,40,40,jl.days_end_month[7]:jl.days_end_month[8]]
    print values_month
#%%
for imon in range(12):
    values_month=





















