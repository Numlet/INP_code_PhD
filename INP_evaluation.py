# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:22:27 2015

@author: eejvt
"""

import numpy as np
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import imp
imp.reload(jl)
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

class vector():
    '''
    array in shape [temps, levels, lat, lon, months]
    '''
    def __init__(self, name, array, lat, lon, pressures, temps=np.arange(-37,1)[::-1], months=np.arange(1,13)):
        self.name=name
        self.array=array
        self.lat=lat
        self.lon=lon
        self.pressures=pressures
        self.temps=temps
        self.months=months
        if len(self.lat)!=len(self.array[0,0,:,0,0]):
            print 'lat shape' ,len(self.lat)
            print 'array shape' ,self.array.shape
            raise NameError('latitude and array do not have the same lenght')
        
        if len(self.lon)!=len(self.array[0,0,:,0,0]):
            print 'lon shape' ,len(self.lon)
            print 'array shape' ,self.array.shape
            raise NameError('longitude and array do not have the same lenght')
        
        if len(self.lat)!=len(self.array[0,0,0,:,0]):
            print 'lat shape' ,len(self.lat)
            print 'array shape' ,self.array.shape
            raise NameError('latitude and array do not have the same lenght')
        if self.pressures.shape!=self.array[0,:,:,:,:].shape:
            print 'pressures shape' ,self.pressures.shape
            print 'array shape' ,self.array.shape
            raise NameError('pressure and array do not have the same shape')
        


INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/Feldspar_distributions/INP_feld_ext_alltemps.npy')*1e6 #m3
pressures_GLOMAP=readsav('/nfs/a107/eejvt/JB_TRAINING/WITHICE)











