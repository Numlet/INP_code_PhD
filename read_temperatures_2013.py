# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:01:23 2015

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
from mpl_toolkits.basemap import Basemap
import datetime
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
import subprocess
import struct

archive_directory='/nfs/a107/eejvt/'
project='TEMPERATURES/OUTPUT'
os.chdir(archive_directory+project)
lat=64
lon=128
lats=np.zeros(lat)
grid=np.zeros((lat,lon))
#dt=np.dtype()
grid=np.empty((lat,lon), dtype='float32')
f= open('era_interim_t_20131100.dat',mode='rb')
recl = np.zeros(1,dtype=np.uint32)

grid = np.fromfile(f, dtype='float32', count=1000)


