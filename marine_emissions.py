# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 10:25:31 2015

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
from mpl_toolkits.basemap import Basemap





archive_directory='/nfs/a107/eejvt/'
project='SATELLITE/MERGED_GlobColour/CHL1/ftp.hermes.acri.fr/824041674/'
os.chdir(archive_directory+project)
#%%












#%%
'''
chlor_file='MY1DMM_CHLORA_2014-11-01_rgb_360x180.CSV'
chlor_data = np.genfromtxt(chlor_file, delimiter=',')
lats=np.linspace(90,-90,180)
lons=np.linspace(-180,180,360)
m=jl.plot(chlor_data,lon=lons,lat=lats,show=1,clevs=np.logspace(-4,2,15).tolist(),cmap=plt.cm.Reds,return_fig=1)
#m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
#    llcrnrlon=-180,urcrnrlon=180,lat_ts=20)
m.fillcontinents(color='green',lake_color='aqua')
'''