# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:27:54 2015

@author: eejvt
"""

import numpy as np
import matplotlib.pyplot as plt
import fortranfile


lon=128
lat=64
date=np.empty((4),dtype='float')
rcat=np.empty(1)
dlon=np.empty(lon,dtype='float')
dlat=np.empty(lat,dtype='float')
inputfile='/nfs/a107/eejvt/chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_T42_L31_MS8_01012001_01022001.PDG'
#f=fortranfile.FortranFile(inputfile)
#total = f.readReals()
with open(inputfile,'r') as f:
    all_data = np.fromfile(f, dtype=np.float)
#f = open(inputfile,'rb')
#date = np.fromfile(f,dtype='float', count=4)
#rcat=np.fromfile(f,count=-1)
#lon= np.fromfile(f,dtype='float',count=lon)
#print dlon