# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 10:57:52 2015

@author: eejvt
"""

import numpy as np
import Jesuslib as jl
import os
from scipy.io.idl import readsav
from glob import glob
import matplotlib.pyplot as plt
archive_directory='/nfs/a107/eejvt/'
project='MARINE_DOMINICK/'


os.chdir(archive_directory+project)


def marine_org_parameterization(T):
    a=12.23968
    b=-0.37848
    INP=np.exp(a+b*T)#[per gram]
    return INP

    
def INP_organic(org_mass,T):
    INP=marine_org_parameterization(T)*org_mass    
    return INP

m=jl.read_data('GEOS_chem')

def grid_dlon(points):
    
    res=np.linspace(0,360,points)
    return res
    
def grid_dlat(points):
    res=np.linspace(-90,90,points)
    return res
dlon=grid_dlon(72)
dlat=grid_dlat(46)
av=6.02214129e23
#multiply by    airdensity cmss=12, divide avo  oc_stm*airde
levels=np.logspace(1,5,15).tolist()

k=276.16680475006905
jl.plot(m.oc_pri[0,:,:,:,:].mean(axis=(2,3))*k,lat=m.lat,lon=m.lon,show=1,clevs=levels,cmap=plt.cm.YlGnBu) 
marine_organic=(m.oc_pri*k-m.oc_std*k)*1e-3*1.8
jl.plot(marine_organic[0,:,:,:,:].mean(axis=(2,3)),lat=m.lat,lon=m.lon,cmap=plt.cm.YlGnBu)


INP_mo=np.zeros((38,46,72))
for i in range(38):
    INP_mo[i,]=INP_organic(marine_organic[0,:,:,:,:].mean(axis=(2,3))*1e-6,-i)
    
jl.plot(INP_mo[15,:,:],lat=m.lat,lon=m.lon,cmap=plt.cm.YlGnBu,cblabel='$m^{-3}$')
B73=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/B73_m3.dat',delimiter="\t",skip_header=1)

def find_nearest_vector_index(array, value):
    n = [abs(i-value) for i in array]
    idx = n.index(min(n))
    return idx


#def fit_grid(data,lat,lon,lat_points_index=3,lon_points_index=4):
#    points=len(data[:,0])




