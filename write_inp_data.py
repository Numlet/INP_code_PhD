# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:28:16 2016

@author: eejvt
"""

'''
Code to save INP data in the same manner as usual

'''

import numpy.ma as ma
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import numpy as np
from random import random
from random import uniform
import random
import numpy.random as nprnd
#import Jesuslib as jl
import matplotlib.pyplot as plt
from glob import glob
import os

os.chdir('/nfs/a107/eejvt/INP_DATA')
#%%
name_file='Conen_chaumont'
header='campaign	temp	IN conc cm-3	lat	lon	alt hpa	notes\n'
tab='\t'
#temp=
f = open(name_file+'.dat', 'w')
f.write(header)
campaing='Conen_chaumont'

ts=[-7,-8,-9,-10,-11,-12]


inp6=np.genfromtxt(jl.home_dir+'data6.dat')*1e-6
inp7=np.genfromtxt(jl.home_dir+'data7.dat')*1e-6
inp8=np.genfromtxt(jl.home_dir+'data8.dat')*1e-6
inp9=np.genfromtxt(jl.home_dir+'data9.dat')*1e-6
inp10=np.genfromtxt(jl.home_dir+'data10.dat')*1e-6
inp11=np.genfromtxt(jl.home_dir+'data11.dat')*1e-6
inp12=np.genfromtxt(jl.home_dir+'data12.dat')*1e-6
lat=47
lon=6.58
hpa=900


t=-6
for i_inp in inp6:
    if np.isnan(i_inp):
        continue
    if i_inp<=0:
        continue
    line=name_file+tab+str(t)+tab+str(i_inp)+tab+str(lat)+tab+str(lon)+tab+str(hpa)+tab+'BACCHUS_data\n'
    print line
    f.write(line)
t=-7
for i_inp in inp7:
    if np.isnan(i_inp):
        continue
    if i_inp<=0:
        continue
    line=name_file+tab+str(t)+tab+str(i_inp)+tab+str(lat)+tab+str(lon)+tab+str(hpa)+tab+'BACCHUS_data\n'
    print line
    f.write(line)

t=-8
for i_inp in inp8:
    if np.isnan(i_inp):
        continue
    if i_inp<=0:
        continue
    line=name_file+tab+str(t)+tab+str(i_inp)+tab+str(lat)+tab+str(lon)+tab+str(hpa)+tab+'BACCHUS_data\n'
    print line
    f.write(line)
t=-9
for i_inp in inp9:
    if np.isnan(i_inp):
        continue
    if i_inp<=0:
        continue
    line=name_file+tab+str(t)+tab+str(i_inp)+tab+str(lat)+tab+str(lon)+tab+str(hpa)+tab+'BACCHUS_data\n'
    print line
    f.write(line)
t=-10
for i_inp in inp10:
    if np.isnan(i_inp):
        continue
    if i_inp<=0:
        continue
    line=name_file+tab+str(t)+tab+str(i_inp)+tab+str(lat)+tab+str(lon)+tab+str(hpa)+tab+'BACCHUS_data\n'
    print line
    f.write(line)
t=-11
for i_inp in inp11:
    if np.isnan(i_inp):
        continue
    if i_inp<=0:
        continue
    line=name_file+tab+str(t)+tab+str(i_inp)+tab+str(lat)+tab+str(lon)+tab+str(hpa)+tab+'BACCHUS_data\n'
    print line
    f.write(line)
t=-12
for i_inp in inp12:
    if np.isnan(i_inp):
        continue
    if i_inp<=0:
        continue
    line=name_file+tab+str(t)+tab+str(i_inp)+tab+str(lat)+tab+str(lon)+tab+str(hpa)+tab+'BACCHUS_data\n'
    print line
    f.write(line)
f.close()
#%%
data=np.genfromtxt(jl.INP_data_dir+'KAD_Israel.csv',delimiter=',',skip_header=2)
temps=data[:,2]
inp_data=data[:,3]*1e-3#cm-3
lat=32
lon=34.5
hpa=1000
name_file='KAD_Israel_myformat'
header='campaign	temp	IN conc cm-3	lat	lon	alt hpa	notes\n'
tab='\t'
#temp=
f = open(name_file+'.dat', 'w')
f.write(header)
campaing='Tel_aviv'

for i in range(len(inp_data)):
    if np.isnan(inp_data[i]):
        continue
    if np.isnan(temps[i]):
        continue
    if inp_data[i]<=0:
        continue
    line=name_file+tab+str(temps[i])+tab+str(inp_data[i])+tab+str(lat)+tab+str(lon)+tab+str(hpa)+tab+'BACCHUS_data\n'
    print line
    
    f.write(line)
possible_temps=np.linspace(-30,0,31)
for t in possible_temps:
    values=[temps==t]
    inp_mean=inp_data[values].mean()
    print t, inp_mean
    line=name_file+'_mean'+tab+str(t)+tab+str(inp_mean)+tab+str(lat)+tab+str(lon)+tab+str(hpa)+tab+'BACCHUS_data\n'
    if not np.isnan(inp_mean):
        f.write(line)


f.close()
#%%
data=np.genfromtxt(jl.INP_data_dir+'KAD_South Pole.csv',delimiter=',',skip_header=3)
temps=data[:,2]
inp_data=data[:,3]*1e-3#cm-3
lat=-89.6
lon=-92
hpa=1000
name_file='KAD_South_Pole_myformat'
header='campaign	temp	IN conc cm-3	lat	lon	alt hpa	notes\n'
tab='\t'
#temp=
f = open(name_file+'.dat', 'w')
f.write(header)
campaing='KAD_South Pole'



for i in range(len(inp_data)):
    if np.isnan(inp_data[i]):
        continue
    if np.isnan(temps[i]):
        continue
    if inp_data[i]<=0:
        continue
    line=name_file+tab+str(temps[i])+tab+str(inp_data[i])+tab+str(lat)+tab+str(lon)+tab+str(hpa)+tab+'BACCHUS_data\n'
    print line
    f.write(line)
possible_temps=np.linspace(-30,0,31)
for t in possible_temps:
    values=[temps==t]
    inp_mean=inp_data[values].mean()
    print t, inp_mean
    line=name_file+'_mean'+tab+str(t)+tab+str(inp_mean)+tab+str(lat)+tab+str(lon)+tab+str(hpa)+tab+'BACCHUS_data\n'
    if not np.isnan(inp_mean):
        f.write(line)


f.close()




