# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:38:28 2015

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

def read_data(simulation):
    s={}
    a=glob(simulation+'/*.sav')
    
    print a
    
    for i in range (len(a)):

        s=readsav(a[i],idict=s)
        
        print i, len(a)
        #np.save(a[i][:-4]+'python',s[keys[i]])
        print a[i]
    keys=s.keys()
    for j in range(len(keys)):
        print keys[j]
        print s[keys[j]].shape, s[keys[j]].ndim
    #variable_list=s.keys()
    #s=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav',idict=s)
    #s=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav',idict=s)
    return s
#%%
mace_head_latlon_index=[13,124]
amsterdam_island_latlon_index=[45,27]

'''
NO USAR ESTOS ARRAY, HACER SIMULACION DECENTE Y CAMBIAR
'''


ams_POM=np.array([ 0.30733973,  0.21654895,  0.13804628,  0.28574091,  0.45627275,
        0.34623331,  0.47585696,  0.15689419,  0.29635409,  0.25457847,
        0.24977866,  0.29562882])
#ug/m3
mace_POM=np.array([ 0.25145622,  0.2526086,  0.33133027,  0.32593864,  0.49346173,
        0.24990407,  0.25208488,  0.20352785,  0.30584994,  0.45496145,
        0.30464193,  0.26050338]) #ug/m3

'''
NO USAR ESTOS ARRAY, HACER SIMULACION DECENTE Y CAMBIAR
'''

#macemonts,march,apr,may,jun,oc,jan ug/m3
#Vignaty 2010
mace_wiom=np.array([0.08,np.nan,0.1,0.24,0.51,0.2,np.nan,np.nan,np.nan,0.24,np.nan,np.nan])
mace_wiom=np.array([0.08,0.088,0.1,0.24,0.51,0.2,0.183,0.155,0.122,0.24,0.062,0.7])#using data from rinaldi thanks to susannah. December interpolated
ams_obs=[0.06,0.11,0.09]

mace_OC=np.array([0.75,0.3,1.25,1.3,0.4,1.4,0.5,0.4,1.1,0.4,0.45,1.2,2,1.8,1.2,1.6,1.25,0.75,0.5,0.45,2.1,0.4,0.8,0.9])

#%%
#Spraklen 2008
ams_total_carbon=np.array([230,180,150,120,110,90,100,110,105,110,130,180])/1e3
mace_total_carbon=np.array([25,10,50,600,850,600,400,420,430,370,250,0])/1e3
#j.sciare
xs=np.linspace(0,11,12).tolist()
ams_wioc=np.array([189,122,96,68,64,41,49,47,64,54,64,130])/1e3
months_str=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
plt.plot(ams_wioc,'ro',label='obs')
plt.plot(ams_POM/1.9,label='GLOMAP')
plt.xticks(xs,months_str)
plt.ylabel('WIOC $\mu g /m^3$')
plt.title('Amsterdam Island')
plt.show()
#%%
plt.plot(mace_wiom,'ro',label='obs')
plt.plot(mace_POM,label='GLOMAP')
plt.xticks(xs,months_str)
plt.legend()
plt.ylabel('WIOM $\mu g /m^3$')
plt.title('Mace head')
plt.show()
#%%
plt.plot(mace_wiom,'ro')
plt.plot(mace_POM)
plt.show()

#%%




archive_directory='/nfs/a107/eejvt/'
project='MARINE_EMISSIONS/GLOMAP/'
os.chdir(archive_directory+project)
archive_directory='/nfs/a201/eejvt/'
project='MARINE_ORGANIC_EMISSIONS_BURROWS/FIRST_TRY'
os.chdir(archive_directory+project)
#%%








#%%
s=read_data('FIRST_TRY')
mo=s.tot_mc_ss_mm_mode[2,:,:,:,:]
jl.plot(mo[30,:,:,6],show=1)
total_carbon=mo+s.tot_mc_oc_mm
#%%
np.save('total_om',total_carbon)
np.save('mo',mo)
#%%
total_carbon=np.load('total_om.npy')
mo=np.load('mo.npy')
plt.title('Mace head organic matter')
plt.plot(total_carbon[30,13,124,:],label='GLOMAP total')
plt.plot(mo[30,13,124,:],label='GLOMAP mo')
plt.plot(mace_OC[0:12],'bo',label='2009')
plt.plot(mace_OC[12:],'ro',label='2010')
#plt.ylabel('Simulated ($cm^{-3}$)')
plt.ylabel('$\mu g/ m^3$')
plt.legend()
plt.savefig('Mace head organic matter')
plt.show()
#%%
r=np.corrcoef(s.tot_mc_oc_mm[30,13,124,:],mace_OC[0:12])
print r
r=np.corrcoef(total_carbon[30,13,124,:],mace_OC[12:])
print r 