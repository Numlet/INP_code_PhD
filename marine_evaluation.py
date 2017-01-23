# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:30:22 2015

@author: eejvt
"""

import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
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


mace_wiom=np.array([0.08,0.088,0.1,0.24,0.51,0.2,0.183,0.155,0.122,0.24,0.062,0.07])
ams_wioc=np.array([189,122,96,68,64,41,49,47,64,54,64,130])/1e3
ams_wiom=ams_wioc*1.9
point_reyes_wiom=np.array([np.nan,0.177429,np.nan,0.053704,0.098196,0.281652,0.01127,0.56441,0.220598,np.nan,np.nan,np.nan])

mace_head_latlon_index=[13,124]
amsterdam_island_latlon_index=[45,27]
point_reyes_latlon_index=[18,84]
months_str=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

class marine_organic_parameterization():
    def __init__(self,name,location=None,array_surface=None):
        self.name=name
        if array_surface==None:
            s=jl.read_data(location)
            self.array_surface=s.tot_mc_ss_mm_mode[2,30,:,:,:]
        else:
            self.array_surface=array_surface
        if self.array_surface.shape[-1]==12:
            self.wiom_mace=self.array_surface[mace_head_latlon_index[0],mace_head_latlon_index[1],:]
            self.wiom_ams=self.array_surface[amsterdam_island_latlon_index[0],amsterdam_island_latlon_index[1],:]
            self.wiom_reyes=self.array_surface[point_reyes_latlon_index[0],point_reyes_latlon_index[1],:]
        else:
            self.wiom_mace=self.array_surface[:,mace_head_latlon_index[0],mace_head_latlon_index[1]]
            self.wiom_ams=self.array_surface[:,amsterdam_island_latlon_index[0],amsterdam_island_latlon_index[1]]
            self.wiom_reyes=self.array_surface[:,point_reyes_latlon_index[0],point_reyes_latlon_index[1]]

archive_directory='/nfs/a201/eejvt/'
project='MARINE_PARAMETERIZATION'
os.chdir(archive_directory+project)
#s=jl.read_data('MO_GANTT_SR')
#ss=s.tot_mc_ss_mm_mode[2,:,:,:,:]
#%%
directory='/nfs/a201/eejvt/HUMMEL/'
file_name='NFPTAERO.cam.h0.OM_AC.lev29.2000-YA.nc'
os.chdir(directory)

mb=netcdf.netcdf_file(directory+file_name,'r')
from scipy.interpolate import interpn

air_density_IUPAC=1.2754#kg/m3 IUPAC

hummel = type('test', (), {})()
hummel.name='hummel'
hummel.wiom_mace=np.zeros(12)
hummel.wiom_ams=np.zeros(12)
hummel.wiom_reyes=np.zeros(12)

for month in range(12):
    data_values=mb.variables['OM_AC'][month,0,:,:]*air_density_IUPAC*1e9#ug/m3

    grid_x=mb.variables['lat'].data
    grid_y=mb.variables['lon'].data
    points=[jl.mace_head_latlon_values,jl.amsterdam_island_latlon_values,jl.point_reyes_latlon_values]
    hummel.wiom_mace[month],hummel.wiom_ams[month],hummel.wiom_reyes[month]=interpn((grid_x,grid_y),data_values,points)

GANTT_SR=marine_organic_parameterization('GANTT_SR',archive_directory+project+'/'+'MO_GANTT_SR')

#%%
archive_directory='/nfs/a107/eejvt/'
project='JB_TRAINING'
os.chdir(archive_directory+project)
#s=jl.read_data('WITH_ICE_SCAV2')
#ss=s.tot_mc_ss_mm_mode[2,:,:,:,:]

sea_salt=marine_organic_parameterization('Acc mode sea-salt surface',archive_directory+project+'/'+'WITH_ICE_SCAV2')
#jl.grid_earth_map(ss[30,:,:])

archive_directory='/nfs/a201/eejvt/'
project='MARINE_PARAMETERIZATION'
os.chdir(archive_directory+project)
#s=jl.read_data('THIRD_TRY')
#s=jl.read_data('MO_PORTION_SS')
#mo=s.tot_mc_ss_mm_mode[2,:,:,:,:]
#mb=netcdf.netcdf_file('/nfs/a201/eejvt/MARINE/FIFTH_TRY/ntraer30_processed_data.nc','r')
#mo=mb.variables['NaCl_mode'].data[:,:,:,2,:]
MO_PORTION_SS=marine_organic_parameterization('MO_PORTION_SS',archive_directory+project+'/'+'MO_PORTION_SS')



archive_directory='/nfs/a201/eejvt/'
project='MARINE_PARAMETERIZATION'
os.chdir(archive_directory+project)
#s=jl.read_data('THIRD_TRY')
#s=jl.read_data('FOURTH_TRY')
#mo=s.tot_mc_ss_mm_mode[2,:,:,:,:]
#mb=netcdf.netcdf_file('/nfs/a201/eejvt/MARINE/FIFTH_TRY/ntraer30_processed_data.nc','r')
#mo=mb.variables['NaCl_mode'].data[:,:,:,2,:]
my_param=marine_organic_parameterization('Mine',archive_directory+project+'/'+'FOURTH_TRY')

archive_directory='/nfs/a201/eejvt/'
project='MARINE'
os.chdir(archive_directory+project)
#s=jl.read_data('THIRD_TRY')
#s=jl.read_data('FOURTH_TRY')
#mo=s.tot_mc_ss_mm_mode[2,:,:,:,:]
rinaldi13_no_wind=marine_organic_parameterization('Rinaldi13_no_wind',archive_directory+project+'/'+'FOURTH_TRY')

archive_directory='/nfs/a201/eejvt/'
project='MARINE'
os.chdir(archive_directory+project)
#s=jl.read_data('THIRD_TRY')
#s=jl.read_data('FIFTH_TRY')
#mo=s.tot_mc_ss_mm_mode[2,:,:,:,:]
rinaldi13=marine_organic_parameterization('Rinaldi13',archive_directory+project+'/'+'FIFTH_TRY')
'''
archive_directory='/nfs/a107/eejvt/'
project='MARINE_EMISSIONS/GLOMAP'
os.chdir(archive_directory+project)
s=jl.read_data('FIRST_TRY')
mo=s.tot_mc_ss_mm_mode[2,:,:,:,:]
rinaldi13=marine_organic_parameterization('Rinaldi13',mo[30,])
'''
archive_directory='/nfs/a201/eejvt/'
project='MARINE_ORGANIC_EMISSIONS_BURROWS'
os.chdir(archive_directory+project)
POM_mm=np.load('burrows_2013_POM.npy')
#s=jl.read_data('THIRD_TRY')
#mo=s.tot_mc_ss_mm_mode[2,:,:,:,:]
burrows2013=marine_organic_parameterization('Burrows13',array_surface=POM_mm[:,89,])
burrows2014=marine_organic_parameterization('Burrows14',archive_directory+project+'/'+'THIRD_TRY')
'''
model_wiom_mace=POM_mm[:,89,mace_head_latlon_index[0],mace_head_latlon_index[1]]
model_wiom_ams=POM_mm[:,89,amsterdam_island_latlon_index[0],amsterdam_island_latlon_index[1]]
model_wiom_mace=mo[30,mace_head_latlon_index[0],mace_head_latlon_index[1],:]
model_wiom_ams=mo[30,amsterdam_island_latlon_index[0],amsterdam_island_latlon_index[1],:]
'''
#%%
plt.figure()
plt.title('Mace Head WIOM')
plt.plot(hummel.wiom_mace,label=hummel.name)
plt.plot(GANTT_SR.wiom_mace,label=GANTT_SR.name)
#plt.plot(MO_PORTION_SS.wiom_mace,label=MO_PORTION_SS.name)
plt.plot(my_param.wiom_mace,label=my_param.name)
#plt.plot(rinaldi13_no_wind.wiom_mace,label=rinaldi13_no_wind.name)
plt.plot(rinaldi13.wiom_mace,label=rinaldi13.name)
#plt.plot(burrows2013.wiom_mace,label=burrows2013.name)
#plt.plot(burrows2014.wiom_mace,label=burrows2014.name)
plt.plot(mace_wiom,'bo',label='Observations')
plt.plot(sea_salt.wiom_mace,'k--',label=sea_salt.name)
plt.ylabel('$\mu g/ m^3$')
plt.xlabel('Month')
plt.xticks(np.arange(12),months_str)
plt.legend()
plt.savefig('Mace Head WIOM')
plt.show()
plt.figure()
plt.title('Amsterdam island WIOM')
plt.plot(hummel.wiom_ams,label=hummel.name)
plt.plot(GANTT_SR.wiom_ams,label=GANTT_SR.name)
#plt.plot(MO_PORTION_SS.wiom_ams,label=MO_PORTION_SS.name)
plt.plot(my_param.wiom_ams,label=my_param.name)
#plt.plot(rinaldi13_no_wind.wiom_ams,label=rinaldi13_no_wind.name)
plt.plot(rinaldi13.wiom_ams,label=rinaldi13.name)
#plt.plot(burrows2013.wiom_ams,label=burrows2013.name)
#plt.plot(burrows2014.wiom_ams,label=burrows2014.name)
plt.plot(sea_salt.wiom_ams,'k--',label=sea_salt.name)
plt.plot(ams_wiom,'bo',label='Observations')
plt.ylabel('$\mu g/ m^3$')
plt.xlabel('Month')
plt.xticks(np.arange(12),months_str)
plt.legend()
plt.savefig('Amsterdam island WIOM')
plt.show()
plt.figure()
plt.title('Point Reyes WIOM')
plt.plot(hummel.wiom_reyes,label=hummel.name)
plt.plot(GANTT_SR.wiom_reyes,label=GANTT_SR.name)
#plt.plot(MO_PORTION_SS.wiom_reyes,label=MO_PORTION_SS.name)
plt.plot(my_param.wiom_reyes,label=my_param.name)
#plt.plot(rinaldi13_no_wind.wiom_reyes,label=rinaldi13_no_wind.name)
plt.plot(rinaldi13.wiom_reyes,label=rinaldi13.name)
#plt.plot(burrows2013.wiom_reyes,label=burrows2013.name)
#plt.plot(burrows2014.wiom_reyes,label=burrows2014.name)
plt.plot(sea_salt.wiom_reyes,'k--',label=sea_salt.name)
plt.plot(point_reyes_wiom,'bo',label='Observations')
plt.ylabel('$\mu g/ m^3$')
plt.xlabel('Month')
plt.xticks(np.arange(12),months_str)
plt.legend()
plt.savefig('Point Reyes WIOM')
plt.show()


#%%


plt.figure()
plt.plot(ams_wiom,my_param.wiom_ams,'ro')
plt.plot(mace_wiom,my_param.wiom_mace,'ro')
plt.plot(np.linspace(0.01,1),np.linspace(0.01,1),'k-')
plt.plot(np.linspace(0.01,1),np.linspace(0.01,1)*2.,'k--')
plt.plot(np.linspace(0.01,1),np.linspace(0.01,1)/2.,'k--')
plt.xlim(0.01,1)
plt.ylim(0.01,1)
plt.xlabel('Observed WIOM $(\mu g/ m^3)$')
plt.ylabel('Modelled WIOM $(\mu g/ m^3)$')
#plt.xscale('log')
#plt.yscale('log')
plt.savefig('/nfs/see-fs-01_users/eejvt/marine_parameterization/models/121.png')
#plt.savefig('121 Hummel')
plt.show()



#%%
'''
plt.title('INP Feldspar simulated using GLOMAP compared with observations at Cape Verde')
plt.plot(INP_feldext[15,30,26,8,:],'bo',label='-15C')
plt.axhline(5,c='b',ls='--',label='observed -15')
plt.axhline(500,c='b',ls='--')
plt.plot(INP_feldext[10,30,26,8,:],'ro',label='-10C')
plt.axhline(0.1,c='r',ls='--',label='observed -10')
plt.axhline(7,c='r',ls='--')
plt.ylabel('$m^3$',fontsize=20)
plt.xlabel('Month')
plt.yscale('log')
plt.ylim(0.01,10000)
plt.xticks(np.arange(12),months_str)
plt.legend(loc="best")
plt.show()
'''


#%%
full_path='/nfs/a201/eejvt/MARINE_ORGANIC_EMISSIONS_BURROWS/THIRD_TRY/'
full_path='/nfs/a201/eejvt/MARINE/FOURTH_TRY/'
name_file='masprimssaccsol'
months_str=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
months_str_upper_case=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ss_emissions=np.zeros((64,128,12))
i=0
for month in months_str_upper_case:
    s=readsav(full_path+name_file+'_'+month+'.sav')
    #print s.masprimssaccsol[30,].shape
    print month

    ss_emissions[:,:,i]=s.masprimssaccsol[30,:,:]
    i=i+1
jl.grid_earth_map(ss_emissions)
