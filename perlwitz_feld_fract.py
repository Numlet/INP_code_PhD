# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:27:21 2016

@author: eejvt
"""

import numpy as np
import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
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
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import csv
import random
import datetime
from scipy.interpolate import interp1d
def read_ncdf(file_name):
    mb=netcdf.netcdf_file(file_name,'r') 
    return mb

#%%
levels=[i/100. for i in range(101) if not (i%5)]
levels=[i/200. for i in range(21) if not (i%2)]
levels=[i/2000. for i in range(401) if not (i%5)]
print levels
levels=np.arange(0,0.2,0.01).tolist()
#%%
mb=netcdf.netcdf_file('/nfs/a201/eejvt/mineral_fractions_perlwitz/mineralfractionsatemission_AMFmethod_NASAGISS_201412_4320x2160.nc','r')
lat=mb.variables['latitude'].data
lon=mb.variables['longitude'].data
fel_clay=np.copy(mb.variables['fracClayFeld'].data)
fel_silt1=np.copy(mb.variables['fracSilt1Feld'].data)
fel_silt2=np.copy(mb.variables['fracSilt2Feld'].data)
fel_silt3=np.copy(mb.variables['fracSilt3Feld'].data)
fel_silt4=np.copy(mb.variables['fracSilt4Feld'].data)
fel_clay[fel_clay==mb.variables['fracClayFeld'].missing_value]=0
#%%
list_4=['fracSilt4Calc', 'fracSilt4Feld', 'fracSilt4Feox', 'fracSilt4Gyps', 'fracSilt4Illi', 'fracSilt4Kaol', 'fracSilt4Quar', 'fracSilt4Smec']
list_names=['Calc', 'Feld', 'Feox', 'Gyps', 'Illi', 'Kaol', 'Quar', 'Smec']
a=0
n='4'
ns=['1','2','3','4']
for name in list_names:
    #a=0
    print name
    b=0
    a=a+mb.variables['fracClay'+name].data[1365,2200]
    for n in ns:    
        a=a+mb.variables['fracSilt'+n+name].data[1365,2200]
        b=b+mb.variables['fracSilt'+n+name].data[1365,2200]
        print 'total',a
        print name, b
#%%
l=[]
levels=np.linspace(0,1,21).tolist()
for mineral in list_names:
    total_clay=np.zeros(mb.variables['fracClay'+mineral].data.shape)
    for name in list_names:
        clay_mineral=np.copy(mb.variables['fracClay'+name].data)
        clay_mineral[clay_mineral==mb.variables['fracClay'+name].missing_value]=0
        total_clay=total_clay+clay_mineral
    fel_clay=np.copy(mb.variables['fracClay'+mineral].data)
    fel_clay[fel_clay==mb.variables['fracClay'+mineral].missing_value]=0
    print total_clay[1365,2201]
    
    fract_clay_feld_from_tot_dust=fel_clay/total_clay
    jl.plot(fract_clay_feld_from_tot_dust,title=mineral,lat=lat,lon=lon,clevs=levels,cmap=plt.cm.rainbow)
    l.append(fract_clay_feld_from_tot_dust)
total=np.zeros(l[0].shape)
for mm in l:
    total=total+mm
#jl.plot(fract_clay_feld_from_tot_dust,lat=lat,lon=lon)
jl.plot(total,lat=lat,lon=lon,clevs=[0,0.9,0.99,1.01,1.1,10])
print fract_clay_feld_from_tot_dust[1365,2201]
#%%
jl.plot(fract_clay_feld_from_tot_dust,lat=lat,lon=lon)#,clevs=levels)
#%%
total_silt=np.zeros(mb.variables['fracSilt1Feld'].data.shape)
fel_silt_total=np.zeros(mb.variables['fracSilt1Feld'].data.shape)
for name in list_names:
    for n in ns:
        silt_mineral=np.copy(mb.variables['fracSilt'+n+name].data)
        silt_mineral[silt_mineral==mb.variables['fracSilt'+n+name].missing_value]=0
        total_silt=total_silt+silt_mineral
for n in ns:
    fel_silt_n=np.copy(mb.variables['fracSilt'+n+'Feld'].data)
    fel_silt_n[fel_silt_n==mb.variables['fracSilt'+n+'Feld'].missing_value]=0
    fel_silt_total=fel_silt_total+fel_silt_n
    
fract_silt_feld_from_tot_dust=fel_silt_total/total_silt
#%%
jl.plot(fract_silt_feld_from_tot_dust,lat=lat,lon=lon)#,clevs=levels)
#%%
jl.plot(jl.congrid(fract_silt_feld_from_tot_dust,(64,128)))#,clevs=levels)

#%%
def interpolate_grid(var,big_lon,big_lat,sml_lon,sml_lat):
    big_lon=np.copy(big_lon)    
    big_lon[big_lon<0]=360+big_lon[big_lon<0]#doing lon positive
    lon_idx=np.zeros(big_lon.shape)
    lat_idx=np.zeros(big_lat.shape)

    for ilon in range(len(big_lon)):
        lon_idx[ilon]=jl.find_nearest_vector_index(sml_lon,big_lon[ilon])

    for ilat in range(len(big_lat)):
        lat_idx[ilat]=jl.find_nearest_vector_index(sml_lat,big_lat[ilat])

    var_sml=np.zeros((len(sml_lat),len(sml_lon)))
    ilon=0
    ilat=0
    for ilon in range(len(sml_lon)):
        print ilon
        for ilat in range(len(sml_lat)):
            print ilat

            lons=np.array(lon_idx==ilon)

            lats=np.array(lat_idx==ilat)
            total=0
            n_obs=0
            for intlons in range(len(lons)):
                for intlats in range(len(lats)):
                    if lons[intlons]:
                        if lats[intlats]:
                            if not np.isnan(var[intlats,intlons]):
                                total=total+var[intlats,intlons]
                                n_obs=n_obs+1
            if n_obs!=0:
                
                mean=total/n_obs
            else:
                mean=0
            var_sml[ilat,ilon]=mean
    return var_sml


#fract_silt_feld_from_tot_dust_glo=interpolate_grid(fract_silt_feld_from_tot_dust,lon,lat,jl.lon,jl.lat)

fract_clay_feld_from_tot_dust_glo=interpolate_grid(fract_clay_feld_from_tot_dust,lon,lat,jl.lon,jl.lat)


np.savetxt('/nfs/a201/eejvt/mineral_fractions_perlwitz/Silt_fraction_GLOMAP.txt',fract_silt_feld_from_tot_dust)
np.savetxt('/nfs/a201/eejvt/mineral_fractions_perlwitz/Clay_fraction_GLOMAP.txt',fract_clay_feld_from_tot_dust)
np.save('/nfs/a201/eejvt/mineral_fractions_perlwitz/Silt_fraction_GLOMAP',fract_silt_feld_from_tot_dust)
np.save('/nfs/a201/eejvt/mineral_fractions_perlwitz/Clay_fraction_GLOMAP',fract_clay_feld_from_tot_dust)



#%%

fel_silt1[fel_silt1==mb.variables['fracSilt1Feld'].missing_value]=0
fel_silt2[fel_silt2==mb.variables['fracSilt2Feld'].missing_value]=0
fel_silt3[fel_silt3==mb.variables['fracSilt3Feld'].missing_value]=0
fel_silt4[fel_silt4==mb.variables['fracSilt4Feld'].missing_value]=0

plt.plot(fel_silt1[:,2200],'o')
plt.plot(fel_silt2[:,2200],'o')
plt.plot(fel_silt3[:,2200],'o')
plt.plot(fel_silt4[:,2200],'o')
plt.plot((fel_silt4[:,2200]+fel_silt3[:,2200]+fel_silt2[:,2200]+fel_silt1[:,2200])/4,'^')


#%%
proportions_07=[0.65339480050819199, 0.26502493704082108, 0.073263146566154236, 0.008321576665577755]
proportions_17=[0.19439870671332321, 0.36058205732743681, 0.31755671837931526, 0.12753822404161372]
proportions_20=[0.14021933654335661, 0.32820507567757329, 0.353083156102247, 0.1786725067725341]
#proportions=proportions_17
#proportions=proportions_07
fel_silt_propotional_17=fel_silt1*proportions_17[0]+fel_silt2*proportions_17[1]+fel_silt3*proportions_17[2]+fel_silt4*proportions_17[3]
fel_silt_propotional_07=fel_silt1*proportions_07[0]+fel_silt2*proportions_07[1]+fel_silt3*proportions_07[2]+fel_silt4*proportions_07[3]
fel_silt_propotional_20=fel_silt1*proportions_20[0]+fel_silt2*proportions_20[1]+fel_silt3*proportions_20[2]+fel_silt4*proportions_20[3]

jl.plot(fel_silt_propotional_07/fel_silt_propotional_20,lat=lat,lon=lon,clevs=[0.1,0.49,0.53,0.8,0.95,1,1.2,1.5,2,10])
jl.plot(fel_silt_propotional_07,lat=lat,lon=lon,clevs=levels)
jl.plot(fel_silt_propotional_17-fel_silt_propotional_07,lat=lat,lon=lon,clevs=levels)
#jl.plot(fel_silt_propotional,lat=lat,lon=lon,clevs=levels)
#%%
jl.plot(fel_clay,lat=lat,lon=lon)#,clevs=levels)
jl.plot(fel_clay,lat=lat,lon=lon,clevs=levels)
jl.plot(fel_silt1,lat=lat,lon=lon,clevs=levels)
jl.plot(fel_silt2,lat=lat,lon=lon,clevs=levels)
jl.plot(fel_silt3,lat=lat,lon=lon,clevs=levels)
jl.plot(fel_silt4,lat=lat,lon=lon,clevs=levels)

fel_clay_glo=jl.congrid(fel_clay,(64,128))

jl.plot(fel_clay_glo,clevs=levels)

#%%

#SMF_clay=np.genfromtxt('/nfs/a201/eejvt/mineral_fractions_perlwitz/DUST_CLAY_FRAC_T42_PERLWITZ.dat')
SMF_clay=np.genfromtxt(jl.home_dir+'FELD_CLAY_FRAC_T42.dat')
SMF_silt=np.genfromtxt(jl.home_dir+'FELD_SILT_FRAC_T42.dat')
jl.plot(SMF_clay.reshape(128,64).swapaxes(0,1))
jl.plot(SMF_clay.reshape(128,64).swapaxes(0,1),clevs=levels)
jl.plot(SMF_silt.reshape(128,64).swapaxes(0,1))#,clevs=levels)
jl.plot((SMF_clay/SMF_silt).reshape(128,64).swapaxes(0,1))

plt.plot(SMF_silt.reshape(128,64).swapaxes(0,1)[:,0],'o')


limits=[4e-6,8e-6,16e-6]
jl.plot(SMF_clay.reshape(128,64).swapaxes(0,1),clevs=levels)
jl.plot(fract_clay_feld_from_tot_dust_glo,clevs=levels)

jl.plot(SMF_silt.reshape(128,64).swapaxes(0,1))#,clevs=levels)
jl.plot(fract_silt_feld_from_tot_dust_glo,clevs=levels)



np.savetxt('/nfs/a201/eejvt/mineral_fractions_perlwitz/FELD_CLAY_FRAC_T42_PERLWITZ.dat',fract_clay_feld_from_tot_dust_glo.swapaxes(0,1).reshape(8192),fmt='%.5f')
np.savetxt(jl.home_dir+'FELD_CLAY_FRAC_T42_PERLWITZ.dat',fract_clay_feld_from_tot_dust_glo.swapaxes(0,1).reshape(8192),fmt='%.5f')






np.savetxt('/nfs/a201/eejvt/mineral_fractions_perlwitz/FELD_SILT_FRAC_T42_PERLWITZ.dat',fract_silt_feld_from_tot_dust_glo.swapaxes(0,1).reshape(8192),fmt='%.5f')
np.savetxt(jl.home_dir+'FELD_SILT_FRAC_T42_PERLWITZ.dat',fract_silt_feld_from_tot_dust_glo.swapaxes(0,1).reshape(8192),fmt='%.5f')







fract_silt_feld_from_tot_dust_glo.swapaxes(0,1).reshape(8192)












