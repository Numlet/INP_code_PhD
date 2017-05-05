# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 10:56:59 2015

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

plt.ion()
archive_directory='/nfs/a107/eejvt/'
project='MARINE_BURROWS/'
os.chdir(archive_directory+project)
def marine_org_parameterization(T):
    a=12.23968
    b=-0.37848   
    INP=np.exp(a+b*T)#[per gram]
    return INP
    
def INP_organic(org_mass,T):
    INP=marine_org_parameterization(T)*org_mass    
    return INP
    
    
rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
rhocomp =rhocomp*1e+3#ug/cm3 or g/m3 es lo mismo
'''
;           Sulphate    Carbon    1.4*BC   NaCl      SiO2,      SOC   Feldspar                               
    mm_aer = [0.098,    0.012,   0.0168 , 0.05844,    0.10,    0.0168 ,0.1] ;SOC was 0.15  ; molar masses in Kg/mol
    rhocomp  = [1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.];masss density of components in Kg/m3
'''
mb=netcdf.netcdf_file('sea_POC_mass_conc_ug_timavg.nc','r')
POC=mb.variables['POC_aqua']
POC=POC[0,:,:,:]*1e-6#g/m3
data = netcdf.netcdf_file('INP_marine_and_feldext.nc', 'r')
INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/Feldspar_distributions/INP_feld_ext_alltemps.npy').mean(axis=-1)*1e6#data.variables['INP_feldext'][:,:,:,:]
#INP_feldext=data.variables['INP_feldext'][:,:,:,:]
INP_marineorganics=data.variables['INP_marineorganics'][:,:,:,:]/1.9
INP_marineorganics=np.load('/nfs/a201/eejvt/COMPILATION_PAPER/DAILY_RUN/INP_marine_alltemps.npy').mean(axis=-1)#glomap#cm3
INP_marineorganics=np.load('/nfs/a201/eejvt//MARINE/THIRD_TRY/INP_marine_alltemps.npy').mean(axis=-1)
'''
INP_total= INP_marineorganics[:,89,:,:]*1e-3+INP_feldext[:,30,:,:,:].mean(axis=-1)
np.save('INP_mo_feldext_surface_L.npy',INP_total)
INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/Feldspar_distributions/INP_feld_ext_alltemps.npy')*1e3#data.variables['INP_feldext'][:,:,:,:]
jl.artic_plot(INP_feldext[20,30,:,:,5],clevs=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],cblabel='L-1')
jl.artic_plot(INP_marineorganics[20,89,:,:]*1e-3,clevs=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],cblabel='L-1')
jl.artic_plot(INP_marineorganics[20,89,:,:]*1e-3+INP_feldext[20,30,:,:,5],clevs=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],cblabel='L-1',title='INP -20, feldspar in june, marine year_mean')
'''
#%%
#lat=data.variables['lat'][:]
#lon=data.variables['lon'][:]
#pressure=data.variables['pressure_levs'][:,:,:]
lat=jl.lat
lon=jl.lon
pressure=jl.pressure
B73=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/B73_m3.dat',delimiter="\t",skip_header=1)
INP_total=INP_feldext+INP_marineorganics[:,:,:,:]
B73_feld=jl.obtain_points_from_data(INP_feldext,B73,plvs=31)
B73_tot=jl.obtain_points_from_data(INP_total,B73,plvs=31)
Rosinsky=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/data_by_campaing/rosinsky.dat',delimiter="\t")
rosdata=Rosinsky[Rosinsky[:,1]<-6]
rosdata[:,2]=rosdata[:,2]*1e6
ros_feld=jl.obtain_points_from_data(INP_feldext,rosdata,plvs=31)
ros_tot=jl.obtain_points_from_data(INP_total,rosdata,plvs=31)
ros_gulf_data=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/data_by_campaing/burrowspaper/rosinskygulf.dat',delimiter="\t")
ros_gulf_data[:,2]=ros_gulf_data[:,2]*1e6
ros_gulf_feld=jl.obtain_points_from_data(INP_feldext,ros_gulf_data,plvs=31)
ros_gulf_tot=jl.obtain_points_from_data(INP_total,ros_gulf_data,plvs=31)
ros_aus_data=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/data_by_campaing/burrowspaper/rosinskyaustralia.dat',delimiter="\t")
ros_aus_data[:,2]=ros_aus_data[:,2]*1e6
ros_aus_feld=jl.obtain_points_from_data(INP_feldext,ros_aus_data,plvs=31)
ros_aus_tot=jl.obtain_points_from_data(INP_total,ros_aus_data,plvs=31)


#%%
plt.figure()


plt.scatter(rosdata[:,2],ros_feld[:,0],marker='x',c='blue',alpha=1,label='Rosinsky pacific feldspar')
plt.scatter(rosdata[:,2],ros_tot[:,0],marker='o',c='Darkblue',edgecolors='none',alpha=0.7,s=50,label='Rosinsky pacific feldspar+marine')
plt.scatter(ros_gulf_data[:,2],ros_gulf_feld[:,0],marker='s',c='black',s=10,alpha=1,label='Rosinsky gulf of MX feldspar')
plt.scatter(ros_gulf_data[:,2],ros_gulf_tot[:,0],marker='o',c='grey',alpha=1,s=50,label='Rosinsky gulf of MX feldspar+marine')
plt.scatter(ros_aus_data[:,2],ros_aus_feld[:,0],marker='x',c='green',alpha=1,label='Rosinsky southern feldspar')
plt.scatter(ros_aus_data[:,2],ros_aus_tot[:,0],marker='o',c='green',alpha=1,s=50,label='Rosinsky southern feldspar+marine')
x=np.linspace(1e-9,1e8,100)
plt.scatter(B73[:,2],B73_feld[:,0],marker='x',c='red',alpha=1,label='B73 feldspar')
plt.scatter(B73[:,2],B73_tot[:,0],marker='o', edgecolors='none',c='DarkRed',alpha=0.7,s=50,label='B73 feldspar+marine')
plt.ylabel('Simulated ($m^{-3}$)')
plt.xlabel('Observed ($m^{-3}$)')
plt.plot(x,x,'k-')
#plt.legend()
plt.plot(x,10*x,'k--')
plt.plot(x,0.1*x,'k--')
plt.xlim(1e-3,x[-1])
plt.ylim(1e-7,1e6)
plt.xscale('log')
plt.yscale('log')

#plt.savefig('comparison_feldintegrated',format='png')
plt.show()



#plt.close()
#%%
simulated_INP_feld=np.concatenate((B73_feld[:,0],ros_feld[:,0],ros_gulf_feld[:,0]))
simulated_INP_total=np.concatenate((B73_tot[:,0],ros_tot[:,0],ros_gulf_tot[:,0]))

B73_theo=np.array((B73[:,2],B73_feld[:,0],B73_tot[:,0],B73[:,1]))
B73_theo=B73_theo.T
np.savetxt('B73_theo.dat',B73_theo,delimiter=',')



Ross_pacific_theo=np.array((rosdata[:,2],ros_feld[:,0],ros_tot[:,0],rosdata[:,1]))
Ross_pacific_theo=Ross_pacific_theo.T
np.savetxt('Ross_pacific_theo.dat',Ross_pacific_theo,delimiter=',')


Ross_gulf_theo=np.array((ros_gulf_data[:,2],ros_gulf_feld[:,0],ros_gulf_tot[:,0],ros_gulf_data[:,1]))
Ross_gulf_theo=Ross_gulf_theo.T
np.savetxt('Ross_gulf_theo.dat',Ross_gulf_theo,delimiter=',')

Ross_aus_theo=np.array((ros_aus_data[:,2],ros_aus_feld[:,0],ros_aus_tot[:,0],ros_aus_data[:,1]))
Ross_aus_theo=Ross_aus_theo.T
np.savetxt('Ross_aus_theo.dat',Ross_aus_theo,delimiter=',')



largedata=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/IN_obs_2.dat',delimiter=",",skip_header=1)


largedata_feld=jl.obtain_points_from_data(INP_feldext*1e-6,largedata,plvs=31)
largedata_tot=jl.obtain_points_from_data(INP_total*1e-6,largedata,plvs=31)

plt.figure()

x=np.linspace(1e-9,1e8,100)
plt.scatter(largedata[:,2],largedata_feld[:,0],c=largedata[:,1],edgecolors='none')
#plt.scatter(largedata[:,2],largedata_tot[:,0], edgecolors='none',c=largedata[:,1])
plt.ylabel('Simulated ($cm^{-3}$)')
plt.xlabel('Observed ($cm^{-3}$)')
plt.colorbar()
plt.plot(x,x,'k-')
plt.legend()
plt.plot(x,10*x,'k--')
plt.plot(x,0.1*x,'k--')
plt.xlim(1e-9,x[-1])
plt.ylim(1e-12,1e3)
plt.xscale('log')
plt.yscale('log')