# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:29:13 2016

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
from scipy.interpolate import interpn
directory='/nfs/see-fs-02_users/amtgwm/tex/obsdata_various/UMiami/WoodwardData/'

file_means='dust_UMiami_Stephanie_monthlymean.dat'
file_std='dust_UMiami_Stephanie_monthlystdev.dat'
archive_directory='/nfs/a107/eejvt/'
project='JB_TRAINING'
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
import glob as glob
os.chdir(archive_directory+project)


rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3

s=jl.read_data('WITH_ICE_SCAV2')


dust_concentration=s.tot_mc_feldspar_mm_mode[:,:,:,:,:]+s.tot_mc_dust_mm_mode[:,:,:,:,:]
dust_concentration_total_surface_mm=dust_concentration.sum(axis=0)[30,]

dust_obs=np.genfromtxt(directory+file_means)
dust_std_obs=np.genfromtxt(directory+file_std)
lons_observations=dust_obs[:,2]
lons_observations[lons_observations<0]=360+lons_observations[lons_observations<0]
lats_observations=dust_obs[:,3]
values=np.copy(dust_obs[:,7:])
std=np.copy(dust_std_obs[:,:])
values[values==-1]=np.nan
std[std==0]=np.nan
#%%
data_points_error=[]
data_points=[]
simulated_points=[]
data_points_std=[]
lons=[]
lats=[]
bias=[]
bias_fraction=[]
for istation in range(len(lons_observations)):
    for imon in range(12):
        grid_x=jl.lat
        grid_y=jl.lon
        point=[lats_observations[istation],lons_observations[istation]]
        
        '''remember that interpn() just works if the latitudes are ascending so we have to invert them if using GLOMAP '''        
        point[0]=point[0]*(-1)
        if np.isnan(values[istation,imon]):
            print values[istation,imon]
            continue        
        if np.isnan(std[istation,imon]):
            continue        
            print values[istation,imon]
        sim_point=interpn(((-1)*grid_x,grid_y),dust_concentration_total_surface_mm[:,:,imon],point)[0]
        simulated_points.append(sim_point)
        data_points.append(values[istation,imon])
        data_points_std.append(std[istation,imon])
        lats.append(lats_observations[istation])
        lons.append(lons_observations[istation])
        bias.append(sim_point-values[istation,imon])
        bias_fraction.append(sim_point/values[istation,imon])
data_points_error=np.array(data_points_error)
data_points=np.array(data_points)
lons=np.array(lons)
lats=np.array(lats)
data_points_std=np.array(data_points_std)
#%%

cmap=plt.cm.RdBu
data_points_error_negative=-data_points_error
#data_points_error_negative[data_points-data_points_error<0]=0
#plt.errorbar(data_points,simulated_points,
#                 xerr=[data_points+data_points_std*1-data_points,data_points-data_points-data_points_std*1.],
#                linestyle="None",c='k',zorder=0)
plot=plt.scatter(data_points,simulated_points)#,c=data_points,cmap=cmap)



plt.ylabel('Simulated ($\mu g/m^{-3}$)')
plt.xlabel('Observed ($\mu g/m^{-3}$)')

if np.min(simulated_points)>np.min(data_points):
    min_plot=np.min(data_points)
else:
    min_plot=np.min(simulated_points)

if np.max(simulated_points)<np.max(data_points):
    max_plot=np.max(data_points)
else:
    max_plot=np.max(simulated_points)

min_val=1e-2
max_val=1e1
minx=np.min(min_val)
maxx=np.max(max_val)
miny=np.min(min_val)
maxy=np.max(max_val)

x=np.linspace(0.1*min_plot,10*max_plot,100)
#global x     
#r=np.corrcoef(np.log(data_points[:,2]),np.log(simulated_points[:,0]))
#print r
#rmsd=RMSD(data_points[:,2],simulated_points[:,0])
#plt.title('R=%f RMSD=%f'%(r[0,1],rmsd))
plt.plot(x,x,'k-')
plt.plot(x,10*x,'k--')
plt.plot(x,10**1.5*x,'k-.')
plt.plot(x,0.1*x,'k--')
plt.plot(x,10**(-1.5)*x,'k-.')
plt.ylim(miny*0.1,maxy*10)
plt.xlim(minx*0.1,maxx*10)
plt.xscale('log')
plt.yscale('log')
plt.title('Dust Concentrations')
plt.show()


#%%
import random
from mpl_toolkits.basemap import Basemap

fig=plt.figure(figsize=(25, 10))
m = fig.add_subplot(1,1,1)
m = Basemap(projection='cyl',lon_0=0)
m.drawcoastlines()
m.drawcountries()


lons[lons>180]=lons[lons>180]-360


valid=[(lons[:]>-180)&(lons[:]<180)&(lats[:]>-90) & (lats[:]<90)]
#xx,yy=lons[valid],lats[valid]
xx,yy = m(np.copy(lons),np.copy(lats))
print xx.max(),xx.min(),yy.max(),yy.min(),xx.shape,yy.shape
print any(np.isnan(xx)),any(np.isnan(xx))

rand_size=10
for i in range(len(xx)):
    xx[i]=xx[i]+(random.random()-0.5)*rand_size
for i in range(len(yy)):
    yy[i]=yy[i]+(random.random()-0.5)*rand_size

#m.scatter(xx,yy,c=INP_obs[:,1],s=250,marker='^',vmin=-35, vmax=0,cmap=plt.cm.RdBu_r)#,label=camp.name)
#m.scatter(xx[np.abs(bias_mason[:])>1.5],yy[np.abs(bias_mason[:])>1.5],c=bias_mason[:][np.abs(bias_mason[:])>1.5],s=250,marker='^',vmin=-5, vmax=5,cmap=plt.cm.RdBu_r)#,label=camp.name)
m.scatter(xx,yy,s=250,c=np.log10(bias_fraction),marker='^',cmap=plt.cm.RdBu_r,vmin=-1, vmax=1)#,label=camp.name)

cb=plt.colorbar()
#cb.set_label('T  $^oC$')

#%%
fig=plt.figure(figsize=(25, 10))
lat=40
lon=-40
m = fig.add_subplot(1,1,1)
m = Basemap(projection='cyl',lon_0=0)
m.drawcoastlines()
m.drawcountries()
x,y=m(lon,lat)
m.scatter(x,y,s=250,marker='^')#,label=camp.name)











