# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:34:40 2017

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
import datetime
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
from scipy.io import netcdf
import scipy as sc
import multiprocessing


rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3




folder='/nfs/a201/eejvt/FELDSPAR_SOLUBLE_REMOVED/ACCCOR/'
folder='/nfs/a201/eejvt/mineral_fractions_perlwitz/RUN/'
folder='/nfs/a201/eejvt/mineral_fractions_perlwitz/RUN_DUSTFRAC/'

archive_directory='/nfs/a201/eejvt/'
project='MARINE_PARAMETERIZATION/DAILY'
folder='/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/'
project='MARINE_PARAMETERIZATION/FOURTH_TRY'
folder=archive_directory+project
#os.chdir(archive_directory+project)

def area_lognormal(rbar,sigma,Nd):
    #print isinstance(sigma,np.float32)
    if isinstance(sigma,np.float32):
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        
        S=Nd*(2*rbar)**2*y
    else:
        S=np.zeros(rbar.shape)
        #y=np.zeros(len(rbar))
        #global S,y
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        for i in range (len(rbar)):
            S[i,]=Nd[i,]*(2*rbar[i,])**2*y[i]
    return S
    
def area_lognormal_per_particle(rbar,sigma):
    #print isinstance(sigma,float)
    if isinstance(sigma,np.float32):
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        S=(2*rbar)**2*y
    else:
        S=np.zeros(rbar.shape)
        #y=np.zeros(len(rbar))
        #global S,y
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        for i in range (len(rbar)):
            S[i,]=(2*rbar[i,])**2*y[i]
    return S

def feld_parametrization(T):
    ns=np.exp((-1.03802178356815*T)+275.263379304105)
    return ns
    
def volumes_of_modes(s):
    modes_vol=(s.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]
    +s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2]
    +s.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
    +s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]
    +s.tot_mc_oc_mm_mode[:,:,:,:,:]/rhocomp[5]
    +s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])
    return modes_vol
    

def lognormal_PDF(rmean,r,std):
   X=(1/(r*np.log(std)*np.sqrt(2*np.pi)))*np.exp(-(np.log(r)-np.log(rmean))**2/(2*np.log(std)**2))
   return X


def marine_org_parameterization(T):
    a=11.2186
    b=-0.4459
    INP=np.exp(a+b*T)#[per gram]
    return INP
    
#archive_directory='/nfs/a107/eejvt/'
#project='JB_TRAINING/'
#from matplotlib.colors import LogNorm
#from matplotlib import colors, ticker, cm
#import glob as glob
#folder=archive_directory+project+'WITH_ICE_SCAV2'

#s=jl.read_data('WITH_ICE_SCAV2')

s={}
#a=glob(folder+'*.sav')

s=jl.read_data(folder)
#%%

#%%
# INPs per particle

diameter_range=jl.logaritmic_steps(-8,-4,1000)
diameter_range.mid_points
dlogD=(np.log(diameter_range.domine[1:])-np.log(diameter_range.domine[:-1]))/1e-6






lat_point=48.56#Ucluelet
lon_point=-125.32#Ucluelet
title=' Ucluelet'

lat_point=-89.#
lon_point=0.1#
title=' Amundsen'


lat_point=82.45#
lon_point=-62.51#
title='Alert'
lat_point=54.59#
lon_point=-55.61#
title=' Amundsen (Labrador Sea)'
ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
ilon=jl.find_nearest_vector_index(jl.lon180,lon_point)

lev=30

mo_mass=s.tot_mc_ss_mm_mode[:,:,:,:,:].mean(axis=-1)
feld_mass=s.tot_mc_feldspar_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
mo_mass=s.tot_mc_ss_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
n_particles=s.st_nd[:,lev,ilat,ilon,:].mean(axis=-1)
radius=s.rbardry[:,lev,ilat,ilon,:].mean(axis=-1)
modes_vol=volumes_of_modes(s)[:,lev,ilat,ilon,:].mean(axis=-1)
std=s.sigma

#fel_modes=[0,1,2,3,4,5,6]
fel_modes=[2,3]
#step=1e-8

kfeld_volfrac=(0.35*feld_mass/rhocomp[6])/modes_vol
mo_volfrac=(mo_mass/rhocomp[3])/modes_vol
N_feldspar=n_particles*kfeld_volfrac
N_mo=n_particles*mo_volfrac


    

#lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])
N_total_acc_sizes=n_particles[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])
N_total=np.zeros_like(N_total_acc_sizes)
A_total=np.zeros_like(N_total_acc_sizes)
for istd in range(len(std)):
    N_total=N_total+n_particles[istd]*lognormal_PDF(radius[istd],diameter_range.mid_points/2.,std[istd])
    A_total=A_total+n_particles[istd]*lognormal_PDF(radius[istd],diameter_range.mid_points/2.,std[istd])*4*np.pi*(diameter_range.mid_points/2)**2
#A_total_acc_sizes=n_particles[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])
N_feld_acc_sizes=N_feldspar[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])
N_feld_coar_sizes=N_feldspar[3]*lognormal_PDF(radius[3],diameter_range.mid_points/2.,std[3])
N_mo_acc_sizes=N_mo[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])









colors=['k','r','b','m']
temps=[-15,-20,-25,-30]
fig, ax1 = plt.subplots()
ax1.set_xlabel('X data')
ax1.set_ylabel('INP/total particles')
#ax2.set_ylabel('Y2 data', color='b')
#ax2.set_yscale('log')
ax1.set_yscale('log')
plt.title(title)


values=np.logical_and(diameter_range.mid_points>1e-7,diameter_range.mid_points<1e-5)

data=np.zeros(( len(diameter_range.mid_points[values]),13))

data[:,0]=diameter_range.mid_points[values]
#icol=0
header=['D (m)']
for i in range(len(temps)):
    
    T=temps[i]
    c=colors[i]
    #ax2 = ax1.twinx()
    #ax1.plot(x, y1, 'g-')
    #ax2.plot(x, y2, 'b-')
    ff_feld=1-np.exp(-feld_parametrization(T+273.15)*4*np.pi*(diameter_range.mid_points/2.*1e2)**2)

    ff_mo=1-np.exp(-marine_org_parameterization(T)*4/3.*np.pi*(diameter_range.mid_points/2.)**3*rhocomp[3]*1e-9)

    plt.xscale('log')
    marine=(N_mo_acc_sizes*ff_mo)/N_total
    ax1.plot(diameter_range.mid_points,marine,':',c=c)
    feldspar=(N_feld_acc_sizes*ff_feld+N_feld_coar_sizes*ff_feld)/N_total
    ax1.plot(diameter_range.mid_points,feldspar,'--',c=c)
    #ax1.plot(diameter_range.mid_points,N_feld_acc_sizes*ff_feld/N_total,'r')
    ax1.plot(diameter_range.mid_points,marine+feldspar,
    label='T='+str(T)+u"\u00b0C",c=c)
    header.append('T='+str(T)+"C"+' Marine+ Feldspar')
    data[:,1+i*3]=marine[values]+feldspar[values]
    header.append('Feldspar')
    data[:,1+i*3+1]=feldspar[values]
    header.append('Feldspar')
    data[:,1+i*3+2]=marine[values]
    
    #ax1.plot(diameter_range.mid_points,N_total_acc_sizes,'k')

plt.plot(0,0,'k--',label='Feldspar')
plt.plot(0,0,'k.',label='Marine organics')

np.savetxt('/nfs/a201/eejvt/ALLAN/INP_per_particle'+title+'.csv',data,header=','.join(header),delimiter=',')


plt.xlabel('D (m)')
plt.xlim(1e-7,1e-5)
plt.ylim(1e-9,1e-1)


#ax1.plot(diameter_range.mid_points,N_total_acc_sizes,'k')
#
plt.legend(loc='best')
plt.savefig('/nfs/a201/eejvt/ALLAN/INP_per_particle'+title+'.png')
plt.show()



fig, ax1 = plt.subplots()
ax1.set_xlabel('X data')
#ax1.set_ylabel('INP/total particles')
#ax2.set_ylabel('Y2 data', color='b')
#ax2.set_yscale('log')
ax1.set_yscale('log')
plt.title(title)




ax1.set_ylabel('INP/surface area (cm^-2)')

data[:,0]=diameter_range.mid_points[values]
#icol=0

Area_total_cm=A_total*1e4
header=['D (m)']
for i in range(len(temps)):
    
    T=temps[i]
    c=colors[i]
    #ax2 = ax1.twinx()
    #ax1.plot(x, y1, 'g-')
    #ax2.plot(x, y2, 'b-')
    ff_feld=1-np.exp(-feld_parametrization(T+273.15)*4*np.pi*(diameter_range.mid_points/2.*1e2)**2)

    ff_mo=1-np.exp(-marine_org_parameterization(T)*4/3.*np.pi*(diameter_range.mid_points/2.)**3*rhocomp[3]*1e-9)

    plt.xscale('log')
    marine=(N_mo_acc_sizes*ff_mo)/A_total
    ax1.plot(diameter_range.mid_points,marine,':',c=c)
    feldspar=(N_feld_acc_sizes*ff_feld+N_feld_coar_sizes*ff_feld)/A_total
    ax1.plot(diameter_range.mid_points,feldspar,'--',c=c)
    #ax1.plot(diameter_range.mid_points,N_feld_acc_sizes*ff_feld/N_total,'r')
    ax1.plot(diameter_range.mid_points,marine+feldspar,
    label='T='+str(T)+u"\u00b0C",c=c)
    header.append('T='+str(T)+"C"+' Marine+ Feldspar')
    data[:,1+i*3]=marine[values]+feldspar[values]
    header.append('Feldspar')
    data[:,1+i*3+1]=feldspar[values]
    header.append('Feldspar')
    data[:,1+i*3+2]=marine[values]
    
    #ax1.plot(diameter_range.mid_points,N_total_acc_sizes,'k')

plt.plot(0,0,'k--',label='Feldspar')
plt.plot(0,0,'k.',label='Marine organics')

np.savetxt('/nfs/a201/eejvt/ALLAN/INP_per_surface_area'+title+'.csv',data,header=','.join(header),delimiter=',')


plt.xlabel('D (m)')
plt.xlim(1e-7,1e-5)
plt.ylim(1e0,1e8)


#ax1.plot(diameter_range.mid_points,N_total_acc_sizes,'k')
#
plt.legend(loc='best')
plt.savefig('/nfs/a201/eejvt/ALLAN/INP_per_surface_area'+title+'.png')
plt.show()


#%%
# INPs size distr


diameter_range=jl.logaritmic_steps(-8,-4,1000)
diameter_range.mid_points
dlogD=(np.log(diameter_range.domine[1:])-np.log(diameter_range.domine[:-1]))/1e-6



lat_point=-89.#
lon_point=0.1#
title=' Amundsen'


lat_point=48.56#Ucluelet
lon_point=-125.32#Ucluelet
title=' Ucluelet'#august
lat_point=54.59#
lon_point=-55.61#
title=' Amundsen (Labrador Sea)'#July
      
lat_point=82.45#
lon_point=-62.51#
title='Alert'




ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
ilon=jl.find_nearest_vector_index(jl.lon180,lon_point)

lev=30

mo_mass=s.tot_mc_ss_mm_mode[:,:,:,:,:].mean(axis=-1)
feld_mass=s.tot_mc_feldspar_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
mo_mass=s.tot_mc_ss_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
n_particles=s.st_nd[:,lev,ilat,ilon,:].mean(axis=-1)
radius=s.rbardry[:,lev,ilat,ilon,:].mean(axis=-1)
modes_vol=volumes_of_modes(s)[:,lev,ilat,ilon,:].mean(axis=-1)
std=s.sigma

#fel_modes=[0,1,2,3,4,5,6]
fel_modes=[2,3]
#step=1e-8

kfeld_volfrac=(0.35*feld_mass/rhocomp[6])/modes_vol
mo_volfrac=(mo_mass/rhocomp[3])/modes_vol
N_feldspar=n_particles*kfeld_volfrac
N_mo=n_particles*mo_volfrac


    

#lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])
N_total_acc_sizes=n_particles[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])
N_total=np.zeros_like(N_total_acc_sizes)
A_total=np.zeros_like(N_total_acc_sizes)
for istd in range(len(std)):
    N_total=N_total+n_particles[istd]*lognormal_PDF(radius[istd],diameter_range.mid_points/2.,std[istd])
    A_total=A_total+n_particles[istd]*lognormal_PDF(radius[istd],diameter_range.mid_points/2.,std[istd])*4*np.pi*(diameter_range.mid_points/2)**2
#A_total_acc_sizes=n_particles[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])
N_feld_acc_sizes=N_feldspar[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])
N_feld_coar_sizes=N_feldspar[3]*lognormal_PDF(radius[3],diameter_range.mid_points/2.,std[3])
N_mo_acc_sizes=N_mo[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])









colors=['k','r','b','m']
temps=[-15,-20,-25,-30]
fig, ax1 = plt.subplots()
ax1.set_xlabel('X data')
ax1.set_ylabel('INP /dlogD')
#ax2.set_ylabel('Y2 data', color='b')
#ax2.set_yscale('log')
ax1.set_yscale('log')
plt.title(title)


values=np.logical_and(diameter_range.mid_points>1e-7,diameter_range.mid_points<1e-5)

data=np.zeros(( len(diameter_range.mid_points[values]),14))

data[:,0]=diameter_range.mid_points[values]
data[:,1]=dlogD[values]
#icol=0
header=['D (m)','dlogD']
for i in range(len(temps)):
    
    T=temps[i]
    c=colors[i]
    #ax2 = ax1.twinx()
    #ax1.plot(x, y1, 'g-')
    #ax2.plot(x, y2, 'b-')
    ff_feld=1-np.exp(-feld_parametrization(T+273.15)*4*np.pi*(diameter_range.mid_points/2.*1e2)**2)

    ff_mo=1-np.exp(-marine_org_parameterization(T)*4/3.*np.pi*(diameter_range.mid_points/2.)**3*rhocomp[3]*1e-9)

    plt.xscale('log')
    marine=(N_mo_acc_sizes*ff_mo)/dlogD
    ax1.plot(diameter_range.mid_points,marine,':',c=c)
    feldspar=(N_feld_acc_sizes*ff_feld+N_feld_coar_sizes*ff_feld)/dlogD
    ax1.plot(diameter_range.mid_points,feldspar,'--',c=c)
    #ax1.plot(diameter_range.mid_points,N_feld_acc_sizes*ff_feld/N_total,'r')
    ax1.plot(diameter_range.mid_points,marine+feldspar,
    label='T='+str(T)+u"\u00b0C",c=c)
    header.append('T='+str(T)+"C"+' Marine+ Feldspar')
    data[:,2+i*3]=marine[values]+feldspar[values]
    header.append('Feldspar')
    data[:,2+i*3+1]=feldspar[values]
    header.append('Feldspar')
    data[:,2+i*3+2]=marine[values]
    
    #ax1.plot(diameter_range.mid_points,N_total_acc_sizes,'k')

plt.plot(0,0,'k--',label='Feldspar')
plt.plot(0,0,'k.',label='Marine organics')

np.savetxt('/nfs/a201/eejvt/ALLAN/INP_size_distribution'+title+'.csv',data,header=','.join(header),delimiter=',')


plt.xlabel('D (m)')
plt.xlim(1e-7,1e-5)
plt.ylim(1e-9,1)


#ax1.plot(diameter_range.mid_points,N_total_acc_sizes,'k')
#
plt.legend(loc='best')
plt.savefig('/nfs/a201/eejvt/ALLAN/INP_size_distribution'+title+'.png')
plt.show()

#%%

fig, ax1 = plt.subplots()
ax1.set_xlabel('X data')
#ax1.set_ylabel('INP/total particles')
#ax2.set_ylabel('Y2 data', color='b')
#ax2.set_yscale('log')
ax1.set_yscale('log')
plt.title(title)




ax1.set_ylabel('INP/surface area (cm^-2)')

data[:,0]=diameter_range.mid_points[values]
#icol=0

Area_total_cm=A_total*1e4
header=['D (m)']
for i in range(len(temps)):
    
    T=temps[i]
    c=colors[i]
    #ax2 = ax1.twinx()
    #ax1.plot(x, y1, 'g-')
    #ax2.plot(x, y2, 'b-')
    ff_feld=1-np.exp(-feld_parametrization(T+273.15)*4*np.pi*(diameter_range.mid_points/2.*1e2)**2)

    ff_mo=1-np.exp(-marine_org_parameterization(T)*4/3.*np.pi*(diameter_range.mid_points/2.)**3*rhocomp[3]*1e-9)

    plt.xscale('log')
    marine=(N_mo_acc_sizes*ff_mo)/A_total
    ax1.plot(diameter_range.mid_points,marine,':',c=c)
    feldspar=(N_feld_acc_sizes*ff_feld+N_feld_coar_sizes*ff_feld)/A_total
    ax1.plot(diameter_range.mid_points,feldspar,'--',c=c)
    #ax1.plot(diameter_range.mid_points,N_feld_acc_sizes*ff_feld/N_total,'r')
    ax1.plot(diameter_range.mid_points,marine+feldspar,
    label='T='+str(T)+u"\u00b0C",c=c)
    header.append('T='+str(T)+"C"+' Marine+ Feldspar')
    data[:,1+i*3]=marine[values]+feldspar[values]
    header.append('Feldspar')
    data[:,1+i*3+1]=feldspar[values]
    header.append('Feldspar')
    data[:,1+i*3+2]=marine[values]
    
    #ax1.plot(diameter_range.mid_points,N_total_acc_sizes,'k')

plt.plot(0,0,'k--',label='Feldspar')
plt.plot(0,0,'k.',label='Marine organics')

np.savetxt('/nfs/a201/eejvt/ALLAN/INP_per_surface_area'+title+'.csv',data,header=','.join(header),delimiter=',')


plt.xlabel('D (m)')
plt.xlim(1e-7,1e-5)
plt.ylim(1e0,1e8)


#ax1.plot(diameter_range.mid_points,N_total_acc_sizes,'k')
#
plt.legend(loc='best')
plt.savefig('/nfs/a201/eejvt/ALLAN/INP_per_surface_area'+title+'.png')
plt.show()



#%%

































diameter_range=jl.logaritmic_steps(-8,-4,1000)
diameter_range.mid_points
dlogD=(np.log(diameter_range.domine[1:])-np.log(diameter_range.domine[:-1]))/1e-6



lat_point=48.56#Ucluelet
lon_point=-125.32#Ucluelet
title=' Ucluelet'


lat_point=-89.#
lon_point=0.1#
title=' Amundsen'
#lat_point=14#Ucluelet
#lon_point=23#Ucluelet
#title=' Ucluelet'
##48°56′9″N 125°32′36″W


T=-25

ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
ilon=jl.find_nearest_vector_index(jl.lon180,lon_point)

lev=30

mo_mass=s.tot_mc_ss_mm_mode[:,:,:,:,:].mean(axis=-1)
feld_mass=s.tot_mc_feldspar_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
mo_mass=s.tot_mc_ss_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
n_particles=s.st_nd[:,lev,ilat,ilon,:].mean(axis=-1)
radius=s.rbardry[:,lev,ilat,ilon,:].mean(axis=-1)
modes_vol=volumes_of_modes(s)[:,lev,ilat,ilon,:].mean(axis=-1)
std=s.sigma

#fel_modes=[0,1,2,3,4,5,6]
fel_modes=[2,3]
#step=1e-8

kfeld_volfrac=(0.35*feld_mass/rhocomp[6])/modes_vol
mo_volfrac=(mo_mass/rhocomp[3])/modes_vol
N_feldspar=n_particles*kfeld_volfrac
N_mo=n_particles*mo_volfrac


    

#lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])
N_total_acc_sizes=n_particles[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])
N_total=np.zeros_like(N_total_acc_sizes)
for istd in range(len(std)):
    N_total=N_total+n_particles[istd]*lognormal_PDF(radius[istd],diameter_range.mid_points/2.,std[istd])
#A_total_acc_sizes=n_particles[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])
N_feld_acc_sizes=N_feldspar[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])
N_feld_coar_sizes=N_feldspar[3]*lognormal_PDF(radius[3],diameter_range.mid_points/2.,std[3])
N_mo_acc_sizes=N_mo[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])


ff_feld=1-np.exp(-feld_parametrization(T+273.15)*4*np.pi*(diameter_range.mid_points/2.*1e2)**2)
ff_mo=1-np.exp(-marine_org_parameterization(T)*4/3.*np.pi*(diameter_range.mid_points/2.)**3*rhocomp[3]*1e-9)


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
#ax1.plot(x, y1, 'g-')
#ax2.plot(x, y2, 'b-')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data', color='g')
ax2.set_ylabel('Y2 data', color='b')
ax2.set_yscale('log')
ax1.set_yscale('log')

plt.show()
ax1.plot(diameter_range.mid_points,N_total_acc_sizes,'k')
ax1.plot(diameter_range.mid_points,N_feld_acc_sizes,'g')
ax1.plot(diameter_range.mid_points,N_mo_acc_sizes,'b')
ax1.plot(diameter_range.mid_points,N_feld_acc_sizes*ff_feld,'r')
ax1.plot(diameter_range.mid_points,N_mo_acc_sizes*ff_mo,'y')
ax2.plot(diameter_range.mid_points,ff_feld)
ax2.plot(diameter_range.mid_points,ff_mo,'y')
plt.xscale('log')
#(diameter_range.mid_points/2.)**2*np.pi
#%%
























diameter_range=jl.logaritmic_steps(-8,-4,10000)
diameter_range.mid_points
dlogD=(np.log(diameter_range.domine[1:])-np.log(diameter_range.domine[:-1]))/1e-6



lat_point=48.56#Ucluelet
lon_point=-125.32#Ucluelet
title=' Ucluelet'

#lat_point=14#Ucluelet
#lon_point=23#Ucluelet
#title=' Ucluelet'
##48°56′9″N 125°32′36″W


ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
ilon=jl.find_nearest_vector_index(jl.lon180,lon_point)

lev=30

mo_mass=s.tot_mc_ss_mm_mode[:,:,:,:,:].mean(axis=-1)
feld_mass=s.tot_mc_feldspar_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
mo_mass=s.tot_mc_ss_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
n_particles=s.st_nd[:,lev,ilat,ilon,:].mean(axis=-1)
radius=s.rbardry[:,lev,ilat,ilon,:].mean(axis=-1)
modes_vol=volumes_of_modes(s)[:,lev,ilat,ilon,:].mean(axis=-1)
std=s.sigma

#fel_modes=[0,1,2,3,4,5,6]
fel_modes=[2,3]
#step=1e-8









kfeld_volfrac=(0.35*feld_mass/rhocomp[6])/modes_vol
kfeld_volfrac=(mo_mass/rhocomp[1])/modes_vol

plt.figure()
ax=plt.subplot(311)
bx=plt.subplot(312)
cx=plt.subplot(313)

N_feldspar=n_particles*kfeld_volfrac

N_feld_acc=N_feldspar[2,]
N_feld_acc_sizes=N_feldspar[2]*lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[2])

ax.plot(diameter_range.mid_points,N_feld_acc_sizes/dlogD,'b')
plt.xscale('log')
#plt.figure(2)
#plt.figure(2)
bx.plot(diameter_range.mid_points,N_feld_acc_sizes/dlogD*(diameter_range.mid_points/2.)**2*np.pi,'b')
#plt.plot(diameter_range.mid_points,N_feld_acc_sizes,'k')
bx.set_xscale('log')
#bx.set_yscale('log')
#plt.figure(3)
cx.plot(diameter_range.mid_points,N_feld_acc_sizes/dlogD*(diameter_range.mid_points/2.)**3*np.pi/6,'b')
plt.xscale('log')


N_feld_coar=N_feldspar[3,]
N_feld_coar_sizes=N_feldspar[3]*lognormal_PDF(radius[3],diameter_range.mid_points/2.,std[3])
#plt.figure(1)
ax.plot(diameter_range.mid_points,N_feld_coar_sizes/dlogD,'r')
#plt.figure(2)
bx.plot(diameter_range.mid_points,N_feld_coar_sizes/dlogD*(diameter_range.mid_points/2.)**2*np.pi,'r')
#plt.figure(3)
cx.plot(diameter_range.mid_points,N_feld_coar_sizes/dlogD*(diameter_range.mid_points/2.)**3*np.pi/6,'r')
M=4/3.*np.pi*(diameter_range.mid_points/2.)**3
ff=1-np.exp(-marine_org_parameterization(-15)*M)
PDF=lognormal_PDF(radius[2],diameter_range.mid_points/2.,std[i])
dINP=PDF*ff
cx.plot(diameter_range.mid_points,dINP,'k')
cx.plot(diameter_range.mid_points,dINP/(n_particles[2]/dlogD*(diameter_range.mid_points/2.)**3*np.pi/6),'g')
cx.set_yscale('log')

#plt.plot(diameter_range.mid_points,N_feld_coar_sizes,'g')
#plt.plot(diameter_range.mid_points,N_feld_coar_sizes,'y')
ax.set_xscale('log')
#ax.set_yscale('log')


#for mode in range(7):
#    print mode
    

#%%



jl.plot(s.rbardry[2,30,].mean(axis=-1)-s.rbardry[3,30,].mean(axis=-1))
large_coar=s.rbardry[3,30,]
large_coar[large_coar<1e-6]=0
jl.plot(large_coar[:,:,0])
jl.plot(large_coar[:,:,0])
jl.plot(s.rbardry[2,30,].mean(axis=-1)-s.rbardry[3,30,].mean(axis=-1))
jl.plot(s.rbardry[6,30,].mean(axis=-1))









