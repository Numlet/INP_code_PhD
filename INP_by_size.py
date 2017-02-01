# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:55:04 2016

@author: eejvt
"""



import numpy as np
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
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
#%%
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
    


s={}
a=glob(folder+'*.sav')

s=jl.read_data(folder)
#%%
#diameter_range=np.linspace(1e-7,1e-5,1000)


'''
Size distribution by normalized by number of particles and surface area


'''
plt.close()
plt.close()
plt.close()
plt.close()
lat=53

lon=180+-0

#title=' Alert, Canada (Artic lat=82.20,lon=-62.21) '
#lat_point=82.20#alert
#lon_point=-62.21#alert

lat_point=-89.#
lon_point=0.1#
title=' Amundsen'
lat_point=48.56#Ucluelet
lon_point=-125.32#Ucluelet
title=' Ucluelet'
#48°56′9″N 125°32′36″W

runs=[]
run=[48.56,-125.32,' Ucluelet']
runs.append(run)
#lat_point=48.56#Ucluelet
#lon_point=-125.32#Ucluelet
#title=' Ucluelet'
#lat_point=-89.#
#lon_point=0.1#
#title=' Admunsen'
run=[-89.,0.1,' Amundsen']
runs.append(run)

for run in runs:
    lat_point=run[0]
    lon_point=run[1]
    title=run[2]
    ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
    ilon=jl.find_nearest_vector_index(jl.lon180,lon_point)
    
    lev=30
    
    mo_mass=s.tot_mc_ss_mm_mode[:,:,:,:,:].mean(axis=-1)
    feld_mass=s.tot_mc_feldspar_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
    mo_mass=s.tot_mc_ss_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
    n_particles=s.st_nd[:,lev,ilat,ilon,:].mean(axis=-1)
    radius=s.rbardry[:,lev,ilat,ilon,:].mean(axis=-1)
    
    
    fel_modes=[0,1,2,3,4,5,6]
    fel_modes=[2,3]
    step=1e-8
    def calculate_INP_feld_by_sizes_old(T,rmin=1e-7,rmax=1e-4,step=1e-8):
        std=s.sigma[:]
        #T=258
        rs=np.arange(rmin,rmax,step)
        modes_vol=volumes_of_modes(s)[:,lev,ilat,ilon,:].mean(axis=-1)
        kfeld_volfrac=(0.35*feld_mass/rhocomp[6])/modes_vol
        Nd=n_particles*kfeld_volfrac
        rmean=radius*1e2#factor 1e2 because of cm in feldspar parameterization
        ns=feld_parametrization(T)
        INP=np.zeros(rs.shape)
        N_size=np.zeros(rs.shape)
        INP_total=0
        for i in fel_modes:
            print 'Mode',i
            A=4*np.pi*rs**2
            ff=1-np.exp(-ns*A)
            PDF=lognormal_PDF(rmean[i],rs,std[i])
            dINP=PDF*Nd[i]*ff
            N_size=N_size+PDF*n_particles[i]        
            INP=INP+(dINP)
            INP_total=INP_total+(dINP*step).sum()
            
        return INP,rs,INP_total,N_size
        
        
    def calculate_INP_feld_by_sizes(T,rmin=1e-8,rmax=1e-4,step=1e-8):
        std=s.sigma[:]
        #T=258
        rs=np.arange(rmin,rmax,step)
        modes_vol=volumes_of_modes(s)[:,lev,ilat,ilon,:].mean(axis=-1)
        kfeld_volfrac=(0.35*feld_mass/rhocomp[6])/modes_vol
        Nd=n_particles*kfeld_volfrac
        rmean=radius
        ns=feld_parametrization(T)*1e4#factor 1e4 because of cm in feldspar parameterization Now in metters
        INP=np.zeros(rs.shape)
        N_size=np.zeros(rs.shape)
        A_size=np.zeros(rs.shape)
        INP_total=0
        for i in fel_modes:
            print 'Mode',i
            A=4*np.pi*rs**2
            ff=1-np.exp(-ns*A)
            PDF=lognormal_PDF(rmean[i],rs,std[i])
            dINP=PDF*Nd[i]*ff
            N_size=N_size+PDF*n_particles[i]
            INP=INP+(dINP)
            INP_total=INP_total+(dINP*step).sum()
    
            A_size=A_size+PDF*n_particles[i]*A*1e4
            
        return INP,rs,INP_total,N_size,A_size
        
    def calculate_INP_mo_by_sizes(T,rmin=1e-8,rmax=1e-4,step=1e-8):
        std=s.sigma[:]
        #T=258
        rs=np.arange(rmin,rmax,step)
    #    rbar_volume=rmean[i]*np.exp(3.0*np.log(std[i]))
        rmean=radius
    
    #    contribution_to_PM25=jl.lognormal_cummulative(mo_mass[i],rs,rbar_volume,std[i])
        INP=np.zeros(rs.shape)
        N_size=np.zeros(rs.shape)
        INP_total=0
        mo_volfrac=(mo_mass/rhocomp[3])/modes_vol
        Nd=n_particles*mo_volfrac
        for i in [2]:
            M=4/3.*np.pi*rs**3
            ff=1-np.exp(-marine_org_parameterization(T)*M)
            PDF=lognormal_PDF(rmean[i],rs,std[i])
            dINP=PDF*Nd[i]*ff
            N_size=N_size+PDF*n_particles[i]
            INP=INP+(dINP)
            INP_total=INP_total+(dINP*step).sum()

        return INP,rs,INP_total

    def calculate_INP_mo_by_sizes(T,rmin=1e-8,rmax=1e-4,step=1e-8):
        std=s.sigma[:]
        #T=258
        rs=np.arange(rmin,rmax,step)
    #    rbar_volume=rmean[i]*np.exp(3.0*np.log(std[i]))
        rmean=radius
    
    #    contribution_to_PM25=jl.lognormal_cummulative(mo_mass[i],rs,rbar_volume,std[i])
        INP=np.zeros(rs.shape)
        N_size=np.zeros(rs.shape)
        INP_total=0
        for i in [2]:
    #        if i==3:
    #            continue
            print 'Mode',i
            rbar_volume=rmean[i]*np.exp(3.0*np.log(std[i]))
    
            mass_pdf=jl.lognormal_PDF(rs,rbar_volume,std[i])*mo_mass[i]*1e-6#g
#            Mr=4/3.*np.pi*rs**3*rhocomp[3]*1e-6#g
#            ff=1-np.exp(-marine_org_parameterization(T)*Mr)
#            N_bin=mass_pdf/rhocomp[3]#m3
            
#            dINP=N_bin*ff*1e6#cm
            dINP=mass_pdf*marine_org_parameterization(T)*1e-6#cm
    #        N_size=N_size+PDF*Nd[i]        
            INP=INP+(dINP)
            INP_total=INP_total+(dINP*step).sum()
            
        return INP,rs,INP_total
    n1=np.random.randint(10000)
    n2=np.random.randint(10000)
    plt.figure(n1)
    plt.figure(n2)
    T=-25+273.15
    temps=[-15+273.15,-20+273.15,-25+273.15,-30+273.15]
    colors=['k','r','b','m']
    i=0
    for T in temps:
        plt.figure(n1)
        
        INP,rs,tot,N_size,A_size=calculate_INP_feld_by_sizes(T)
        INP_mo,rs,tot=calculate_INP_mo_by_sizes(T-273.15)
        A=4*np.pi*(rs*1e2)**2
        #INP2,rs2,tot2,N_size2=calculate_INP_feld_by_sizes2(T)
        #plt.plot(rs,INP,'ro')
    #    plt.xscale('log')
    #    plt.yscale('log')
        INP_cum=np.cumsum(INP*step)
        INP_total=INP+INP_mo
        N_total=n_particles.sum()
        plt.plot(rs*2.,(INP+INP_mo)/N_size, label='T='+str(T-273.15)+u"\u00b0C",c=colors[i])
        plt.plot(rs*2.,(INP_mo)/N_size,ls=':', label='Marine',c=colors[i])
        plt.plot(rs*2.,(INP)/N_size,ls='--', label='Feldspar',c=colors[i])
        plt.figure(n2)
        plt.plot(rs*2.,(INP+INP_mo)/(A*N_size), label='T='+str(T-273.15)+u"\u00b0C",c=colors[i])
        plt.plot(rs*2.,(INP_mo)/(A*N_size),ls=':', label='Marine',c=colors[i])
        plt.plot(rs*2.,(INP)/(A*N_size),ls='--', label='Feldspar',c=colors[i])
        
    
    
    
        i=i+1
    
    #    plt.plot(rs,(INP)/N_size, label=T-273.15,c='k')
    #    plt.plot(rs,/N_size, label=T-273.15)
    #plt.plot(rs2,INP2/N_size2, label='2')
    #plt.axhline(tot/N_total)
    
    plt.figure(n1)
    print tot
#    plt.ylim(1e-9,1e1)
#    plt.xlim(1e-8,1e-4)
#    plt.ylim(1e-7,1)
    plt.xlim(1e-7,1e-5)
    plt.ylabel('INP/total particles')
    plt.xlabel('$D_p (m)$')
    plt.xscale('log')
    plt.yscale('log')
    #plt.title()
    #plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    #plt.legend(loc='lower right')
#    plt.legend(loc='upper left')
    print np.sum(INP*step)*1e3
    print N_total
    plt.title(title)
    plt.savefig(jl.home_dir+title+'_for_Meng.png')
    
    plt.figure(n2)
#    plt.ylim(1e-1,1e7)
#    plt.xlim(1e-8,1e-4)
#    plt.ylim(1e2,1e6)
    plt.xlim(1e-7,1e-5)
    plt.ylabel('INP/surface area')
    plt.xlabel('$D_p (m)$')
    plt.xscale('log')
    plt.yscale('log')
    #plt.title()
    #plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
#    plt.legend(loc='upper left')
    print np.sum(INP*step)*1e3
    print N_total
    plt.title(title)
    plt.savefig(jl.home_dir+title+'surface_area_for_Meng.png')


#%%

'''
INP size distribution for Marine and Feldspar


'''


lat=53

lon=180+-0

#title=' Alert, Canada (Artic lat=82.20,lon=-62.21) '
#lat_point=82.20#alert
#lon_point=-62.21#alert

lat_point=-89.#
lon_point=0.1#
title=' Amundsen'
lat_point=48.56#Ucluelet
lon_point=-125.32#Ucluelet
title=' Ucluelet'
#48°56′9″N 125°32′36″W

runs=[]
run=[48.56,-125.32,' Ucluelet']
runs.append(run)
#lat_point=48.56#Ucluelet
#lon_point=-125.32#Ucluelet
#title=' Ucluelet'
#lat_point=-89.#
#lon_point=0.1#
#title=' Admunsen'
run=[-89.,0.1,' Amundsen']
runs.append(run)
sr=1e-2
er=1e2
st=1e-2
sr=1e-8
er=1e-4
st=1e-8
for run in runs:
    lat_point=run[0]
    lon_point=run[1]
    title=run[2]
    ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
    ilon=jl.find_nearest_vector_index(jl.lon180,lon_point)
    
    lev=30
    
    mo_mass=s.tot_mc_ss_mm_mode[:,:,:,:,:].mean(axis=-1)
    feld_mass=s.tot_mc_feldspar_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
    mo_mass=s.tot_mc_ss_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
    n_particles=s.st_nd[:,lev,ilat,ilon,:].mean(axis=-1)
    radius=s.rbardry[:,lev,ilat,ilon,:].mean(axis=-1)
    
    
    fel_modes=[0,1,2,3,4,5,6]
    step=1e-8
        
        
    def calculate_INP_feld_by_sizes(T,rmin=sr,rmax=er,step=st):
        std=s.sigma[:]
        #T=258
        rs=np.arange(rmin,rmax,step)
        modes_vol=volumes_of_modes(s)[:,lev,ilat,ilon,:].mean(axis=-1)
        kfeld_volfrac=(0.35*feld_mass/rhocomp[6])/modes_vol
        Nd=n_particles*kfeld_volfrac
        rmean=radius
        ns=feld_parametrization(T)*1e4#factor 1e4 because of cm in feldspar parameterization Now in metters
        INP=np.zeros(rs.shape)
        N_size=np.zeros(rs.shape)
        A_size=np.zeros(rs.shape)
        INP_total=0
        for i in fel_modes:
            print 'Mode',i
            A=4*np.pi*rs**2
            ff=1-np.exp(-ns*A)
            PDF=lognormal_PDF(rmean[i],rs,std[i])
            dINP=PDF*Nd[i]*ff
            N_size=N_size+PDF*n_particles[i]
            INP=INP+(dINP)
            INP_total=INP_total+(dINP*step).sum()
    
            A_size=A_size+PDF*n_particles[i]*A*1e4
            
        return INP,rs,INP_total,N_size,A_size
        
    def calculate_INP_mo_by_sizes(T,rmin=sr,rmax=er,step=st):
        std=s.sigma[:]
        #T=258
        rs=np.arange(rmin,rmax,step)
    #    rbar_volume=rmean[i]*np.exp(3.0*np.log(std[i]))
        rmean=radius
    
    #    contribution_to_PM25=jl.lognormal_cummulative(mo_mass[i],rs,rbar_volume,std[i])
        INP=np.zeros(rs.shape)
        N_size=np.zeros(rs.shape)
        INP_total=0
        for i in fel_modes:
    #        if i==3:
    #            continue
            print 'Mode',i
            rbar_volume=rmean[i]*np.exp(3.0*np.log(std[i]))
    
            mass_pdf=jl.lognormal_PDF(rs,rbar_volume,std[i])*mo_mass[i]
            Mr=4/3.*np.pi*rs**3*rhocomp[3]*1e-6#g
            ff=1-np.exp(-marine_org_parameterization(T)*Mr)
            N_bin=mass_pdf/rhocomp[3]#m3
            
            dINP=N_bin*ff*1e6#cm
    #        N_size=N_size+PDF*Nd[i]        
            INP=INP+(dINP)
            INP_total=INP_total+(dINP*step).sum()
            
        return INP,rs,INP_total
    n1=np.random.randint(10000)
#    n2=np.random.randint(10000)
    plt.figure(n1)
#    plt.figure(n2)
    T=-25+273.15
    temps=[-15+273.15,-20+273.15,-25+273.15,-30+273.15]
    colors=['k','r','b','m']
    i=0
    
    for T in temps:
        plt.figure(n1)
        
        INP,rs,tot,N_size,A_size=calculate_INP_feld_by_sizes(T)
        INP_mo,rs,tot=calculate_INP_mo_by_sizes(T-273.15)
        A=4*np.pi*(rs*1e2)**2
        #INP2,rs2,tot2,N_size2=calculate_INP_feld_by_sizes2(T)
        #plt.plot(rs,INP,'ro')
    #    plt.xscale('log')
    #    plt.yscale('log')
#        INP=np.cumsum(INP*step)
#        INP_mo=np.cumsum(INP*step)
        INP_total=INP+INP_mo
        N_total=n_particles.sum()
        plt.plot(rs*2.,(INP+INP_mo), label='Total T='+str(T-273.15)+u"\u00b0C",c=colors[i])
        if T==temps[0]:        
            plt.plot(rs*2.,(INP),ls='--', label='Feldspar',c=colors[i])
            plt.plot(rs*2.,(INP_mo),ls=':', label='Marine' ,c=colors[i])
        else:
            plt.plot(rs*2.,(INP),ls='--',c=colors[i])
            plt.plot(rs*2.,(INP_mo),ls=':' ,c=colors[i])
#        plt.figure(n2)
#        plt.plot(rs*2.,(INP+INP_mo), label='T='+str(T-273.15)+u"\u00b0C",c=colors[i])
        
    
    
    
        i=i+1
    
    #    plt.plot(rs,(INP)/N_size, label=T-273.15,c='k')
    #    plt.plot(rs,/N_size, label=T-273.15)
    #plt.plot(rs2,INP2/N_size2, label='2')
    #plt.axhline(tot/N_total)
    
    plt.figure(n1)
    print tot
#    plt.ylim(1e-9,1e1)
#    plt.xlim(1e-8,1e-4)
#    plt.ylim(1e-7,1)
#    plt.xlim(1e-7,1e-5)
    plt.ylabel('INP/total particles')
    plt.xlabel('$D_p (m)$')
    plt.xscale('log')
    plt.yscale('log')
    #plt.title()
    #plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    #plt.legend(loc='lower right')
    plt.legend(loc='best')
    print np.sum(INP*step)*1e3
    print N_total
    plt.title(title)
    plt.savefig(jl.home_dir+'Allan/'+title+'_for_Meng.png')
    
#    plt.figure(n2)
##    plt.ylim(1e-1,1e7)
##    plt.xlim(1e-8,1e-4)
##    plt.ylim(1e2,1e6)
##    plt.xlim(1e-7,1e-5)
#    plt.ylabel('INP')
#    plt.xlabel('$D_p (m)$')
#    plt.xscale('log')
#    plt.yscale('log')
#    #plt.title()
#    #plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
#    plt.legend(loc='best')
#    print np.sum(INP*step)*1e3
#    print N_total
#    plt.title(title)
#    plt.savefig(jl.home_dir+'Allan/'+title+'surface_area_for_Meng.png')





limits=rs-step/2.
limits=limits.tolist().append(rs[-1]+step/2.)
dlogD=np.log(diameter_range.domine[1:])-np.log(diameter_range.domine[:-1])












#%%

diameter_range=jl.logaritmic_steps(-7,-5,10000)
diameter_range.mid_points
dlogD=np.log(diameter_range.domine[1:])-np.log(diameter_range.domine[:-1])
runs=[]
run=[48.56,-125.32,' Ucluelet']
runs.append(run)
#lat_point=48.56#Ucluelet
#lon_point=-125.32#Ucluelet
#title=' Ucluelet'
plt.figure()
#lat_point=-89.#
#lon_point=0.1#
#title=' Admunsen'
run=[-89.,0.1,' Amundsen']
runs.append(run)

for run in runs:
    lat_point=run[0]
    lon_point=run[1]
    title=run[2]
#48°56′9″N 125°32′36″W
    ilat=jl.find_nearest_vector_index(jl.lat,lat_point)
    ilon=jl.find_nearest_vector_index(jl.lon180,lon_point)
    std=s.sigma[:]
    lev=30
    
    mo_mass=s.tot_mc_ss_mm_mode[:,:,:,:,:].mean(axis=-1)
    feld_mass=s.tot_mc_feldspar_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
    mo_mass=s.tot_mc_ss_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)
    n_particles=s.st_nd[:,lev,ilat,ilon,:].mean(axis=-1)
    radius=s.rbardry[:,lev,ilat,ilon,:].mean(axis=-1)
    
    dNtot=np.zeros_like(diameter_range)
    for i in range(len(std)):
        PDF=lognormal_PDF(radius[i],diameter_range.mid_points/2.,std[i])
        dNtot=dNtot+PDF*n_particles[i]*diameter_range.grid_steps_width
    
    plt.plot(diameter_range.mid_points,dNtot/dlogD,label=title+' Ntot (cm-3)=%1.2f'%dNtot.sum())
    plt.xscale('log')
#    plt.ylim(1e-4,1e5)
#    plt.xlim(1e-7,1e-5)
    plt.yscale('log')
plt.ylabel('dN/dlogDp (cm-3)')
plt.xlabel('$D_p (m)$')
plt.legend(loc='best')
#%%

#mo_mass=s.tot_mc_ss_mm_mode[:,:,:,:,:].mean(axis=-1)
mo_mass=s.tot_mc_ss_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)

rmin=1e-8
rmax=1e-2
step=1e-8

std=s.sigma[:]

rs=np.arange(rmin,rmax,step)

rmean=radius
modes_vol=volumes_of_modes(s)[:,lev,ilat,ilon,:].mean(axis=-1)
mo_volfrac=(mo_mass/rhocomp[3])/modes_vol
Nd_mo=n_particles*mo_volfrac



ns=marine_org_parameterization(T)
INP=np.zeros(rs.shape)
N_size=np.zeros(rs.shape)
INP_total=0
i=2
print 'Mode',i

Total_mass=s.tot_mc_ss_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)+s.tot_mc_dust_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)+s.tot_mc_su_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)+s.tot_mc_feldspar_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)+s.tot_mc_oc_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)+s.tot_mc_bc_mm_mode[:,lev,ilat,ilon,:].mean(axis=-1)

A=4*np.pi*rs**2
M=rhocomp[3]*4/3.*np.pi*rs**3
#M=np.zeros_like(rs)
#for ir in range(len(rhocomp)):
#    M=M+rhocomp[ir]*4/3.*np.pi*rs**3
#


#ff=1-np.exp(-ns*A)

jl.lognormal_cummulative
PDF=lognormal_PDF(rmean[i]*np.exp(3.0*np.log(std[i])**2),rs,std[i])
PDFm=lognormal_PDF(rmean[i],rs,std[i])/(rhocomp[3]*4*np.pi*rs**2)

mass_integrated=(PDF*step*M*n_particles[i]).sum()#kg/m3

mass_modelled=mo_mass[i]*1e-9#kg/m3
T_mass=Total_mass*1e-9
print mass_integrated,T_mass[i]
print mass_integrated,mass_modelled


plt.plot(rs,PDF*step*M*Nd_mo[i])
plt.xscale('log')
plt.yscale('log')


#%%
rbar_volume=rmean[i]*np.exp(3.0*np.log(std[i]))

contribution_to_PM25=jl.lognormal_cummulative(1.,rs,rbar_volume,std[i])
pdf=jl.lognormal_PDF(rs,rbar_volume,std[i])



#plt.plot(rs,contribution_to_PM25,label='cumulative')
plt.plot(rs,pdf)
plt.legend()
plt.xscale('log')
plt.yscale('log')





d={}
d['asd']='fgh_jkl'