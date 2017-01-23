# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:18:07 2015

@author: eejvt
"""

import os
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
archive_directory='/nfs/a107/eejvt/'
project='JB_TRAINING/'
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
os.chdir(archive_directory+project)

rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3
#levels=np.logspace(-4,1,14).tolist()
#levels=[5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1,5,10,50,100]

#jl.plot(feldext[20,20,:,:,:].mean(axis=-1)*1e3,cblabel='$L^{-1}$',clevs=levels,colorbar_format_sci=1,title='INP feldspar T=$-20^o$C, pressure=600hpa')
#%%
'''
R esta en metros 
st_nd en cm-3


'''
def area_lognormal(rbar,sigma,Nd):
    #print isinstance(sigma,float)
    if isinstance(sigma,float):
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
    
feld_paper_clevs=[0,0.0001,0.0002,0.0005,0.0010,0.002,0.005,0.01,0.02,0.05,0.1]
def lognormal_PDF(rmean,r,std):
   X=(1/(r*np.log(std)*np.sqrt(2*np.pi)))*np.exp(-(np.log(r)-np.log(rmean))**2/(2*np.log(std)**2))
   return X
   
   
#%%
rs=np.linspace(0.1,100,1000)
std=2
rmean=0.1
rmin=np.log(rmean*0.01)
rmax=np.log(rmean*10)
rmin=0.0001*rmean
rmax=1000*rmean
rmin=rmean/10/std
rmax=rmean*10*std
rs=np.arange(rmin,rmax,rmin)
ns=0.2
A=4*np.pi*rs
ff=1-np.exp(-ns*A)
PDF=lognormal_PDF(rmean,rs,std)
#plt.plot(rs,ff)
print lognormal_PDF(rmean,rs,std).sum()*rmin
#plt.xscale('log')
#plt.yscale('log')
plt.show()
#%%(1-exp(-4*pi*n*T*r^2))*N*(1/(r*s*sqrt(2*pi))*exp(-(ln(r)-ln(rm))^2/(2*s^2))
s=jl.read_data('WITH_ICE_SCAV2')
s=jl.read_data('/nfs/a201/eejvt/FELDSPAR_SOLUBLE_REMOVED')

sigma=s.sigma[:]
#%%
def calculate_INP_feld_ext(T):
    fel_modes=[2,3,5,6]
    #def INP_feldspar_ext(s,T):
    std=s.sigma[:]
    #T=258
    modes_vol=volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*kfeld_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
    ns=feld_parametrization(T)
    INP=np.zeros(Nd.shape)
    for i in fel_modes:
        print 'Mode',i    
        rmin=rmean[i,]/4/std[i]
        rmax=rmean[i,]*4*std[i]
        step=rmin
        for ilev in range(len(step[:,0,0,0])):
            print 'ilev',ilev
            for ilat in range(len(step[0,:,0,0])):
                for ilon in range(len(step[0,0,:,0])):
                    for imon in range(len(step[0,0,0,:])):
                        rs=np.arange(rmin[ilev,ilat,ilon,imon],rmax[ilev,ilat,ilon,imon],step[ilev,ilat,ilon,imon])
                        A=4*np.pi*rs**2
                        ff=1-np.exp(-ns*A)
                        PDF=lognormal_PDF(rmean[i,ilev,ilat,ilon,imon],rs,std[i])
                        #print PDF.sum()*step[ilev,ilat,ilon,imon]
                        dINP=PDF*Nd[i,ilev,ilat,ilon,imon]*ff
                        INP[i,ilev,ilat,ilon,imon]=INP[i,ilev,ilat,ilon,imon]+(dINP.sum()*step[ilev,ilat,ilon,imon])
                        #print 'INP',INP[i,ilev,ilat,ilon,imon]
    return INP
INP_feld_ext=np.zeros((38,7, 31, 64, 128, 12))
for t in range(38):
    INP_feld_ext[t,:,:,:,:,:]=calculate_INP_feld_ext(-t+273.15)

INP_total_feld_ext=INP_feld_ext.sum(axis=1)
np.save('INP_feld_ext_alltemps',INP_total_feld_ext)

    #plt.plot(rs,dINP)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.show()
#return INP
#INP_feld_ext=INP_feldspar_ext(s,253)
#%%
pressures=s.pl_m
feld_paper_clevs=np.logspace(-4,5,15).tolist()#[0,0.0001,0.0002,0.0005,0.0010,0.002,0.005,0.01,0.02,0.05,0.1]
feld_paper_clevs=[0,0.0001,0.0002,0.0005,0.0010,0.002,0.005,0.01,0.02,0.05]
#jl.plot(INP[:,21,:,:,:].mean(axis=3).sum(axis=0),show=1,clevs=feld_paper_clevs,cmap=plt.cm.RdBu_r,cblabel='$cm^{-3}$',title='ext 15 800',colorbar_format_sci=0)
#%%
feld_paper_clevs=np.logspace(-4,5,15).tolist()
feld_paper_clevs=[0,0.0001,0.0002,0.0005,0.0010,0.002,0.005,0.01,0.02,0.05]
#INP_20_ext=np.load('INP_20_external.npy')
#jl.plot(INP[:,20,:,:,:].mean(axis=3).sum(axis=0),show=1,clevs=feld_paper_clevs,cmap=plt.cm.RdBu_r,cblabel='$cm^{-3}$',title='int 20 600',colorbar_format_sci=0)
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')

#%%INP internal
def calculate_INP_feld_int(T):
    fel_modes=[2,3]#,5,6]
    std=s.sigma[:]
    #T=253
    modes_vol=volumes_of_modes(s)
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]
    rmean=s.rbardry[:,:,:,:,:]*1e2
    ns=feld_parametrization(T)
    INP=np.zeros(Nd.shape)
    for i in fel_modes:
        print 'Mode',i    
        rmin=rmean[i,]/4/std[i]
        rmax=rmean[i,]*4*std[i]
        step=rmin
        for ilev in range(len(step[:,0,0,0])):
            print 'ilev',ilev
            for ilat in range(len(step[0,:,0,0])):
                for ilon in range(len(step[0,0,:,0])):
                    for imon in range(len(step[0,0,0,:])):
                        Amax=4*np.pi*rmax[ilev,ilat,ilon,imon]**2*kfeld_volfrac[i,ilev,ilat,ilon,imon]
                        exponent=ns*Amax
                        if exponent<0.05:
                            dINP=ns*area_lognormal(rmean[i,ilev,ilat,ilon,imon],std[i],Nd[i,ilev,ilat,ilon,imon])
                        else:
                            rs=np.arange(rmin[ilev,ilat,ilon,imon],rmax[ilev,ilat,ilon,imon],step[ilev,ilat,ilon,imon])
                            A=4*np.pi*rs**2*kfeld_volfrac[i,ilev,ilat,ilon,imon]
                            ff=1-np.exp(-ns*A)
                            PDF=lognormal_PDF(rmean[i,ilev,ilat,ilon,imon],rs,std[i])
                            #print PDF.sum()*step[ilev,ilat,ilon,imon]
                            dINP=PDF*Nd[i,ilev,ilat,ilon,imon]*ff
                            dINP=dINP.sum()*step[ilev,ilat,ilon,imon]
                        INP[i,ilev,ilat,ilon,imon]=INP[i,ilev,ilat,ilon,imon]+dINP
                        #print 'INP',INP[i,ilev,ilat,ilon,imon]
            print INP[i,ilev,:,:,:].mean()
    return INP

INP_feld_int=np.zeros((38,7, 31, 64, 128, 12))
for t in range(38):
    INP_feld_int[t,:,:,:,:,:]=calculate_INP_feld_int(-t+273.15)

INP_total_feld_int=INP_feld_int.sum(axis=1)

np.save('INP_feld_int_alltemps',INP_total_feld_int)

#np.save('INP_20_int_sol',INP)

#%%

INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_int_alltemps.npy').mean(axis=-1)
INP_observations=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/MURRAY.dat',delimiter="\t",skip_header=1)
#ros_gulf_data[:,2]=ros_gulf_data[:,2]*1e6
modeled_feld=jl.obtain_points_from_data(INP_feldext,INP_observations,plvs=31)



plt.figure()


plt.scatter(INP_observations[:,2],modeled_feld[:,0],marker='o',c=INP_observations[:,1],alpha=1,s=50)#label='Rosinsky gulf of MX feldspar+marine')
x=np.linspace(1e-9,1e8,100)
plt.ylabel('Simulated ($m^{-3}$)')
plt.xlabel('Observed ($m^{-3}$)')
plt.plot(x,x,'k-')
plt.legend()
plt.plot(x,10*x,'k--')
plt.plot(x,0.1*x,'k--')
plt.xlim(1e-7,10)
plt.ylim(1e-11,1e6)
plt.xscale('log')
plt.yscale('log')
plt.show()
#%%
levels=np.logspace(-5,1,7).tolist()
jl.plot(INP_feldext[0,30,:,:],show=1,cmap=plt.cm.RdBu_r,clevs=levels,colorbar_format_sci=1)

#%%
INP_feldext_integrated=np.load('/nfs/a107/eejvt/JB_TRAINING/Feldspar_distributions/INP_feld_ext_alltemps.npy').mean(axis=-1)
INP_feldext_old=np.load('/nfs/a107/eejvt/JB_TRAINING/inp_dust_alltemp_ym_ext.npy')

ratio=INP_feldext_old/INP_feldext_integrated
logclevs=np.logspace(-5,5,11).tolist()
jl.plot(ratio[20,20,:,:],clevs=logclevs,colorbar_format_sci=1,cblabel='Ratio old/new',title='600hpa 20C')
jl.plot(ratio[15,30,:,:],clevs=logclevs,colorbar_format_sci=1,cblabel='Ratio old/new',title='Surf 15C')
jl.plot(ratio[20,30,:,:],clevs=logclevs,colorbar_format_sci=1,cblabel='Ratio old/new',title='Surf 20C')
#%%

 s.tot_mc_feldspar_mm[20,32:,118,[11,0,1]].mean()
INP_feldext_ambient=np.load('/nfs/a107/eejvt/JB_TRAINING/Feldspar_distributions/INP_ambient_feld_ext.npy').sum(axis=0)

modes_vol=volumes_of_modes(s)
kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
Np_kfeld=(s.st_nd*kfeld_volfrac).sum(axis=0)
jl.plot(Np_kfeld[30,:,:,:].mean(axis=-1),clevs=[0,0.001,0.1,1,10,100,1000],title='Conc surface cm-3')
