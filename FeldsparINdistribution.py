# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:07:22 2014

@author: eejvt
"""

#INP CALCULATION BASED ON FELDSPAR
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
rhocomp =rhocomp*1e+3#ug/cm3
'''
;           Sulphate    Carbon    1.4*BC   NaCl      SiO2,      SOC   Feldspar                               
    mm_aer = [0.098,    0.012,   0.0168 , 0.05844,    0.10,    0.0168 ,0.1] ;SOC was 0.15  ; molar masses in Kg/mol
    rhocomp  = [1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.];masss density of components in Kg/m3
'''

feld_paper_clevs=[0,0.0001,0.0002,0.0005,0.0010,0.002,0.005,0.01,0.02,0.05,0.1]

def lognormal_PDF(rmean,r,std):
   X=(1/(r*std*np.sqrt(2*np.pi)))*np.exp(-(np.log(r)-rmean)**2/(2*std**2))
   return X
   
#rs=np.linspace(-5,5,100)
#plt.plot(rs,lognormal_PDF(-2,rs,np.sqrt(0.5)))
#plt.show()
    
#%%
s=jl.read_data('WITH_ICE_SCAV2')
sigma=s.sigma[:]
def area_lognormal(rbar,sigma,Nd):
    print isinstance(sigma,float)
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

def kfeld_INP(s,T):
    modes_vol=volumes_of_modes(s)
    print 'modevol',modes_vol[:,15,50,50,1]
    nd=s.st_nd[:,:,:,:,:]
    
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
    print s.rbardry.shape,s.sigma.shape,(nd*kfeld_volfrac).shape
    kfeld_s_ext=area_lognormal(s.rbardry[:,:,:,:,:]*1e2,s.sigma[:],nd*kfeld_volfrac)
    print kfeld_s_ext.shape
    kfeld_s_int=area_lognormal(s.rbardry[:,:,:,:,:]*1e2,s.sigma[:],nd)*kfeld_volfrac
    ns=feld_parametrization(T)
    global kfeld_s_ext, kfeld_s_int
    if ns.shape==(31,64,128,365):
        month_days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        iold=0
        kfeld_s_ext_daily=np.zeros((7,31,64,128,365))
        kfeld_s_int_daily=np.zeros((7,31,64,128,365))
        kfeld_volfrac_daily=np.zeros((7,31,64,128,365))
        nd_daily=np.zeros((7,31,64,128,365))
        for i in range(len(month_days)):
            print i
            for j in range (month_days[i]):
                kfeld_s_ext_daily[:,:,:,:,iold+j]=kfeld_s_ext[:,:,:,:,i]
                kfeld_s_int_daily[:,:,:,:,iold+j]=kfeld_s_int[:,:,:,:,i]
                kfeld_volfrac_daily[:,:,:,:,iold+j]=kfeld_volfrac[:,:,:,:,i]
                nd_daily[:,:,:,:,iold+j]=nd[:,:,:,:,i]
            iold=iold+month_days[i]
        #kfeld_volfrac=np.copy(kfeld_volfrac_daily)
        #kfeld_INP_ext=kfeld_s_ext_daily*ns#*1e-6
        #kfeld_INP_int=kfeld_s_int_daily*ns#*1e-6
        ff_ext=1.0-np.exp(-(ns*kfeld_s_ext_daily))
        kfeld_INP_ext=ff_ext*nd_daily*kfeld_volfrac_daily
        #ff_int=1.0-np.exp(-(ns*kfeld_s_int_daily))
    else:    
        #kfeld_INP_ext=kfeld_s_ext*ns#*1e-6
        #kfeld_INP_int=kfeld_s_int*ns#*1e-6
        ff_ext=1.0-np.exp(-(ns*kfeld_s_ext))
    #print ff_ext[:,15,50,50,1]
        ff_int=1.0-np.exp(-(ns*kfeld_s_int))
        kfeld_INP_ext=ff_ext*nd*kfeld_volfrac
    print 'ff calculado'
    print 'external calculado'    
    #kfeld_INP_int=ff_int*nd#chequear esto
    kfeld_INP_int=0
    return kfeld_INP_ext,kfeld_INP_int

#%%
joclevs=[0,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]#liters-1
inp_ext,inp_int=kfeld_INP(s,258)
inp_ext_600_L=inp_ext.sum(axis=0).mean(axis=-1)[18,]*1e3
#inp_int_600_L=inp_int.sum(axis=0).mean(axis=-1)[18,]*1e3
inp_ext_tot=inp_ext.sum(axis=0).mean(axis=0).mean(axis=-1)
#inp_int_tot=inp_int.sum(axis=0).mean(axis=0).mean(axis=-1)
#jl.plot(inp_ext_600_L,show=1,clevs=joclevs,projection='merc',title='ext',file_name='Feldspar_600hp_2000_ext',cblabel='$L^{-1}$')
#jl.plot(inp_int_600_L,show=1,clevs=joclevs,projection='merc',title='int',file_name='Feldspar_600hp_2000_int',cblabel='$L^{-1}$')

inp_ext_tot_ins=(inp_ext[4,]+inp_ext[5,]+inp_ext[6,]).mean(axis=0).mean(axis=-1)

temps_daily=np.load('/nfs/a107/eejvt/JB_TRAINING/temperatures_daily.npy')

temps=temps_daily-273.15
mixed_phase_range_up=temps<-6
mixed_phase_range_down=temps>-25
mixed_phase_range=mixed_phase_range_down*mixed_phase_range_up
mixed_out_from_param1=temps<-25
mixed_out_from_param2=temps>-37
mixed_out_from_param=mixed_out_from_param1*mixed_out_from_param2
mixed_false=mixed_phase_range[True-mixed_phase_range]
#jl.plot(mixed_phase_range[30,:,:],show=1,title='Mean temperature')

#jl.plot(temps[15,:,:],show=1,clevs=np.linspace(0,-40,15).tolist())
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
glopressure_mean=s.pl_m.mean(axis=(-1,-2,-3))*1e-2
#%%







#jl.plot(inp_ext_tot_ins,show=1,projection='merc',title='int',file_name='Feldspar_600hp_2000_extins',cblabel='$L^{-1}$')
#INP_data=np.genfromtxt("/nfs/a107/eejvt/INP_DATA/MURRAY.dat",delimiter='\t',skip_header=1)

'''
ts=temps_daily*mixed_phase_range
ts[mixed_out_from_param]=248
ts[ts==0]=90000
INP_feldext_glotemps,INP_feldint_glotemps=kfeld_INP(s,ts)
print 'tenemos los INP'
INP_feldext_glotemps[np.isnan(INP_feldext_glotemps)]=0
INP_feldext_ambient=INP_feldext_glotemps.sum(axis=0)*1e6
np.save('INP_feldext_ambient.npy',INP_feldext_ambient)
INP_feldext_glotemps=INP_feldext_glotemps.sum(axis=0).mean(axis=-1)*1e6#m3
print 'calculos hechos'
#INP_feldint_glotemps=INP_feldint_glotemps.sum(axis=0).mean(axis=-1)
np.save('INP_feldext_glotemps_daily',INP_feldext_glotemps)
#np.save('INP_feldint_glotemps_daily',INP_feldint_glotemps)
print 'salvado'
INP_feldext_glotemps=np.load('INP_feldext_glotemps_daily.npy')
levels=np.logspace(-1,5,15).tolist()
jl.plot(INP_feldext_glotemps[20,:,:],show=1,clevs=levels)
#levels=np.logspace(-8,2,10).tolist()
X, Y = np.meshgrid(lat.glat, glopressure_mean)
fig=plt.figure()
ax=plt.subplot(1,1,1)
plt.contourf(X,Y,INP_feldext_glotemps.mean(axis=-1),levels,norm= colors.BoundaryNorm(levels, 256))
#plt.contourf(X,Y,s.st_nd.sum(axis=(0,-1)).mean(axis=-1)*1e6,levels,norm = LogNorm())
plt.colorbar(ax=ax,ticks=levels,drawedges=1)

plt.gca().invert_yaxis()
plt.show()
'''
#INP_feldext_glotemps=np.load('INP_feldext_glotemps'+'.npy')
#INP_feldint_glotemps=np.load('INP_feldint_glotemps' + '.npy')
'''
inps_int=np.zeros((38,7,31,64,128,12))
inps_ext=np.zeros((38,7,31,64,128,12))
for i in range(38):
    inps_ext[i,],inps_int[i,]=kfeld_INP(s,273.15-i)
inp_int_ym_ts=inps_int.sum(axis=1)
inp_ext_ym_ts=inps_ext.sum(axis=1)
inp_ext_ym_ts=inp_ext_ym_ts.mean(axis=-1)
inp_int_ym_ts=inp_int_ym_ts.mean(axis=-1)
'''

#np.save('inp_dust_alltemp_ym_ext',inp_ext_ym_ts)
#np.save('inp_dust_alltemp_ym_int',inp_int_ym_ts)


#np.load(archive_directory+'JB_TRAINING/'+'inp_dust_alltemp_ym_ext' + '.npy')
#np.load(archive_directory+'JB_TRAINING/'+'inp_dust_alltemp_ym_int' + '.npy')
#inp_int_ym_ts=np.concatenate()
#inp_ext_ym_ts
def level_anual_mode_mean(array):
    if array.ndim==5:
        return array.sum(axis=0).mean(axis=0).mean(axis=-1)
    elif array.ndim==4:
        return array.mean(axis=0).mean(axis=-1)
            
def nan_to_0(a):
    a[np.isnan(a)]=0
    return a
#%%

'''
modes_vol=volumes_of_modes(s)
dust_vol_frac=s.tot_mc_dust_mm_mode/rhocomp[4]
dust_particles=dust_vol_frac*modes_vol*s.st_nd
dust_particles_means=level_anual_mode_mean(dust_particles)
'''
inp_ext=np.load('/nfs/a107/eejvt/JB_TRAINING/Feldspar_distributions/INP_feld_ext_alltemps.npy')
INPconc=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MURRAY.dat",header=1)
INPconc15=INPconc[INPconc[:,1]<-15]
#feld_mass_frac=s.tot_mc_feldspar_mm_mode/s.tot_mc_dust_mm_mode
sim,obs=obtain_points_from_data()
#sim,obv=jl.fitandplot_comparison(inp_ext.sum(axis=0),INPconc15,return_arrays=1)
plt.title('ext')
plt.show()
plt.figure()
#jl.fitandplot_comparison

#plt.title('int')
#plt.show()
INPconc=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MURRAY.dat",header=1)
#np.corrcoef(sim[:,0],obv[:,2])



