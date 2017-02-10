# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 17:31:55 2015

@author: eejvt
"""
import numpy.ma as ma
import sys
import random
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
#import cartopy.crs as ccrs
#import cartopy.feature as cfeat
from scipy.io.idl import readsav
from scipy.optimize import minimize
import scipy as sp
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from glob import glob
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
import Jesuslib as jl
archive_directory='/nfs/a107/eejvt/'
project='JB_TRAINING/'


os.chdir(archive_directory+project)

rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
rhocomp =rhocomp*1e+9#ug/m3
'''
;           Sulphate    Carbon    1.4*BC   NaCl      SiO2,      SOC   Feldspar                               
    mm_aer = [0.098,    0.012,   0.0168 , 0.05844,    0.10,    0.0168 ,0.1] ;SOC was 0.15  ; molar masses in Kg/mol
    rhocomp  = [1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.];masss density of components in Kg/m3
'''
#%%
s1=jl.read_data('WITH_ICE_SCAV2')

def volumes_of_modes(s):
    modes_vol=(s.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]
    +s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2]
    +s.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
    +s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]
    +s.tot_mc_oc_mm_mode[:,:,:,:,:]/rhocomp[5]
    +s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])
    return modes_vol
'''
def meyers_param(T,units_cm=0):
    if units_cm:
        return np.exp(-2.8+0.262*(273.15-T))*1e-3
    else:
        return np.exp(-2.8+0.262*(273.15-T))#L-1
'''        
def meyers_param(T,units_cm=0):
    a=-0.639
    b=0.1296
    return np.exp(a+b*(100*(jl.saturation_ratio_C(T)-1)))#L-1
#%%
#print 'FINE'
#demott=np.load(archive_directory+'JB_TRAINING/'+'demott' + '.npy')
#ss_vol=s1.tot_mc_su_mm_mode/rhocomp[1]+s1.tot_mc_ss_mm_mode/rhocomp[3]
#ss_vol=s1.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]+s1.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2]+s1.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]+s1.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]+s1.tot_mc_oc_mm_mode[:,:,:,:,:]/rhocomp[5]+s1.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6]
ss_vol=s1.tot_mc_dust_mm_mode/rhocomp[4]
ss_vol=s1.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]#+s1.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
ss_vol=s1.tot_mc_dust_mm_mode/rhocomp[4]+s1.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6]
ss_vol=+s1.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
modes_vol=volumes_of_modes(s1)#m3
volfrac_ss=ss_vol/modes_vol
ss_particles_ext=volfrac_ss*s1.st_nd

ss_particles_05=ss_particles_ext[3,]+(ss_particles_ext[2,]-jl.lognormal_cummulative(ss_particles_ext[2,],250e-9,s1.rbardry[2,:,:,:,:],s1.sigma[2]))


partial_acc=s1.st_nd[2,:,:,:,:]-jl.lognormal_cummulative(s1.st_nd[2,:,:,:,:],250e-9,s1.rbardry[2,:,:,:,:],s1.sigma[2])
#+s1.st_nd[5,:,:,:,:]-jl.lognormal_cummulative(s1.st_nd[5,:,:,:,:],250e-9,s1.rbardry[5,:,:,:,:],s1.sigma[5])

#n05=s1.st_nd[3,:,:,:,:]+partial_acc#+s1.st_nd[6,:,:,:,:]#-ss_particles_05-ss_particles_05
n05=s1.st_nd[3,:,:,:,:]+partial_acc#-ss_particles_05#+s1.st_nd[6,:,:,:,:]#-ss_particles_05
#%%
jl.plot(ss_particles_05.mean(axis=-1)[30,:,:]/n05.mean(axis=-1)[30,:,:],clevs=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],cblabel='Fraction of SS particles')
jl.plot(ss_particles_05.mean(axis=-1)[30,:,:]/n05.mean(axis=-1)[30,:,:],clevs=np.logspace(-4,1,6).tolist(),cblabel='Fraction of SS particles')
jl.plot(partial_acc.mean(axis=-1)[30,:,:]/s1.st_nd[3,].mean(axis=-1)[30,:,:],clevs=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
#%%
n05=s1.st_nd[3,:,:,:,:]+partial_acc-ss_particles_05#+s1.st_nd[6,:,:,:,:]#-ss_particles_05
#n05=n05
ts=40
demott=np.zeros((ts, 31, 64, 128,12))
meyers=np.zeros((ts, 31, 64, 128))
for t in range(ts):
    print t
    demott[t,]=jl.demott_parametrization(n05,273.15-t)*1e-6
    meyers[t,]=meyers_param(-t)*1e-3#cm3

np.save('/nfs/a201/eejvt/demott.npy',demott)
np.save('/nfs/a201/eejvt/meyers.npy',meyers)
np.save('demott',demott)
#%%
meyers=np.load('/nfs/a201/eejvt/meyers.npy')
demott=np.load('/nfs/a201/eejvt/demott.npy')
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MURRAY.dat",header=1)
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MURRAY_for_plotting.dat",header=1)
INP_obs_mason=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/R_H_MASON",header=1)
INP_obs_mason=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/demott2015",header=1)
INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')#m3
INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e6 #m3
#INP_feldext=np.load('/nfs/a201/eejvt/CLIMATOLOGY/2001/INP_feldext_alltemps_2001.npy')*1e6 #m3
INP_feldspar_climatology=np.load('/nfs/a201/eejvt/CLIMATOLOGY/INP_feldspar_climatology.npy')*1e6 #m3
INP_feldspar_climatology_std=np.load('/nfs/a201/eejvt/CLIMATOLOGY/INP_feldspar_climatology_std.npy')*1e6 #m3

INP_total=INP_marine_alltemps+INP_feldext
INP_total_year_mean=INP_total.mean(axis=-1)*1e-6#cm-3
INP_niemand=np.load('/nfs/a201/eejvt/INP_niemand_ext_alltemps.npy')
INP_osullivan=np.load('/nfs/a201/eejvt/INP_osullivan_ext_alltemps.npy')
#%%
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/Conen_JFJ.dat",header=1)
'''
jl.fitandplot_comparison(demott,INP_obs)
jl.fitandplot_comparison(INP_total_year_mean,INP_obs,cmap=plt.cm.OrRd,marker='o',marker_size=50)
jl.fitandplot_comparison(INP_total_year_mean,INP_obs_mason,cmap=plt.cm.OrRd,marker='^',marker_size=100)
jl.fitandplot_comparison(INP_feldext.mean(axis=-1)*1e-6,INP_obs,marker='^')
jl.fitandplot_comparison(INP_niemand.mean(axis=-1),INP_obs)#cm3
jl.fitandplot_comparison(INP_niemand.mean(axis=-1)+INP_marine_alltemps.mean(axis=-1)*1e-6,INP_obs)#cm3

jl.fitandplot_comparison(INP_marine_alltemps.mean(axis=-1)*1e-6,INP_obs)#cm3
'''

#%%
#%%
#%%
fig=plt.figure(figsize=(25,10))
m = fig.add_subplot(1,1,1)
m = Basemap(projection='cyl',lon_0=0)
m.drawcoastlines()
rand_size=2

m.drawcountries()
#INP_obs=INPconc
#INP_obs=INPconc_mason
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/TERRESTIAL_INFLUENCED.dat",header=1)
#m.bluemarble()  

#INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",header=1)
lons=INP_obs[:,4]
lats=INP_obs[:,3]
if np.any((lons<-180)):
    lons=INP_obs[:,4]
    lons[lons<-180]=lons[lons<-180]+360
    #lons=lons+360                 
if np.any((lons[:]>180)):
    lons=INP_obs[:,4]
    lons[lons>180]=lons[lons>180]-360
    #lons=lons+360                 

valid=[(lons[:]>-180)&(lons[:]<180)&(lats[:]>-90) & (lats[:]<90)]
#xx,yy=lons[valid],lats[valid]
xx,yy = m(lons[valid],lats[valid])
print xx.max(),xx.min(),yy.max(),yy.min(),xx.shape,yy.shape
print any(np.isnan(xx)),any(np.isnan(xx))

for i in range(len(xx)):
    xx[i]=xx[i]+random.random()*rand_size
for i in range(len(yy)):
    yy[i]=yy[i]+random.random()*rand_size


m.scatter(xx,yy,c=INP_obs[:,1],s=180,marker='o',vmin=-35, vmax=0,cmap=plt.cm.RdBu_r)#,label=camp.name)


INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",header=1)
lons=INP_obs[:,4]
lats=INP_obs[:,3]
if np.any((lons<-180)):
    lons=INP_obs[:,4]
    lons[lons<-180]=lons[lons<-180]+360
    #lons=lons+360                 
if np.any((lons[:]>180)):
    lons=INP_obs[:,4]
    lons[lons>180]=lons[lons>180]-360
    #lons=lons+360                 

valid=[(lons[:]>-180)&(lons[:]<180)&(lats[:]>-90) & (lats[:]<90)]
#xx,yy=lons[valid],lats[valid]
xx,yy = m(lons[valid],lats[valid])
print xx.max(),xx.min(),yy.max(),yy.min(),xx.shape,yy.shape
print any(np.isnan(xx)),any(np.isnan(xx))

for i in range(len(xx)):
    xx[i]=xx[i]+random.random()*rand_size
for i in range(len(yy)):
    yy[i]=yy[i]+random.random()*rand_size





m.scatter(xx,yy,c=INP_obs[:,1],s=250,marker='^',vmin=-35, vmax=0,cmap=plt.cm.RdBu_r)#,label=camp.name)



cb=plt.colorbar()
cb.set_label('T  $^oC$')

plt.savefig('/nfs/see-fs-01_users/eejvt/terrestial_marine/Locations_marine_terrestial.png')

#%%
#%%
plt.show()


#%%



simulated_values=INP_osullivan.mean(axis=-1)*0.01+INP_marine_alltemps.mean(axis=-1)*1e-6
title='OSullivan001+marine'



simulated_values=INP_osullivan.mean(axis=-1)*0.20
title='OSullivan'





simulated_values=INP_feldspar_climatology.mean(axis=-1)*1e-6
simulated_values_max=INP_feldspar_climatology.max(axis=-1)*1e-6
simulated_values_min=INP_feldspar_climatology.min(axis=-1)*1e-6
errors=1
title='Feldspar climatology'
#%%

bacteria_surface=np.zeros((38, 31, 64, 128))

def psyringae_param(T):
    if T<-5:
        T=-5
    return np.exp(-0.4325*T**2-5.1067*T-10.399)

conc_bact=1e5#m-3
conc_bact=1e6*0.01#m-3
conc_bact=conc_bact*1e-6#cm-3
radius_bact=(10**-4)/2.#cm

param=[]
for t in range (38):
    print t
    param.append(psyringae_param(-t))
    #plt.plot(param)
    
    bacteria_surface[t,30,:,:]=conc_bact*(1-np.exp(-psyringae_param(-t)*4*np.pi*radius_bact**2))
    
simulated_values=INP_total_year_mean+bacteria_surface*jl.terrestrial_grid

simulated_values_max=INP_total.max(axis=-1)*1e-6
simulated_values_min=INP_total.min(axis=-1)*1e-6
errors=0#
title='Marine+Feldspar+bacteria surface'


#%%





simulated_values=meyers
title='Meyers'


#%%
title='Feldspar'
simulated_values_max=0
simulated_values_min=0
simulated_values_max=INP_niemand.max(axis=-1)
simulated_values_min=INP_niemand.min(axis=-1)
simulated_values=INP.mean(axis=-1)#*1e-6
simulated_values=INP_feldext.mean(axis=-1)*1e-6
simulated_values=INP_feld_ext[:,3,].mean(axis=-1)+INP_marine_alltemps.mean(axis=-1)*1e-6+INP_feld_ext[:,2,].mean(axis=-1)/5
errors=0
title='Feldspar sol_scavenged'

simulated_values=INP_niemand.mean(axis=-1)
simulated_values_max=INP_niemand.max(axis=-1)
simulated_values_min=INP_niemand.min(axis=-1)
errors=1

title='Niemand dust'



simulated_values=INP_niemand.mean(axis=-1)/10+INP_marine_alltemps.mean(axis=-1)*1e-6
simulated_values_max=INP_niemand.max(axis=-1)/10+INP_marine_alltemps.mean(axis=-1)*1e-6
simulated_values_min=INP_niemand.min(axis=-1)/10+INP_marine_alltemps.mean(axis=-1)*1e-6
errors=1

title='Niemand dust /10'
errors=0#

simulated_values=demott.mean(axis=-1)
simulated_values_max=demott.max(axis=-1)
simulated_values_min=demott.min(axis=-1)
errors=1
title='DeMott 2010'
#%%



simulated_values=INP_total_year_mean
simulated_values_max=INP_total.max(axis=-1)*1e-6
simulated_values_min=INP_total.min(axis=-1)*1e-6
errors=1#
title='Marine+Feldspar'
'''
simulated_values=INP_feld_solrem.mean(axis=-1)
jl.plot((INP_feld_solrem.mean(axis=-1)[20,30,])/(INP_feldext.mean(axis=-1)[20,30,]),clevs=np.logspace(-0.0001,0.0001,25).tolist())
'''
INP_obs_mason=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/KAD_South_Pole_myformat.dat",header=1)
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/KAD_Israel_myformat.dat",header=1)
INP_obs_mason=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",header=1)
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/TERRESTIAL_INFLUENCED.dat",header=1)


INP_obs=INP_obs[INP_obs[:,1]>-26]
plt.figure()
cmap=plt.cm.RdBu_r
marker='o'
marker_mason='^'
marker_size=50
marker_size_mason=120

INPconc=INP_obs
INPconc_mason=INP_obs_mason
simulated_points=jl.obtain_points_from_data(simulated_values,INPconc,plvs=0,surface_level_comparison_on=True)
simulated_points_max=jl.obtain_points_from_data(simulated_values_max,INPconc,plvs=0,surface_level_comparison_on=True)
simulated_points_min=jl.obtain_points_from_data(simulated_values_min,INPconc,plvs=0,surface_level_comparison_on=True)
simulated_points_mason=jl.obtain_points_from_data(simulated_values,INPconc_mason,plvs=0,surface_level_comparison_on=True)
simulated_points_mason_max=jl.obtain_points_from_data(simulated_values_max,INPconc_mason,plvs=0,surface_level_comparison_on=True)
simulated_points_mason_min=jl.obtain_points_from_data(simulated_values_min,INPconc_mason,plvs=0,surface_level_comparison_on=True)
data_points=INPconc
data_points_mason=INPconc_mason
bias=np.log10(simulated_points[:,0])-np.log10(data_points[:,2])
bias_mason=np.log10(simulated_points_mason[:,0])-np.log10(data_points_mason[:,2])

if errors:
    plt.errorbar(data_points_mason[:,2],simulated_points_mason[:,0],
                 yerr=[simulated_points_mason[:,0]-simulated_points_mason_min[:,0],simulated_points_mason_max[:,0]-simulated_points_mason[:,0]],
                 linestyle="None",c='k',zorder=0)
    plt.errorbar(data_points[:,2],simulated_points[:,0],
                 yerr=[simulated_points[:,0]-simulated_points_min[:,0],simulated_points_max[:,0]-simulated_points[:,0]],
                linestyle="None",c='k',zorder=0)
plot=plt.scatter(data_points[:,2],simulated_points[:,0],c=data_points[:,1],cmap=cmap,marker=marker,s=marker_size,vmin=-35, vmax=0)
plot=plt.scatter(data_points_mason[:,2],simulated_points_mason[:,0],c=data_points_mason[:,1],cmap=cmap,marker=marker_mason,s=marker_size_mason,vmin=-35, vmax=0)
    #plt.errorbar(data_points[:,2],simulated_points[:,0],yerr=[simulated_points_min[:,0],simulated_points_max[:,0]], linestyle="None",c='k')
    
#plot=plt.scatter(data_points[:,2],simulated_points[:,0],c=bias,cmap=cmap,marker=marker,s=marker_size,vmin=-5, vmax=5)
#plot=plt.scatter(data_points_mason[:,2],simulated_points_mason[:,0],c=bias_mason,cmap=cmap,marker=marker_mason,s=marker_size_mason,vmin=-5, vmax=5)


plt.colorbar(plot,label='Temperature $C$')

plt.ylabel('Simulated ($cm^{-3}$)')
plt.xlabel('Observed ($cm^{-3}$)')

if np.min(simulated_points[:,0])>np.min(data_points[:,2]):
    min_plot=np.min(data_points[:,2])
else:
    min_plot=np.min(simulated_points[:,0])

if np.max(simulated_points[:,0])<np.max(data_points[:,2]):
    max_plot=np.max(data_points[:,2])
else:
    max_plot=np.max(simulated_points[:,0])
    
#minx=np.min(data_points[:,2])
#maxx=np.max(data_points[:,2])
#miny=np.min(simulated_points[:,0])
#maxy=np.max(simulated_points[:,0])
min_val=1e-9
max_val=1e1
minx=np.min(min_val)
maxx=np.max(max_val)
miny=np.min(min_val)
maxy=np.max(max_val)
min_plot=min_val
max_plot=max_val


plt.title(title)
x=np.linspace(0.1*min_plot,10*max_plot,100)
#global x     
r=np.corrcoef(np.log(data_points[:,2]),np.log(simulated_points[:,0]))
print r
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
#plt.savefig('/nfs/see-fs-01_users/eejvt/terrestial_marine/'+title+'.png')
plt.show()
#%%





fig=plt.figure(figsize=(25, 10))
m = fig.add_subplot(1,1,1)
m = Basemap(projection='cyl',lon_0=0)
m.drawcoastlines()
m.drawcountries()

INP_obs=INPconc_mason
INP_obs=INPconc
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/TERRESTIAL_INFLUENCED.dat",header=1)
INP_obs=INP_obs[INP_obs[:,1]>-26]
#m.bluemarble()  

lons=INP_obs[:,4]
lats=INP_obs[:,3]
if np.any((lons<-180)):
    lons=INP_obs[:,4]
    lons[lons<-180]=lons[lons<-180]+360
    #lons=lons+360                 
if np.any((lons[:]>180)):
    lons=INP_obs[:,4]
    lons[lons>180]=lons[lons>180]-360
    #lons=lons+360                 

valid=[(lons[:]>-180)&(lons[:]<180)&(lats[:]>-90) & (lats[:]<90)]
#xx,yy=lons[valid],lats[valid]
xx,yy = m(lons[valid],lats[valid])
print xx.max(),xx.min(),yy.max(),yy.min(),xx.shape,yy.shape
print any(np.isnan(xx)),any(np.isnan(xx))

rand_size=5
for i in range(len(xx)):
    xx[i]=xx[i]+random.random()*rand_size
for i in range(len(yy)):
    yy[i]=yy[i]+random.random()*rand_size

#m.scatter(xx,yy,c=INP_obs[:,1],s=250,marker='^',vmin=-35, vmax=0,cmap=plt.cm.RdBu_r)#,label=camp.name)
#m.scatter(xx[np.abs(bias[:])>1.5],yy[np.abs(bias[:])>1.5],c=bias[np.abs(bias[:])>1.5],s=180,marker='o',vmin=-5, vmax=5,cmap=plt.cm.RdBu_r)#,label=camp.name)
m.scatter(xx[(bias[:])<-1.5],yy[(bias[:])<-1.5],c=INP_obs[:,1][(bias[:])<-1.5],s=180,marker='o',vmin=-35, vmax=0,cmap=plt.cm.RdBu_r)#,label=camp.name)
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",header=1)

lons=INP_obs[:,4]
lats=INP_obs[:,3]
if np.any((lons<-180)):
    lons=INP_obs[:,4]
    lons[lons<-180]=lons[lons<-180]+360
    #lons=lons+360                 
if np.any((lons[:]>180)):
    lons=INP_obs[:,4]
    lons[lons>180]=lons[lons>180]-360
    #lons=lons+360                 

valid=[(lons[:]>-180)&(lons[:]<180)&(lats[:]>-90) & (lats[:]<90)]
#xx,yy=lons[valid],lats[valid]
xx,yy = m(lons[valid],lats[valid])
print xx.max(),xx.min(),yy.max(),yy.min(),xx.shape,yy.shape
print any(np.isnan(xx)),any(np.isnan(xx))

rand_size=5
for i in range(len(xx)):
    xx[i]=xx[i]+random.random()*rand_size
for i in range(len(yy)):
    yy[i]=yy[i]+random.random()*rand_size

#m.scatter(xx,yy,c=INP_obs[:,1],s=250,marker='^',vmin=-35, vmax=0,cmap=plt.cm.RdBu_r)#,label=camp.name)
#m.scatter(xx[np.abs(bias_mason[:])>1.5],yy[np.abs(bias_mason[:])>1.5],c=bias_mason[:][np.abs(bias_mason[:])>1.5],s=250,marker='^',vmin=-5, vmax=5,cmap=plt.cm.RdBu_r)#,label=camp.name)
m.scatter(xx[(bias_mason[:])<-1.5],yy[(bias_mason[:])<-1.5],c=INP_obs[:,1][(bias_mason[:])<-1.5],s=250,marker='^',vmin=-35, vmax=0,cmap=plt.cm.RdBu_r)#,label=camp.name)

plt.title('b)Understimated concentrations',fontsize=25)
cb=plt.colorbar()
cb.set_label('T  $^oC$',fontsize=20)
cb.ax.tick_params(labelsize=20)
plt.show()
plt.savefig('/nfs/see-fs-01_users/eejvt/terrestial_marine/Understimation_places.png')
#%%





fig=plt.figure(figsize=(25, 10))
m = fig.add_subplot(1,1,1)
m = Basemap(projection='cyl',lon_0=0)
m.drawcoastlines()
m.drawcountries()

INP_obs=INPconc_mason
INP_obs=INPconc
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/TERRESTIAL_INFLUENCED.dat",header=1)
INP_obs=INP_obs[INP_obs[:,1]>-36]
#m.bluemarble()  

lons=INP_obs[:,4]
lats=INP_obs[:,3]
if np.any((lons<-180)):
    lons=INP_obs[:,4]
    lons[lons<-180]=lons[lons<-180]+360
    #lons=lons+360                 
if np.any((lons[:]>180)):
    lons=INP_obs[:,4]
    lons[lons>180]=lons[lons>180]-360
    #lons=lons+360                 

valid=[(lons[:]>-180)&(lons[:]<180)&(lats[:]>-90) & (lats[:]<90)]
#xx,yy=lons[valid],lats[valid]
xx,yy = m(lons[valid],lats[valid])
print xx.max(),xx.min(),yy.max(),yy.min(),xx.shape,yy.shape
print any(np.isnan(xx)),any(np.isnan(xx))
rand_size=5
for i in range(len(xx)):
    xx[i]=xx[i]+random.random()*rand_size
for i in range(len(yy)):
    yy[i]=yy[i]+random.random()*rand_size

#m.scatter(xx,yy,c=INP_obs[:,1],s=250,marker='^',vmin=-35, vmax=0,cmap=plt.cm.RdBu_r)#,label=camp.name)
#m.scatter(xx[np.abs(bias[:])>1.5],yy[np.abs(bias[:])>1.5],c=bias[np.abs(bias[:])>1.5],s=180,marker='o',vmin=-5, vmax=5,cmap=plt.cm.RdBu_r)#,label=camp.name)
m.scatter(xx[(bias[:])>1.5],yy[(bias[:])>1.5],c=INP_obs[:,1][(bias[:])>1.5],s=180,marker='o',vmin=-35, vmax=0,cmap=plt.cm.RdBu_r)#,label=camp.name)
print len(xx[(bias[:])>1.5]),len(xx[(bias[:])>1])
print len(xx[(bias[:])<-1.5]),len(xx[(bias[:])<-1])
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",header=1)

lons=INP_obs[:,4]
lats=INP_obs[:,3]
if np.any((lons<-180)):
    lons=INP_obs[:,4]
    lons[lons<-180]=lons[lons<-180]+360
    #lons=lons+360                 
if np.any((lons[:]>180)):
    lons=INP_obs[:,4]
    lons[lons>180]=lons[lons>180]-360
    #lons=lons+360                 

valid=[(lons[:]>-180)&(lons[:]<180)&(lats[:]>-90) & (lats[:]<90)]
#xx,yy=lons[valid],lats[valid]
xx,yy = m(lons[valid],lats[valid])
print xx.max(),xx.min(),yy.max(),yy.min(),xx.shape,yy.shape
print any(np.isnan(xx)),any(np.isnan(xx))

rand_size=5
for i in range(len(xx)):
    xx[i]=xx[i]+random.random()*rand_size
for i in range(len(yy)):
    yy[i]=yy[i]+random.random()*rand_size

#m.scatter(xx,yy,c=INP_obs[:,1],s=250,marker='^',vmin=-35, vmax=0,cmap=plt.cm.RdBu_r)#,label=camp.name)
#m.scatter(xx[np.abs(bias_mason[:])>1.5],yy[np.abs(bias_mason[:])>1.5],c=bias_mason[:][np.abs(bias_mason[:])>1.5],s=250,marker='^',vmin=-5, vmax=5,cmap=plt.cm.RdBu_r)#,label=camp.name)
m.scatter(xx[(bias_mason[:])>1.5],yy[(bias_mason[:])>1.5],c=INP_obs[:,1][(bias_mason[:])>1.5],s=250,marker='^',vmin=-35, vmax=0,cmap=plt.cm.RdBu_r)#,label=camp.name)
print len(xx[(bias_mason[:])>1.5]),len(xx[(bias_mason[:])>1])
print len(xx[(bias_mason[:])<-1.5]),len(xx[(bias_mason[:])<-1])
plt.title('a)Overstimated concentrations',fontsize=25)
cb=plt.colorbar()
cb.set_label('T  $^oC$',fontsize=20)
cb.ax.tick_params(labelsize=20)
plt.savefig('/nfs/see-fs-01_users/eejvt/terrestial_marine/Overstimation_places.png')
#%%
ratio=INP_total_year_mean[20,]/meyers[20,]
jl.plot(ratio[30,:,:],clevs=np.logspace(-3,3,13).tolist())