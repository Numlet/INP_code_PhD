# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:11:33 2015

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
archive_directory='/nfs/a107/eejvt/'
project='BC_INP/'
os.chdir(archive_directory+project)

sm=readsav('/nfs/a173/earjbr/mode10_setup/GLOMAP_mode_mol_no_mode10_2001.sav')

'''



NO USAR ESTE CODIGO!!!!!!!!!!!!!!!!!!!!!!! ESTA MAL!!!!!!!!!!!!!!!!!!!!!!!!






'''

def ns_BC(T):
    A=-20.27
    B=1.20
    return np.exp(A-B*T)

def INP_bc_int(sa,nd,T):
     ns=ns_BC(T)
     ff=1-np.exp(-ns*sa)
     INP=ff*nd
     return INP
     
     
#%%

#%%
     
os.chdir(archive_directory)
s=jl.read_data(project)
os.chdir(archive_directory+project)
temp_files=glob('/nfs/a173/earjbr/daily_run_ntraer30/hindcast3_temp_feldspar_*')    
month_days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
#%%
temp_daily=np.zeros((31, 64, 128, 365))
days_cumulative=0
for i in range (12):
    
    td=readsav(temp_files[i])
    for j in range (month_days[i]):
        
        temp_daily[:,:,:,days_cumulative+j]=td.t3d_mm[:,:,:,j]#The name is _mm but the array is in daily temperature
        

    days_cumulative=days_cumulative+month_days[i]
    
temp_daily=temp_daily-273.15
np.save('temperatures_daily.npy',temp_daily)
#%%
temp_files=glob('/nfs/a173/earjbr/daily_run_ntraer30/hindcast3_temp_feldspar_*')    
month_days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
press_files=glob('/nfs/a173/earjbr/daily_run_ntraer30/GLOMAP_mode_pressure_mp_*')
press_daily=np.zeros((31, 64, 128, 365))
days_cumulative=0
for i in range (12):
    
    ps=readsav(press_files[i])
    for j in range (month_days[i]):
        
        press_daily[:,:,:,days_cumulative+j]=ps.pl_m[:,:,:,j]#The name is _mm but the array is in daily temperature
        

    days_cumulative=days_cumulative+month_days[i]
press_daily=press_daily*0.01
np.save('press_daily',press_daily)
#%%
month_days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
INP_bc_ambient=np.zeros((11,31, 64, 128, 365))
temp_range=np.array(temp_daily,copy=True)
temp_range[temp_range>-16]=100
temp_range[temp_range<-37]=100
temp_range[temp_range<-23.5]=-23.5

for imode in range(7):
    imon=0
    days_cummulative=0
    for imon in range(12):  
        print imon
        for iday in range(month_days[imon]):
            INP_bc_ambient[imode,:,:,:,days_cummulative+iday]=INP_bc_int(s.bc_sa_int_na_nms[imode,:,:,:,imon],s.nd_na_nms[imode,:,:,:,imon],temp_range[:,:,:,days_cummulative+iday])

        days_cummulative=days_cummulative+month_days[imon]
        
INP_bc_ambient_sum=INP_bc_ambient.sum(axis=0)*1e6
np.save('INP_bc_ambient_na.npy',INP_bc_ambient_sum)

INP_bc_ambient=np.zeros((11,31, 64, 128, 365))
for imode in range(7):
    imon=0
    days_cummulative=0
    for imon in range(12):  
        print imon
        for iday in range(month_days[imon]):
            INP_bc_ambient[imode,:,:,:,days_cummulative+iday]=INP_bc_int(s.bc_sa_int_pd_nms[imode,:,:,:,imon],s.nd_pd_nms[imode,:,:,:,imon],temp_range[:,:,:,days_cummulative+iday])

        days_cummulative=days_cummulative+month_days[imon]
        
INP_bc_ambient_sum=INP_bc_ambient.sum(axis=0)*1e6
np.save('INP_bc_ambient_pd.npy',INP_bc_ambient_sum)
INP_bc_ambient=np.zeros((11,31, 64, 128, 365))
for imode in range(7):
    imon=0
    days_cummulative=0
    for imon in range(12):  
        print imon
        for iday in range(month_days[imon]):
            INP_bc_ambient[imode,:,:,:,days_cummulative+iday]=INP_bc_int(s.bc_sa_int_pi_nms[imode,:,:,:,imon],s.nd_pi_nms[imode,:,:,:,imon],temp_range[:,:,:,days_cummulative+iday])

        days_cummulative=days_cummulative+month_days[imon]
        
INP_bc_ambient_sum=INP_bc_ambient.sum(axis=0)*1e6
np.save('INP_bc_ambient_pi.npy',INP_bc_ambient_sum)
#%%
month_days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
INP_bc_pi=np.zeros((38,11,31, 64, 128, 12),dtype='float64')
INP_bc_pd=np.zeros((38,11,31, 64, 128, 12),dtype='float64')
INP_bc_na=np.zeros((38,11,31, 64, 128, 12),dtype='float64')
T=np.arange(0,38,1)
for t in T:
    print 'T',-t
    for imode in range(7):
        imon=0
        days_cummulative=0
        for imon in range(12):  
            temp=-t            
            if -t<-24:
                temp=-24                
            print imon
            INP_bc_pi[t,imode,:,:,:,imon]=INP_bc_int(s.bc_sa_int_pi_nms[imode,:,:,:,imon],s.nd_pi_nms[imode,:,:,:,imon],temp)
            INP_bc_na[t,imode,:,:,:,imon]=INP_bc_int(s.bc_sa_int_na_nms[imode,:,:,:,imon],s.nd_na_nms[imode,:,:,:,imon],temp)
            INP_bc_pd[t,imode,:,:,:,imon]=INP_bc_int(s.bc_sa_int_pd_nms[imode,:,:,:,imon],s.nd_pd_nms[imode,:,:,:,imon],temp)
            print INP_bc_pd[t,imode,30,50,50,imon]
        days_cummulative=days_cummulative+month_days[imon]
#%%
INP_bc_pi=INP_bc_pi.sum(axis=1)
INP_bc_pd=INP_bc_pd.sum(axis=1)
INP_bc_na=INP_bc_na.sum(axis=1)

np.save('INP_bc_pi',INP_bc_pi)
np.save('INP_bc_pd',INP_bc_pd)
np.save('INP_bc_na',INP_bc_na)


#%%


        
#%%    

glolevs=np.array([   4.47640944,   28.3048172 ,   48.96452713,   69.30890656,
 89.56233978,  110.04908752,  131.62251282,  154.64620972,
179.33183289,  205.97129822,  234.46916199,  264.84896851,
297.05499268,  330.97183228,  366.49978638,  403.52679443,
441.94363403,  481.63827515,  522.48620605,  564.35626221,
607.08886719,  650.46594238,  694.17602539,  737.8137207 ,
780.80426025,  822.40307617,  861.61694336,  897.16723633,
927.43457031,  950.37841797,  963.48803711])
INP_bc_ambient_pd=np.load('INP_bc_ambient_pd.npy')

INP_feldext_ambient=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feldext_ambient.npy')
levelsfel=0
levelsbc=[1,10,20,30,40,50,60,70,80,90,100,1000]
levelsfel=[10,100,1000]
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
fs=10
fig=plt.figure()
cx=plt.subplot(1,1,1)
Xmo,Ymo= np.meshgrid(lat.glat, glolevs)
Xfel,Yfel=np.meshgrid(lat.glat,glolevs)
cx.set_title('Year mean')
CS=cx.contour(Xfel,Yfel,INP_feldext_ambient[:,:,:,:].mean(axis=(-1,-2)),levelsfel,linewidths=np.linspace(2, 6, 3),colors='k',hold='on',)
plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
plt.setp(CS.collections )
CF=cx.contourf(Xmo,Ymo,INP_bc_ambient_pd[:,:,:,:].mean(axis=(-1,-2)),levelsbc,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsbc, 256))
CB=plt.colorbar(CF,ticks=levelsbc,drawedges=1,label='$m^{-3}$')

cx.invert_yaxis()
cx.set_ylim(ymax=200)
cx.tick_params(axis='both', which='major', labelsize=fs)
cx.set_ylabel('Pressure level $(hPa)$')
cx.set_xlabel('Latitude')
cx.xaxis.set_ticks(np.arange(-90,100,20))
plt.savefig('year_mean_lat_mean.ps',dpi=300,formatker='ps')
plt.show()
#plt.close()


#%%

#PRESENT DAY

INP_bc_ambient_pd=np.load('INP_bc_ambient_pd.npy')
INP_feldext_ambient_mean=INP_feldext_ambient.mean(axis=-1)
fs=13
#[:,:,180:211]
DJF=np.arange(-31,59,1)
mdays=[0,31,59,90,120,151,181,212,243,273,304,334,365]
mnames=['January','February','March','April','May','June','July','August','September','October','November','December']
#levelsmo=[1,10,20,30,40,50,60,70,80,90,100,500]
levelsbc=[0.0001,1,5,10,50,100,500,1000,10000]
levelsfel=[10,100,1000]
#os.system('cd gifs_maker')
#os.system('mkdir AT')
#os.chdir('gifs_maker')
#os.system('mkdir latmean2')
#os.chdir('latmean2')
#os.system('cp /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/AT/animate_psfiles /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/latmean2/')

#levelsmo=[1,10,20,30,40,50,60,70,80,90,100,400]
fs=10
for i in range(12):
    #fig=plt.figure()
    print mnames[i]
    cx=plt.subplot(4,3,i+1)
    Xmo,Ymo= np.meshgrid(lat.glat, glolevs)
    Xfel,Yfel=np.meshgrid(lat.glat,glolevs)
    cx.set_title(mnames[i])
    CS=cx.contour(Xfel,Yfel,INP_feldext_ambient[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsfel,linewidths=np.linspace(0.5, 3, 3),colors='k',hold='on',)
    plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
    plt.setp(CS.collections )
    CF=cx.contourf(Xmo,Ymo,INP_bc_ambient_pd[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsbc,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsbc, 256))
    if i==2 or i==5 or i==8 or i==11:
        CB=plt.colorbar(CF,ticks=levelsbc,drawedges=1,label='$m^{-3}$')
    
    cx.invert_yaxis()
    cx.set_ylim(ymax=200)
    cx.tick_params(axis='both', which='major', labelsize=fs)
    if i==0 or i==3 or i==6 or i==9:
        cx.set_ylabel('Pressure level $(hPa)$')
    if i>8:    
        cx.set_xlabel('Latitude')
    cx.xaxis.set_ticks(np.arange(-90,100,20))
plt.figtext(0.4,0.95,'PRESENT DAY',fontsize=20)
plt.show()
plt.savefig('latmean_grid_pd.svg',dpi=1200,format='svg')
plt.savefig('latmean_grid_pd.png',dpi=1200,format='png')
#plt.close()
#%%



INP_bc_ambient_pi=np.load('INP_bc_ambient_pi.npy')

INP_feldext_ambient_mean=INP_feldext_ambient.mean(axis=-1)
fs=13
#[:,:,180:211]
DJF=np.arange(-31,59,1)
mdays=[0,31,59,90,120,151,181,212,243,273,304,334,365]
mnames=['January','February','March','April','May','June','July','August','September','October','November','December']
#levelsmo=[1,10,20,30,40,50,60,70,80,90,100,500]
levelsbc=[0.0001,1,5,10,50,100,500,1000,10000]
levelsfel=[10,100,1000]
#os.system('cd gifs_maker')
#os.system('mkdir AT')
#os.chdir('gifs_maker')
#os.system('mkdir latmean2')
#os.chdir('latmean2')
#os.system('cp /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/AT/animate_psfiles /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/latmean2/')

#levelsmo=[1,10,20,30,40,50,60,70,80,90,100,400]
fs=10
for i in range(12):
    #fig=plt.figure()
    print mnames[i]
    cx=plt.subplot(4,3,i+1)
    Xmo,Ymo= np.meshgrid(lat.glat, glolevs)
    Xfel,Yfel=np.meshgrid(lat.glat,glolevs)
    cx.set_title(mnames[i])
    CS=cx.contour(Xfel,Yfel,INP_feldext_ambient[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsfel,linewidths=np.linspace(0.5, 3, 3),colors='k',hold='on',)
    plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
    plt.setp(CS.collections )
    CF=cx.contourf(Xmo,Ymo,INP_bc_ambient_pi[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsbc,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsbc, 256))
    if i==2 or i==5 or i==8 or i==11:
        CB=plt.colorbar(CF,ticks=levelsbc,drawedges=1,label='$m^{-3}$')
    
    cx.invert_yaxis()
    cx.set_ylim(ymax=200)
    cx.tick_params(axis='both', which='major', labelsize=fs)
    if i==0 or i==3 or i==6 or i==9:
        cx.set_ylabel('Pressure level $(hPa)$')
    if i>8:    
        cx.set_xlabel('Latitude')
    cx.xaxis.set_ticks(np.arange(-90,100,20))
plt.figtext(0.4,0.95,'PRE INDUSTRIAL',fontsize=20)
plt.show()
plt.savefig('latmean_grid_pi.svg',dpi=1200,format='svg')
plt.savefig('latmean_grid_pi.png',dpi=1200,format='png')
#plt.close()



#%%

INP_bc_ambient_na=np.load('INP_bc_ambient_na.npy')
INP_feldext_ambient_mean=INP_feldext_ambient.mean(axis=-1)
fs=13
#[:,:,180:211]
DJF=np.arange(-31,59,1)
mdays=[0,31,59,90,120,151,181,212,243,273,304,334,365]
mnames=['January','February','March','April','May','June','July','August','September','October','November','December']
#levelsmo=[1,10,20,30,40,50,60,70,80,90,100,500]
levelsbc=[0.0001,1,5,10,50,100,500,1000,10000]
levelsfel=[10,100,1000]
#os.system('cd gifs_maker')
#os.system('mkdir AT')
#os.chdir('gifs_maker')
#os.system('mkdir latmean2')
#os.chdir('latmean2')
#os.system('cp /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/AT/animate_psfiles /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/latmean2/')

#levelsmo=[1,10,20,30,40,50,60,70,80,90,100,400]
fs=10
for i in range(12):
    #fig=plt.figure()
    print mnames[i]
    cx=plt.subplot(4,3,i+1)
    Xmo,Ymo= np.meshgrid(lat.glat, glolevs)
    Xfel,Yfel=np.meshgrid(lat.glat,glolevs)
    cx.set_title(mnames[i])
    CS=cx.contour(Xfel,Yfel,INP_feldext_ambient[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsfel,linewidths=np.linspace(0.5, 3, 3),colors='k',hold='on',)
    plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
    plt.setp(CS.collections )
    CF=cx.contourf(Xmo,Ymo,INP_bc_ambient_na[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsbc,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsbc, 256))
    if i==2 or i==5 or i==8 or i==11:
        CB=plt.colorbar(CF,ticks=levelsbc,drawedges=1,label='$m^{-3}$')
    
    cx.invert_yaxis()
    cx.set_ylim(ymax=200)
    cx.tick_params(axis='both', which='major', labelsize=fs)
    if i==0 or i==3 or i==6 or i==9:
        cx.set_ylabel('Pressure level $(hPa)$')
    if i>8:    
        cx.set_xlabel('Latitude')
    cx.xaxis.set_ticks(np.arange(-90,100,20))
plt.figtext(0.4,0.95,'NO ANTROPHOGENIC',fontsize=20)
plt.show()
plt.savefig('latmean_grid_na.svg',dpi=1200,format='svg')
plt.savefig('latmean_grid_na.png',dpi=1200,format='png')
#plt.close()

#%%


jl.plot(INP_bc_ambient_na[20,:,:,:].mean(axis=-1),show=1)
jl.plot(INP_bc_ambient_pd[20,:,:,:].mean(axis=-1),show=1)
#%%




def constant_pressure_level(array,pressures,levels=31):
    step=1000./levels
    ps=np.linspace(0,levels,levels)*step
    array_constant_index=np.zeros(array.shape)
    for i1 in range (len (pressures[:,0,0,0])):
        for i2 in range (len (pressures[0,:,0,0])):
            for i3 in range (len (pressures[0,0,:,0])):
                for i4 in range (len (pressures[0,0,0,:])):
                    array_constant_index[i1,i2,i3,i4]=jl.find_nearest_vector_index(ps,pressures[i1,i2,i3,i4])
    return array_constant_index
    #np.apply_along_axis(jl.find_nearest_vector_index, axis, arr, *args, **kwargs)[source]
    #vecfunc = np.vectorize(find_nearest_vector_index)
    #array_constant_index=find_nearest_vector_index(pressures,ps)
    #print array_constant_index
prueba=constant_pressure_level(np.zeros(press_daily.shape),press_daily,levels=31)







#%%

data = netcdf.netcdf_file('INP_marine_and_feldext.nc', 'r')
INP_feldext=data.variables['INP_feldext'][:,:,:,:]
INP_mo=data.variables['INP_marineorganics'][:,:,:,:]
lat=data.variables['lat'][:]
lon=data.variables['lon'][:]
pressure=data.variables['pressure_levs'][:,:,:]
feldext_20=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_20_external.npy').sum(axis=0)
ratio=INP_feldext[20,89,:,:]/feldext_20[30,:,:,:].mean(axis=-1)
loglevs=np.logspace(-10,10,21).tolist()
jl.plot(ratio,show=1,clevs=loglevs)
#%%
INP_bc_pd=np.load('INP_bc_pd.npy').mean(axis=-1)
INP_total=INP_feldext[:,59:,:,:]*1e-6+INP_bc_pd+INP_mo[:,59:,:,:]*1e-6

largedata=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/IN_obs_2.dat',delimiter=",",skip_header=1)
largedata_feld=jl.obtain_points_from_data(INP_feldext*1e-6,largedata,plvs=31)
largedata_tot=jl.obtain_points_from_data(INP_total,largedata,plvs=31)
largedata_bc=jl.obtain_points_from_data(INP_bc_pd,largedata[largedata[:,1]>-24],plvs=31)

plt.figure

x=np.linspace(1e-9,1e8,100)
#plt.scatter(largedata[:,2],largedata_feld[:,0],c=largedata[:,1])#,edgecolors='none')
#plt.scatter(largedata[:,2],largedata_tot[:,0], edgecolors='none',c=largedata[:,1])

plt.scatter(largedata[largedata[:,1]>-24][:,2],largedata_bc[:,0], edgecolors='none',c=largedata[largedata[:,1]>-24][:,1])
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
plt.show()
#%%
press_daily=np.load('press_daily.npy')
INP_bc_ambient_pd=np.load('INP_bc_ambient_pd.npy')
def find_nearest_vector_index(array, value):
    n = np.array([abs(i-value) for i in array])
    nindex=np.apply_along_axis(np.argmin,0,n) 

    return nindex
    
    

def constant_pressure_level(array,pressures,levels=31):
    step=1000./levels
    ps=np.linspace(0,levels,levels)*step
    array_constant_index=np.zeros(array.shape)
    #array_constant=np.zeros(array.shape)
    array_constant_index=find_nearest_vector_index(ps,pressures)

    return array_constant_index,ps
    
def fit_array_to_pressure(array,pressure,levels=31,pressure_index=0,ps=0):
    if not pressure_index:
        pressure_index,ps=constant_pressure_level(array,pressure,levels=31)
        print 'Index array calculated'
    
    array_fitted=np.zeros(INP_bc_ambient_pd.shape)
    for iday in range(len(INP_bc_ambient_pd[0,0,0,:])):
        print iday
        for ilat in range(len(INP_bc_ambient_pd[0,:,0,0])):
            for ilon in range(len(INP_bc_ambient_pd[0,0,:,0])):
                for ilev in range(len(INP_bc_ambient_pd[:,0,0,0])):
                    array_fitted[ilev,ilat,ilon,iday]=INP_bc_ambient_pd[pressure_index[ilev,ilat,ilon,iday],ilat,ilon,iday]
    return array_fitted,ps
    #np.apply_along_axis(jl.find_nearest_vector_index, axis, arr, *args, **kwargs)[source]
    #vecfunc = np.vectorize(find_nearest_vector_index)
    #array_constant_index=find_nearest_vector_index(pressures,ps)
    #print array_constant_index
INP_bc_ambient_pd_fitted,_=fit_array_to_pressure(INP_bc_ambient_pd,press_daily)
daily_pressure_index,ps=constant_pressure_level(INP_bc_ambient_pd,press_daily,levels=31)
INP_bc_ambient_pd_fitted=np.zeros(INP_bc_ambient_pd.shape)
np.save('daily_pressure_index',daily_pressure_index)
#%%
#for iday in INP_bc_ambient_pd[0,0,0,:]:
#    print iday
#    for ilat in INP_bc_ambient_pd[0,:,0,0]:
#        for ilon in INP_bc_ambient_pd[0,0,:,0]:
press_daily_fitted=np.zeros(INP_bc_ambient_pd.shape)
for iday in range(len(INP_bc_ambient_pd[0,0,0,:])):
    print iday
    for ilat in range(len(INP_bc_ambient_pd[0,:,0,0])):
        for ilon in range(len(INP_bc_ambient_pd[0,0,:,0])):
            for ilev in range(len(INP_bc_ambient_pd[:,0,0,0])):
    
                press_daily_fitted[ilev,ilat,ilon,iday]=press_daily[daily_pressure_index[ilev,ilat,ilon,iday],ilat,ilon,iday]
np.save('press_daily_fitted',press_daily_fitted)
#%%

np.save('INP_bc_ambient_pd_daily',INP_bc_ambient_pd_fitted)
#%%
INP_feldext_ambient=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feldext_ambient.npy')
INP_bc_ambient_pd=np.load('INP_bc_ambient_pd_daily.npy')#INP_bc_ambient_pd_fitted[0]#
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
INP_feldext_ambient_mean=INP_feldext_ambient.mean(axis=-1)
fs=13
#[:,:,180:211]
DJF=np.arange(-31,59,1)
mdays=[0,31,59,90,120,151,181,212,243,273,304,334,365]
mnames=['January','February','March','April','May','June','July','August','September','October','November','December']
#levelsmo=[1,10,20,30,40,50,60,70,80,90,100,500]
levelsbc=[0.0001,1,5,10,50,100,500,1000,10000]
levelsfel=[10,100,1000]
glolevs=ps
#os.system('cd gifs_maker')
#os.system('mkdir AT')
#os.chdir('gifs_maker')
#os.system('mkdir latmean2')
#os.chdir('latmean2')
#os.system('cp /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/AT/animate_psfiles /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/latmean2/')

#levelsmo=[1,10,20,30,40,50,60,70,80,90,100,400]
fs=10
for i in range(12):
    #fig=plt.figure()
    print mnames[i]
    cx=plt.subplot(4,3,i+1)
    Xmo,Ymo= np.meshgrid(lat.glat, ps)
    Xfel,Yfel=np.meshgrid(lat.glat,glolevs)
    cx.set_title(mnames[i])
    CS=cx.contour(Xfel,Yfel,INP_feldext_ambient[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsfel,linewidths=np.linspace(0.5, 3, 3),colors='k',hold='on',)
    plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
    plt.setp(CS.collections )
    CF=cx.contourf(Xmo,Ymo,INP_bc_ambient_pd[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsbc,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsbc, 256))
    if i==2 or i==5 or i==8 or i==11:
        CB=plt.colorbar(CF,ticks=levelsbc,drawedges=1,label='$m^{-3}$')
    
    cx.invert_yaxis()
    #plt.ylim
    cx.set_ylim(ymin=800,ymax=300)
    cx.tick_params(axis='both', which='major', labelsize=fs)
    if i==0 or i==3 or i==6 or i==9:
        cx.set_ylabel('Pressure level $(hPa)$')
    if i>8:    
        cx.set_xlabel('Latitude')
    cx.xaxis.set_ticks(np.arange(-90,100,20))
plt.figtext(0.4,0.95,'PRESENT DAY',fontsize=20)

plt.savefig('latmean_grid_pd_fitted.svg',dpi=600,format='svg')
plt.savefig('latmean_grid_pd_fitted.png',dpi=600,format='png')
plt.show()
#%%
lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
jl.plot(INP_bc_ambient_pd[18,:,:,:].mean(axis=-1),show=1)
#%%
jl.plot(press_daily[15,:,:,:].mean(axis=-1),show=1)