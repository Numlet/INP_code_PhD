# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:05:14 2015

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
from scipy import stats
from scipy.optimize import curve_fit
import scipy

saving_folder='/nfs/see-fs-01_users/eejvt/marine_parameterization/'

def rinaldis(chl,wind):
    return (56.9*chl+(-4.64*wind)+40.9)/100

def OMF_gantt(chl,wind):
    return 1/(1+np.exp(-2.63*chl+0.18*wind))
def OMF_gantt_sr(chl,wind,d):
    return 1/(1+np.exp(-2.63*chl+0.18*wind))/(1+0.03*np.exp(6.81*d)+0.03/(1+np.exp(-2.63*chl+0.18*wind)))
#%%


mace_head_latlon_index=[13,124]
amsterdam_island_latlon_index=[45,27]
point_reyes_latlon_index=[18,84]
months_str=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

mace_wiom=np.array([0.08,0.088,0.1,0.24,0.51,0.2,0.183,0.155,0.122,0.24,0.062,0.07])
ams_wioc=np.array([189,122,96,68,64,41,49,47,64,54,64,130])/1e3
ams_wiom=ams_wioc*1.9


class marine_organic_parameterization():
    def __init__(self,name,array_surface):
        self.name=name
        if array_surface.shape[-1]==12:
            self.wiom_mace=array_surface[mace_head_latlon_index[0],mace_head_latlon_index[1],:]
            self.wiom_ams=array_surface[amsterdam_island_latlon_index[0],amsterdam_island_latlon_index[1],:]
            self.wiom_reyes=array_surface[point_reyes_latlon_index[0],point_reyes_latlon_index[1],:]
        else:
            self.wiom_mace=array_surface[:,mace_head_latlon_index[0],mace_head_latlon_index[1]]
            self.wiom_ams=array_surface[:,amsterdam_island_latlon_index[0],amsterdam_island_latlon_index[1]]
            self.wiom_reyes=array_surface[:,point_reyes_latlon_index[0],point_reyes_latlon_index[1]]



archive_directory='/nfs/a107/eejvt/'
project='JB_TRAINING'
os.chdir(archive_directory+project)
s=jl.read_data('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2')
ss=s.tot_mc_ss_mm_mode[2,:,:,:,:]

sea_salt=marine_organic_parameterization('Acc mode sea-salt surface',ss[30,:,:,:])



full_path='/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2/'
name_file='masprimssaccsol'
months_str_upper_case=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ss_emissions=np.zeros((64,128,12))
i=0
for month in months_str_upper_case:
    s=readsav(full_path+name_file+'_'+month+'.sav')
    #print s.masprimssaccsol[30,].shape
    print month
    ss_emissions[:,:,i]=s.masprimssaccsol[30,:,:]
    i=i+1
#jl.grid_earth_map(ss_emissions)
#%%
corr_ss_MH=np.zeros((64,128))
corr_ss_AI=np.zeros((64,128))
for ilon in range(128):
    for ilat in range(64):
        corr_ss_AI[ilat,ilon]=np.corrcoef(ss_emissions[ilat,ilon,:],sea_salt.wiom_ams)[0,1]
        corr_ss_MH[ilat,ilon]=np.corrcoef(ss_emissions[ilat,ilon,:],sea_salt.wiom_mace)[0,1]
#%%
a=np.argwhere(corr_ss_MH>0.9)
a=a[a[:,1]>100]
a=np.delete(a,0,axis=0)
lats_corr=jl.lat[a[:,0]]
lons_corr=jl.lon[a[:,1]]-360

scatter_index_MH=np.zeros((len(lats_corr),2),dtype=int)
scatter_index_MH[:,0]=a[:,0]
scatter_index_MH[:,1]=a[:,1]
scatter_MH=np.zeros((len(lats_corr),2))
scatter_MH[:,0]=lats_corr
scatter_MH[:,1]=lons_corr
jl.plot2(corr_ss_MH,scatter_points2=scatter_MH,file_name=saving_folder+'corr_MH',saving_format='png',cblabel='R',title='Gridboxes influencing Mace Head')
#%%
a=np.argwhere(corr_ss_AI>0.90)
a=a[a[:,0]>35]
lats_corr=jl.lat[a[:,0]]
lons_corr=jl.lon[a[:,1]]
scatter_index_AI=np.zeros((len(lats_corr),2),dtype=int)
scatter_index_AI[:,0]=a[:,0]
scatter_index_AI[:,1]=a[:,1]
scatter_AI=np.zeros((len(lats_corr),2))
scatter_AI[:,0]=lats_corr
scatter_AI[:,1]=lons_corr
jl.plot2(corr_ss_AI,scatter_points2=scatter_AI,file_name=saving_folder+'corr_AI',saving_format='png',cblabel='R',title='Gridboxes influencing Amsterdam Island')

#%%
output_file='/nfs/a107/eejvt/SATELLITE/MERGED_GlobColour/CHL1_MONTHLY/T42_congrid/'
os.chdir(output_file)
files=glob('*2001*')
year_str='2001'
chl=np.zeros((64,128,12))
for i in range(len(files)):
    print 'CHL_T42_'+jl.months_str[i]+'_'+year_str+'.dat'
    array=np.genfromtxt('CHL_T42_'+jl.months_str[i]+'_'+year_str+'.dat')
    chl[:,:,i]=np.reshape(array,(64,128))

#jl.grid_earth_map(chl,levels=np.logspace(-2,1,10).tolist(),cmap=plt.cm.Greens)
#%%
array_chl=chl[scatter_index_AI[:,0],scatter_index_AI[:,1],:].mean(axis=0)
print np.corrcoef(array_chl,ams_wiom)
#plt.plot(ams_wiom,'bo')
#plt.plot(array_chl,'go')
#%%


array_chl=chl[scatter_index_MH[:,0],scatter_index_MH[:,1],:].mean(axis=0)
print np.corrcoef(array_chl,mace_wiom)
#plt.plot(mace_wiom,'k-')
#plt.plot(array_chl,'go')


#%%

os.chdir('/nfs/a201/eejvt/MARINE_ORGANIC_EMISSIONS_BURROWS/EMISSION_FILES_T42')
a=glob('omf*')
omf=np.zeros((64,128,12))
months_str=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
for i in range(len(months_str)):
    print i
    array=np.genfromtxt('omf_%s.dat'%months_str[i])
    omf[:,:,i]=np.reshape(array,(64,128))

#%%
mb=netcdf.netcdf_file('/nfs/a201/eejvt/2001_Winds.nc','r')
u=mb.variables['u10'].data[:,:,:]*mb.variables['u10'].scale_factor+mb.variables['u10'].add_offset
v=mb.variables['v10'].data[:,:,:]*mb.variables['v10'].scale_factor+mb.variables['v10'].add_offset
si=mb.variables['si10'].data[:,:,:]*mb.variables['si10'].scale_factor+mb.variables['si10'].add_offset
U10M=np.sqrt(u**2+v**2).swapaxes(0,1).swapaxes(1,2)
U10M=si.swapaxes(0,1).swapaxes(1,2)
big_lat=mb.variables['latitude'].data
big_lon=mb.variables['longitude'].data
#jl.plot2(u,lat=big_lat,lon=big_lon,scatter_points2=scatter_AI,scatter_points=scatter_MH)
#jl.plot2(v,lat=big_lat,lon=big_lon,scatter_points2=scatter_AI,scatter_points=scatter_MH)
#jl.plot2(U10M,lat=big_lat,lon=big_lon,scatter_points2=scatter_AI,scatter_points=scatter_MH)
#jl.grid_earth_map(U10M,lat=big_lat,lon=big_lon)
X,Y=np.meshgrid(big_lon,big_lat)
X_glo,Y_glo=np.meshgrid(jl.lon,jl.lat)
#U10M_GLOMAP=jl.interpolate_grid(U10M,big_lon=X,big_lat=Y,sml_lon=X_glo,sml_lat=Y_glo)
#%%
U10M_glo=scipy.misc.imresize(U10M[:,:,0],(64,128))
U10M_glo=np.zeros((64,128,12))
for i in range(len(U10M[0,0,:])):
    U10M_glo[:,:,i]=jl.congrid(U10M[:,:,i],(64,128))
    #U10M_glo[:,:,i]=scipy.misc.imresize(U10M[:,:,i],(64,128))

#jl.grid_earth_map(U10M_glo)
#%%
array_omf_burrows_AI=chl[scatter_index_AI[:,0],scatter_index_AI[:,1],:].mean(axis=0)
array_chl_AI=chl[scatter_index_AI[:,0],scatter_index_AI[:,1],:].mean(axis=0)
array_ss_emissions_AI=ss_emissions[scatter_index_AI[:,0],scatter_index_AI[:,1],:].mean(axis=0)
array_wind_AI=U10M[jl.find_nearest_vector_index(big_lat,jl.lat[scatter_index_AI[:,0]]),jl.find_nearest_vector_index(big_lon,jl.lon[scatter_index_AI[:,1]]),:].mean(axis=0)
print np.corrcoef(array_chl_AI,ams_wiom)
plt.plot(ams_wiom,'bo')
plt.plot(array_chl_AI,'go')
#%%

array_omf_burrows_MH=chl[scatter_index_MH[:,0],scatter_index_MH[:,1],:].mean(axis=0)
array_chl_MH=chl[scatter_index_MH[:,0],scatter_index_MH[:,1],:].mean(axis=0)
array_ss_emissions_MH=ss_emissions[scatter_index_MH[:,0],scatter_index_MH[:,1],:].mean(axis=0)
array_wind_MH=U10M[jl.find_nearest_vector_index(big_lat,jl.lat[scatter_index_MH[:,0]]),jl.find_nearest_vector_index(big_lon,jl.lon[scatter_index_MH[:,1]]),:].mean(axis=0)
print np.corrcoef(array_chl_MH,mace_wiom)
plt.plot(mace_wiom,'k-')
plt.plot(array_chl_MH,'go')

#%%
OMF_MH=mace_wiom/(mace_wiom+sea_salt.wiom_mace)
OMF_AI=ams_wiom/(ams_wiom+sea_salt.wiom_ams)
print np.corrcoef(OMF_MH,array_chl_MH)
#plt.plot(OMF_MH,array_chl_MH,'ko')
print np.corrcoef(OMF_AI,array_chl_AI)
#plt.plot(OMF_AI,array_chl_AI,'ko')
chls=np.concatenate((array_chl_AI,array_chl_MH))
omfs=np.concatenate((OMF_AI,OMF_MH))
winds=np.concatenate((array_wind_AI,array_wind_MH))
plt.figure()
#plt.title('all')
plt.scatter(chls,omfs,c=winds,s=chls*1000,cmap=plt.cm.Reds)
plt.ylabel('OMF')
plt.title('a)')
plt.xlabel('Chl ($\mu/m^{-3}$)')
plt.colorbar(label='Wind $m/s$')
plt.savefig(saving_folder+'obs_all.png')
plt.figure()
plt.title('MH')
plt.scatter(array_chl_MH,OMF_MH,c=array_wind_MH,s=array_chl_MH*1000,cmap=plt.cm.Reds)
plt.ylabel('OMF')
plt.xlabel('chl')
plt.colorbar(label='Wind 10M m/s')
plt.savefig(saving_folder+'obs_MH.png')
plt.figure()
plt.title('AI')
plt.scatter(array_chl_AI,OMF_AI,c=array_wind_AI,s=array_chl_AI*1000,cmap=plt.cm.Reds)
plt.xlabel('Chl $\mu/m^{-3}$')
plt.ylabel('OMF')
plt.savefig(saving_folder+'obs_AI.png')
plt.colorbar(label='Wind $m/s$')

#%%
corr_chl_MH=np.zeros((64,128))
corr_chl_AI=np.zeros((64,128))
for ilon in range(128):
    for ilat in range(64):
        corr_chl_AI[ilat,ilon]=np.corrcoef(chl[ilat,ilon,:],ams_wiom)[0,1]
        corr_chl_MH[ilat,ilon]=np.corrcoef(chl[ilat,ilon,:],mace_wiom)[0,1]
#jl.plot(corr_chl_AI,file_name=saving_folder+'corr_AI',saving_format='eps',cblabel='R')
#jl.plot(corr_chl_MH,file_name=saving_folder+'corr_MH',saving_format='eps',cblabel='R')
#jl.plot(corr_chl_AI,cblabel='R')
#jl.plot(corr_chl_MH,cblabel='R')
#%%
def func(x,a,b,c):
    return a*x**2+b*x+c    
plt.plot(chls,omfs,'bo')
plt.xlabel('CHL')
#%%
popt,pcov = curve_fit(func, chls, omfs)



#%%
def func(x,u,v):
  a = x[0]
  b = x[1]
  #c = x[2]
  #d = x[3]
  #e = x[4]
  return 1./(1+np.exp(-a*u+b*v))#a*u**b*v**c

def func(x,u,v):
  a = x[0]
  b = x[1]
  c = x[2]
  d = x[3]
  e = x[4]
  return a*u+b*v**c+d#a*u**b*v**c

def f(x,u,v,z_data):
  modelled_z = func(x,u,v)
  diffs = modelled_z - z_data
  return diffs.flatten() # it expects a 1D array out. 
       # it doesn't matter that it's conceptually 2D, provided flatten it consistently

result = scipy.optimize.leastsq(f,[2.6,0.18,1.0,1.0,1.0], # initial guess at starting point
                        args = (chls,winds,omfs)) # alternatively you can do this with closure variables in f if you like)
print result[0]
chlsx=np.linspace(0,1,100)
windsx=np.linspace(0,10,100)
omf_fitted=func(result[0],chls,winds)
plt.scatter(omfs,omf_fitted,c=winds,s=chls*1000,cmap=plt.cm.Reds)
print np.corrcoef(omfs,omf_fitted)
x=np.linspace(0.1,1)
#plt.title('all')
plt.plot(x,x,'k-')
#plt.title('R=%1.3f'%np.corrcoef(omfs,omf_fitted)[0,1])
plt.title('b) ')
plt.xlabel('OMF')
plt.xlim(0.1,1)
plt.ylim(0.1,1)
plt.ylabel('Parameterized OMF')
plt.savefig(saving_folder+'one_to_one_all.png')

# result is the best fit point
#%%
#result = scipy.optimize.leastsq(f,[1.0,1.0,1.0], # initial guess at starting point
#                        args = (array_chl_AI,array_wind_AI,OMF_AI))
omf_fitted=func(result[0],array_chl_AI,array_wind_AI)
#omf_fitted=rinaldis(array_chl_AI,array_wind_AI)
plt.scatter(OMF_AI,omf_fitted,c=array_wind_AI,s=array_chl_AI*1000,cmap=plt.cm.Reds)
plt.title('R=%1.3f'%np.corrcoef(OMF_AI,omf_fitted)[0,1])
print np.corrcoef(OMF_AI,omf_fitted)
plt.plot(x,x,'k-')
plt.xlabel('observed')
plt.ylabel('parametrized')
plt.savefig(saving_folder+'one_to_one_AI.png')
#plt.
plt.figure()
plt.plot(omf_fitted,label='fitted')
plt.plot(array_omf_burrows_AI,label='fitted')
plt.plot(OMF_AI,'bo',label='Observations')
plt.ylabel('OMF')
plt.xlabel('Month')
plt.xticks(np.arange(12),months_str)
plt.legend()
plt.savefig(saving_folder+'AI_monthly')
#%%

omf_fitted=func(result[0],array_chl_MH,array_wind_MH)
#omf_fitted=rinaldis(array_chl_MH,array_wind_MH)
plt.scatter(OMF_MH,omf_fitted,c=array_wind_MH,s=array_chl_MH*1000,cmap=plt.cm.Reds)
print np.corrcoef(OMF_MH,omf_fitted)
plt.figure()
plt.title('R=%1.3f'%np.corrcoef(OMF_MH,omf_fitted)[0,1])
plt.plot(x,x,'k-')
plt.xlabel('observed')
plt.ylabel('parametrized')
plt.savefig(saving_folder+'one_to_one_MH.png')

plt.figure()
plt.plot(omf_fitted,label='fitted')
plt.plot(OMF_MH,'bo',label='Observations')
plt.ylabel('OMF')
plt.xlabel('Month')
plt.xticks(np.arange(12),months_str)
plt.legend()
plt.savefig(saving_folder+'MH_monthly')

#%%




chlsx=np.linspace(1.5,1.5,100)
windsx=np.linspace(6,12,100)
omf_fitted=func(result[0],chlsx,windsx)
omf_fitted_rinaldi=rinaldis(chlsx,windsx)
omf_fitted_gantt=OMF_gantt(chlsx,windsx)
omf_fitted_gantt_sr=OMF_gantt_sr(chlsx,windsx,0.2)
#plt.scatter(windsx,omf_fitted_gantt_sr,c=windsx,s=chlsx*1000,cmap=plt.cm.bone)
plt.scatter(windsx,omf_fitted_gantt,c=windsx,s=chlsx*1000,cmap=plt.cm.Blues)
#plt.scatter(windsx,omf_fitted_rinaldi,c=windsx,s=chlsx*1000,cmap=plt.cm.Greens)
plt.scatter(windsx,omf_fitted,c=windsx,s=chlsx*1000,cmap=plt.cm.Reds)




chlsx=np.linspace(0.5,0.5,100)
windsx=np.linspace(6,12,100)
omf_fitted=func(result[0],chlsx,windsx)
#omf_fitted_rinaldi=rinaldis(chlsx,windsx)
omf_fitted_gantt=OMF_gantt(chlsx,windsx)
#omf_fitted_gantt_sr=OMF_gantt_sr(chlsx,windsx,0.2)
#plt.scatter(windsx,omf_fitted_gantt_sr,c=windsx,s=chlsx*1000,cmap=plt.cm.bone)
plt.scatter(windsx,omf_fitted_gantt,c=windsx,s=chlsx*1000,cmap=plt.cm.Blues)
#plt.scatter(windsx,omf_fitted_rinaldi,c=windsx,s=chlsx*1000,cmap=plt.cm.Greens)
plt.scatter(windsx,omf_fitted,c=windsx,s=chlsx*1000,cmap=plt.cm.Reds)

'''
chlsx=np.linspace(0.2,0.2,100)
omf_fitted=func(result[0],chlsx,windsx)
omf_fitted_rinaldi=rinaldis(chlsx,windsx)
omf_fitted_gantt_sr=OMF_gantt_sr(chlsx,windsx,0.2)
plt.scatter(windsx,omf_fitted_gantt_sr,c=windsx,s=chlsx*1000,cmap=plt.cm.bone)
plt.scatter(windsx,omf_fitted_gantt,c=windsx,s=chlsx*1000,cmap=plt.cm.Blues)
plt.scatter(windsx,omf_fitted_rinaldi,c=windsx,s=chlsx*1000,cmap=plt.cm.Greens)
plt.scatter(windsx,omf_fitted,c=windsx,s=chlsx*1000,cmap=plt.cm.Reds)
'''
chlsx=np.linspace(0.05,0.05,100)
omf_fitted=func(result[0],chlsx,windsx)
omf_fitted_rinaldi=rinaldis(chlsx,windsx)
omf_fitted_gantt_sr=OMF_gantt_sr(chlsx,windsx,0.2)
#plt.scatter(windsx,omf_fitted_gantt_sr,c=windsx,s=chlsx*1000,cmap=plt.cm.bone)
plt.scatter(windsx,omf_fitted_gantt,c=windsx,s=chlsx*1000,cmap=plt.cm.Blues)
#plt.scatter(windsx,omf_fitted_rinaldi,c=windsx,s=chlsx*1000,cmap=plt.cm.Greens)
plt.scatter(windsx,omf_fitted,c=windsx,s=chlsx*1000,cmap=plt.cm.Reds)
plt.ylabel('OMF')
plt.xlabel('wind')
#plt.savefig(saving_folder+'example.png')

#%%
'''

RINALDIS part

'''



omf_fitted=rinaldis(array_chl_AI,array_wind_AI)
plt.scatter(OMF_AI,omf_fitted,c=array_wind_AI,s=array_chl_AI*1000,cmap=plt.cm.Reds)
plt.title('R=%1.3f'%np.corrcoef(OMF_AI,omf_fitted)[0,1])
print np.corrcoef(OMF_AI,omf_fitted)
plt.plot(x,x,'k-')
plt.savefig(saving_folder+'one_to_one_AI_rinaldi.png')
#plt.
plt.figure()
plt.plot(omf_fitted,label='fitted')
plt.plot(OMF_AI,'bo',label='Observations')
plt.ylabel('OMF')
plt.xlabel('Month')
plt.xticks(np.arange(12),months_str)
plt.legend()
plt.savefig(saving_folder+'AI_monthly_rinaldi')
#%%

omf_fitted=rinaldis(array_chl_MH,array_wind_MH)
plt.scatter(OMF_MH,omf_fitted,c=array_wind_MH,s=array_chl_MH*1000,cmap=plt.cm.Reds)
print np.corrcoef(OMF_MH,omf_fitted)
plt.title('R=%1.3f'%np.corrcoef(OMF_MH,omf_fitted)[0,1])
plt.plot(x,x,'k-')
#plt.savefig(saving_folder+'one_to_one_MH_rinaldi.png')

plt.figure()
plt.plot(omf_fitted,label='fitted')
plt.plot(OMF_MH,'bo',label='Observations')
plt.ylabel('OMF')
plt.xlabel('Month')
plt.xticks(np.arange(12),months_str)
plt.legend()
#plt.savefig(saving_folder+'MH_monthly_rinaldi')

omf_map_rinaldi=rinaldis(chl,U10M_glo)
omf_map_rinaldi[omf_map_rinaldi>0.78]=0.78
omf_map_rinaldi[omf_map_rinaldi<0]=0
omf_map_rinaldi[chl==0]=0
jl.grid_earth_map(omf_map_rinaldi)
jl.grid_earth_map(omf_map_rinaldi*ss_emissions,levels=np.logspace(-5,-2,15).tolist())
jl.grid_earth_map(omf*ss_emissions,levels=np.logspace(-5,-2,15).tolist())
jl.grid_earth_map(ss_emissions,levels=np.logspace(-5,-2,15).tolist())

omf_map=func(result[0],chl,U10M_glo)
omf_map[omf_map>0.78]=0.78
omf_map[omf_map<0]=0
omf_map[chl==0]=0
jl.grid_earth_map(omf_map*ss_emissions)

'''

RINALDIS part

'''

MOF=0.241*chl-7.503*U10M_glo**(0.075)+9.274
MOF[chl==0]=0

jl.grid_earth_map(MOF,levels=np.logspace(-1,0,15).tolist())


#%%

for i in np.arange(51,59):#range(len(chl[43:55,0,0])):
    for j in range(len(chl[0,:,0])):
        if not np.all(chl[i,j,:]==0):
            plt.plot(np.arange(0,12),chl[i,j,:],'k',lw=0.1)
            plt.scatter(np.arange(0,12),chl[i,j,:],c='k')



#%%
'''
PLOTING THE OMF

'''

def my_param(chla,wind):
  a = .241
  b = -7.503
  c = 0.075
  d = 9.274
  return a*chla+b*wind**c+d
OMF_mine=my_param(chl,U10M_glo)
OMF_gantt_non_limited=OMF_gantt(chl,U10M_glo)

OMF_gantt_limited=np.copy(OMF_gantt_non_limited)
OMF_gantt_limited[OMF_gantt_non_limited>0.8]=0.8
jl.grid_earth_map(OMF_gantt_limited*marine_points*ss_emissions/(1-OMF_gantt_limited),levels=np.arange(0,0.01,0.001).tolist()),#levels=np.arange(0,0.01,0.001).tolist())
#%%
marine_points=np.zeros(ss_emissions.shape)
marine_points[ss_emissions!=0]=1
OMF_mine_limited=np.copy(OMF_mine)
OMF_mine_limited[OMF_mine>0.8]=0.8
jl.grid_earth_map(OMF_mine_limited*marine_points*ss_emissions,levels=np.arange(0,0.01,0.001).tolist())
jl.grid_earth_map(OMF_mine_limited*marine_points)#,levels=np.arange(0,1,0.01).tolist())
jl.grid_earth_map(1/(1-OMF_mine_limited)*marine_points)#,levels=np.arange(0,1,0.01).tolist())
jl.grid_earth_map(OMF_mine_limited*marine_points*ss_emissions/(1-OMF_mine_limited),levels=np.arange(0,0.02,0.001).tolist())
jl.grid_earth_map(OMF_mine_limited*marine_points/(1-OMF_mine_limited),levels=np.arange(0,3,0.1).tolist())#*ss_emissions
jl.grid_earth_map(OMF_mine*marine_points*ss_emissions/(1-OMF_mine),levels=np.arange(0,0.02,0.001).tolist())
jl.grid_earth_map(ss_emissions,levels=np.arange(0,0.02,0.001).tolist())

jl.plot((OMF_mine_limited*marine_points*ss_emissions/(1-OMF_mine_limited)).mean(axis=-1),clevs=np.logspace(-2.5,-2,10).tolist())

