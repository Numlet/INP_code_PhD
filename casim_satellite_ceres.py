# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:44:00 2016

@author: eejvt
"""

import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib
import sys
import glob
dir_scripts='/nfs/see-fs-01_users/eejvt/UKCA_postproc'#Change this to the downloaded folder
sys.path.append(dir_scripts)
import UKCA_lib as ukl
import iris.quickplot as qp
import numpy as np
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import matplotlib.pyplot as plt
import matplotlib.animation as animationlt
from scipy.io import netcdf
sys.path.append('/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/code')
#from amsre_daily_v7 import AMSREdaily
from pyhdf import SD
import datetime
import scipy as sc
from scipy.io import netcdf
import time
import scipy
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)


def unrotated_grid(cube):
    rotated_cube=isinstance(cube.coord('grid_longitude').coord_system,iris.coord_systems.RotatedGeogCS)
    if rotated_cube:
        pole_lat=cube.coord('grid_longitude').coord_system.grid_north_pole_latitude
        pole_lon=cube.coord('grid_longitude').coord_system.grid_north_pole_longitude
        lons, lats =iris.analysis.cartography.unrotate_pole(cube.coord('grid_longitude').points,cube.coord('grid_latitude').points,pole_lon,pole_lat)
    else:
        lons=cube.coord('grid_longitude').points
        lats=cube.coord('grid_latitude').points
    return lons,lats

lon_offset=7
path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/'
from scipy.io import netcdf
cubes  =iris.load(path+'ceres/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc')
mb=netcdf.netcdf_file(path+'ceres/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc','r') 
mb=netcdf.netcdf_file(path+'ceres_all_SO/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc','r') 

SW=cubes[1]
#model_lons=np.arange(-0.02*250,250*0.02,0.02)[0:]
#model_lats=np.arange(-0.02*250-52,250*0.02-52,0.02)

#model_lons=np.linspace(-7,17)
#model_lats=np.linspace(-47.5,-58)
times_ceres=mb.variables['time'].data*24*60*60

#model_lons=model_lons+lon_offset

SW=np.copy(mb.variables['CERES_SW_TOA_flux___upwards'].data)
SW[SW>1400]=0
lon=mb.variables['lon'].data
lat=mb.variables['lat'].data

ti=13#h
te=23#h

tdi=(datetime.datetime(2014,12,9,ti)-datetime.datetime(1970,1,1)).total_seconds()
tde=(datetime.datetime(2014,12,9,te)-datetime.datetime(1970,1,1)).total_seconds()

t13=(datetime.datetime(2014,12,9,14)-datetime.datetime(1970,1,1)).total_seconds()/3600.
#plt.scatter(lon,lat,c=SW)
#plt.colorbar()
#%%
cloud_top= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/','m01s09i223'))[0]
cube_3ord = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/All_time_steps/','m01s01i208'))[0]
cube_2m = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_MORE/All_time_steps/','m01s01i208'))[0]
cube = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/','m01s01i208'))[0]
#cube_con = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/BASE_CONTACT_242/All_time_steps/','m01s01i208'))[0]
cube_oldm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/OLD_MICRO/All_time_steps/','m01s01i208'))[0]
cube_nh = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/All_time_steps/','m01s01i208'))[0]
cube_ni = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NOICE/All_time_steps/','m01s01i208'))[0]
cube_csb = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/CLOUD_SQUEME/BASE/All_time_steps/','m01s01i208'))[0]

#cube=cube.extract(iris.Constraint(time=jl.find_nearest_vector(cube.coord('time').points,t13)))
cube_csb=cube_csb[13,:,:]
cube=cube[12,:,:]
cube_nh=cube_nh[13,:,:]
cube_3ord=cube_3ord[13,:,:]
cube_2m=cube_2m[13,:,:]
#cube_con=cube_con[13,:,:]
cube_oldm=cube_oldm[13,:,:]
lon_offset=7
#model_lons=cube.coord('grid_longitude').points-180+lon_offset
#model_lats=cube.coord('grid_latitude').points-52
model_lons,model_lats=unrotated_grid(cube)
#times_range=np.argwhere((times_ceres >= tdi) & (times_ceres <=tde))
times_range=np.logical_and([times_ceres >= tdi],[times_ceres <=tde])[0]
sat_lon=lon[times_range]
sat_lat=lat[times_range]
sat_SW=SW[times_range]
coord=np.zeros([len(sat_lon),2])
coord[:,0]=sat_lon
coord[:,1]=sat_lat
cm=plt.cm.RdBu_r
#model_lons=np.linspace(-5,20,500)
X,Y=np.meshgrid(model_lons, model_lats)
lon_old,lat_old=unrotated_grid(cube_oldm)
#%%
Xo,Yo=np.meshgrid(lon_old,lat_old)
coord_model=np.zeros((len(Xo.flatten()),2))
coord_model[:,0]=Xo.flatten()
coord_model[:,1]=Yo.flatten()
data_old= sc.interpolate.griddata(coord_model, cube_oldm.data.flatten(), (X,Y), method='linear')
grid_z0 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='nearest')
grid_z1 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='linear')
grid_z2 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='cubic')
grid_z2[grid_z2<0]=0
grid_z2[grid_z2==np.nan]=0
#plt.imshow(sat_SW)
plt.figure(figsize=(15,13))
levels=np.arange(0,820,82).tolist()
plt.subplot(221)


#levels_ct=np.linspace(0,10,21).tolist()
#
#CS=plt.contour(X,Y,cloud_top[12].data*0.3048,levels_ct,color='k')
#
#plt.clabel(CS, levels_ct,
#           inline=1,
#           fmt='%1.1f',

plt.title('BASE (CS)')
plt.contourf(X,Y,cube_csb.data,levels, origin='lower',cmap=cm)
#plt.title('OLD_MICRO')
#plt.contourf(X,Y,data_old,levels, origin='lower',cmap=cm)
##plt.contourf(X,Y,grid_z1,levels, origin='lower',cmap=cm)
cb=plt.colorbar()




#plt.title('Satellite (CERES)')
#           fontsize=14)

cb.set_label('Outgoing SW radiation $W/m^2$')
plt.subplot(222)
plt.title('ALL_ICE_PROC')
plt.contourf(X,Y,cube.data,levels, origin='lower',cmap=cm)
#plt.contourf(X,Y,grid_z0,levels, origin='lower')
#plt.title('Nearest')
cb=plt.colorbar()
cb.set_label('Outgoing SW radiation $W/m^2$')
plt.subplot(223)
plt.title('2_ORD_MORE')
plt.contourf(X,Y,cube_2m.data,levels, origin='lower',cmap=cm)
#plt.title('con')
cb=plt.colorbar()
cb.set_label('Outgoing SW radiation $W/m^2$')
plt.subplot(224)
plt.title('3_ORD_LESS')
plt.contourf(X,Y,cube_3ord.data,levels, origin='lower',cmap=cm)
#plt.contourf(X,Y,grid_z2,levels, origin='lower')
#plt.title('Cubic')
#plt.gcf().set_size_inches(6, 6)
cb=plt.colorbar()
cb.set_label('Outgoing SW radiation $W/m^2$')
plt.savefig('/nfs/see-fs-01_users/eejvt/CASIM/CERES.png')
#plt.savefig('/nfs/see-fs-01_users/eejvt/CASIM/CERES.png')
plt.show()

#%%

#a=scipy.stats.pdf_moments(np.sort(cube.data.flatten()))
same_bins=np.linspace(0,800,100)

def PDF(data,nbins=100):
    min_val=data.min()
    max_val=data.max()
    if isinstance(nbins,np.ndarray):
        bins=nbins
    else:
        bins=np.linspace(min_val,max_val,nbins)
    size_bin=bins[1:]-bins[:-1]
    bins_midpoint=(bins[1:]-bins[:-1])/2.+bins[1:]
    number_ocurrencies=np.zeros_like(bins_midpoint)
    for ibin in range(len(number_ocurrencies)):
        larger=[data>bins[ibin]]
        smaller=[data<bins[ibin+1]]
        
        number_ocurrencies[ibin]=np.sum(np.logical_and(larger,smaller))
        
    normalized_pdf=number_ocurrencies/float(len(data))/size_bin
    return bins_midpoint,normalized_pdf

bins,pdf=PDF(cube.data.flatten(),same_bins)
#bins_con,pdf_con=PDF(cube_con.data.flatten(),same_bins)
bins_old,pdf_old=PDF(data_old.flatten(),same_bins)
bins_csb,pdf_csb=PDF(cube_csb.data.flatten(),same_bins)
bins_3ord,pdf_3ord=PDF(cube_3ord.data.flatten(),same_bins)
bins_2m,pdf_2m=PDF(cube_2m.data.flatten(),same_bins)
bins_nh,pdf_nh=PDF(cube_nh.data.flatten(),same_bins)
sat_bins, sat_pdf=PDF(grid_z1.flatten(),same_bins)
plt.figure()

plt.plot(bins, pdf,label='ALL_ICE_PROC R=%1.2f'%np.corrcoef(pdf,sat_pdf)[0,1])
plt.plot(bins_csb, pdf_csb,label='BASE (CS) R=%1.2f'%np.corrcoef(pdf_csb,sat_pdf)[0,1])

#plt.plot(bins_con, pdf_con,label='con R=%1.2f'%np.corrcoef(pdf_con[20:],sat_pdf[20:])[0,1])
plt.plot(bins_old, pdf_old,label='OLD_MICROPHYSICS R=%1.2f'%np.corrcoef(pdf_old,sat_pdf)[0,1])
#plt.plot(bins_nh, pdf_nh,label='no hallet R=%1.2f'%np.corrcoef(pdf_nh[20:],sat_pdf[20:])[0,1])
plt.plot(bins_2m, pdf_2m,label='2_ORD_MORE R=%1.2f'%np.corrcoef(pdf_2m,sat_pdf)[0,1])
plt.plot(bins_3ord, pdf_3ord,label='3_ORD_LESS R=%1.2f'%np.corrcoef(pdf_3ord,sat_pdf)[0,1])
plt.plot(sat_bins, sat_pdf,label='Satelite (CERES)')
plt.ylim(0,0.01)
plt.legend()
plt.title('Probability density function of reflected SW radiation')
plt.ylabel('Normalized PDF')
plt.xlabel('Reflected Shortwave Radiation $W/m^{2}$')
plt.savefig('/nfs/see-fs-01_users/eejvt/CASIM/CERES_PDF.png')
#np.sort(cube.data.flatten())
#%%






#cube.add_aux_coord(iris.coords.DimCoord(lons,long_name='longitude_unrotated',var_name='longitude_unrotated',units='degrees'))
