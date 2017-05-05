# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:36:53 2016

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
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
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




path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/'
SDS_NAME  = 'Cloud_Top_Height_Nadir_Night'
SDS_NAME  = 'cloud_top_height_1km'
#hdf = SD.SD(FILE_NAME)
SDS_NAME  = 'Cloud_Water_Path'
SDS_NAME  = 'Cloud_Water_Path_16'
SDS_NAME  = 'Cloud_Top_Temperature'
SDS_NAME  = 'Cloud_Top_Height'
SDS_NAME  = 'cloud_top_height_1km'
SDS_NAME  = 'Cloud_Mask_1km'
hdf  =SD.SD(path+'modis/'+'MYD06_L2.A2014343.1325.006.2014344210847.hdf')
#print hdf.datasets().keys()
for k in hdf.datasets().keys():
    if 'eight' in k:
        print k

sds = hdf.select(SDS_NAME)
data = sds.get()
mask_value=-9999
mask_value=-32767
data[data==mask_value]=np.float64('Nan')
print data.shape
lat = hdf.select('Latitude')
latitude = lat[:,:]
lon = hdf.select('Longitude')
longitude = lon[:,:]


sat_lon=longitude.flatten()
sat_lat=latitude.flatten()
sat_data=data.flatten()
#for att in sds.attributes():
#    print att
#%%

cloud_top= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/','m01s09i223'))[0]
cube_3ord = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/All_time_steps/','m01s09i223'))[0]
cube_2m = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_MORE/All_time_steps/','m01s09i223'))[0]
cube = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/','m01s09i223'))[0]
cube_con = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/BASE_CONTACT_242/All_time_steps/','m01s09i223'))[0]
#cube_oldm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/OLD_MICRO/All_time_steps/','m01s09i223'))[0]
cube_nh = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/All_time_steps/','m01s09i223'))[0]
#%%
#
#path='/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/'
#potential_temperature=iris.load(ukl.Obtain_name(path,'m01s00i004'))[0]
#air_pressure=iris.load(ukl.Obtain_name(path,'m01s00i408'))[0]
#p0 = iris.coords.AuxCoord(1000.0,
#                          long_name='reference_pressure',
#                          units='hPa')
#p0.convert_units(air_pressure.units)
#
#Rd=287.05 # J/kg/K
#cp=1005.46 # J/kg/K
#Rd_cp=Rd/cp
#
#temperature=potential_temperature*(air_pressure/p0)**(Rd_cp)
#print temperature.data[0,0,0,0]
#temperature._var_name='temperature'
#R_specific=iris.coords.AuxCoord(287.058,
#                          long_name='R_specific',
#                          units='J-kilogram^-1-kelvin^-1')#J/(kg·K)
#
#air_density=(air_pressure/(temperature*R_specific))
#
#print temperature.data[0,0,0,0]
#temperature.long_name='Temperature'
#temperature._var_name='temperature'
#
#
#
#
##%%
#
#cube_l = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/','m01s00i254'))[0]
#cube_i = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/','m01s00i012'))[0]
#print cube
#data=cube_l.data[:,:,:,:]+cube_i.data[:,:,:,:]
##data[data==0]=999
#
#temp_cloud=temperature.data
#temp_cloud[data==0]=999
#temp_cloud.min(axis=1)
#plt.imshow(temp_cloud.min(axis=1)[13])
#plt.colorbar()
##%%
#cloud_top_temp=temp_cloud.min(axis=1)
#cloud_top_temp[cloud_top_temp==999]=np.nan
#cloud_top_temp[cloud_top_temp<258]=np.nan
#plt.imshow(cloud_top_temp[13]-273.15)
#
#plt.colorbar()
#plt.show()


#%%
#plt.contourf(data)
#plt.imshow(data)

#subset_data=

#sat_lon=lon[times_range]
#sat_lat=lat[times_range]
#sat_SW=SW[times_range]
coord=np.zeros([len(sat_lon),2])
coord[:,0]=sat_lon
coord[:,1]=sat_lat
cm=plt.cm.RdBu_r
model_lons,model_lats=unrotated_grid(cube)
#model_lons=np.linspace(-5,20,500)
X,Y=np.meshgrid(model_lons, model_lats)
#%%
#lon_old,lat_old=unrotated_grid(cube_oldm)
#Xo,Yo=np.meshgrid(lon_old,lat_old)
#coord_model=np.zeros((len(Xo.flatten()),2))
#coord_model[:,0]=Xo.flatten()
#coord_model[:,1]=Yo.flatten()
#data_old= sc.interpolate.griddata(coord_model, cube_oldm.data.flatten(), (X,Y), method='linear')
#grid_z0 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='nearest')
grid_z1 = sc.interpolate.griddata(coord, sat_data, (X,Y), method='linear')
#grid_z2 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='cubic')
#grid_z2[grid_z2<0]=0
grid_z1[np.isnan(grid_z1)]=0

#%%
plt.figure(figsize=(15,13))
plt.subplot(221)
plt.contourf(X,Y,cube.data[12]*0.304*1000.,levels, origin='lower',cmap=cm)
cb=plt.colorbar()
cb.set_label('Cloud top height $km$')
#%%

#plt.figure()
plt.figure(figsize=(15,13))
levels=np.arange(0.1,4.3,0.3).tolist()
plt.subplot(221)
plt.contourf(X,Y,grid_z1/1000.,levels, origin='lower',cmap=cm)
plt.title('Satellite (MODIS)')
cb=plt.colorbar()
cb.set_label('Cloud top height $km$')

plt.subplot(222)
#plt.subplot(223)
plt.title('ALL_ICE_PROC')
plt.contourf(X,Y,cube.data[12]*0.304,levels, origin='lower',cmap=cm)
cb=plt.colorbar()
cb.set_label('Cloud top height $km$')
plt.subplot(223)
plt.title('2_ORD_MORE')
plt.contourf(X,Y,cube_2m.data[13]*0.304,levels, origin='lower',cmap=cm)
cb=plt.colorbar()
cb.set_label('Cloud top height $km$')
plt.subplot(224)
plt.contourf(X,Y,cube_3ord.data[13]*0.304,levels, origin='lower',cmap=cm)
plt.title('3_ORD_LESS')
#plt.contourf(X,Y,grid_z1/1000.-cube.data[12]*0.304,levels, origin='lower',cmap=cm)
#plt.title('diff')
#plt.title('con')
#plt.contourf(X,Y,data_old,levels, origin='lower',cmap=cm)
#plt.subplot(224)
#plt.title('3 ord')
#plt.contourf(X,Y,cube_3ord.data,levels, origin='lower',cmap=cm)
#plt.contourf(X,Y,grid_z2,levels, origin='lower')
#plt.title('Cubic')
#plt.gcf().set_size_inches(6, 6)
cb=plt.colorbar()
cb.set_label('Cloud top height $km$')
plt.savefig('/nfs/see-fs-01_users/eejvt/CASIM/MODIS.png')
plt.show()

#%%

same_bins=np.linspace(0,12,50)



bins,pdf=PDF(cube[12].data.flatten()*0.304,same_bins)
bins_con,pdf_con=PDF(cube_con[13].data.flatten()*0.304,same_bins)
#bins_old,pdf_old=PDF(data_old[13].flatten()*0.304,same_bins)
bins_3ord,pdf_3ord=PDF(cube_3ord[13].data.flatten()*0.304,same_bins)
bins_2m,pdf_2m=PDF(cube_2m[13].data.flatten()*0.304,same_bins)
#bins_nh,pdf_nh=PDF(cube_nh.data.flatten(),same_bins)
sat_bins, sat_pdf=PDF(grid_z1.flatten()/1000.,same_bins)
plt.figure()

plt.plot(bins, pdf,label='ALL_ICE_PROC R=%1.2f'%np.corrcoef(pdf[:],sat_pdf[:])[0,1])
#plt.plot(bins_con, pdf_con,label='con R=%1.2f'%np.corrcoef(pdf_con[:],sat_pdf[:])[0,1])
#plt.plot(bins_old, pdf_old,label='old R=%1.2f'%np.corrcoef(pdf_old[20:],sat_pdf[20:])[0,1])
#plt.plot(bins_nh, pdf_nh,label='no hallet R=%1.2f'%np.corrcoef(pdf_nh[20:],sat_pdf[20:])[0,1])
plt.plot(bins_2m, pdf_2m,label='2_ORD_MORE R=%1.2f'%np.corrcoef(pdf_2m[:],sat_pdf[:])[0,1])
plt.plot(bins_3ord, pdf_3ord,label='3_ORD_LESS R=%1.2f'%np.corrcoef(pdf_3ord[:],sat_pdf[:])[0,1])
plt.plot(sat_bins, sat_pdf,label='satellite')
plt.legend()
plt.title('Cloud top height')
plt.ylabel('Normalized PDF')
plt.xlabel('Cloud top height $Km$')
plt.xlim(0,5)

plt.savefig('/nfs/see-fs-01_users/eejvt/CASIM/MODIS_PDF.png')


#%%

#plt.contourf(reduced_data)
#plt.colorbar()

plt.figure()
data=jl.congrid(data,latitude.shape)
print data.shape
plt.contourf(longitude,latitude,data,15)

plt.colorbar()