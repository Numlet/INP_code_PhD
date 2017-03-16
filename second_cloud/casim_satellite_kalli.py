# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 12:25:19 2016

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

path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/'


model_lons=np.arange(-0.02*250,250*0.02,0.02)[0:]
model_lats=np.arange(-0.02*250-52,250*0.02-52,0.02)
model_lons=np.linspace(-7,17)
model_lats=np.linspace(-47.5,-58)
#ks = f.keys()

#%%
'''
CERES
'''
#1st of jan 1970 days



SDS_NAME  = 'CERES SW TOA flux - upwards'
hdf  =SD.SD(path+'ceres/'+'FLASH_SSF_Aqua-FM3-MODIS_Version3B_112102.2014120913.nc')
hdf  =SD.SD(path+'ceres/'+'FLASH_SSF_Aqua-FM3-MODIS_Version3B_112102.2014120915.nc')
sds = hdf.select(SDS_NAME)
data = sds.get()
sds.units
#SDS_NAME  = 'Time of observation'
time = hdf.select('Time of observation').get()

for name in hdf.datasets().keys():
    if 'ime' in name:
        print name
        
#datetime.datetime()
#time.gmtime([secs]

#datetime.datetime(1970,01,01)


lon = hdf.select('Longitude of CERES FOV at surface').get()#-180
lat = hdf.select('Colatitude of CERES FOV at surface').get()-90
from scipy.io import netcdf
#cubes  =iris.load(path+'ceres/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc')
#mb=netcdf.netcdf_file(path+'ceres/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc','r') 

SW=hdf.select(SDS_NAME).get()

#times_ceres=mb.variables['time'].data*24*60*60
lon = hdf.select('Longitude of subsatellite point at surface at observation').get()
lat = hdf.select('Colatitude of subsatellite point at surface at observation').get()-90

lat_range=np.logical_and([lat >= -58],[lat<=-47])[0]
lon_range=np.logical_and([lon>= 360-7],[lon <=17])[0]
total_range=np.logical_and(lon_range,lat_range)
print total_range.sum()
sat_lon=lon[total_range]
sat_lat=lat[total_range]
sat_SW=SW[total_range]
coord=np.zeros([len(sat_lon),2])
coord[:,0]=sat_lon
coord[:,1]=sat_lat
X,Y=np.meshgrid(model_lons, model_lats)
grid_z1 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='linear')
plt.imshow(grid_z1)
plt.colorbar()
#%%
'''
CERES MINE
'''
from scipy.io import netcdf
cubes  =iris.load(path+'ceres/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc')
mb=netcdf.netcdf_file(path+'ceres/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc','r') 
mb=netcdf.netcdf_file(path+'ceres_all_SO/'+'CERES_SSF_Aqua-XTRK_Edition4A_Subset_2014120900-2014121023.nc','r') 

SW=cubes[1]
model_lons=np.arange(-0.02*250,250*0.02,0.02)[0:]
model_lats=np.arange(-0.02*250-52,250*0.02-52,0.02)

model_lons=np.linspace(-7,17)
model_lats=np.linspace(-47.5,-58)
times_ceres=mb.variables['time'].data*24*60*60

model_lons=model_lons+lon_offset

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

#cloud_top= iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/All_time_steps/','m01s09i223'))[0]
cube_3ord = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/All_time_steps/','m01s01i208'))[0]
cube = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/','m01s01i208'))[0]
cube_old = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/BASE_CONTACT_242/All_time_steps/','m01s01i208'))[0]
cube_nh = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/All_time_steps/','m01s01i208'))[0]
#cube=cube.extract(iris.Constraint(time=jl.find_nearest_vector(cube.coord('time').points,t13)))
cube=cube[12,:,:]
cube_nh=cube_nh[13,:,:]
cube_3ord=cube_3ord[13,:,:]
cube_old=cube_old[13,:,:]
lon_offset=7
model_lons=cube.coord('grid_longitude').points-180+lon_offset
model_lats=cube.coord('grid_latitude').points-52
#times_range=np.argwhere((times_ceres >= tdi) & (times_ceres <=tde))
times_range=np.logical_and([times_ceres >= tdi],[times_ceres <=tde])[0]
sat_lon=lon[times_range]
sat_lat=lat[times_range]
sat_SW=SW[times_range]
coord=np.zeros([len(sat_lon),2])
coord[:,0]=sat_lon
coord[:,1]=sat_lat
cm=plt.cm.RdBu_r
plt.figure()
#model_lons=np.linspace(-5,20,500)
X,Y=np.meshgrid(model_lons, model_lats)
grid_z0 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='nearest')
grid_z1 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='linear')
grid_z2 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='cubic')
grid_z2[grid_z2<0]=0
grid_z2[grid_z2==np.nan]=0
#plt.imshow(sat_SW)
levels=np.arange(0,820,82).tolist()
plt.subplot(221)
plt.title('All_ice_proc')
plt.contourf(X,Y,cube.data,levels, origin='lower',cmap=cm)
plt.subplot(222)
#plt.contourf(X,Y,grid_z0,levels, origin='lower')
#plt.title('Nearest')
plt.contourf(X,Y,grid_z1,levels, origin='lower',cmap=cm)
plt.title('Satellite')
plt.subplot(223)
plt.title('old')
plt.contourf(X,Y,cube_old.data,levels, origin='lower',cmap=cm)
plt.subplot(224)
plt.title('3 ord')
plt.contourf(X,Y,cube_3ord.data,levels, origin='lower',cmap=cm)
#plt.contourf(X,Y,grid_z2,levels, origin='lower')
#plt.title('Cubic')
#plt.gcf().set_size_inches(6, 6)
plt.colorbar()
plt.show()
#%%
import scipy
#a=scipy.stats.pdf_moments(np.sort(cube.data.flatten()))
same_bins=np.linspace(0,800,100)
def PDF(data,nbins=100):
    min_val=data.min()
    max_val=data.max()
    if isinstance(nbins,np.ndarray):
        bins=nbins
    else:
        bins=np.linspace(min_val,max_val,nbins)
    bins_midpoint=(bins[1:]-bins[:-1])/2.+bins[1:]
    number_ocurrencies=np.zeros_like(bins_midpoint)
    for ibin in range(len(number_ocurrencies)):
        larger=[data>bins[ibin]]
        smaller=[data<bins[ibin+1]]
        
        number_ocurrencies[ibin]=np.sum(np.logical_and(larger,smaller))
        
    normalized_pdf=number_ocurrencies/float(len(data))
    return bins_midpoint,normalized_pdf

bins,pdf=PDF(cube.data.flatten(),same_bins)
bins_old,pdf_old=PDF(cube_old.data.flatten(),same_bins)
bins_3ord,pdf_3ord=PDF(cube_3ord.data.flatten(),same_bins)
bins_nh,pdf_nh=PDF(cube_nh.data.flatten(),same_bins)
sat_bins, sat_pdf=PDF(grid_z1.flatten(),same_bins)
plt.figure()
plt.plot(bins, pdf,label='all_ice_proc R=%1.2f'%np.corrcoef(pdf[20:],sat_pdf[20:])[0,1])
plt.plot(bins_old, pdf_old,label='old R=%1.2f'%np.corrcoef(pdf_old[20:],sat_pdf[20:])[0,1])
#plt.plot(bins_nh, pdf_nh,label='no hallet R=%1.2f'%np.corrcoef(pdf_nh[20:],sat_pdf[20:])[0,1])
plt.plot(bins_3ord, pdf_3ord,label='3ord_less R=%1.2f'%np.corrcoef(pdf_3ord[20:],sat_pdf[20:])[0,1])
plt.plot(sat_bins, sat_pdf,label='satellite')
plt.legend()
plt.title('Probability density function of reflected SW radiation')
plt.ylabel('Normalized PDF')
plt.xlabel('Reflected Shortwave Radiation $W/m^{2}$')
#np.sort(cube.data.flatten())

#%%

def func(x, y):
     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
     
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
points = np.random.rand(1000, 2)
values = func(points[:,0], points[:,1])


from scipy.interpolate import griddata
grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

import matplotlib.pyplot as plt
plt.subplot(221)
plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
plt.plot(points[:,0], points[:,1], 'k.', ms=1)
plt.title('Original')
plt.subplot(222)
plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
plt.title('Nearest')
plt.subplot(223)
plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
plt.title('Linear')
plt.subplot(224)
plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
plt.title('Cubic')
plt.gcf().set_size_inches(6, 6)
plt.show()
#%%
fig, ax1 = plt.subplots()


ax1.plot(times_ceres)
#ax1.plot(t, s1, 'b-')

plt.axhline(tdi)

plt.axhline(tde)
ax2 = ax1.twinx()


ax2.plot(lon,'g')
ax2.plot(lat,'r')
#%%
 
#%%
for name in hdf.datasets().keys():
    if '' in name:
        print name
        
    
#%%
'''

MODIS
'''
SDS_NAME  = 'Cloud_Top_Height_Nadir_Night'
SDS_NAME  = 'cloud_top_height_1km'
#hdf = SD.SD(FILE_NAME)
SDS_NAME  = 'Cloud_Water_Path'
SDS_NAME  = 'Cloud_Water_Path_16'
SDS_NAME  = 'Cloud_Top_Height'
hdf  =SD.SD(path+'modis/'+'MYD06_L2.A2014343.1325.006.2014344210847.hdf')
#print hdf.datasets().keys()
for k in hdf.datasets().keys():
    if 'Cloud_W' in k:
        print k

sds = hdf.select(SDS_NAME)
data = sds.get()
mask_value=-9999
mask_value=-32767
data[data==mask_value]=np.float64('Nan')
print data.shape
#%%
for att in sds.attributes():
    print att
#%%
#plt.contourf(data)
#plt.imshow(data)

#subset_data=


#plt.contourf(reduced_data)
#plt.colorbar()
lat = hdf.select('Latitude')
latitude = lat[:,:]
lon = hdf.select('Longitude')
longitude = lon[:,:]
plt.figure()
data=jl.congrid(data,latitude.shape)
print data.shape
plt.contourf(longitude,latitude,data,15)
plt.colorbar()
#fig=plt.figure(figsize=(20, 12))
#m = fig.add_subplot(1,1,1)
#m = Basemap(projection='cyl',lon_0=0)
#m= Basemap(projection='ortho',lat_0=-45,lon_0=0,resolution='l')
lats = [ -37, -57, -57, -37 ]
lons = [ -5, -5, 5, 5 ]

draw_screen_poly( lats, lons, m )
lats = [ -47.5, -57.5,-57.5,-47.5]
lons = [ -5, -5, 15, 15 ]

draw_screen_poly( lats, lons, m )
#m.drawcoastlines()


#X,Y=np.meshgrid(lon,lat)
#cs=m.contourf(longitude,latitude,data,9,cmap=plt.cm.RdBu_r)#,norm= colors.BoundaryNorm(clevs, 256))
#cb = m.colorbar(cs,format='%.2e')#,ticks=clevs)
#plt.show()


#%%
#%%
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#%%

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

path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/'



def show_dimensions(ds):
    print
    print 'Dimensions'
    for dim in ds.dimensions:
        print ' '*4, dim, ':', ds.dimensions[dim]

def show_variables(ds):
    print
    print 'Variables:'
    for var in ds.variables:
        print ' '*4, var, ':', ds.variables[var].long_name

def show_validrange(ds):
    print
    print 'Valid min and max and units:'
    for var in ds.variables:
        print ' '*4, var, ':', \
              ds.variables[var].valid_min, 'to', \
              ds.variables[var].valid_max,\
              '(',ds.variables[var].units,')'

def show_somedata(ds):
    print
    print 'Show some data for:',avar
    print 'index range: (' + str(iasc) + ', ' + \
          str(ilat[0]) + ':' + str(ilat[1]) + ' ,' + \
          str(ilon[0]) + ':' + str(ilon[1]) + ')'
    print ds.variables[avar][iasc, ilat[0]:ilat[1]+1, ilon[0]:ilon[1]+1]



sys.path.append('/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/code')
from amsr2_daily_v7 import AMSR2daily
amsr_data='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/code/data.remss.com/amsr2/bmaps_v07.2/y2014/m12/'
glob.glob(amsr_data+'*')
def read_data(filename=amsr_data+'f34_20141209v7.2.gz'):
    dataset = AMSR2daily(filename, missing=missing)
    if not dataset.variables: sys.exit('problem reading file')
    return dataset

ilon = (169,174)
ilat = (273,277)
iasc = 0
avar = 'vapor'
missing = -999.
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm

from matplotlib.patches import Polygon

def draw_screen_poly( lats, lons, m):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='green', alpha=0.1 )
    plt.gca().add_patch(poly)

lats = [ -37, -57, -57, -37 ]
lons = [ -5, -5, 15, 15 ]

lats = [ -47.5, -57.5,-57.5,-47.5]
lons = [ -5, -5, 5, 5 ]
lons=np.array(lons)+lon_offset
#lons = [ 180-5, 180-5, 180+15, 180+15 ]
#m = Basemap(projection='sinu',lon_0=0)
#m.drawcoastlines()
#m.drawmapboundary()


lat_max=-37.0
lat_min=-57.0
lon_max=5
lon_min=-5
import sys
dataset = read_data()
show_dimensions(dataset)
show_variables(dataset)
show_validrange(dataset)
show_somedata(dataset)
print
print 'done'

times=dataset.variables['time'][0,]
times[times==missing]=0
LWP=dataset.variables['cloud'][0,]
LWP[LWP==missing]=0
lon=dataset.variables['longitude']
lat=dataset.variables['latitude']



fig=plt.figure()#figsize=(20, 12))
m = fig.add_subplot(1,1,1)
#m = Basemap(projection='lcc',
#            resolution='c',urcrnrlat=-57.5,llcrnrlat=-47.5,llcrnrlon=-5,ucrnrlon=15.)
#m = Basemap(projection='cyl',lon_0=0)#,urcrnrlat=-57.5,llcrnrlat=-47.5,llcrnrlon=-5,ucrnrlon=15.)
#m = Basemap(projection='cea',llcrnrlat=-58.5,urcrnrlat=-47.5,\
#            llcrnrlon=-5,urcrnrlon=15,resolution='c')
m= Basemap(projection='ortho',lat_0=-25,lon_0=0,resolution='l')
draw_screen_poly( lats, lons, m )
#m = Basemap(projection='stere',lon_0=0,lat_0=90.,lat_ts=lat_0,\
#            llcrnrlat=latcorners[0],urcrnrlat=latcorners[2],\
#            llcrnrlon=loncorners[0],urcrnrlon=loncorners[2],\
#            rsphere=6371200.,resolution='l',area_thresh=10000)
m.drawcoastlines()
m.drawparallels(np.arange(-90.,120.,10.))
m.drawmeridians(np.arange(0.,360.,10))

plt.title('Liquid water path retrieved by AMSR')
#LWP[times<12]=0
#LWP[times>14]=0

X,Y=np.meshgrid(lon,lat)
#X,Y=np.meshgrid(X[times==13],Y[times==13])
levels=np.arange(0,0.4,0.05).tolist()
cs=m.contourf(X,Y,LWP,levels,latlon=True)#,cmap=plt.cm.Blues)#,norm= colors.BoundaryNorm(clevs, 256))

cb = m.colorbar(cs,format='%.2f')#,ticks=clevs)
cb.set_label('$kg/m^{-3}$')
#    cs=m.contourf(X,Y,data,15,latlon=True,cmap=cmap)
#    cb = m.colorbar(cs)





