# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:23:55 2016

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


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/'
lon_offset=7


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
    poly = Polygon( xy, facecolor='green', alpha=0.4 )
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


#%%

cube_3ord = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/L1/','LWP'))[0]
cube_2m = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_MORE/L1/','LWP'))[0]
cube = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/L1/','LWP'))[0]
cube_con = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/BASE_CONTACT_242/L1/','LWP'))[0]
#cube_oldm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/OLD_MICRO/L1/','LWP'))[0]
#cube_nh = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/L1/','LWP'))[0]
cube_ni = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NOICE/L1/','LWP'))[0]

cube_csb = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/CLOUD_SQUEME/BASE/L1/','LWP'))[0]





times=dataset.variables['time'][0,]
#times[times==missing]=0
LWP=dataset.variables['cloud'][0,]
LWP[LWP==missing]=np.nan
LWP[times<12]=np.nan
LWP[times>15]=np.nan
lon=dataset.variables['longitude']
lat=dataset.variables['latitude']


Xsat,Ysat=np.meshgrid(lon,lat)
Xsat[np.isnan(LWP)]=np.nan
Ysat[np.isnan(LWP)]=np.nan

Xsat_flat=Xsat.flatten()
Ysat_flat=Ysat.flatten()
LWP_flat=LWP.flatten()
Xsat_flat=Xsat_flat[np.logical_not(np.isnan(Xsat_flat))]
Ysat_flat=Ysat_flat[np.logical_not(np.isnan(Ysat_flat))]
LWP_flat=LWP_flat[np.logical_not(np.isnan(LWP_flat))]




cube=cube[12,]
#cube_nh=cube_nh[13,:,:]
cube_3ord=cube_3ord[13,]
cube_2m=cube_2m[13,]
cube_csb=cube_csb[13,]
cube_con=cube_con[13,:,:]
#cube_oldm=cube_oldm[13,:,:]
model_lons,model_lats=unrotated_grid(cube)

coord=np.zeros([len(Xsat_flat),2])
coord[:,0]=Xsat_flat
coord[:,1]=Ysat_flat
cm=plt.cm.RdBu_r
#model_lons=np.linspace(-5,20,500)
X,Y=np.meshgrid(model_lons, model_lats)
grid_z1 = sc.interpolate.griddata(coord, LWP_flat, (X,Y), method='linear')

#plt.figure()
plt.figure(figsize=(15,13))
levels=np.arange(0,0.45,0.05).tolist()
plt.subplot(321)
plt.contourf(X,Y,grid_z1,levels, origin='lower',cmap=cm)
plt.title('Satellite (AMRS2)')
cb=plt.colorbar()
cb.set_label('LWP $kg/m^2$')
#           fontsize=14)

plt.subplot(322)
plt.title('ALL_ICE_PROC')
plt.contourf(X,Y,cube.data,levels, origin='lower',cmap=cm)

#levels_ct=np.linspace(0,10,21).tolist()
#
#CS=plt.contour(X,Y,cloud_top[12].data*0.3048,levels_ct,color='k')
#
#plt.clabel(CS, levels_ct,
#           inline=1,
#           fmt='%1.1f',
cb=plt.colorbar()
cb.set_label('LWP $kg/m^2$')
#plt.contourf(X,Y,grid_z0,levels, origin='lower')
#plt.title('Nearest')
plt.subplot(323)
#plt.title('BASE_CONTACT')
#plt.contourf(X,Y,cube_con.data,levels, origin='lower',cmap=cm)
plt.title('2_ORD_MORE')
plt.contourf(X,Y,cube_2m.data,levels, origin='lower',cmap=cm)
cb=plt.colorbar()
cb.set_label('LWP $kg/m^2$')
plt.subplot(324)
plt.title('3_ORD_LESS')
plt.contourf(X,Y,cube_3ord.data,levels, origin='lower',cmap=cm)
#plt.contourf(X,Y,grid_z2,levels, origin='lower')
#plt.title('Cubic')
#plt.gcf().set_size_inches(6, 6)
cb=plt.colorbar()
cb.set_label('LWP $kg/m^2$')
plt.subplot(325)
plt.title('BASE (CS)')
plt.contourf(X,Y,cube_csb.data,levels, origin='lower',cmap=cm)
#plt.contourf(X,Y,grid_z2,levels, origin='lower')
#plt.title('Cubic')
#plt.gcf().set_size_inches(6, 6)
cb=plt.colorbar()
cb.set_label('LWP $kg/m^2$')
plt.savefig('/nfs/see-fs-01_users/eejvt/CASIM/AMSR2.png')
plt.show()



#%%


levels=np.arange(0,0.45,0.05).tolist()
same_bins=np.linspace(0,0.5,100)

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
bins_con,pdf_con=PDF(cube_con.data.flatten(),same_bins)
#bins_old,pdf_old=PDF(data_old.flatten(),same_bins)
bins_3ord,pdf_3ord=PDF(cube_3ord.data.flatten(),same_bins)
bins_csb,pdf_csb=PDF(cube_csb.data.flatten(),same_bins)
bins_2m,pdf_2m=PDF(cube_2m.data.flatten(),same_bins)
#bins_nh,pdf_nh=PDF(cube_nh.data.flatten(),same_bins)
sat_bins, sat_pdf=PDF(grid_z1.flatten(),same_bins)
plt.figure(figsize=(15,10))

plt.plot(bins, pdf,label='ALL_ICE_PROC R=%1.2f'%np.corrcoef(pdf[:],sat_pdf[:])[0,1])
#plt.plot(bins_con, pdf_con,label='con R=%1.2f'%np.corrcoef(pdf_con[:],sat_pdf[:])[0,1])
#plt.plot(bins_old, pdf_old,label='old R=%1.2f'%np.corrcoef(pdf_old[20:],sat_pdf[20:])[0,1])
#plt.plot(bins_nh, pdf_nh,label='no hallet R=%1.2f'%np.corrcoef(pdf_nh[20:],sat_pdf[20:])[0,1])
plt.plot(bins_csb, pdf_csb,label='BASE(CS) R=%1.2f'%np.corrcoef(pdf_2m[:],sat_pdf[:])[0,1])
plt.plot(bins_2m, pdf_2m,label='2_ORD_LESS R=%1.2f'%np.corrcoef(pdf_2m[:],sat_pdf[:])[0,1])
plt.plot(bins_3ord, pdf_3ord,label='3_ORD_LESS R=%1.2f'%np.corrcoef(pdf_3ord[:],sat_pdf[:])[0,1])
plt.plot(sat_bins, sat_pdf,label='Satellite')
plt.legend()
plt.xlim(0,0.3)
plt.title('Probability density function of LWP AMSR2')
plt.ylabel('Normalized PDF')
plt.xlabel('LWP $mm$')


plt.savefig('/nfs/see-fs-01_users/eejvt/CASIM/AMSR2_PDF.png')

#%%


#sat_lon=lon[times_range]
#sat_lat=lat[times_range]

fig=plt.figure()#figsize=(20, 12))
m = fig.add_subplot(1,1,1)
#m = Basemap(projection='lcc',
#            resolution='c',urcrnrlat=-57.5,llcrnrlat=-47.5,llcrnrlon=-5,ucrnrlon=15.)
#m = Basemap(projection='cyl',lon_0=0)#,urcrnrlat=-57.5,llcrnrlat=-47.5,llcrnrlon=-5,ucrnrlon=15.)
#m = Basemap(projection='cea',llcrnrlat=-58.5,urcrnrlat=-47.5,\
#            llcrnrlon=-5,urcrnrlon=15,resolution='c')
m= Basemap(projection='ortho',lat_0=-35,lon_0=0,resolution='l')
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
LWP=dataset.variables['cloud'][0,]
#LWP[LWP==missing]=np.nan
#X,Y=np.meshgrid(X[times==13],Y[times==13])
levels=np.arange(0,0.4,0.05).tolist()
cs=m.contourf(X,Y,LWP,levels,latlon=True,cmap=plt.cm.Blues)#,norm= colors.BoundaryNorm(clevs, 256))

cb = m.colorbar(cs,format='%.2f')#,ticks=clevs)
cb.set_label('$kg/m^2$')
#    cs=m.contourf(X,Y,data,15,latlon=True,cmap=cmap)
#    cb = m.colorbar(cs)



