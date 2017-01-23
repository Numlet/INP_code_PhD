# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:02:46 2016

@author: eejvt
"""

###########IMPORT STATEMENTS
import matplotlib.pyplot as plt
import sys
import glob
dir_scripts='/nfs/see-fs-01_users/eejvt/UKCA_postproc'#Change this to the downloaded folder
sys.path.append(dir_scripts)
import numpy as np
sys.path.append('/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/code')
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm


from matplotlib.patches import Polygon
path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/'

sys.path.append('/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/code')
from amsr2_daily_v7 import AMSR2daily




########DEFINITION OF FUNCTIONS
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

def read_data(filename=amsr_data+'f34_20141212v7.2.gz'):
    dataset = AMSR2daily(filename, missing=missing)
    if not dataset.variables: sys.exit('problem reading file')
    return dataset



def draw_screen_poly( lats, lons, m):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='green', alpha=0.7 )
    plt.gca().add_patch(poly)




########Main script


#data where the files are
amsr_data='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/code/data.remss.com/amsr2/bmaps_v07.2/y2014/m12/'


glob.glob(amsr_data+'*')

lats = [ -37, -57, -57, -37 ]
lons = [ -5, -5, 15, 15 ]

lats = [ -47.5, -57.5,-57.5,-47.5]
lons = [ -5, -5, 5, 5 ]

ilon = (169,174)
ilat = (273,277)
iasc = 0
avar = 'vapor'
missing = -999.

lat_max=-37.0
lat_min=-57.0
lon_max=5
lon_min=-5
import sys



#READING DATASET
dataset = read_data(amsr_data+'f34_20141212v7.2.gz')
show_dimensions(dataset)
show_variables(dataset)
show_validrange(dataset)
show_somedata(dataset)


#CHOSING VARIABLES
times=dataset.variables['time'][0,]
times[times==missing]=0
LWP=dataset.variables['cloud'][0,]
LWP[LWP==missing]=0
lon=dataset.variables['longitude']
lat=dataset.variables['latitude']


#PLOTING EXAMPLE

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
cs=m.contourf(X,Y,LWP,levels,latlon=True,cmap=plt.cm.Blues)#,norm= colors.BoundaryNorm(clevs, 256))

cb = m.colorbar(cs,format='%.2f')#,ticks=clevs)
cb.set_label('$kg/m^{-3}$')
#    cs=m.contourf(X,Y,data,15,latlon=True,cmap=cmap)
#    cb = m.colorbar(cs)





