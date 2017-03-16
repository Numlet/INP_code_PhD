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
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE/Satellite_Comparison')
import satellite_comparison_suite as stc

from collections import OrderedDict

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

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
amsr_data='/nfs/a201/eejvt/CASIM/THIRD_CLOUD/SATELLITE/AMSR2/'
glob.glob(amsr_data+'*')
def read_data(filename=amsr_data+'f34_20150110v7.2.gz'):
    dataset = AMSR2daily(filename, missing=missing)
    if not dataset.variables: sys.exit('problem reading file')
    return dataset

ilon = (169,174)
ilat = (273,277)
iasc = 0
avar = 'vapor'
missing = -999.

dataset = read_data()
#dataset = read_data('/nfs/a201/eejvt/CASIM/SECOND_CLOUD/SATELLITE/AMSR2/f34_20150308v7.2.gz')
#dataset = read_data('/nfs/a201/eejvt/CASIM/SECOND_CLOUD/SATELLITE/AMSR2/f34_20150301v7.2.gz')
show_dimensions(dataset)
show_variables(dataset)
show_validrange(dataset)
show_somedata(dataset)


#%%

sim_path='/nfs/a201/eejvt/CASIM/THIRD_CLOUD/'
sub_folder='L1/'
code='LWP'

cube_DM10 = iris.load(ukl.Obtain_name(sim_path+'/DM10/'+sub_folder,code))[0]
cube_GLO_HIGH = iris.load(ukl.Obtain_name(sim_path+'/GLO_HIGH/'+sub_folder,code))[0]
cube_GLO_MEAN = iris.load(ukl.Obtain_name(sim_path+'/GLO_MEAN/'+sub_folder,code))[0]
cube_GLO_MIN = iris.load(ukl.Obtain_name(sim_path+'/GLO_MIN/'+sub_folder,code))[0]
cube_GP_HAM_DMDUST = iris.load(ukl.Obtain_name(sim_path+'/DM_DUST/'+sub_folder,code))[0]
cube_MEYERS = iris.load(ukl.Obtain_name(sim_path+'/MEYERS/'+sub_folder,code))[0]
#%%


times=dataset.variables['time'][0,]
#times[times==missing]=0
LWP=dataset.variables['cloud'][0,]
#LWP[LWP==missing]=np.nan
LWP[times<15]=np.nan
LWP[times>18]=np.nan

plt.imshow(LWP)
#%%
lon=dataset.variables['longitude']

lon[lon>180]=lon[lon>180]-360
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



def check_nan(array):
    return np.isnan(array).any()

model_lons,model_lats=stc.unrotated_grid(cube_DM10)
max_lon,min_lon=model_lons.max(),model_lons.min()
max_lat,min_lat=model_lats.max(),model_lats.min()

coord=np.zeros([len(Xsat_flat),2])
coord[:,0]=Xsat_flat
coord[:,1]=Ysat_flat
cm=plt.cm.RdBu_r
#model_lons=np.linspace(-5,20,500)
X,Y=np.meshgrid(model_lons, model_lats)
grid_z1 = sc.interpolate.griddata(coord, LWP_flat, (X,Y), method='linear')
plt.imshow(grid_z1)
it=15
runs_dict=OrderedDict()
runs_dict['Satellite']=grid_z1
runs_dict['DM10']=cube_DM10[it].data
runs_dict['GLO_HIGH']=cube_GLO_HIGH[it].data
#runs_dict['MEYERS (CS)']=cube_csbm[13].data
#runs_dict['MEYERS']=cube_m[13].data
#runs_dict['3_ORD_LESS']=cube_3ord[13].data
#runs_dict['2_ORD_LESS']=cube_2l[13].data
#runs_dict['2_ORD_MORE']=cube_2m[13].data
#runs_dict['OLD_MICRO']=cube_oldm[13].data
#runs_dict['GLOPROF']=cube_gloprof[13].data
runs_dict['GLO_MEAN']=cube_GLO_MEAN[it].data
runs_dict['GLO_MIN']=cube_GLO_MIN[it].data
runs_dict['GP_HAM_DMDUST']=cube_GP_HAM_DMDUST[it].data
runs_dict['MEYERS']=cube_MEYERS[it].data
         
levels=np.arange(0.00,0.5,0.05).tolist()
same_bins=np.linspace(0.00,0.5,100)

#levels=np.linspace(runs_dict['Satellite (AMSR2)'].min(),runs_dict['Satellite (AMSR2)'].max(),15)
stc.plot_map(runs_dict,levels,lat=X,lon=Y,variable_name='LWP mm')
stc.plot_PDF(runs_dict,same_bins,
             variable_name='LWP mm')









#%%
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
times=dataset.variables['time'][0,]
times[times==missing]=0
LWP=dataset.variables['cloud'][0,]
LWP[LWP==missing]=0
lon=dataset.variables['longitude']
lat=dataset.variables['latitude']


def draw_screen_poly( lats, lons, m):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='green', alpha=0.7 )
    plt.gca().add_patch(poly)




#%%
#PLOTING EXAMPLE

lats = [ -47.5, -57.5,-57.5,-47.5]
lons = [ -5, -5, 5, 5 ]


fig=plt.figure()#figsize=(20, 12))
m = fig.add_subplot(1,1,1)
#m = Basemap(projection='lcc',
#            resolution='c',urcrnrlat=-57.5,llcrnrlat=-47.5,llcrnrlon=-5,ucrnrlon=15.)
#m = Basemap(projection='cyl',lon_0=0)#,urcrnrlat=-57.5,llcrnrlat=-47.5,llcrnrlon=-5,ucrnrlon=15.)
#m = Basemap(projection='cea',llcrnrlat=-58.5,urcrnrlat=-47.5,\
#            llcrnrlon=-5,urcrnrlon=15,resolution='c')
m= Basemap(projection='ortho',lat_0=-25,lon_0=-50,resolution='l')
draw_screen_poly( lats, lons, m )
#m = Basemap(projection='stere',lon_0=0,lat_0=90.,lat_ts=lat_0,\
#            llcrnrlat=latcorners[0],urcrnrlat=latcorners[2],\
#            llcrnrlon=loncorners[0],urcrnrlon=loncorners[2],\
#            rsphere=6371200.,resolution='l',area_thresh=10000)
m.drawcoastlines()
m.drawparallels(np.arange(-90.,120.,10.))
m.drawmeridians(np.arange(0.,360.,10))

plt.title('Liquid water path retrieved by AMSR2')
#LWP[times<12]=0
#LWP[times>14]=0

X,Y=np.meshgrid(lon,lat)
#X,Y=np.meshgrid(X[times==13],Y[times==13])
levels=np.arange(0,0.4,0.05).tolist()
cs=m.contourf(X,Y,LWP,levels,latlon=True,cmap=plt.cm.Blues)#,norm= colors.BoundaryNorm(clevs, 256))

cb = m.colorbar(cs,format='%.2f')#,ticks=clevs)
cb.set_label('$kg/m^{-3}$')


#%%
for _ in range(100):
    plt.close()















