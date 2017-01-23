# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:26:44 2016

@author: eejvt
"""

import numpy as np
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import os
from glob import glob
import pylab
import matplotlib.pyplot as plt
import scipy as sc
from glob import glob
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
from scipy.io.idl import readsav
reload(jl)
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
import numpy.ma as ma
import datetime
import matplotlib

#INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e3 #l
#data_map=INP_feldext[20,20,:,:,:].mean(axis=-1)
#np.save('data_map_test',data_map)
data_map=np.load('data_map_test.npy')
#%%
def plot(data,title=' ',projection='cyl',file_name=datetime.datetime.now().isoformat(),
         show=1,cblabel='$\mu g/ m^3$',cmap=plt.cm.CMRmap_r,logscale=0,
         clevs=np.zeros(1),return_fig=0,dpi=300,lon=0,lat=0,colorbar_format_sci=0,saving_format='svg',
         scatter_points=0,f_size=20):
    fig=plt.figure(figsize=(20, 12))
    m = fig.add_subplot(1,1,1)
    if projection=='merc':
        m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20)
    else:
        m = Basemap(projection=projection,lon_0=0)
    m.drawcoastlines()

    if isinstance(lon, int):
        lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    if isinstance(lat, int):
        lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')

        X,Y=np.meshgrid(lon.glon,lat.glat)
    else:
        if lon.ndim==1:
            X,Y=np.meshgrid(lon,lat)
        else:
            X=np.copy(lon)
            Y=np.copy(lat)
    if type(clevs) is list:
        cs=m.contourf(X,Y,data,clevs,latlon=True,cmap=cmap,norm= colors.BoundaryNorm(clevs, 256))
        if colorbar_format_sci:
            cb = m.colorbar(cs,format='%.2e',ticks=clevs)
          
        else:
            cb = m.colorbar(cs,format='%.2e',ticks=clevs)
            #cb.set_ticks(clevs)
            #cb.set_ticklabels(clevs)
    else:
        if logscale:
            cs=m.contourf(X,Y,data,30,latlon=True,cmap=cmap,norm=matplotlib.colors.LogNorm())
        else:
            cs=m.contourf(X,Y,data,15,latlon=True,cmap=cmap)
        cb = m.colorbar(cs)

    if not isinstance(scatter_points,int):
        m.scatter(scatter_points[:,0],scatter_points[:,100])
    cb.set_label(cblabel,fontsize=f_size)
    cb.ax.tick_params(labelsize=f_size)
    plt.title(title,fontsize=f_size)
    if os.path.isdir("PLOTS/"):
        plt.savefig('PLOTS/'+file_name+'.'+saving_format,format=saving_format,dpi=dpi)
        plt.savefig('PLOTS/'+file_name+'.svg',format='svg')
    else:
        plt.savefig(file_name+'.'+saving_format,format=saving_format,dpi=dpi)
        plt.savefig(file_name+'.svg',format='svg')
    if show:
        plt.show()
    if return_fig:
        return fig
#%%

def log_levels(data_map,levels_per_order=2):
    maxmap=data_map.max()
    minmap=data_map.min()
    lim_max=int(1000+np.log10(maxmap))-1000+1
    lim_min=int(1000+np.log10(minmap))-1000
    orders_of_magnitude=lim_max-lim_min
    levels=np.logspace(lim_min,lim_max,levels_per_order*orders_of_magnitude+1)
    return levels.tolist()
    
    
    
    
    
    
    
    
    
