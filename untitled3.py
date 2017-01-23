# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 17:00:04 2014

@author: eejvt
"""

import numpy.ma as ma
#import cartopy.crs as ccrs
#import cartopy.feature as cfeat
from scipy.io.idl import readsav
from scipy.optimize import anneal
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


os.chdir('/nfs/see-fs-01_users/eejvt')

feld=np.genfromtxt('FELD_SILT_FRAC_T42.dat')



feld=feld.reshape(128,64)
feld=feld.T

fig=plt.figure()
m = fig.add_subplot(1,1,1)
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='robin',lon_0=0)
m.drawcoastlines()

#m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,120.,10.))
m.drawmeridians(np.arange(0.,360.,60.))
#m.drawmapboundary(fill_color='aqua')
#if (np.log(np.amax(data))-np.log(np.amin(data)))!=0:
    #clevs=logscale(data)
    #s=m.contourf(X,Y,data,30,latlon=True,cmap=cmap,clevs=clevs)#locator=ticker.LogLocator(),
#else:
lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
X,Y=np.meshgrid(lon.glon,lat.glat)
cs=m.contourf(X,Y,feld,30,latlon=True,cmap=plt.cm.RdBu_r)
cb = m.colorbar(cs,"right")

cb.set_label('dd')

plt.title('none')

#plt.savefig(archive_directory+project+'PLOTS/'+file_name+'.png',format='png',dpi=300)

plt.show()