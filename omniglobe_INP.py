#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:29:08 2017

@author: eejvt
"""
import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import Jesuslib as jl
#%%
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg",force=1)
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

cmap=plt.cm.coolwarm
plt.plot([])
plt.show()
INP_feldspar_alltemps_daily=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e6#m3
#%%
INP_feldspar_alltemps_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_feldext_alltemps_daily.npy')*1e6#m3
INP_marine_alltemps_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_alltemps_daily.npy')#m3
#

#%%
import numpy.ma as ma
#import cartopy.crs as ccrs
#import cartopy.feature as cfeat
from scipy.io.idl import readsav
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
from sklearn.metrics import mean_squared_error
import datetime



def plot(data,title=' ',projection='cyl',file_name=datetime.datetime.now().isoformat(),show=1,cblabel='$\mu g/ m^3$',cmap=plt.cm.RdBu_r,clevs=np.zeros(1),return_fig=0,dpi=300,lon=0,lat=0,colorbar_format_sci=0,saving_format='svg',scatter_points=0,f_size=20):
#    fig=plt.figure(figsize=(20, 12))
    m = fig.add_subplot(1,1,1)
#    if projection=='merc':
#        m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
#            llcrnrlon=-180,urcrnrlon=180,lat_ts=20)
#    else:
#        m = Basemap(projection=projection,lon_0=0)
    m = Basemap(resolution='c', llcrnrlon=-180.0,llcrnrlat=-90, urcrnrlon=180.0,urcrnrlat=90)

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
#        if colorbar_format_sci:
#            def fmt(x, pos):
#                a, b = '{:.1e}'.format(x).split('e')
#                b = int(b)
#                return r'${} \times 10^{{{}}}$'.format(a, b)
#            cb = m.colorbar(cs,"right",format=ticker.FuncFormatter(fmt),ticks=clevs)
#        else:
#            cb = m.colorbar(cs,format='%.2e',ticks=clevs)
#    else:
#        cs=m.contourf(X,Y,data,15,latlon=True,cmap=cmap)
#        cb = m.colorbar(cs)
#
#    cb.set_label(cblabel,fontsize=f_size)
#    cb.ax.tick_params(labelsize=f_size)
#    plt.title(title,fontsize=f_size)

import os
def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def date(iday):
    month=0
    for imonth in range(len(jl.days_end_month)):
        if iday-1>=jl.days_end_month[imonth]:        
            month=np.copy(imonth)
    month_str=jl.months_str_upper_case[month]
    
    day_str=str(iday-jl.days_end_month[month])
    
    day_end='th'
    if day_str=='1'or day_str=='21' or day_str=='31':
        day_end='st'
    if day_str=='2' or day_str=='22':
        day_end='nd'
    if day_str=='3' or day_str=='23':
        day_end='rd'
    if len(day_str)==1:
        date='0'+day_str+day_end+' '+month_str
    else:
        date=day_str+day_end+' '+month_str
    return date
print date(1)
print date(10)
print date(34)
print date(31)
print date(365)
print date(330)

h_inches = 30.0  # 3000 by 1500 i.e. OmniGlobe at 100dpi
v_inches = 15.0
#%%
sl=15
levels=[1*(4*(i%2)**2+1)*10**(i/2)for i in np.arange(-12,2,1)]
levels[0]=0.0
levels[-1]=500
sl=20
matplotlib.rcParams.update({'font.size': sl})
#%%
#fig=plt.figure(figsize=(20, 12))
fig=plt.figure(figsize=(h_inches,v_inches))

name='Feldspar_15'
folder=jl.a201+'OMNI_FELD/'
create_folder(folder)
#for i in range(INP_feldspar_alltemps_daily.shape[-1]):
for i in range(1):

    fig=plt.figure(figsize=(h_inches,v_inches))
    print i
    plot(INP_feldspar_alltemps_daily[15,30,:,:,i]*1e-3,clevs=levels,cblabel='$[INP]_{-15}/L^{-1}$',
    file_name=jl.a201+'feldspar_inp_surface_15',colorbar_format_sci=1,
    cmap=cmap,saving_format='png',f_size=sl,dpi=100)
    title='Vergara-Temprado et al. (2017) ACP'
    title2='Feldspar $[INP]_{-15^oC}$ Surface Level'
#    plt.text(0.5,0.5,title,size='x-large')
    plt.annotate(title, xy=(0.41, 0.47), xycoords='axes fraction',rotation=0,size=10)
    plt.annotate(title2, xy=(0.41, 0.44), xycoords='axes fraction',rotation=0,size=10)
    plt.annotate(date(i+1), xy=(0.42, 0.7), xycoords='axes fraction',rotation=0,size=15)
    plt.annotate(date(i+1), xy=(0, 0.7), xycoords='axes fraction',rotation=0,size=15)
    plt.annotate(date(i+1), xy=(0, 0.3), xycoords='axes fraction',rotation=0,size=15)
    plt.annotate(date(i+1), xy=(0.42, 0.3), xycoords='axes fraction',rotation=0,size=15)
    plt.savefig(folder+'Feldspar_%03i.png'%i,bbox_inches="tight",pad_inches=0.0)
#    plt.show()        
    plt.close()        
#%%


name='marine_15'
folder=jl.a201+'OMNI_MARINE/'
create_folder(folder)
for i in range(INP_marine_alltemps_daily.shape[-1]):

    fig=plt.figure(figsize=(h_inches,v_inches))
    print i
    plot(INP_marine_alltemps_daily[15,30,:,:,i]*1e-3,clevs=levels,cblabel='$[INP]_{-15}/L^{-1}$',
    file_name=jl.a201+'marine_inp_surface_15',colorbar_format_sci=1,
    cmap=cmap,saving_format='png',f_size=sl,dpi=100)
    title='Vergara-Temprado et al. (2017) ACP'
    title2='Marine $[INP]_{-15^oC}$ Surface Level'
#    plt.text(0.5,0.5,title,size='x-large')
    plt.annotate(title, xy=(0.41, 0.47), xycoords='axes fraction',rotation=0,size=10)
    plt.annotate(title2, xy=(0.41, 0.44), xycoords='axes fraction',rotation=0,size=10)
    plt.annotate(date(i+1), xy=(0.42, 0.7), xycoords='axes fraction',rotation=0,size=15)
    plt.annotate(date(i+1), xy=(0, 0.7), xycoords='axes fraction',rotation=0,size=15)
    plt.annotate(date(i+1), xy=(0, 0.3), xycoords='axes fraction',rotation=0,size=15)
    plt.annotate(date(i+1), xy=(0.42, 0.3), xycoords='axes fraction',rotation=0,size=15)
    plt.savefig(folder+'marine_%03i.png'%i,bbox_inches="tight",pad_inches=0.0)
    plt.close()      

#%%plt.figure(figsize=(h_inches,v_inches))

name='total_15'
folder=jl.a201+'OMNI_TOTAL/'
create_folder(folder)
INP_total_alltemps_daily=INP_marine_alltemps_daily+INP_feldspar_alltemps_daily
for i in range(INP_total_alltemps_daily.shape[-1]):

    fig=plt.figure(figsize=(h_inches,v_inches))
    print i
    plot(INP_total_alltemps_daily[15,30,:,:,i]*1e-3,clevs=levels,cblabel='$[INP]_{-15}/L^{-1}$',
    file_name=jl.a201+'total_inp_surface_15',colorbar_format_sci=1,
    cmap=cmap,saving_format='png',f_size=sl,dpi=100)
    title='Vergara-Temprado et al. (2017) ACP'
    title2='Total $[INP]_{-15^oC}$ Surface Level'
#    plt.text(0.5,0.5,title,size='x-large')
    plt.annotate(title, xy=(0.41, 0.47), xycoords='axes fraction',rotation=0,size=10)
    plt.annotate(title2, xy=(0.41, 0.44), xycoords='axes fraction',rotation=0,size=10)
    plt.annotate(date(i+1), xy=(0.42, 0.7), xycoords='axes fraction',rotation=0,size=15)
    plt.annotate(date(i+1), xy=(0, 0.7), xycoords='axes fraction',rotation=0,size=15)
    plt.annotate(date(i+1), xy=(0, 0.3), xycoords='axes fraction',rotation=0,size=15)
    plt.annotate(date(i+1), xy=(0.42, 0.3), xycoords='axes fraction',rotation=0,size=15)
    plt.savefig(folder+'total_%03i.png'%i,bbox_inches="tight",pad_inches=0.0)
    plt.close()      


##%%
#fig=plt.figure(figsize=(20, 12))
##sl=15
##levels=[1*(4*(i%2)**2+1)*10**(i/2)for i in np.arange(-12,2,1)]
##levels[0]=0.0
##writer = FFMpegWriter(fps=5, metadata=metadata)
##sl=20
##matplotlib.rcParams.update({'font.size': sl})
#name='Marine_15'
#create_folder(jl.a201+'OMNI_MARINE')
#for i in range(INP_feldspar_alltemps_daily.shape[-1]):
#    print i
#    plot(INP_marine_alltemps_daily[15,30,:,:,i]*1e-3,clevs=levels,cblabel='$[INP]_{-15}/L^{-1}$',
#    title='Vergara-Temprado et al. (2017) ACP Marine Organics $[INP]_{-15^oC}$ Surface Level '+date(i+1),
#    file_name=jl.a201+'feldspar_inp_surface_15',colorbar_format_sci=1,
#    cmap=cmap,saving_format='png',f_size=sl,dpi=100)
#    
#plt.close()        
##%%
#fig=plt.figure(figsize=(20, 12))
#name='Total_15'
#create_folder(jl.a201+'OMNI_TOTAL')
#with writer.saving(fig,'/nfs/a201/eejvt/'+name+".mp4", 200):
#    for i in range(INP_feldspar_alltemps_daily.shape[-1]):
#        print i
#        plot(INP_marine_alltemps_daily[15,30,:,:,i]*1e-3+INP_feldspar_alltemps_daily[15,30,:,:,i]*1e-3,clevs=levels,cblabel='$[INP]_{-15}/L^{-1}$',
#        title='Vergara-Temprado et al. (2017) ACP Total $[INP]_{-15^oC}$ Surface Level '+date(i+1),
#        file_name=jl.a201+'feldspar_inp_surface_15',colorbar_format_sci=1,
#        cmap=cmap,saving_format='png',f_size=sl,dpi=100)
#        writer.grab_frame()
#plt.close()        
#
jl.send_email()
