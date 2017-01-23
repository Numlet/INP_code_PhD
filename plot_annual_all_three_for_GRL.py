import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from math import *
import os
from netCDF4 import Dataset as nd
import iris
import iris.plot as iplt
from mpl_toolkits.basemap import Basemap
import pylab as pl
from tables import *
import Image
import cartopy.feature as cfeat


month_list=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

pre_yr1='1850'
pre_yr2='1978'
pre_yr3='1998'

lonlist= np.arange(1.25,361.25,2.5)

latlist=np.arange(-88.75,91.25,2.5)

lonlist[143]=360
lonlist[0]=0

nlon= 144
nlat= 72
npar= 31

#########################Reading in data#####################

dir_in='/nfs/a134/mm11lr/aerosol_PPE/Global_averages/Annual_averages/'

infile1= dir_in+'2008_'+pre_yr1+'_annual_average_netRF_raw_data.txt'
avg_RF1=np.loadtxt(infile1,skiprows=0,dtype='float32')

infile2= dir_in+'2008_'+pre_yr2+'_annual_average_netRF_raw_data.txt'
avg_RF2=np.loadtxt(infile2,skiprows=0,dtype='float32')

infile3= dir_in+'2008_'+pre_yr3+'_annual_average_netRF_raw_data.txt'
avg_RF3=np.loadtxt(infile3,skiprows=0,dtype='float32')


###############Combined plots#################################
## subplots require the 3 parameter index
# 1st the number of rows, then number of columns - should be the same for each subplot
# then comes the subplot number.

avg_RF1_2d= (avg_RF1).reshape([nlon,nlat]).transpose()
avg_RF2_2d= (avg_RF2).reshape([nlon,nlat]).transpose()
avg_RF3_2d= (avg_RF3).reshape([nlon,nlat]).transpose()

outdir= '/nfs/a134/mm11lr/aerosol_PPE/Global_averages/Annual_averages/'
fname_out= '2008_'+pre_yr1+'_'+pre_yr2+'_'+pre_yr3+'_average_netRF'
levels2= np.array([-2,-1.6,-1.3,-1,-0.6,-0.2,0.2,0.6,1,2,3,4])
levels1= np.array([-7,-5,-3,-2,-1,-0.2,0.2,0.36,0.52,0.68,0.84,1])
cmap_a_inv= plt.get_cmap('RdBu_r')
norm_main1= BoundaryNorm(levels1, ncolors=cmap_a_inv.N, clip=True)
norm_main2= BoundaryNorm(levels2, ncolors=cmap_a_inv.N, clip=True)

fig= plt.figure()
fig.subplots_adjust(bottom=0.9,top=1.0,left=0,right=0.1)

font_size=4
mp.rcParams['axes.labelsize'] = font_size
mp.rcParams['font.size'] = font_size
mp.rcParams['axes.linewidth'] = font_size*0.21 # 0.25 too big
mp.rcParams['axes.titlesize'] = font_size*1.1
mp.rcParams['legend.fontsize'] = font_size
mp.rcParams['xtick.labelsize'] = font_size*0.8
mp.rcParams['ytick.labelsize'] = font_size

ax1= plt.subplot(3,1,1,projection=ccrs.Mollweide())
CS = ax1.pcolormesh(lonlist, latlist, avg_RF1_2d, cmap=cmap_a_inv, norm=norm_main1, transform=ccrs.PlateCarree())
CB = plt.colorbar(CS, shrink= 0.3, orientation= 'horizontal', pad=0.05, ticks=levels1,use_gridspec=True)
CB.ax.set_xlabel('CAE radiative forcing W m$^{-2}$')
plt.figtext(0.501,0.942,'60N',fontsize=3) # 0.94 too low
plt.figtext(0.501,0.908,'30N',fontsize=3) # 0.905 too low
plt.figtext(0.501,0.828,'30S',fontsize=3) #0.83 too high
plt.figtext(0.501,0.793,'60S',fontsize=3) #0.79 too low
plt.figtext(0.4371,0.869,'60W',fontsize=3) #0.437 too far left
plt.figtext(0.385,0.869,'120W',fontsize=3) #0.38 too far right
plt.figtext(0.55,0.869,'60E',fontsize=3) 
plt.figtext(0.597,0.869,'120E',fontsize=3) #0.6 slightly too far right
plt.figtext(0.35,0.975,'a)',fontsize=6) 
ax1.add_feature(cfeat.COASTLINE,linewidth=0.3)
ax1.gridlines()
plt.title(pre_yr1+"-2008 (Global mean: -1.02 W m$^{-2}$)")

ax2= plt.subplot(3,1,2,projection=ccrs.Mollweide())
CS = ax2.pcolormesh(lonlist, latlist, avg_RF2_2d, cmap=cmap_a_inv, norm=norm_main2, transform=ccrs.PlateCarree())
CB = plt.colorbar(CS, shrink= 0.3, orientation= 'horizontal', pad=0.05, ticks=levels2,use_gridspec=True)
CB.ax.set_xlabel('CAE radiative forcing W m$^{-2}$')
plt.figtext(0.501,0.612,'60N',fontsize=3)
plt.figtext(0.501,0.578,'30N',fontsize=3)
plt.figtext(0.501,0.498,'30S',fontsize=3)
plt.figtext(0.501,0.463,'60S',fontsize=3)
plt.figtext(0.4371,0.539,'60W',fontsize=3)
plt.figtext(0.385,0.539,'120W',fontsize=3)
plt.figtext(0.55,0.539,'60E',fontsize=3) 
plt.figtext(0.597,0.539,'120E',fontsize=3)
ax2.add_feature(cfeat.COASTLINE,linewidth=0.3)
ax2.gridlines()
plt.title(pre_yr2+"-2008 (Global mean: 0.00 W m$^{-2}$)")
plt.figtext(0.35,0.645,'b)',fontsize=6) # 0.65 ever so slightly too high

ax3= plt.subplot(3,1,3,projection=ccrs.Mollweide())
CS = ax3.pcolormesh(lonlist, latlist, avg_RF3_2d, cmap=cmap_a_inv, norm=norm_main2, transform=ccrs.PlateCarree())
CB = plt.colorbar(CS, shrink= 0.3, orientation= 'horizontal', pad=0.05, ticks=levels2,use_gridspec=True)
CB.ax.set_xlabel('CAE radiative forcing W m$^{-2}$')
ax3.add_feature(cfeat.COASTLINE,linewidth=0.3)
plt.figtext(0.501,0.282,'60N',fontsize=3)
plt.figtext(0.501,0.248,'30N',fontsize=3)
plt.figtext(0.501,0.168,'30S',fontsize=3)
plt.figtext(0.501,0.133,'60S',fontsize=3)
plt.figtext(0.4371,0.209,'60W',fontsize=3)
plt.figtext(0.385,0.209,'120W',fontsize=3)
plt.figtext(0.55,0.209,'60E',fontsize=3) 
plt.figtext(0.597,0.209,'120E',fontsize=3)
ax3.gridlines()
plt.title(pre_yr3+"-2008 (Global mean: 0.014 W m$^{-2}$)")
fig.tight_layout()
plt.figtext(0.35,0.315,'c)',fontsize=6) # 0.32 too high.
#plt.savefig(outdir+fname_out+'.ps', bbox_inches=0,orientation='portrait',dpi=fig.dpi)
#pl.savefig(outdir+fname_out+'.png',bbox_inches=0)

plt.savefig(outdir+fname_out+'.eps', format='eps', dpi=fig.dpi,bbox_inches='tight')
#Image.open(outdir+fname_out+'.png').save(outdir+fname_out+'.jpg','JPEG')

plt.show()

