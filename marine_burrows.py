# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 10:20:09 2015

@author: eejvt
"""
import numpy as np
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import imp
imp.reload(jl)

import os
from scipy.io.idl import readsav
from glob import glob
from scipy.io import netcdf
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats.stats import pearsonr
from glob import glob
from scipy.io.idl import readsav
from mpl_toolkits.basemap import Basemap
import datetime
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
archive_directory='/nfs/a107/eejvt/'
project='MARINE_BURROWS/'
os.chdir(archive_directory+project)
lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
plt.rcdefaults()
mace_head_latlon_index=[13,124]
ps=np.linspace(0,1000,21).tolist()
amsterdam_island_latlon_index=[45,27]
ams_POM=np.array([ 0.30733973,  0.21654895,  0.13804628,  0.28574091,  0.45627275,
        0.34623331,  0.47585696,  0.15689419,  0.29635409,  0.25457847,
        0.24977866,  0.29562882])
#ug/m3
mace_POM=np.array([ 0.35145622,  0.46526086,  0.33133027,  0.32593864,  0.49346173,
        0.24990407,  0.25208488,  0.20352785,  0.30584994,  0.45496145,
        0.30464193,  0.26050338]) #ug/m3
#macemonts,march,apr,may,jun,oc,jan ug/m3
mace_obs=[0.1,0.24,0.51,0.2,0.24,0.08]

#ams  oct-nov dec-feb marc-jun
ams_obs=[0.06,0.11,0.09]
#%%
def printarr(a):
    for i in range(len(a)):
        print a[i]

#%%
def hpa_to_emacpl(hpa):
    index=int(find_nearest_vector_index(mlevs_mean,hpa))
    return index
 #chequedado

#a=11.2186
#b=-0.4459 susannah
#a=12.23968
#b=-0.37848theo
def marine_org_parameterization(T):
    a=11.2186
    b=-0.4459
    INP=np.exp(a+b*T)#[per gram]
    return INP


def INP_organic(org_mass,T):
    INP=marine_org_parameterization(T)*org_mass
    return INP


rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
rhocomp =rhocomp*1e+3#ug/cm3 or g/m3 es lo mismo
'''
;           Sulphate    Carbon    1.4*BC   NaCl      SiO2,      SOC   Feldspar
    mm_aer = [0.098,    0.012,   0.0168 , 0.05844,    0.10,    0.0168 ,0.1] ;SOC was 0.15  ; molar masses in Kg/mol
    rhocomp  = [1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.];masss density of components in Kg/m3
'''
def plot(data,title='None',projection='cyl',file_name=datetime.datetime.now().isoformat(),show=0,cblabel='$\mu g/ m^3$',cmap=plt.cm.RdBu_r,clevs=np.zeros(1),return_fig=0,dpi=300,lon=0,lat=0,colorbar_format_sci=0,saving_format='ps',scatter_points=0,scatter_points3=0,scatter_points2=0,contour=0,contourlevs=[10,100,1000]):
    # lon_0 is central longitude of projection.

    #clevs=np.logspace(np.amax(data),np.amin(data),levels)
    #print np.amax(data),np.amin(data)
    fig=plt.figure()
    #fig.set_size_inches(18.5,10.5)
    m = fig.add_subplot(1,1,1)
    # resolution = 'c' means use crude resolution coastlines.
    if projection=='merc':
        m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20)
    else:
        m = Basemap(projection=projection,lon_0=0)
    m.drawcoastlines()

    #m.fillcontinents(color='coral',lake_color='aqua')
    # draw parallels and meridians.
    #m.drawparallels(np.arange(-90.,120.,10.))
    #m.drawmeridians(np.arange(0.,360.,60.))
    #m.drawmapboundary(fill_color='aqua')
    #if (np.log(np.amax(data))-np.log(np.amin(data)))!=0:
        #clevs=logscale(data)
        #s=m.contourf(X,Y,data,30,latlon=True,cmap=cmap,clevs=clevs)#locator=ticker.LogLocator(),
    #else:
    if isinstance(lon, int):
        lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    if isinstance(lat, int):
        lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
        X,Y=np.meshgrid(lon.glon,lat.glat)
    else:
        X,Y=np.meshgrid(lon,lat)
    if type(clevs) is list:

        #clevs=clevs.tolist()

        cs=m.contourf(X,Y,data,clevs,latlon=True,cmap=cmap,norm= colors.BoundaryNorm(clevs, 256))

        if colorbar_format_sci:
            def fmt(x, pos):
                a, b = '{:.2e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)
            cb = m.colorbar(cs,"right",format=ticker.FuncFormatter(fmt),ticks=clevs)
        else:
            cb = m.colorbar(cs,"right")
            cb.set_ticks(clevs)

            #cb.set_ticklabels(clevs)



    else:
        cs=m.contourf(X,Y,data,15,latlon=True,cmap=cmap)
        cb = m.colorbar(cs)

    #cb = m.colorbar(cs,"right",ticks=clevs)#,size="5%", pad="2%"
    cb.set_label(cblabel,size=20)


    #csc=m.scatter(B73[:,4],B73[:,3],c=B73[:,2],cmap=plt.cm.Reds)
    #cb2=m.colorbar(csc,"right")
    #cb2.set_ticks(clevs)
    if not isinstance(scatter_points,int):
        m.scatter(scatter_points[:,1],scatter_points[:,0],s=20,marker='^',c='grey')
    if not isinstance(scatter_points2,int):
        m.scatter(scatter_points2[:,1],scatter_points2[:,0],s=20,marker='o',c='black')
    if not isinstance(scatter_points3,int):
        m.scatter(scatter_points3[:,1],scatter_points3[:,0],s=20,marker='s',c='blue')
    plt.title(title)
    if not isinstance(contour,int):
         #lon.glon[lon.glon>180]=lon.glon[lon.glon>180]-360

        X,Y=np.meshgrid(lon.glon,lat.glat)
        lala=m.contour(X,Y,contour,contourlevs,colors='k',hold='on',latlon=1)
        plt.clabel(lala, inline=1,fmt='%1.0f',fontsize=14)
        plt.setp(lala.collections , linewidths=2)
    if os.path.isdir("PLOTS/"):
        plt.savefig('PLOTS/'+file_name+'.'+saving_format,format=saving_format,dpi=dpi)
    else:
        plt.savefig(file_name+'.'+saving_format,format=saving_format,dpi=dpi)
    if show:
        plt.show()
    #print clevs

    if return_fig:
        return m
    #else:
    #    plt.close()

vol=(100e-9)**3*4/3*np.pi#m-3
grams_per_particle=vol*rhocomp[2]

mb=netcdf.netcdf_file('sea_POC_mass_conc_ug_timavg.nc','r')
POC=mb.variables['POC_aqua']
POC=POC[0,:,:,:]*1e-6/1.9#g/m3
#POC units='g/m3'
n_morgparticles=POC/grams_per_particle
press=netcdf.netcdf_file('monthly_4y_mean_echam.nc','r')
aps=press.variables['aps'][:,]
aps=aps.mean(axis=0)
hyam=mb.variables['hyam'][:,]
hybm=mb.variables['hybm'][:,]
mlevs=np.zeros((90,64,128))



mm=netcdf.netcdf_file('/nfs/a107/eejvt/MARINE_BURROWS/monthly_mean_marine.nc','r')
#mm=netcdf.netcdf_file('P:\eejvt\MARINE_BURROWS\monthly_mean_marine.nc')
POM_mm=mm.variables['POM_aqua']
POC_mm=POM_mm[:,:,:,:]*1e-6/1.9#g/m3










glolevs=np.array([   4.47640944,   28.3048172 ,   48.96452713,   69.30890656,
 89.56233978,  110.04908752,  131.62251282,  154.64620972,
179.33183289,  205.97129822,  234.46916199,  264.84896851,
297.05499268,  330.97183228,  366.49978638,  403.52679443,
441.94363403,  481.63827515,  522.48620605,  564.35626221,
607.08886719,  650.46594238,  694.17602539,  737.8137207 ,
780.80426025,  822.40307617,  861.61694336,  897.16723633,
927.43457031,  950.37841797,  963.48803711])

for i in range(len (hybm)):
    mlevs[i,]=hyam[i]+hybm[i]*aps



real_pressures=np.zeros(mlevs.shape)
for i in range(len(real_pressures[0,:,0])):
    real_pressures[:,i,:]=mlevs[:,-(i+1),:]

mlevs=np.copy(real_pressures)
mlevs_mean=mlevs.mean(axis=(1,2))*1e-2
mlevs=mlevs*1e-2

#%%
def find_nearest_vector_index(array, value):
    n = [abs(i-value) for i in array]
    idx = n.index(min(n))
    return idx
#emacpl_31=np.zeros(31)
emacpl_31_index=np.zeros(31)
for i in range (len(glolevs)):
    emacpl_31_index[i]=find_nearest_vector_index(mlevs_mean,glolevs[i])
emacpl_31_index=emacpl_31_index.tolist()
emacpl_31=mlevs_mean[emacpl_31_index]



a=glob('*.sav')
t=readsav(a[0])
tglo=t.t3d_mm
temac=np.zeros((90,64,128,12))


s={}
s1={}
a=glob(archive_directory+'JB_TRAINING/WITH_ICE_SCAV'+'/*.sav')
s=readsav(a[5],idict=s)
s=readsav(a[6],idict=s)
s1=readsav(a[6],idict=s1)
s1=readsav(a[7],idict=s1)
#glopl_m=s.plt_m.mean(axis=-1)*1e-2
#glopl_int=s.plt_mm.mean(axis=-1)*1e-2
#pressures_mid=s1.plt_m
#glopl_midpoint=pressures_mid.mean(axis=-1)*1e-2
#np.save('glopl_midpoint',glopl_midpoint)
glopl_midpoint=np.load('glopl_midpoint.npy')

temp_files=glob('/nfs/a173/earjbr/daily_run_ntraer30/hindcast3_temp_feldspar_*')



'''
#join arrays
month_days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])

temp_daily=np.zeros((31, 64, 128, 365))
days_cumulative=0
for i in range (12):

    td=readsav(temp_files[i])
    for j in range (month_days[i]):

        temp_daily[:,:,:,days_cumulative+j]=td.t3d_mm[:,:,:,j]
    days_cumulative=days_cumulative+month_days[i]

#move from glolevels to emac levels
temac_daily=np.zeros((90,64,128,365))
for ilat in range(len(temp_daily[0,:,0,0])):
    print 'ilat',ilat
    for ilon in range(len(temp_daily[0,0,:,0])):c
        for ilev in range(90):
            level=0
            for iglo in range (31):
                if (mlevs[ilev,ilat,ilon]>glopl_int[iglo,ilat,ilon] and mlevs[ilev,ilat,ilon]<glopl_int[iglo+1,ilat,ilon]):
                    level=iglo

            if level!=0:
                print level
            temac_daily[ilev,ilat,ilon,:]=temp_daily[level,ilat,ilon,:]
temac_daily=temac_daily-273.15
np.save('temac_daily',temac_daily)
'''


month_days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])

temp_daily=np.zeros((31, 64, 128, 365))
days_cumulative=0
for i in range (12):

    td=readsav(temp_files[i])
    for j in range (month_days[i]):

        temp_daily[:,:,:,days_cumulative+j]=td.t3d_mm[:,:,:,j]
    days_cumulative=days_cumulative+month_days[i]


'''

temac_daily=np.zeros((90,64,128,365))
for ilat in range(len(temp_daily[0,:,0,0])):
    print 'ilat',ilat
    for ilon in range(len(temp_daily[0,0,:,0])):
        for ilev in range(len(temac_daily[:,0,0,0])):
            index=find_nearest_vector_index(glopl_midpoint[:,ilat,ilon],mlevs[ilev,ilat,ilon])

            temac_daily[ilev,ilat,ilon,:]=temp_daily[index,ilat,ilon,:]
temac_daily=temac_daily-273.15
np.save('temac_daily',temac_daily)


'''








temac_daily=np.load('temac_daily.npy')
'''
#daily mean temp INP ambient
INP_mo_ambientjun=np.zeros((90, 64, 128, 30))
temac_daily=temac_daily
for ilat in range(len(temac_daily[0,:,0,0])):
    print 'ilat',ilat
    for ilon in range(len(temac_daily[0,0,:,0])):
        for ilev in range(90):
            idm=0
            for iday in np.arange(150,180,1):

                #print iday
                if temac_daily[ilev,ilat,ilon,iday]<-6 and temac_daily[ilev,ilat,ilon,iday]>-25:

                    INP_mo_ambientjun[ilev,ilat,ilon,idm]=INP_organic(POC[ilev,ilat,ilon],temac_daily[ilev,ilat,ilon,iday])
                idm=idm+1
np.save('INP_mo_ambient_ym_dailyjun',INP_mo_ambientjun)


#daily mean temp INP ambient
INP_mo_ambientjun=np.zeros((90, 64, 128, 30))
temac_daily=temac_daily
for ilat in range(len(temac_daily[0,:,0,0])):
    print 'ilat',ilat
    for ilon in range(len(temac_daily[0,0,:,0])):
        for ilev in range(90):
            idm=0
            for iday in np.arange(150,180,1):

                #print iday
                if temac_daily[ilev,ilat,ilon,iday]<-6 and temac_daily[ilev,ilat,ilon,iday]>-25:

                    INP_mo_ambientjun[ilev,ilat,ilon,idm]=INP_organic(POC[ilev,ilat,ilon],temac_daily[ilev,ilat,ilon,iday])
                elif temac_daily[ilev,ilat,ilon,iday]<-27 and temac_daily[ilev,ilat,ilon,iday]>-37:
                    INP_mo_ambientjun[ilev,ilat,ilon,idm]=INP_organic(POC[ilev,ilat,ilon],-27)
                idm=idm+1
np.save('INP_mo_ambient_ym_dailyjul_theo',INP_mo_ambientjun)
plot_lonmean(INP_mo_ambientjun,name='INP_mo_ambientjul_theo')

INP_mo_ambientjan=np.zeros((90, 64, 128, 31))
temac_daily=temac_daily
for ilat in range(len(temac_daily[0,:,0,0])):
    print 'ilat',ilat
    for ilon in range(len(temac_daily[0,0,:,0])):
        for ilev in range(90):
            idm=0
            for iday in np.arange(0,30,1):

                #print iday
                if temac_daily[ilev,ilat,ilon,iday]<-6 and temac_daily[ilev,ilat,ilon,iday]>-25:

                    INP_mo_ambientjan[ilev,ilat,ilon,idm]=INP_organic(POC[ilev,ilat,ilon],temac_daily[ilev,ilat,ilon,iday])
                elif temac_daily[ilev,ilat,ilon,iday]<-27 and temac_daily[ilev,ilat,ilon,iday]>-37:
                    INP_mo_ambientjan[ilev,ilat,ilon,idm]=INP_organic(POC[ilev,ilat,ilon],-27)
                idm=idm+1
np.save('INP_mo_ambient_ym_dailyjan_theo',INP_mo_ambientjan)
plot_lonmean(INP_mo_ambientjan,name='INP_mo_ambientjan_theo')

INP_mo_ambientmar=np.zeros((90, 64, 128, 31))
temac_daily=temac_daily
for ilat in range(len(temac_daily[0,:,0,0])):
    print 'ilat',ilat
    for ilon in range(len(temac_daily[0,0,:,0])):
        for ilev in range(90):
            idm=0
            for iday in np.arange(58,89,1):

                #print iday
                if temac_daily[ilev,ilat,ilon,iday]<-6 and temac_daily[ilev,ilat,ilon,iday]>-25:

                    INP_mo_ambientmar[ilev,ilat,ilon,idm]=INP_organic(POC[ilev,ilat,ilon],temac_daily[ilev,ilat,ilon,iday])
                elif temac_daily[ilev,ilat,ilon,iday]<-27 and temac_daily[ilev,ilat,ilon,iday]>-37:
                    INP_mo_ambientmar[ilev,ilat,ilon,idm]=INP_organic(POC[ilev,ilat,ilon],-27)
                idm=idm+1
np.save('INP_mo_ambient_ym_dailymar',INP_mo_ambientmar)
plot_lonmean(INP_mo_ambientmar,name='INP_mo_ambientmar')


INP_mo_ambient2001=np.zeros((90, 64, 128, 365))
temac_daily=temac_daily
for ilat in range(len(temac_daily[0,:,0,0])):
    print 'ilat',ilat
    for ilon in range(len(temac_daily[0,0,:,0])):
        for ilev in range(90):
            idm=0
            for iday in np.arange(0,365,1):

                #print iday
                if temac_daily[ilev,ilat,ilon,iday]<-6 and temac_daily[ilev,ilat,ilon,iday]>-25:

                    INP_mo_ambient2001[ilev,ilat,ilon,idm]=INP_organic(POC[ilev,ilat,ilon],temac_daily[ilev,ilat,ilon,iday])
                elif temac_daily[ilev,ilat,ilon,iday]<-27 and temac_daily[ilev,ilat,ilon,iday]>-37:
                    INP_mo_ambient2001[ilev,ilat,ilon,idm]=INP_organic(POC[ilev,ilat,ilon],-27)
                idm=idm+1
np.save('INP_mo_ambient_ym_daily2001_theo_param',INP_mo_ambient2001)
plot_lonmean(INP_mo_ambient2001,name='INP_mo_ambient2001_theo_param')


'''
'''
month_days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
iold=0
POC_mm_daily=np.zeros((90,64,128,365))
for i in range(len(month_days)):
    print i
    for j in range (month_days[i]):
        POC_mm_daily[:,:,:,iold+j]=POC_mm[i,:,:,:]
    iold=iold+month_days[i]
'''
#np.save('POC_mm_daily_sus',POC_mm_daily)
POC_mm_daily_theo=np.load('POC_mm_daily_theo.npy')
POC_mm_daily_sus=np.load('POC_mm_daily_sus.npy')


temps=temac_daily
mixed_phase_range_up=temps<-6
mixed_phase_range_down=temps>-25
mixed_phase_range=mixed_phase_range_down*mixed_phase_range_up
mixed_out_from_param1=temps<-25
mixed_out_from_param2=temps>-37
mixed_out_from_param=mixed_out_from_param1*mixed_out_from_param2
ts=temac_daily*mixed_phase_range
ts[mixed_out_from_param]=-27
ts[ts==0]=90000

INP_mo_daily_sus=INP_organic(POC_mm_daily_sus,ts)

np.save('INP_mo_daily_sus',INP_mo_daily_sus)
#%%
INP_mo_daily_sus=np.load('INP_mo_daily_sus.npy')


#np.save('INP_mo_daily_theo',INP_mo_daily_theo)
#INP_mo_daily_theo=np.load('INP_mo_daily_theo.npy')

#INP_mo_ambient_daily=np.load('INP_mo_ambient_ym_daoly.npy')
INP_mo_ambient2001=np.load('INP_mo_ambient_ym_daily2001.npy')
INP_mo_ambientjan=np.load('INP_mo_ambient_ym_dailyjan.npy')
INP_mo_ambient_dailyjul_theo=np.load('INP_mo_ambient_ym_dailyjul_theo.npy')
INP_mo_ambient_dailyjul_theo=np.load('INP_mo_ambient_ym_dailyjan_theo.npy')
INP_mo_ambientjun=np.load('INP_mo_ambient_ym_dailyjun.npy')
INP_mo_ambient_2001sus=np.load('INP_mo_ambient_ym_daily2001_susannah_param.npy')
INP_mo_ambient_2001theo=np.load('INP_mo_ambient_ym_daily2001_theo_param.npy')

#INP_feldext_ambient=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feldext_ambient.npy')
INP_feldext_ambient=np.load('/nfs/a107/eejvt/JB_TRAINING/Feldspar_distributions/INP_ambient_feld_ext.npy').sum(axis=0)

fig=plt.figure()
ax=plt.subplot(1,2,1)
bx=plt.subplot(1,2,2)
ax.set_title('January')
bx.set_title('July')
ax.plot(INP_mo_ambient_2001theo[:,53,:,0:30].mean(axis=(-1,-2)),mlevs_mean,label='marine theo')
ax.plot(INP_mo_ambient_2001sus[:,53,:,0:30].mean(axis=(-1,-2)),mlevs_mean,label='marine susannah')
ax.plot(INP_feldext_ambient[:,53,:,0:30].mean(axis=(-1,-2)),glolevs,label='feldext')

bx.plot(INP_mo_ambient_2001theo[:,53,:,181:212].mean(axis=(-1,-2)),mlevs_mean,label='marine theo')
bx.plot(INP_mo_ambient_2001sus[:,53,:,181:212].mean(axis=(-1,-2)),mlevs_mean,label='marine susannah')
bx.plot(INP_feldext_ambient[:,53,:,181:212].mean(axis=(-1,-2)),glolevs,label='feldext')



ax.invert_yaxis()
ax.set_xlim(xmax=100)
ax.legend()
bx.invert_yaxis()
bx.set_xlim(xmax=100)
bx.legend()
#plt.show()
plt.close()





temac_daily_mean=temac_daily.mean(axis=-1)
mixed=((temac_daily_mean<0) & (temac_daily_mean>-37))
#temac_daily_mean[1-mixed]=np.nan
t0_indexes=np.zeros((64,128))
for i in range (64):
    for j in range (128):
        t0_indexes[i,j]=find_nearest_vector_index(temac_daily_mean[:,i,j],0)



#y0=mlevs_mean[t0_indexes]


#%%
#Atlantic Transect

INP_feldext_ambient=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_ambient_feld_ext.npy').sum(axis=0)
press_mm=np.load('/nfs/a107/eejvt/pressure_mm.npy')*1e-2

lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
INP_feldext_ambient_constantpress,a,b=jl.constant_pressure_level_array(INP_feldext_ambient,press_mm,levels=21)

np.save('/nfs/a107/eejvt/JB_TRAINING/Feldspar_distributions/INP_feldext_ambient_constantpress',INP_feldext_ambient_constantpress)

#%%
INP_mo_daily_sus=np.load('INP_mo_daily_sus.npy')
mlevs365=np.zeros((90, 64, 128, 365))
for i in range(len(mlevs365[0,0,0,:])):
    mlevs365[:,:,:,i]=mlevs[:,:,:]

INP_mo_daily_sus_constantpress=np.zeros((21,64,128,365))
for i in range(len(mlevs365[0,0,0,:])):
    print i
    INP_mo_daily_sus_constantpress[:,:,:,i],c,d=jl.constant_pressure_level_array(INP_mo_daily_sus[:,:,:,i],mlevs,levels=21)
np.save('INP_mo_daily_sus_constantpress',INP_mo_daily_sus_constantpress)

#%%
#INP_feldext_ambient_constantpress_old=np.load('INP_feldext_ambient_constantpress.npy')*1e6
INP_mo_daily_sus_constantpress=np.load('INP_mo_daily_sus_constantpress.npy')
#AT_feldext=INP_feldext_ambient[:,:,118,:].mean(axis=(-1))
#AT_mo_sus=INP_mo_daily_sus[:,:,118,:].mean(axis=(-1))
ps=np.linspace(0,1000,21).tolist()

AT_feldext=INP_feldext_ambient_constantpress[:,:,118,:]*1e6
AT_mo_sus=INP_mo_daily_sus_constantpress[:,:,118,:]

#levelsmo=[0,1,20,40,60,80,100,120,140,160,180,200]
levelsmo=[1,10,20,30,40,50,60,70,80,90,100,1000]
levelsfel=[10,100,1000,10000]



#[:,:,0:30]
fs=13
#[:,:,180:211]
DJF=np.arange(-31,59,1)
DJF_months=np.array([11,0,1])
feb=np.arange(31,59,1)
fig=plt.figure()
cx=plt.subplot(1,1,1)
Xmo,Ymo= np.meshgrid(lat.glat, ps)
Xfel,Yfel=np.meshgrid(lat.glat,ps)
cx.set_title('Dec-Jan-Feb')
CS=cx.contour(Xfel,Yfel,AT_feldext[:,:,DJF_months].mean(axis=-1),levelsfel,colors='k',hold='on',linewidths=[2,2,2])#linewidths=np.linspace(2, 6, 3)
plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
plt.setp(CS.collections)
CF=cx.contourf(Xmo,Ymo,AT_mo_sus[:,:,DJF].mean(axis=-1),levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
CB=plt.colorbar(CF,ticks=levelsmo,drawedges=1,label='$m^{-3}$')

cx.invert_yaxis()
cx.set_ylim(ymax=200)
cx.tick_params(axis='both', which='major', labelsize=fs)
cx.set_ylabel('Pressure level /hPa')
cx.set_xlabel('Latitude')
cx.xaxis.set_ticks(np.arange(-90,100,20))
plt.savefig('AT_winter.png',dpi=600,format='png')
plt.savefig('AT_winter.svg',dpi=600,format='svg')
plt.show()
#%%
JJA=np.arange(150,242,1)
JJA_months=np.array([5,6,7])
plt.figure()
dx=plt.subplot(1,1,1)
Xmo,Ymo= np.meshgrid(lat.glat, ps)
Xfel,Yfel=np.meshgrid(lat.glat,ps)
dx.set_title('Jun-Jul-Aug')
CS1=dx.contour(Xfel,Yfel,AT_feldext[:,:,JJA_months].mean(axis=-1),levelsfel,colors='k',hold='on',linewidths=[2,2,2])#,linewidths=np.linspace(2, 6, 3)
plt.clabel(CS1, inline=1, fontsize=fs,fmt='%1.0f')
plt.setp(CS1.collections)
CF1=dx.contourf(Xmo,Ymo,AT_mo_sus[:,:,JJA].mean(axis=-1),levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
CB1=plt.colorbar(CF1,ticks=levelsmo,drawedges=1,label='$m^{-3}$')
dx.invert_yaxis()
dx.set_ylim(ymax=200)
dx.tick_params(axis='both', which='major', labelsize=fs)
dx.set_ylabel('Pressure level /hPa')
dx.set_xlabel('Latitude')
dx.xaxis.set_ticks(np.arange(-90,100,20))
plt.savefig('AT_sumer.png',dpi=600,format='png')
plt.savefig('AT_sumer.svg',dpi=600,format='svg')

plt.show()
#plt.close()


'''
temperatures[20,32:,[5,6,7]][temperatures[20,32:,[5,6,7]]<1000].mean()


'''




#%%
doit1=0
if doit1:

    INP_feldext_ambient_mean=INP_feldext_ambient.mean(axis=-1)
    fs=13
    #[:,:,180:211]
    DJF=np.arange(-31,59,1)
    mdays=[0,31,59,90,120,151,181,212,243,273,304,334,365]
    mnames=['January','February','March','April','May','June','July','August','September','October','November','December']
    levelsmo=[1,10,20,30,40,50,60,70,80,90,100,500]
    levelsfel=[10,100,1000]
    #os.system('cd gifs_maker')
    #os.system('mkdir AT')
    os.chdir('gifs_maker')
    os.system('mkdir latmean2')
    os.chdir('latmean2')
    #os.system('cp /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/AT/animate_psfiles /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/latmean2/')

    os.system('rm *.gif')
    for i in range(12):
        fig=plt.figure()
        cx=plt.subplot(1,1,1)
        Xmo,Ymo= np.meshgrid(lat.glat, mlevs_mean)
        Xfel,Yfel=np.meshgrid(lat.glat,glolevs)
        cx.set_title(mnames[i])
        CS=cx.contour(Xfel,Yfel,INP_feldext_ambient[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsfel,linewidths=np.linspace(2, 6, 3),colors='k',hold='on',)
        plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
        plt.setp(CS.collections )
        CF=cx.contourf(Xmo,Ymo,INP_mo_daily_sus[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
        CB=plt.colorbar(CF,ticks=levelsmo,drawedges=1,label='$m^{-3}$')

        cx.invert_yaxis()
        cx.set_ylim(ymax=200)
        cx.tick_params(axis='both', which='major', labelsize=fs)
        cx.set_ylabel('Pressure level $(hPa)$')
        cx.set_xlabel('Latitude')
        cx.xaxis.set_ticks(np.arange(-90,100,20))
        plt.savefig('latmean_forgift_%i.ps'%(i+10),dpi=300,format='ps')
        plt.close()
    #plt.show()

    os.system('bash animate_psfiles')
    os.chdir(archive_directory+project)





#%%
doit1=0
if doit1:
    fs=13
    #[:,:,180:211]
    DJF=np.arange(-31,59,1)
    mdays=[0,31,59,90,120,151,181,212,243,273,304,334,365]
    mnames=['January','February','March','April','May','June','July','August','September','October','November','December']
    levelsmo=[1,10,20,30,40,50,60,70,80,90,100,500]
    levelsfel=[10,100,1000]
    #os.system('cd gifs_maker')
    #os.system('mkdir AT')
    os.chdir('gifs_maker')
    os.system('mkdir latmean2')
    os.chdir('latmean2')
    os.system('cp /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/AT/animate_psfiles /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/latmean2/')

    os.system('rm *.gif')
    for i in range(12):
        fig=plt.figure()
        cx=plt.subplot(1,1,1)
        Xmo,Ymo= np.meshgrid(lat.glat, mlevs_mean)
        Xfel,Yfel=np.meshgrid(lat.glat,glolevs)
        cx.set_title(mnames[i])
        CS=cx.contour(Xfel,Yfel,INP_feldext_ambient[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsfel,linewidths=np.linspace(2, 6, 3),colors='k',hold='on',)
        plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
        plt.setp(CS.collections )
        CF=cx.contourf(Xmo,Ymo,INP_mo_daily_sus[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
        CB=plt.colorbar(CF,ticks=levelsmo,drawedges=1,label='$m^{-3}$')

        cx.invert_yaxis()
        cx.set_ylim(ymax=200)
        cx.tick_params(axis='both', which='major', labelsize=fs)
        cx.set_ylabel('Pressure level $(hPa)$')
        cx.set_xlabel('Latitude')
        cx.xaxis.set_ticks(np.arange(-90,100,20))
        plt.savefig('latmean_forgift_%i.ps'%(i+10),dpi=300,format='ps')
        plt.close()
    #plt.show()

    os.system('bash animate_psfiles')
    os.chdir(archive_directory+project)

#%%
doit2=1
if doit2:
    levelsmo=[1,10,20,30,40,50,60,70,80,90,100,400]
    fs=10
    mdays=[0,31,59,90,120,151,181,212,243,273,304,334,365]
    mnames=['January','February','March','April','May','June','July','August','September','October','November','December']
    levelsmo=[1,10,20,30,40,50,60,70,80,90,100,500]
    levelsfel=[10,100,1000]
    for i in range(12):
        #fig=plt.figure()
        print mnames[i]
        cx=plt.subplot(4,3,i+1)
        Xmo,Ymo= np.meshgrid(lat.glat,ps)# mlevs_mean)
        Xfel,Yfel=np.meshgrid(lat.glat,ps)#glolevs)
        cx.set_title(mnames[i])
        CS=cx.contour(Xfel,Yfel,INP_feldext_ambient_constantpress[:,:,:,i].mean(axis=(-1)),levelsfel,linewidths=np.linspace(0.5, 3, 3),colors='k',hold='on',)
        plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
        plt.setp(CS.collections )
        CF=cx.contourf(Xmo,Ymo,INP_mo_daily_sus_constantpress[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2)),levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
        if i==2 or i==5 or i==8 or i==11:
            CB=plt.colorbar(CF,ticks=levelsmo,drawedges=1,label='$m^{-3}$')

        cx.invert_yaxis()
        cx.set_ylim(ymax=200)
        cx.tick_params(axis='both', which='major', labelsize=fs)
        if i==0 or i==3 or i==6 or i==9:
            cx.set_ylabel('Pressure level $(hPa)$')
        if i>8:
            cx.set_xlabel('Latitude')
        cx.xaxis.set_ticks(np.arange(-90,100,20))
    plt.show()
    plt.savefig('latmean_grid.ps',dpi=1200,format='ps')
    plt.close()
#%%
doit3=0
if doit3:
    DJF=np.arange(-31,59,1)
    mdays=[0,31,59,90,120,151,181,212,243,273,304,334,365]
    mnames=['January','February','March','April','May','June','July','August','September','October','November','December']
    levelsmo=[1,10,20,30,40,50,60,70,80,90,100,500]
    levelsfel=[10,100,1000]
    #os.system('cd gifs_maker')
    #os.system('mkdir AT')
    os.chdir('gifs_maker')
    os.system('mkdir latmean')
    os.chdir('latmean')
    os.system('cp /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/AT/animate_psfiles /nfs/a107/eejvt/MARINE_BURROWS/gifs_maker/latmean/')

    for i in range(12):
        fig=plt.figure()
        cx=plt.subplot(1,1,1)
        Xmo,Ymo= np.meshgrid(lat.glat, mlevs_mean)
        Xfel,Yfel=np.meshgrid(lat.glat,glolevs)
        cx.set_title(mnames[i])
        CS=cx.contour(Xfel,Yfel,AT_feldext[:,:,mdays[i]:mdays[i+1]].mean(axis=-1),levelsfel,colors='k',hold='on',)
        plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
        plt.setp(CS.collections , linewidths=2)
        CF=cx.contourf(Xmo,Ymo,AT_mo_sus[:,:,mdays[i]:mdays[i+1]].mean(axis=-1),levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
        CB=plt.colorbar(CF,ticks=levelsmo,drawedges=1,label='$m^{-3}$')

        cx.invert_yaxis()
        cx.set_ylim(ymax=200)
        cx.tick_params(axis='both', which='major', labelsize=fs)
        cx.set_ylabel('Pressure level $(hPa)$')
        cx.set_xlabel('Latitude')
        cx.xaxis.set_ticks(np.arange(-90,100,20))
        plt.savefig('AT_forgift_%i.ps'%(i+10),dpi=300,format='ps')
        plt.close()
    #plt.show()

    os.system('animate_psfiles')
    os.chdir(archive_directory+project)
#%%
i=8
if i in [0,5,6,8]:
    print 'vale'
#%%
fs=10
levelsmo=[1,10,20,30,40,50,60,70,80,90,100,500]
levelsfel=[10,100,1000]
fig=plt.figure()
cx=plt.subplot(1,1,1)
Xmo,Ymo= np.meshgrid(lat.glat,ps)# mlevs_mean)
Xfel,Yfel=np.meshgrid(lat.glat,ps)#glolevs)
cx.set_title('Year mean')
CS=cx.contour(Xfel,Yfel,INP_feldext_ambient_constantpress[:,:,:,:].mean(axis=(-1,-2)),levelsfel,linewidths=np.linspace(2, 6, 3),colors='k',hold='on',)
plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
plt.setp(CS.collections )
CF=cx.contourf(Xmo,Ymo,INP_mo_daily_sus_constantpress[:,:,:,:].mean(axis=(-1,-2)),levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
CB=plt.colorbar(CF,ticks=levelsmo,drawedges=1,label='$m^{-3}$')

cx.invert_yaxis()
cx.set_ylim(ymax=200)
cx.tick_params(axis='both', which='major', labelsize=fs)
cx.set_ylabel('Pressure level $(hPa)$')
cx.set_xlabel('Latitude')
cx.xaxis.set_ticks(np.arange(-90,100,20))
plt.savefig('year_mean_lat_mean.ps',dpi=300,formatker='ps')
plt.show()
#plt.close()


#%%
def plot_lonmean(INP,name='noname'):
    INP_tm=INP.mean(axis=-1)

    INP_tm_lonm=INP_tm.mean(axis=-1)
    lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
    levels=np.linspace(0.1,150,15).tolist()
    levels=np.logspace(-1,5,15).tolist()
    #levels=np.logspace(-8,2,10).tolist()
    X, Y = np.meshgrid(lat.glat, mlevs_mean)
    fig=plt.figure()
    ax=plt.subplot(1,1,1)
    plt.contourf(X,Y,INP_tm_lonm,levels,norm= colors.BoundaryNorm(levels, 256))
    plt.colorbar(ax=ax,ticks=levels,drawedges=1)
    plt.gca().invert_yaxis()
    plt.savefig(name)
    plt.close()














'''
temac_daily=np.load('temac_daily.npy')
#montly mean temp INP ambient
INP_mo_ambient=np.zeros(temac_daily.shape)
temac_daily=temac_daily-273.15
for ilat in range(len(temac_daily[0,:,0,0])):
    print 'ilat',ilat
    for ilon in range(len(temac_daily[0,0,:,0])):
        for ilev in range(90):
            level=0
            for iglo in range (31):
                for iday in range (365):
                    if temac_daily[ilev,ilat,ilon,iday]<6 and temac_daily[ilev,ilat,ilon,iday]>-25:
                        INP_mo_ambient[ilev,ilat,ilon,iday]=INP_organic(POC[ilev,ilat,ilon],temac_daily[ilev,ilat,ilon,iday])

np.save('INP_mo_ambient_ym_daoly',INP_mo_ambient)
'''




'''
#go from temps glopl to temps emac pl
for ilat in range(len(tglo[0,:,0,0])):
    print 'ilat',ilat
    for ilon in range(len(tglo[0,0,:,0])):
        for ilev in range(90):
            level=0
            for iglo in range (31):
                if (mlevs[ilev,ilat,ilon]>glopl_int[iglo,ilat,ilon] and mlevs[ilev,ilat,ilon]<glopl_int[iglo+1,ilat,ilon]):
                    level=iglo
            if level!=0:
                print level
            temac[ilev,ilat,ilon,:]=tglo[level,ilat,ilon,:]
temac=temac-273.15
np.save('temac',temac)
'''
temac=np.load('temac.npy')
'''
#montly mean temp INP ambient
INP_mo_ambient=np.zeros(temac.shape)
for ilat in range(len(tglo[0,:,0,0])):
    print 'ilat',ilat
    for ilon in range(len(tglo[0,0,:,0])):
        for ilev in range(90):
            level=0
            for iglo in range (31):
                for imon in range (12):
                    if temac[ilev,ilat,ilon,imon]<6 and temac[ilev,ilat,ilon,imon]>-30:
                        INP_mo_ambient[ilev,ilat,ilon,imon]=INP_organic(POC[ilev,ilat,ilon],temac[ilev,ilat,ilon,imon])
#np.save('INP_mo_ambient_ym_mm',INP_mo_ambient)
'''
#INP_mo_ambient=np.load('INP_mo_ambient_ym_mm.npy')
INP_mo=np.zeros((38,90, 64, 128))
for i in range(38):
    INP_mo[i,]=INP_organic(POC[:,:,:],-i)



INP_mo_glolevs=INP_mo[:,emacpl_31_index,]
inp_per_particles15=INP_mo[15,]/n_morgparticles
B73=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/B73_m3.dat',delimiter="\t",skip_header=1)
Rosinsky=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/data_by_campaing/rosinsky.dat',delimiter="\t")
rosdata=Rosinsky[Rosinsky[:,1]<-6]
rosdata[:,2]=rosdata[:,2]*1e6
ros_gulf_data=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/data_by_campaing/burrowspaper/rosinskygulf.dat',delimiter="\t")
ros_gulf_data[:,2]=ros_gulf_data[:,2]*1e6
ros_aus_data=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/data_by_campaing/burrowspaper/rosinskyaustralia.dat',delimiter="\t")
ros_aus_data[:,2]=ros_aus_data[:,2]*1e6
#levels=np.linspace(POC[89,:,:].min()*1e6,POC[89,:,:].max()*1e6,15).tolist()
#levels=INP_organic(levels,-20)

data_points=np.concatenate((rosdata,ros_aus_data))

scatter_points=np.array((data_points[:,3],data_points[:,4],data_points[:,2]))
scatter_points2=np.array((B73[:,3],B73[:,4],B73[:,2]))
scatter_points3=np.array((ros_gulf_data[:,3],ros_gulf_data[:,4],ros_gulf_data[:,2]))
scatter_points=scatter_points.T
scatter_points2=scatter_points2.T
scatter_points3=scatter_points3.T
levels=np.array([0,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.6])*1e-6/1.9
levels_inp=INP_organic(levels,15)
levels_inp800=INP_organic(levels,15)
plot(POC[89,:,:]*1e6/1.9,show=0,cmap=plt.cm.Reds,title='surface marine organics',cblabel='$\mu g/m^{-3}$',clevs=levels,colorbar_format_sci=0,scatter_points=scatter_points,scatter_points2=scatter_points2,scatter_points3=scatter_points3,file_name='POC',dpi=1200,saving_format='svg')
plot(INP_mo[15,89,:,:]/1.9,show=0,cmap=plt.cm.Reds,title='surface INP from marine organics',cblabel='$m^{-3}$',clevs=levels_inp,colorbar_format_sci=1,scatter_points=scatter_points,scatter_points2=scatter_points2,scatter_points3=scatter_points3,file_name='INP_surface',dpi=1200,saving_format='svg')
plot(INP_mo[15,85,:,:]/1.9,show=0,cmap=plt.cm.Reds,title='800hpa INP from marine organics',cblabel='$m^{-3}$',clevs=levels_inp800,colorbar_format_sci=1,file_name='INP_800',dpi=1200,saving_format='png')

#plot(INP_mo_ambient[86,:,:,:].mean(axis=-1),cmap=plt.cm.Reds,show=1,cblabel='$m^{-3}$',title='850hpa')


'''

#plot INP_ambient with monthly mean temperatures
INP_mo_ambient_ym=INP_mo_ambient.mean(axis=-1)

INP_mo_ambient_ym_lonm=INP_mo_ambient_ym.mean(axis=-1)
lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
levels=np.linspace(0.1,150,15).tolist()
levels=np.logspace(-1,5,15).tolist()
#levels=np.logspace(-8,2,10).tolist()
X, Y = np.meshgrid(lat.glat, mlevs_mean)
fig=plt.figure()
ax=plt.subplot(1,1,1)
plt.contourf(X,Y,INP_mo_ambient_ym_lonm,levels,norm= colors.BoundaryNorm(levels, 256))
plt.colorbar(ax=ax,ticks=levels,drawedges=1)
plt.gca().invert_yaxis()


'''


'''

s={}

a=glob(archive_directory+'JB_TRAINING/WITH_ICE_SCAV'+'/*.sav')
s=readsav(a[5],idict=s)
s=readsav(a[6],idict=s)
#glopl_m=s.plt_m.mean(axis=-1)*1e-2
glopl_int=s.plt_mm.mean(axis=-1)*1e-2



#a[5]
#a[6]
inp_ext_ym_ts=np.load(archive_directory+'JB_TRAINING/'+'inp_dust_alltemp_ym_ext' + '.npy')
INP_feldext_mlevs=np.zeros((38,90,64,128))

for ilat in range(len(inp_ext_ym_ts[0,0,:,0])):
    print 'ilat',ilat
    for ilon in range(len(inp_ext_ym_ts[0,0,0,:])):
        for ilev in range(90):
            level=0
            for iglo in range (31):
                if (mlevs[ilev,ilat,ilon]>glopl_int[iglo,ilat,ilon] and mlevs[ilev,ilat,ilon]<glopl_int[iglo+1,ilat,ilon]):
                    level=iglo
            if level!=0:
                print level
            INP_feldext_mlevs[:,ilev,ilat,ilon]=inp_ext_ym_ts[:,level,ilat,ilon]
#inp_ext_ym_ts

'''

'''



f = netcdf.netcdf_file('INP_marine_and_feldext.nc', 'w')
f.createDimension('temperature',38)
f.createDimension('levels',90)
f.createDimension('lat',64)
f.createDimension('lon',128)

INP_feldext = f.createVariable('INP_feldext', 'float', ('temperature','levels','lat','lon'))
INP_feldext[:,:,:,:]=INP_feldext_mlevs[:,:,:,:]*1e6
INP_feldext.units='m3'
INP_marineorganics=f.createVariable('INP_marineorganics','float',('temperature','levels','lat','lon'))
INP_marineorganics[:,:,:,:]=INP_mo[:,:,:,:]
INP_marineorganics.units='m3'
glon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
glat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
lat=f.createVariable('lat','float',('lat',))
lon=f.createVariable('lon','float',('lon',))
lat[:]=glat.glat[:]
lon[:]=glon.glon[:]
pressure_levs=f.createVariable('pressure_levs','float',('levels','lat','lon'))
pressure_levs[:,:,:]=mlevs[:,:,:]
pressure_levs.units='hpa'
f.close()


# f = netcdf.netcdf_file('INP_marine_and_feldext.nc', 'r')

'''




#jl.plot(INP_mo[15,89,:,:],show=1,cmap=plt.cm.Blues,cblabel='$m^{-3}$',title='surface INP from marine organics')
#B73=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/B73_m3.dat',delimiter="\t",skip_header=1)
#B73_sim=jl.obtain_points_from_data(INP_mo,B73,plvs=89)
#plt.figure()
#plt.title('Comparison marine organics-B73')
#jl.fitandplot_comparison(INP_mo,B73,plvs=89,show=1)
#jl.plot_comparison(B73_sim,B73)
#upper=INP_mo[:,:,:,:]*10
#lower=INP_mo[:,:,:,:]*0.1
B73=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/B73_m3.dat',delimiter="\t",skip_header=1)
#upper_sim=jl.obtain_points_from_data(upper,B73,plvs=89)
#B73=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/B73_m3.dat',delimiter="\t",skip_header=1)
#lower_sim=jl.obtain_points_from_data(lower,B73,plvs=89)
#jl.plot_comparison(B73_sim,B73,yerrup=upper_sim,yerrdown=lower_sim)
#jl.fitandplot_comparison(INP_mo,B73,plvs=89,simdownerr=lower,simuperr=upper)
#jl.plot_comparison(B73_sim,B73)

#print 'inp_ext'
#total_inp=INP_mo_glolevs+inp_ext_ym_ts*1e6
#percentage_mo=INP_mo_glolevs/total_inp*100
#inp_int_ym_ts=np.load(archive_directory+'JB_TRAINING/'+'inp_dust_alltemp_ym_int' + '.npy')
#print 'inp_int'

#plt.scatter(B73[:,2],B73_sim[:,0])
def hpa_to_glopl(hpa):
    x=(hpa-10)/(990./31)
    x=int(round(x))
    x=x-1
    return x
def glopl_to_hpa(glopl):
    glopl=glopl+1
    hpa=glopl*990/31.93+10
    return hpa

def glopl_to_emacpl(glopl):
    glopl=glopl+1
    x=glopl*1000./31
    emacpl=x*90/1000.
    emacpl=int(round(emacpl))
    emacpl=emacpl-1
    return emacpl



def latplot(data,levels=31):
    data=data.mean(axis=2)
    lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
    levarray=np.zeros(levels)
    for i in range(levels):
        levarray[i]=glopl_to_hpa(i)
    print levarray
    X,Y=np.meshgrid(lat.glat,levarray)
    plt.contourf(X,Y,data)
    plt.gca().invert_yaxis()
    plt.colorbar()




'''
t=15
hpa=1000
glopl=hpa_to_glopl(hpa)
emacpl=glopl_to_emacpl(glopl)
tot_surface_INP=np.copy(INP_mo[:,:,:,:])
tot_surface_INP[:,89,:,:]=tot_surface_INP[:,89,:,:]#+inp_int_ym_ts[:,30,:,:]*1e6
B73_sim=jl.obtain_points_from_data(tot_surface_INP,B73,plvs=89)
r,a=pearsonr(B73_sim[:,0],B73[:,2])
sq=jl.RMSD(B73[:,2],B73_sim[:,0])
plt.title('R=%f RMSD=%f pvalue=%f'%(r,sq,a))
jl.plot_comparison(B73_sim,B73)
print np.corrcoef(B73_sim[:,0],B73[:,2]),jl.RMSD(B73[:,2],B73_sim[:,0])
print pearsonr(B73_sim[:,0],B73[:,2])
'''

#plt.title()
#jl.fitandplot_comparison(tot_surface_INP,B73,plvs=89,show=1)
levels=np.linspace(0,100,11).tolist()

#jl.plot(100*porcentaje,show=0,cmap=plt.cm.RdBu,file_name='animates/Percentage_marine_organics_total_INP_pl=%i'%hpa,title='Percentage of marine organics to the total INP pl=%i'%hpa,cblabel='%',clevs=levels)


#
#plt.close()
'''

INP_mo_glolevels=np.zeros((38, 31, 64, 128))
for i in range(31):
    x=i
    y=89./30*i INP_mo_glolevs=INP_mo[:,emacpl_31_index,]
    INP_mo_glolevels[:,i,:,:]=INP_mo[:,int(round(y)),:,:]
    #print i
'''

#total_inp=total_inp*1e-6#cm3
#INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MURRAY.dat",header=1)
#sim=jl.obtain_points_from_data(total_inp,INP_obs)
#r=np.corrcoef(sim[:,0],INP_obs[:,2])
#rmsd=jl.RMSD(sim[:,0],INP_obs[:,2])
#jl.fitandplot_comparison(total_inp,INP_obs)
#plt.show()
#plt.close()
'''
POC_glolevels=np.zeros(( 31, 64, 128))
for i in range(30):
    x=i
    y=89./30*i
    POC_glolevels[i,:,:,:]=POC[int(round(y)),:,:,:]

#INP_mo_glotemps=INP_organic(POC_glolevels,s.t3d_mm.mean(axis=-1)*mixed_phase_range)
#np.save('INP_mo_glotemps',INP_mo_glotemps)
INP_mo_glotemps=np.load('INP_mo_glotemps' + '.npy')

'''
'''
h_levels=np.zeros(s.g3d_mm.shape)
for i in range(len(s.g3d_mm[:,2,2,1])-1):
    h_levels[29-i,:,:,:]=s.g3d_mm[30-i,:,:,:]+s.g3d_mm[29-i,:,:,:]
'''

'''
t=10
jl.plot(INP_mo_glolevels[t,24,:,:],file_name='tot_mo_800hpa_%i'%t,cmap=plt.cm.YlGnBu,show=1,title='Total marine INP 800hpa t=%iC $m^{-3}$'%t,clevs=np.logspace(-4,1,20).tolist(),colorbar_format_sci=1,cblabel='$m^{-3}$',dpi=300)
t=15
jl.plot(INP_mo_glolevels[t,24,:,:],file_name='tot_mo_800hpa_%i'%t,cmap=plt.cm.YlGnBu,show=1,title='Total marine INP 800hpa t=%iC $m^{-3}$'%t,clevs=np.logspace(-4,1,20).tolist(),colorbar_format_sci=1,cblabel='$m^{-3}$',dpi=300)
t=20
jl.plot(INP_mo_glolevels[t,24,:,:],file_name='tot_mo_800hpa_%i'%t,cmap=plt.cm.YlGnBu,show=1,title='Total marine INP 800hpa t=%iC $m^{-3}$'%t,clevs=np.logspace(-4,1,20).tolist(),colorbar_format_sci=1,cblabel='$m^{-3}$',dpi=300)
t=25
jl.plot(INP_mo_glolevels[t,24,:,:],file_name='tot_mo_800hpa_%i'%t,cmap=plt.cm.YlGnBu,show=1,title='Total marine INP 800hpa t=%iC $m^{-3}$'%t,clevs=np.logspace(-4,1,20).tolist(),colorbar_format_sci=1,cblabel='$m^{-3}$',dpi=300)
'''


#jl.plot(INP_mo_glolevs[15,24,:,:],show=1,title='INP from marine organics 800hpa T=-15 $m^{-3}$',cblabel='$m^{-3}$',clevs=np.logspace(-4,5,15).tolist(),colorbar_format_sci=1,file_name='INP_mo_15_800hpa')

doit=0

if doit:
    for i in range(len(glolevs)):
        t=15
        hpa=glolevs[i]
        print i,hpa
        j=i+10
        #tot_surface_INP=INP_mo[:,emacpl,:,:]+inp_ext_ym_ts[:,glopl,:,:]*1e6
        #porcentaje=INP_mo[t,emacpl,:,:]/tot_surface_INP[t,:,:]
        #levels=np.logspace(-4,3,17).tolist()
        levels=np.linspace(0,100,11).tolist()
        month=0
        #jl.plot(INP_mo_glolevs[t,i,:,:],show=0,cmap=plt.cm.RdBu_r,file_name='animates/INP_marine_i=%i'%j,title='Marine INP pl=%i $m^{-3}$'%hpa,cblabel='$m^{-3}$',clevs=levels,colorbar_format_sci=1,saving_format='ps')
        #jl.plot(inp_ext_ym_ts[t,i,:,:]*1e6,show=0,cmap=plt.cm.RdBu_r,file_name='animates/INP_feldext_pl=%i'%j,title='Feldspar INP pl=%i $m^{-3}$'%hpa,cblabel='$m^{-3}$',clevs=levels,colorbar_format_sci=1,saving_format='ps')
        jl.plot(percentage_mo[t,i,:,:],show=0,cmap=plt.cm.RdBu,file_name='animates/percentage%i'%j,title='Percentage contribution of Marine organics to INP pl=%i T=-15'%hpa,cblabel='%',clevs=levels,saving_format='ps')

        plt.close()

#for i in glolevs:
#    print i




#jl.plot(inp_ext_ym_ts[15,25,:,:]*1e6,show=1,cblabel='$m^{-3}$',clevs=np.logspace(-4,5,15).tolist(),colorbar_format_sci=1,title='Feldspar INP distribution 816hpa T=-15')
#jl.plot(INP_mo_glolevels[15,25,:,:],show=1,cblabel='$m^{-3}$',clevs=np.logspace(-4,5,15).tolist(),colorbar_format_sci=1,title='Marine INP distribution 816hpa T=-15')
#7*14200000000*360*24*3600/1e12/60/365/2





fel850=readsav('/nfs/a107/eejvt/IDL_CODE/850hpafel.sav')
mo850=readsav('/nfs/a107/eejvt/IDL_CODE/850hpamo.sav')

INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/Feldspar_distributions/INP_feld_ext_alltemps.npy').mean(axis=-1)*1e6



#plot(mo850.mo_array2d_800hpa,cmap=plt.cm.Reds,show=1,contour=fel850.fd_array2d_800hpa,title='850hPa Marine Organic [INP]-20C',clevs=[0,5,10,15,20,25,30,35,40,45,50])
plot(mo850.mo_array2d_800hpa,cmap=plt.cm.Reds,show=1,contour=INP_feldext[20,26,:,:],title='850hPa Marine Organic [INP]-20C',clevs=[0,5,10,15,20,25,30,35,40,45,50],cblabel='$m^{-3}$')

#mo850.mo_array2d_800hpa

#fel850.fd_array2d_800hpa
