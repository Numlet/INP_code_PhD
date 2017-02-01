# -*- coding: utf-8 -*-#
"""
Created on Fri Jan 22 12:35:46 2016

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

saving_folder='/nfs/see-fs-01_users/eejvt/terrestial_marine/distribution_plots/'
sl=20
#%%
matplotlib.rcParams.update({'font.size': sl})
#%%
#'''EXAMPLE COLORMAPS'''
#
#pylab.cm.datad
#maps=[m for m in pylab.cm.datad]
#maps.sort()
#folder='/nfs/see-fs-01_users/eejvt/example_colormap/'
#INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e3 #l
#levels=[1*(4*(i%2)**2+1)*10**(i/2)for i in np.arange(-7,3,1)]
##%%
#for cmap in maps:
#    jl.plot(INP_feldext[20,20,:,:,:].mean(axis=-1),clevs=levels,cblabel='$L^{-1}$',
#            title=cmap,
#            file_name=folder+cmap,
#            cmap=pylab.get_cmap(cmap),saving_format='png',
#            f_size=sl,dpi=100,show=0)
#%%
'''121 marine'''
max_MH=[0.17094988, 0.28302482, 0.2793586, 0.84606493, 0.5165779, 0.51200157, 0.39861295, 0.28370959, 0.2559191, 0.25218725, 0.21540299, 0.15642051]
min_MH=[0.026291696, 0.010237678, 0.050976824, 0.030408056, 0.00812867, 0.054171387, 0.038709678, 0.03089226, 0.0063462765, 0.020354403, 0.036825355, 0.033150878]
max_AI=[0.20845467, 0.21709569, 0.20782858, 0.18590406, 0.14794946, 0.16367373, 0.11563908, 0.14727378, 0.12697276, 0.14866787, 0.16019118, 0.19510968]
min_AI=[0.027016662, 0.013924136, 0.016355444, 0.0074105528, 0.044033412, 0.036094427, 0.056427792, 0.058463085, 0.021073973, 0.040624764, 0.01883061, 0.081256323]

mace_wiom=np.array([0.08,0.088,0.1,0.24,0.51,0.2,0.183,0.155,0.122,0.24,0.062,0.07])
ams_wioc=np.array([189,122,96,68,64,41,49,47,64,54,64,130])/1e3
ams_wiom=ams_wioc*1.9
point_reyes_wiom=np.array([np.nan,0.177429,np.nan,0.053704,0.098196,0.281652,0.01127,0.56441,0.220598,np.nan,np.nan,np.nan])

mace_head_latlon_index=[13,124]
amsterdam_island_latlon_index=[45,27]
point_reyes_latlon_index=[18,84]
months_str=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

class marine_organic_parameterization():
    def __init__(self,name,location=None,array_surface=None):
        self.name=name
        if array_surface==None:
            s=jl.read_data(location)
            self.array_surface=s.tot_mc_ss_mm_mode[2,30,:,:,:]
        else:
            self.array_surface=array_surface
        if self.array_surface.shape[-1]==12:
            self.wiom_mace=self.array_surface[mace_head_latlon_index[0],mace_head_latlon_index[1],:]
            self.wiom_ams=self.array_surface[amsterdam_island_latlon_index[0],amsterdam_island_latlon_index[1],:]
            self.wiom_reyes=self.array_surface[point_reyes_latlon_index[0],point_reyes_latlon_index[1],:]
        else:
            self.wiom_mace=self.array_surface[:,mace_head_latlon_index[0],mace_head_latlon_index[1]]
            self.wiom_ams=self.array_surface[:,amsterdam_island_latlon_index[0],amsterdam_island_latlon_index[1]]
            self.wiom_reyes=self.array_surface[:,point_reyes_latlon_index[0],point_reyes_latlon_index[1]]
marine_archive_directory='/nfs/a201/eejvt/'
marine_project='MARINE_PARAMETERIZATION'
my_param=marine_organic_parameterization('Mine',marine_archive_directory+marine_project+'/'+'FOURTH_TRY')
#%%
plt.figure()
plt.plot(ams_wiom,my_param.wiom_ams,'ro',label='Amsterdam Island')
plt.plot(mace_wiom,my_param.wiom_mace,'bo',label='Mace Head')
plt.plot(np.linspace(0.01,1),np.linspace(0.01,1),'k-')
plt.errorbar(ams_wiom,my_param.wiom_ams,yerr=[my_param.wiom_ams-min_AI,max_AI-my_param.wiom_ams],
                    linestyle="None",c='k',zorder=0)
plt.errorbar(mace_wiom,my_param.wiom_mace,yerr=[my_param.wiom_mace-min_MH,max_MH-my_param.wiom_mace],
                    linestyle="None",c='k',zorder=0)
plt.plot(np.linspace(0.01,1),np.linspace(0.01,1)*2.,'k--')
plt.plot(np.linspace(0.01,1),np.linspace(0.01,1)/2.,'k--')
plt.xlim(0.01,1)
plt.ylim(0.01,1)
plt.legend()
plt.xlabel('Observed WIOM $(\mu g/ m^3)$')
plt.ylabel('Modelled WIOM $(\mu g/ m^3)$')
#plt.xscale('log')
#plt.yscale('log')
plt.savefig('/nfs/see-fs-01_users/eejvt/marine_parameterization/models/121.png')
#plt.savefig('121 Hummel')
plt.show()

#%%
'''

FELDSPAR INP DISTRIBUTIONS

'''
INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e3 #l
INP_feldext_ambient=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feldext_ambient_constantpress.npy')*1e3 #l3

levels=[1*(4*(i%2)**2+1)*10**(i/2)for i in np.arange(-8,6,1)]
levels[0]=0
cmap=plt.cm.RdBu_r
cmap=plt.cm.coolwarm
cmap=plt.cm.CMRmap_r

jl.plot(INP_feldext[20,20,:,:,:].mean(axis=-1),clevs=levels,cblabel='$[INP]_{-20}/L^{-1}$',
        title='',#INP feldspar T=-20C pressure=600hpa
        file_name=saving_folder+'feldspar_600_20',colorbar_format_sci=1,
        cmap=cmap,saving_format='png',
        f_size=sl)


jl.plot(INP_feldext_ambient[12,:,:,:].mean(axis=-1),clevs=levels,cblabel='$[INP]_{ambient}/L^{-1}$',
        title='',#INP_ambient feldspar pressure=600hpa
        file_name=saving_folder+'feldspar_600_ambient',colorbar_format_sci=1,
        saving_format='png',f_size=sl,cmap=cmap)

#%%
'''
MARINE DISTRIBUTION
'''
dir_data='/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/'
names=[#'tot_mc_oc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'tot_mc_bc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'tot_mc_su_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
'tot_mc_ss_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'tot_mc_feldspar_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'GLOMAP_mode_mol_no_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'mean_dry_r_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'sigmachained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
'hindcast3_temp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
'GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
]
names=[
'tot_mc_ss_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
'hindcast3_temp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
'GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
]
s={}
a=glob('*.sav')
for name in names:
    print name
    s=readsav(dir_data+name,idict=s)

total_marine_mass=s.tot_mc_ss_mm_mode[2,]#+s.tot_mc_ss_mm_mode[3,]#ug/m3
total_marine_mass_year_mean=total_marine_mass.mean(axis=-1)
total_marine_mass_monthly_mean=jl.from_daily_to_monthly(total_marine_mass)
winter_months=[9,10,11,0,1,2]
summer_months=[3,4,5,6,7,8]
#jl.grid_earth_map(total_marine_mass_monthly_mean[30,],cblabel='$\mu g/m^3$',levels=np.logspace(-1.5,np.log10(0.6),15).tolist(),cmap=plt.cm.Greens)
#jl.plot(total_marine_mass_monthly_mean[30,:,:,summer_months].mean(axis=0),cblabel='$\mu g/m^3$',clevs=np.logspace(-1.5,np.log10(0.6),10).tolist(),cmap=plt.cm.Greens)
#jl.plot(total_marine_mass_monthly_mean[30,:,:,winter_months].mean(axis=0),cblabel='$\mu g/m^3$',clevs=np.logspace(-1.5,np.log10(0.6),10).tolist(),cmap=plt.cm.Greens)
jl.plot(total_marine_mass_year_mean[30,:,:],title='Year mean concentration',cblabel='$\mu g/m^3$',clevs=np.logspace(-1.5,np.log10(0.6),10).tolist(),cmap=plt.cm.Greens,f_size=sl,
        file_name=saving_folder+'Wiom_year_mean',saving_format='png')

#%%
'''

MARINE INP DISTRIBUTIONS

'''
cmap=plt.cm.RdBu_r
cmap=plt.cm.CMRmap_r
cmap=plt.cm.hot_r
cmap=plt.cm.YlGnBu
cmap=plt.cm.coolwarm
levels=[1*(4*(i%2)**2+1)*10**int((i-1)/2)for i in np.arange(-11,1,1)]
levels=[1*(4*(i%2)**2+1)*10**(i/2)for i in np.arange(-12,2,1)]
INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e3 #l
INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')*1e-3#l
INP_marine_ambient_constantpress_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_ambient_constant_press.npy')*1e-3 #l

jl.plot(INP_marine_alltemps[15,30,:,:,:].mean(axis=-1),clevs=levels,cblabel='$[INP]_{-15}/L^{-1}$',
   #     title='b)[INP] marine organic T=-15C surface',
        title='b)',
        file_name=saving_folder+'marine_inp_surface_15',colorbar_format_sci=1,
        cmap=cmap,saving_format='png',f_size=sl,dpi=600)


jl.plot(INP_feldext[15,30,:,:,:].mean(axis=-1),clevs=levels,cblabel='$[INP]_{-15}/L^{-1}$',
    #    title='a)[INP] feldspar T=-15C surface',
        title='a)',
        file_name=saving_folder+'feldspar_inp_surface_15',colorbar_format_sci=1,
        cmap=cmap,saving_format='png',f_size=sl,dpi=600)

jl.plot(INP_feldext[15,30,:,:,:].mean(axis=-1)+INP_marine_alltemps[15,30,:,:,:].mean(axis=-1),clevs=levels,cblabel='$[INP]_{-15}/L^{-1}$',
     #   title='c)[INP] total T=-15C surface',
        title='c)',
        file_name=saving_folder+'total_inp_surface_15',colorbar_format_sci=1,
        cmap=cmap,saving_format='png',f_size=sl,dpi=600)




jl.plot(INP_marine_ambient_constantpress_daily[12,:,:,:].mean(axis=-1),clevs=levels,cblabel='$[INP]_{ambient}/L^{-1}$',
        title='INP_ambient marine pressure=600hpa',
        file_name=saving_folder+'marine_600_ambient',colorbar_format_sci=1,
        saving_format='png',f_size=sl,dpi=600)

#%%
'''

NUMBER OF DAYS MARINE DOMINATES

'''

levels=[1*(4*(i%2)**2+1)*10**(i/2)for i in np.arange(-8,6,1)]
INP_marine_ambient_constantpress_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_ambient_constant_press.npy')*1e-3 #l
INP_feldext_ambient_constantpress_daily=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_feld_extamb_constant_press.npy')*1e3#l


lim=0.01
INP_marine_ambient_constantpress_daily[INP_marine_ambient_constantpress_daily<lim]=0
INP_feldext_ambient_constantpress_daily[INP_feldext_ambient_constantpress_daily<lim]=0
INP_ambient_total=INP_marine_ambient_constantpress_daily+INP_feldext_ambient_constantpress_daily
ratio=INP_marine_ambient_constantpress_daily/INP_feldext_ambient_constantpress_daily

ratio[np.isnan(ratio)]=0

print np.array([ratio==1]).sum()
ratio[ratio>1]=1
ratio[ratio<1]=0
ratio=ratio.sum(axis=-1)

#jl.plot(ratio[16,:,:])
#jl.plot(INP_marine_ambient_constantpress_daily[16,:,:,:].mean(axis=-1)*1e-3,clevs=levels,colorbar_format_sci=1)
levels_days=[1,10,30,60,90,120,150,180,210,240,270,300,330,360,365]
#%%
for i in range (20):
    jl.plot2(ratio[i,:,:],contour=INP_ambient_total[i,:,:,:].mean(axis=-1),contourlevs=[0.01,0.1,1,10,100,1000],title='%i hpa'%((i+1)*50),clevs=levels_days,cmap=plt.cm.brg,file_name=saving_folder+'days_mo-Feld_%i'%((i+1)*50),saving_format='png',cblabel='Days')
    plt.close()
#%%
ratio=INP_marine_ambient_constantpress_daily/INP_feldext_ambient_constantpress_daily
days_inp_not_small=np.zeros((21, 64, 128, 12))
for i in range(12):
    days_inp_not_small[:,:,:,i]=np.logical_not(np.isnan(ratio[:,:,:,jl.days_end_month[i]:jl.days_end_month[i+1]])).sum(axis=-1)


ratio[np.isnan(ratio)]=0

ratio[ratio>1]=1
ratio[ratio<1]=0

ratio_days_per_month=np.zeros((21, 64, 128, 12))
for i in range(12):
    ratio_days_per_month[:,:,:,i]=ratio[:,:,:,jl.days_end_month[i]:jl.days_end_month[i+1]].sum(axis=-1)

#a=np.array([ratio!=0]).sum()
#%%
i=12

jl.grid_earth_map_with_countourlines(ratio_days_per_month[i,],contour_map=days_inp_not_small[i,],
                                     contour_map_lines=[10,20,28],cmap=plt.cm.Blues#,file_name=saving_folder+'days_per_month_%i'%((i+1)*50),saving_format='png'
                                     ,levels=np.arange(1,31,2).tolist(),big_title='%i hpa'%((i+1)*50)
                                     ,cblabel='days')

jl.grid_earth_map(ratio_days_per_month[13,])
jl.grid_earth_map(ratio_days_per_month[13,]/days_inp_not_small[13,])
#%%
seasons={'DJF':[11,0,1],'MAM':[2,3,4],'JJA':[5,6,7],'SON':[8,9,10]}
ratio_days_per_month_season=np.zeros((21,64,128))
days_inp_not_small_season=np.zeros((21,64,128))
ilev=12
levels_temp=[-40,-20,-10,0]
#press_mm=np.load('/nfs/a107/eejvt/pressure_mm.npy')*1e-2
#temperatures=np.load('/nfs/a107/eejvt/temperatures_daily.npy')
#temperatures_monthly=jl.from_daily_to_monthly(temperatures)
#temperatures_monthly,_,_=jl.constant_pressure_level_array(temperatures_monthly,press_mm,levels=21)#seguir aqui
#np.save('/nfs/a107/eejvt/temperatures_monthly_constantpl.npy',temperatures_monthly)
temperatures_monthly_constantpl=np.load('/nfs/a107/eejvt/temperatures_monthly_constantpl.npy')

levels=[i*10 for i in range(11)]
for months in seasons.iterkeys():
    ratio_days_per_month_season=np.zeros((21,64,128))
    temperatures_season=np.zeros((21,64,128))
    days_inp_not_small_season=np.zeros((21,64,128))
    for imon in seasons[months]:
        print imon
        ratio_days_per_month_season=ratio_days_per_month_season+ratio_days_per_month[:,:,:,imon]
        days_inp_not_small_season=days_inp_not_small_season+days_inp_not_small[:,:,:,imon]
        temperatures_season=temperatures_season+temperatures_monthly_constantpl[:,:,:,imon]
    temperatures_season=temperatures_season/3
    #jl.plot(ratio_days_per_month_season[ilev,]/days_inp_not_small_season[ilev,],title=months,cmap=plt.cm.OrRd,cblabel=' ',file_name=saving_folder+'Percentage_days_%i_%s'%(((i+1)*50),months))
    days_inp_not_small_season[days_inp_not_small_season<3]=0
    fig=plt.figure()
    ax=plt.subplot(1,1,1)
    X,Y= np.meshgrid(jl.lat, jl.pressure_constant_levels)
    Xtem,Ytem= np.meshgrid(jl.lat, jl.pressure_constant_levels)
    ax.set_title(months)
    CS=ax.contour(Xtem,Ytem,temperatures_season[:,:,:].mean(axis=-1),levels_temp,colors='k',hold='on',linewidths=[2,2,2,2])#linewidths=np.linspace(2, 6, 3)
    plt.clabel(CS, inline=1, fmt='%1.0f')
    plt.setp(CS.collections)
    CF=ax.contourf(X,Y,100*(ratio_days_per_month_season.mean(axis=-1)/days_inp_not_small_season.mean(axis=-1)),levels,cmap=plt.cm.OrRd,norm=colors.BoundaryNorm(levels, 256))
    #CF=ax.contourf(X,Y,(days_inp_not_small_season).mean(axis=-1),cmap=plt.cm.OrRd)
    #CF=ax.contourf(X,Y,(ratio_days_per_month_season.mean(axis=-1)/days_inp_not_small_season.mean(axis=-1)),cmap=plt.cm.OrRd)
    #CB=plt.colorbar(CF)
    CB=plt.colorbar(CF,ticks=levels,drawedges=1,label='%')
    ax.invert_yaxis()

    ax.set_ylim(ymax=200)
    #ax.tick_params(axis='both', which='major')#, labelsize=fs)
    ax.set_ylabel('Pressure level $hPa$')
    ax.set_xlabel('Latitude')
    ax.xaxis.set_ticks(np.arange(-90,100,20))
    plt.savefig(saving_folder+'Vertical_profile_percentage_days'+months+'.png',dpi=300,format='png')
    #plt.savefig('AT_winter.svg',dpi=600,format='svg')
    plt.show()


#jl.plot(ratio_days_per_month[ilev,].sum(axis=-1)/days_inp_not_small[ilev,].sum(axis=-1),title='Year mean',cmap=plt.cm.RdGy_r)


#%%
for i in range (20):
    jl.grid_earth_map_with_countourlines(ratio_days_per_month[i,],contour_map=days_inp_not_small[i,],
                                         contour_map_lines=[10,20,28],cmap=plt.cm.Blues,file_name=saving_folder+'days_per_month_%i'%((i+1)*50),saving_format='png'
                                         ,levels=np.arange(1,31,2).tolist(),big_title='%i hpa'%((i+1)*50)
                                         ,cblabel='days')
    plt.close()

#%%

'''

LATITUDINAL MEAN VERTICAL PROFILE SEASONAL WINTER-SUMER

'''
INP_marine_ambient_constantpress_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_ambient_constant_press.npy')*1e-3 #l
INP_feldext_ambient_constantpress_daily=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_feld_extamb_constant_press.npy')*1e3#l

levelsmo=[0.001,0.01,0.02,0.030,0.040,0.050,0.060,0.070,0.080,0.090,0.100,1]
levelsfel=[0.010,0.100,1,10]
ps=np.linspace(0,1000,21).tolist()
lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')

levelsmo=[1,2,5,10,20,30,40,50,60,70,80,90,100,500]
levelsfel=[10,100,1000,10000]


fs=13
DJF=np.arange(-31,59,1)
DJF_months=np.array([11,0,1])
feb=np.arange(31,59,1)
fig=plt.figure()
cx=plt.subplot(1,1,1)
Xmo,Ymo= np.meshgrid(lat.glat, ps)
Xfel,Yfel=np.meshgrid(lat.glat,ps)
cx.set_title('Dec-Jan-Feb')
CS=cx.contour(Xfel,Yfel,INP_feldext_ambient_constantpress_daily[:,:,:,DJF].mean(axis=(-1,-2))*1e3,levelsfel,colors='k',hold='on',linewidths=[2,2,2])#linewidths=np.linspace(2, 6, 3)
plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
plt.setp(CS.collections)
CF=cx.contourf(Xmo,Ymo,INP_marine_ambient_constantpress_daily[:,:,:,DJF].mean(axis=(-1,-2))*1e3,levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
CB=plt.colorbar(CF,ticks=levelsmo,drawedges=1,label='$m^{-3}$')

cx.invert_yaxis()
cx.set_ylim(ymax=200)
cx.tick_params(axis='both', which='major', labelsize=fs)
cx.set_ylabel('Pressure level /hPa')
cx.set_xlabel('Latitude')
cx.xaxis.set_ticks(np.arange(-90,100,20))
plt.savefig(saving_folder+'AT_winter.png',dpi=600,format='png')
plt.savefig(saving_folder+'AT_winter.svg',dpi=600,format='svg')
plt.show()
#%%
JJA=np.arange(150,242,1)
JJA_months=np.array([5,6,7])
plt.figure()
dx=plt.subplot(1,1,1)
Xmo,Ymo= np.meshgrid(lat.glat, ps)
Xfel,Yfel=np.meshgrid(lat.glat,ps)
dx.set_title('Jun-Jul-Aug')
CS1=dx.contour(Xfel,Yfel,INP_feldext_ambient_constantpress_daily[:,:,:,JJA].mean(axis=(-1,-2))*1e3,levelsfel,colors='k',hold='on',linewidths=[2,2,2])#,linewidths=np.linspace(2, 6, 3)
plt.clabel(CS1, inline=1, fontsize=fs,fmt='%1.0f')
plt.setp(CS1.collections)
CF1=dx.contourf(Xmo,Ymo,INP_marine_ambient_constantpress_daily[:,:,:,JJA].mean(axis=(-1,-2))*1e3,levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
CB1=plt.colorbar(CF1,ticks=levelsmo,drawedges=1,label='$m^{-3}$')
dx.invert_yaxis()
dx.set_ylim(ymax=200)
dx.tick_params(axis='both', which='major', labelsize=fs)
dx.set_ylabel('Pressure level /hPa')
dx.set_xlabel('Latitude')
dx.xaxis.set_ticks(np.arange(-90,100,20))
plt.savefig(saving_folder+'AT_sumer.png',dpi=600,format='png')
plt.savefig(saving_folder+'AT_sumer.svg',dpi=600,format='svg')
#%%
MAM=np.arange(59,150,1)
MAM_months=np.array([2,3,4])
plt.figure()
dx=plt.subplot(1,1,1)
Xmo,Ymo= np.meshgrid(lat.glat, ps)
Xfel,Yfel=np.meshgrid(lat.glat,ps)
dx.set_title('March-April-May')
CS1=dx.contour(Xfel,Yfel,INP_feldext_ambient_constantpress_daily[:,:,:,MAM].mean(axis=(-1,-2))*1e3,levelsfel,colors='k',hold='on',linewidths=[2,2,2])#,linewidths=np.linspace(2, 6, 3)
plt.clabel(CS1, inline=1, fontsize=fs,fmt='%1.0f')
plt.setp(CS1.collections)
CF1=dx.contourf(Xmo,Ymo,INP_marine_ambient_constantpress_daily[:,:,:,MAM].mean(axis=(-1,-2))*1e3,levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
CB1=plt.colorbar(CF1,ticks=levelsmo,drawedges=1,label='$m^{-3}$')
dx.invert_yaxis()
dx.set_ylim(ymax=200)
dx.tick_params(axis='both', which='major', labelsize=fs)
dx.set_ylabel('Pressure level /hPa')
dx.set_xlabel('Latitude')
dx.xaxis.set_ticks(np.arange(-90,100,20))
plt.savefig(saving_folder+'AT_spring.png',dpi=600,format='png')
plt.savefig(saving_folder+'AT_spring.svg',dpi=600,format='svg')

#plt.show()
#%%
SON=np.arange(242,334,1)
SON_months=np.array([8,9,10])
plt.figure()
dx=plt.subplot(1,1,1)
Xmo,Ymo= np.meshgrid(lat.glat, ps)
Xfel,Yfel=np.meshgrid(lat.glat,ps)
dx.set_title('September-October-November')
CS1=dx.contour(Xfel,Yfel,INP_feldext_ambient_constantpress_daily[:,:,:,SON].mean(axis=(-1,-2))*1e3,levelsfel,colors='k',hold='on',linewidths=[2,2,2])#,linewidths=np.linspace(2, 6, 3)
plt.clabel(CS1, inline=1, fontsize=fs,fmt='%1.0f')
plt.setp(CS1.collections)
CF1=dx.contourf(Xmo,Ymo,INP_marine_ambient_constantpress_daily[:,:,:,SON].mean(axis=(-1,-2))*1e3,levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
CB1=plt.colorbar(CF1,ticks=levelsmo,drawedges=1,label='$m^{-3}$')
dx.invert_yaxis()
dx.set_ylim(ymax=200)
dx.tick_params(axis='both', which='major', labelsize=fs)
dx.set_ylabel('Pressure level /hPa')
dx.set_xlabel('Latitude')
dx.xaxis.set_ticks(np.arange(-90,100,20))
plt.savefig(saving_folder+'AT_autumn.png',dpi=600,format='png')
plt.savefig(saving_folder+'AT_autumn.svg',dpi=600,format='svg')

#plt.show()
#%%

'''

LATITUDINAL MEAN VERTICAL PROFILE FOR EVERY MONTH

'''
INP_marine_ambient_constantpress_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_ambient_constant_press.npy')*1e-3 #l
INP_feldext_ambient_constantpress_daily=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_feld_extamb_constant_press.npy')*1e3#l

levelsmo=[1,10,20,30,40,50,60,70,80,90,100,400]
fs=10
mdays=[0,31,59,90,120,151,181,212,243,273,304,334,365]
mnames=['January','February','March','April','May','June','July','August','September','October','November','December']
levelsmo=[1,10,20,30,40,50,60,70,80,90,100,500]
levelsmo=[1,5,10,20,30,40,50,100]
levelsfel=[10,100]
ps=jl.pressure_constant_levels
fig=plt.figure(figsize=(30, 20))
for i in range(12):
    #fig=plt.figure()
    print mnames[i]
    cx=plt.subplot(4,3,i+1)
    Xmo,Ymo= np.meshgrid(jl.lat,ps)# mlevs_mean)
    Xfel,Yfel=np.meshgrid(jl.lat,ps)#glolevs)
    cx.set_title(mnames[i])
    CS=cx.contour(Xfel,Yfel,INP_feldext_ambient_constantpress_daily[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2))*1e3,levelsfel,linewidths=np.linspace(0.5, 2, 2),colors='k',hold='on',)
    plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
    plt.setp(CS.collections )
    CF=cx.contourf(Xmo,Ymo,INP_marine_ambient_constantpress_daily[:,:,:,mdays[i]:mdays[i+1]].mean(axis=(-1,-2))*1e3,levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
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
#plt.savefig(saving_folder+'latmean_grid.png',dpi=600,format='png')
#plt.savefig(saving_folder+'latmean_grid.ps',format='ps')
#plt.close()
#%%

'''

MODEL VS OBSERVED feld/marine influence

'''
#matplotlib.rcParams.update({'font.size': sl})
INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')#m3
INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e6 #m3
INP_obs_mason=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",header=1)
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/TERRESTIAL_INFLUENCED.dat",header=1)
INP_obs=INP_obs[INP_obs[:,1]>-26]
plt.figure()
cmap=plt.cm.BrBG
marker='o'
marker_mason='^'
marker_size=50
marker_size_mason=120
INP_total=INP_marine_alltemps+INP_feldext
INP_total_year_mean=INP_total.mean(axis=-1)*1e-6#cm-3
simulated_values=INP_total_year_mean
simulated_values_max=INP_total.max(axis=-1)*1e-6
simulated_values_min=INP_total.min(axis=-1)*1e-6


simulated_values_feld=INP_feldext.min(axis=-1)*1e-6
simulated_values_marine=INP_marine_alltemps.min(axis=-1)*1e-6
errors=1#
title='Marine+Feldspar'


INPconc=INP_obs
INPconc_mason=INP_obs_mason

simulated_points_feld=jl.obtain_points_from_data(simulated_values_feld,INPconc,plvs=0)
simulated_points_marine=jl.obtain_points_from_data(simulated_values_marine,INPconc,plvs=0)

simulated_points_feld_mason=jl.obtain_points_from_data(simulated_values_feld,INPconc_mason,plvs=0)
simulated_points_marine_mason=jl.obtain_points_from_data(simulated_values_marine,INPconc_mason,plvs=0)

ratio=np.log10(simulated_points_marine[:,0]/simulated_points_feld[:,0])
ratio_mason=np.log10(simulated_points_marine_mason[:,0]/simulated_points_feld_mason[:,0])


simulated_points=jl.obtain_points_from_data(simulated_values,INPconc,plvs=0)
simulated_points_max=jl.obtain_points_from_data(simulated_values_max,INPconc,plvs=0)
simulated_points_min=jl.obtain_points_from_data(simulated_values_min,INPconc,plvs=0)
simulated_points_mason=jl.obtain_points_from_data(simulated_values,INPconc_mason,plvs=0)
simulated_points_mason_max=jl.obtain_points_from_data(simulated_values_max,INPconc_mason,plvs=0)
simulated_points_mason_min=jl.obtain_points_from_data(simulated_values_min,INPconc_mason,plvs=0)
data_points=INPconc
data_points_mason=INPconc_mason
bias=np.log10(simulated_points[:,0])-np.log10(data_points[:,2])
bias_mason=np.log10(simulated_points_mason[:,0])-np.log10(data_points_mason[:,2])

if errors:
    plt.errorbar(data_points_mason[:,2],simulated_points_mason[:,0],
                 yerr=[simulated_points_mason[:,0]-simulated_points_mason_min[:,0],simulated_points_mason_max[:,0]-simulated_points_mason[:,0]],
                 linestyle="None",c='k',zorder=0)
    plt.errorbar(data_points[:,2],simulated_points[:,0],
                 yerr=[simulated_points[:,0]-simulated_points_min[:,0],simulated_points_max[:,0]-simulated_points[:,0]],
                linestyle="None",c='k',zorder=0)
plot=plt.scatter(data_points[:,2],simulated_points[:,0],c=ratio,cmap=cmap,marker=marker,s=marker_size,vmin=-3.5, vmax=3.5)
plot=plt.scatter(data_points_mason[:,2],simulated_points_mason[:,0],c=ratio_mason,cmap=cmap,marker=marker_mason,s=marker_size_mason,vmin=-3, vmax=3)
    #plt.errorbar(data_points[:,2],simulated_points[:,0],yerr=[simulated_points_min[:,0],simulated_points_max[:,0]], linestyle="None",c='k')

#plot=plt.scatter(data_points[:,2],simulated_points[:,0],c=bias,cmap=cmap,marker=marker,s=marker_size,vmin=-5, vmax=5)
#plot=plt.scatter(data_points_mason[:,2],simulated_points_mason[:,0],c=bias_mason,cmap=cmap,marker=marker_mason,s=marker_size_mason,vmin=-5, vmax=5)


plt.colorbar(plot,label='$log_{10}([INP_{marine}]/[INP_{feldspar}])$')

plt.ylabel('Simulated ($cm^{-3}$)')
plt.xlabel('Observed ($cm^{-3}$)')

if np.min(simulated_points[:,0])>np.min(data_points[:,2]):
    min_plot=np.min(data_points[:,2])
else:
    min_plot=np.min(simulated_points[:,0])

if np.max(simulated_points[:,0])<np.max(data_points[:,2]):
    max_plot=np.max(data_points[:,2])
else:
    max_plot=np.max(simulated_points[:,0])

min_val=1e-9
max_val=1e1
minx=np.min(min_val)
maxx=np.max(max_val)
miny=np.min(min_val)
maxy=np.max(max_val)
min_plot=min_val
max_plot=max_val


plt.title('b)')
x=np.linspace(0.1*min_plot,10*max_plot,100)
#global x
r=np.corrcoef(np.log(data_points[:,2]),np.log(simulated_points[:,0]))
print r
#rmsd=RMSD(data_points[:,2],simulated_points[:,0])
#plt.title('R=%f RMSD=%f'%(r[0,1],rmsd))
plt.plot(x,x,'k-')
plt.plot(x,10*x,'k--')
plt.plot(x,10**1.5*x,'k-.')
plt.plot(x,0.1*x,'k--')
plt.plot(x,10**(-1.5)*x,'k-.')
plt.ylim(miny*0.1,maxy*10)
plt.xlim(minx*0.1,maxx*10)
plt.xscale('log')
plt.yscale('log')
plt.savefig(saving_folder+title+'.png')
plt.show()

print (np.array(np.abs(bias[INPconc[:,1]<-15])<1.5).sum())/float(len(bias[INPconc[:,1]<-15]))
print (np.array(np.abs(bias)<1).sum()+np.array(np.abs(bias_mason)<1).sum())/float(len(bias)+len(bias_mason))
print (np.array(np.abs(bias_mason)<1).sum())/float(len(bias_mason))
print (np.array(np.abs(bias)<1.5).sum()+np.array(np.abs(bias_mason)<1.5).sum())/float(len(bias)+len(bias_mason))
print (np.array(np.abs(bias)<1.5).sum()+np.array(np.abs(bias_mason)<1.5).sum())/float(len(bias)+len(bias_mason))


#%%

'''
121 temperature
'''
meyers=np.load('/nfs/a201/eejvt/meyers.npy')
demott=np.load('/nfs/a201/eejvt/demott.npy')
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MURRAY.dat",header=1)
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MURRAY_for_plotting.dat",header=1)
INP_obs_mason=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/R_H_MASON",header=1)
INP_obs_mason=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/demott2015",header=1)
#INP_feldext=np.load('/nfs/a201/eejvt/CLIMATOLOGY/2001/INP_feldext_alltemps_2001.npy')*1e6 #m3
INP_feldspar_climatology=np.load('/nfs/a201/eejvt/CLIMATOLOGY/INP_feldspar_climatology.npy')*1e6 #m3
INP_feldspar_climatology_std=np.load('/nfs/a201/eejvt/CLIMATOLOGY/INP_feldspar_climatology_std.npy')*1e6 #m3

INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')#m3
INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e6 #m3
INP_total=INP_marine_alltemps+INP_feldext
INP_total_year_mean=INP_total.mean(axis=-1)*1e-6#cm-3
INP_niemand=np.load('/nfs/a201/eejvt/INP_niemand_ext_alltemps.npy')
INP_osullivan=np.load('/nfs/a201/eejvt/INP_osullivan_ext_alltemps.npy')
INP_feld_perlwitz_solrem=np.load('/nfs/a201/eejvt/mineral_fractions_perlwitz/RUN_SOLREM/INP_feldext_alltemps.npy')
INP_feld_perlwitz=np.load('/nfs/a201/eejvt/mineral_fractions_perlwitz/RUN/INP_feldext_alltemps.npy')
INP_feld_solrem=np.load('/nfs/a201/eejvt/FELDSPAR_SOLUBLE_REMOVED/ACCCOR/INP_feldext_alltemps.npy')
INP_feld_perlwitz_coarse=np.load('/nfs/a201/eejvt/mineral_fractions_perlwitz/RUN/INP_feldext_alltemps_modes.npy')[:,3,:,:,:,:]
INP_feld_perlwitz_acc_005=np.load('/nfs/a201/eejvt/mineral_fractions_perlwitz/RUN/INP_feldext_alltemps_modes.npy')[:,2,:,:,:,:]*0.05

#INP_feld_perlwitz_dustfrac=np.load('/nfs/a201/eejvt/mineral_fractions_perlwitz/RUN_DUSTFRAC/INP_feldext_alltemps.npy')
#%%
params={}
class INP_param():
    def __init__(self,title, simulated_values,errors=0,simulated_values_max=0,simulated_values_min=0):
        self.title=title
        self.simulated_values= simulated_values
        self.errors=errors
        self.simulated_values_max=simulated_values_max
        self.simulated_values_min=simulated_values_min

import matplotlib



simulated_values=INP_osullivan.mean(axis=-1)*0.01+INP_marine_alltemps.mean(axis=-1)*1e-6
title='OSullivan001+marine'
#params[title]=INP_param(title,simulated_values)

simulated_values=INP_osullivan.mean(axis=-1)*0.20
title='OSullivan'
#params[title]=INP_param(title,simulated_values)




simulated_values=demott.mean(axis=-1)
simulated_values_max=demott.max(axis=-1)
simulated_values_min=demott.min(axis=-1)
errors=1
title='DeMott_2010'
params[title]=INP_param(title,simulated_values,errors,simulated_values_max,simulated_values_min)


simulated_values=meyers
title='Meyers'
params[title]=INP_param(title,simulated_values)


simulated_values=INP_niemand.mean(axis=-1)
simulated_values_max=INP_niemand.max(axis=-1)
simulated_values_min=INP_niemand.min(axis=-1)
errors=1
title='Niemand_dust'
params[title]=INP_param(title,simulated_values,errors,simulated_values_max,simulated_values_min)

simulated_values=INP_marine_alltemps.mean(axis=-1)*1e-6+INP_feld_perlwitz_solrem.mean(axis=-1)
errors=0
title='perlwitz solrem'
#params[title]=INP_param(title,simulated_values)

simulated_values=INP_marine_alltemps.mean(axis=-1)*1e-6+INP_feld_perlwitz.mean(axis=-1)
errors=0
title='perlwitz'
#params[title]=INP_param(title,simulated_values)

#simulated_values=INP_feld_perlwitz_dustfrac.mean(axis=-1)
#errors=0
#title='perlwitz dustfrac'
#params[title]=INP_param(title,simulated_values)
#
#simulated_values=INP_marine_alltemps.mean(axis=-1)*1e-6+INP_feld_perlwitz_dustfrac.mean(axis=-1)
#errors=0
#title='perlwitz dustfrac + marine'
#params[title]=INP_param(title,simulated_values)

simulated_values=INP_marine_alltemps.mean(axis=-1)*1e-6+INP_feld_solrem.mean(axis=-1)
errors=0
title='Solrem'
#params[title]=INP_param(title,simulated_values)

simulated_values=INP_marine_alltemps.mean(axis=-1)*1e-6+INP_feld_perlwitz_coarse.mean(axis=-1)
errors=0
title='Perlwitz coarse'
#params[title]=INP_param(title,simulated_values)

simulated_values=INP_marine_alltemps.mean(axis=-1)*1e-6+INP_feld_perlwitz_coarse.mean(axis=-1)+INP_feld_perlwitz_acc_005.mean(axis=-1)
errors=0
title='Perlwitz coarse+0.05acc'
#params[title]=INP_param(title,simulated_values)

errors=0#
simulated_values=INP_total_year_mean
simulated_values_max=INP_total.max(axis=-1)*1e-6
simulated_values_min=INP_total.min(axis=-1)*1e-6
errors=1#
title='Marine+Feldspar'
params[title]=INP_param(title,simulated_values,errors,simulated_values_max,simulated_values_min)

simulated_values=INP_niemand.mean(axis=-1)/10+INP_marine_alltemps.mean(axis=-1)*1e-6
simulated_values_max=INP_niemand.max(axis=-1)/10+INP_marine_alltemps.mean(axis=-1)*1e-6
simulated_values_min=INP_niemand.min(axis=-1)/10+INP_marine_alltemps.mean(axis=-1)*1e-6
errors=1

title='Niemand_dust div10'
params[title]=INP_param(title,simulated_values,errors,simulated_values_max,simulated_values_min)


for param in params.itervalues():
    #print param
    print'\n\n\n'
    INP_obs_mason=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",header=1)
    INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/TERRESTIAL_INFLUENCED.dat",header=1)


    if param.title=='Niemand_dust':
#    if True:
        range_par=np.logical_and((INP_obs[:,1]<-12), (INP_obs[:,1]>-33))
#        range_par=np.logical_and((INP_obs[:,1]<-12), (INP_obs[:,1]>-25))
        out_range_par=np.logical_not(range_par)
        range_par_mason=np.logical_and((INP_obs_mason[:,1]<-12), (INP_obs_mason[:,1]>-33))
#        range_par_mason=np.logical_and((INP_obs_mason[:,1]<-12), (INP_obs_mason[:,1]>-25))
        out_range_par_mason=np.logical_not(range_par_mason)
        INP_obs_out=INP_obs[out_range_par]
        INP_obs=INP_obs[range_par]
#        INP_obs_mason=INP_obs_mason[INP_obs_mason[:,1]<-12]
        INP_obs_mason_out=INP_obs_mason[out_range_par_mason]
        INP_obs_mason=INP_obs_mason[range_par_mason]
#        INP_obs=INP_obs[INP_obs[:,1]>-33]
#        INP_obs_mason=INP_obs_mason[INP_obs_mason[:,1]>-33]
    elif param.title=='Marine+Feldspar':
        range_par=np.logical_and((INP_obs[:,1]<-5), (INP_obs[:,1]>-26))
        out_range_par=np.logical_not(range_par)
        range_par_mason=np.logical_and((INP_obs_mason[:,1]<-5), (INP_obs_mason[:,1]>-26))
        out_range_par_mason=np.logical_not(range_par_mason)
#        print out_range_par
        INP_obs_out=INP_obs[out_range_par]
        INP_obs=INP_obs[range_par]
#        INP_obs_mason=INP_obs_mason[INP_obs_mason[:,1]<-12]
        INP_obs_mason_out=INP_obs_mason[out_range_par_mason]
        INP_obs_mason=INP_obs_mason[range_par_mason]
#        range_par=np.logical_and((INP_obs[:,1]>-12), (INP_obs[:,1]<-25))
#        out_range_par=np.logical_not(range_par)
#        INP_obs=INP_obs[INP_obs[:,1]>-26]
#        INP_obs_mason=INP_obs_mason[INP_obs_mason[:,1]>-26]
    else:
        INP_obs_out=[]
        INP_obs_mason_out=[]

    plt.figure()
    cmap=plt.cm.RdBu_r
    marker='o'
    marker_mason='^'
    marker_size=50
    marker_size_mason=120

    INPconc=INP_obs
    INPconc_mason=INP_obs_mason
    simulated_points=jl.obtain_points_from_data(param.simulated_values,INPconc)#,surface_level_comparison_on=True)
    simulated_points_mason=jl.obtain_points_from_data(param.simulated_values,INPconc_mason)#,surface_level_comparison_on=True)

    if param.errors:
        simulated_points_max=jl.obtain_points_from_data(param.simulated_values_max,INPconc)#,surface_level_comparison_on=True)
        simulated_points_min=jl.obtain_points_from_data(param.simulated_values_min,INPconc)#,surface_level_comparison_on=True)
        simulated_points_mason_max=jl.obtain_points_from_data(param.simulated_values_max,INPconc_mason)#,surface_level_comparison_on=True)
        simulated_points_mason_min=jl.obtain_points_from_data(param.simulated_values_min,INPconc_mason)#,surface_level_comparison_on=True)
    data_points=INPconc
    data_points_mason=INPconc_mason

    bias=np.log10(simulated_points[:,0])-np.log10(data_points[:,2])
    bias_mason=np.log10(simulated_points_mason[:,0])-np.log10(data_points_mason[:,2])

    if param.errors:
        plt.errorbar(data_points_mason[:,2],simulated_points_mason[:,0],
                     yerr=[simulated_points_mason[:,0]-simulated_points_mason_min[:,0],simulated_points_mason_max[:,0]-simulated_points_mason[:,0]],
                     linestyle="None",c='k',zorder=0)
        plt.errorbar(data_points[:,2],simulated_points[:,0],
                     yerr=[simulated_points[:,0]-simulated_points_min[:,0],simulated_points_max[:,0]-simulated_points[:,0]],
                    linestyle="None",c='k',zorder=0)
    plot=plt.scatter(data_points[:,2],simulated_points[:,0],c=data_points[:,1],
                     cmap=cmap,marker=marker,s=marker_size,vmin=-35, vmax=0)
    plot=plt.scatter(data_points_mason[:,2],simulated_points_mason[:,0],c=data_points_mason[:,1],
                     cmap=cmap,marker=marker_mason,s=marker_size_mason,vmin=-35, vmax=0)



    plt.colorbar(plot,label='Temperature $^{o}C$')


    bias_out=np.array([])
    bias_mason_out=np.array([])
    if len(INP_obs_out)!=0:
        INPconc_out=INP_obs_out
        simulated_points_out=jl.obtain_points_from_data(param.simulated_values,INPconc_out)#,surface_level_comparison_on=True)
        data_points_out=INPconc_out
        if param.errors:
            simulated_points_max_out=jl.obtain_points_from_data(param.simulated_values_max,INPconc_out)#,surface_level_comparison_on=True)
            simulated_points_min_out=jl.obtain_points_from_data(param.simulated_values_min,INPconc_out)#,surface_level_comparison_on=True)
            plt.errorbar(data_points_out[:,2],simulated_points_out[:,0],
                         yerr=[simulated_points_out[:,0]-simulated_points_min_out[:,0],simulated_points_max_out[:,0]-simulated_points_out[:,0]],
                        linestyle="None",c='k',zorder=0,alpha=0.3)

        plot=plt.scatter(data_points_out[:,2],simulated_points_out[:,0],c=data_points_out[:,1],
                         cmap=cmap,marker=marker,s=marker_size,vmin=-35, vmax=0,alpha=0.3)
        bias_out=np.log10(simulated_points_out[:,0])-np.log10(data_points_out[:,2])

    if len(INP_obs_mason_out)!=0:
        INPconc_mason_out=INP_obs_mason_out
        simulated_points_mason_out=jl.obtain_points_from_data(param.simulated_values,INPconc_mason_out)#,surface_level_comparison_on=True)
        data_points_out=INPconc_out
        data_points_mason_out=INPconc_mason_out
        if param.errors:
            simulated_points_mason_max_out=jl.obtain_points_from_data(param.simulated_values_max,INPconc_mason_out)#,surface_level_comparison_on=True)
            simulated_points_mason_min_out=jl.obtain_points_from_data(param.simulated_values_min,INPconc_mason_out)#,surface_level_comparison_on=True)
            plt.errorbar(data_points_mason_out[:,2],simulated_points_mason_out[:,0],
                         yerr=[simulated_points_mason_out[:,0]-simulated_points_mason_min_out[:,0],simulated_points_mason_max_out[:,0]-simulated_points_mason_out[:,0]],
                        linestyle="None",c='k',zorder=0,alpha=0.5)
        plot=plt.scatter(data_points_mason_out[:,2],simulated_points_mason_out[:,0],c=data_points_mason_out[:,1],
                         cmap=cmap,marker=marker_mason,s=marker_size_mason,vmin=-35, vmax=0,alpha=0.5)
        bias_mason_out=np.log10(simulated_points_mason_out[:,0])-np.log10(data_points_mason_out[:,2])


        #plt.errorbar(data_points[:,2],simulated_points[:,0],yerr=[simulated_points_min[:,0],simulated_points_max[:,0]], linestyle="None",c='k')

#    print '--------------------------------------------------------------------'
#    print 'outside range'
#    print param.title
#    print 'datapoints',len(bias_out)+len(bias_mason_out)
#    print 'low temp (-15) bias 1.5 fraction',(np.array(np.abs(bias_out[INPconc_out[:,1]<-15])<1.5).sum())/float(len(bias_out[INPconc_out[:,1]<-15]))
#    print 'total marine+terrestrial bias 1',(np.array(np.abs(bias_out)<1).sum()+np.array(np.abs(bias_mason_out)<1).sum())/float(len(bias_out)+len(bias_mason_out))
#    print (np.array(np.abs(bias_mason_out)<1).sum())/float(len(bias_mason_out))
#    print 'total marine+terrestrial bias 1.5',(np.array(np.abs(bias_out)<1.5).sum()+np.array(np.abs(bias_mason_out)<1.5).sum())/float(len(bias_out)+len(bias_mason_out))
#    print (np.array(np.abs(bias_out)<1.5).sum()+np.array(np.abs(bias_mason_out)<1.5).sum())/float(len(bias_out)+len(bias_mason_out))
#    print '--------------------------------------------------------------------'
    #plot=plt.scatter(data_points[:,2],simulated_points[:,0],c=bias,cmap=cmap,marker=marker,s=marker_size,vmin=-5, vmax=5)
#            plt.errorbar(data_points_mason_out[:,2],simulated_points_mason_out[:,0],
#                         yerr=[simulated_points_mason[:,0]-simulated_points_mason_min[:,0],simulated_points_mason_max[:,0]-simulated_points_mason[:,0]],
#                         linestyle="None",c='k',zorder=0)
    #plot=plt.scatter(data_points_mason[:,2],simulated_points_mason[:,0],c=bias_mason,cmap=cmap,marker=marker_mason,s=marker_size_mason,vmin=-5, vmax=5)


    plt.ylabel('Simulated [INP] ($cm^{-3}$)')
    plt.xlabel('Observed [INP] ($cm^{-3}$)')
    if param.title=='Meyers':
        plt.ylabel('Calculated [INP] ($cm^{-3}$)')

    if np.min(simulated_points[:,0])>np.min(data_points[:,2]):
        min_plot=np.min(data_points[:,2])
    else:
        min_plot=np.min(simulated_points[:,0])

    if np.max(simulated_points[:,0])<np.max(data_points[:,2]):
        max_plot=np.max(data_points[:,2])
    else:
        max_plot=np.max(simulated_points[:,0])

    #minx=np.min(data_points[:,2])
    #maxx=np.max(data_points[:,2])
    #miny=np.min(simulated_points[:,0])
    #maxy=np.max(simulated_points[:,0])
    min_val=1e-9
    max_val=1e1
    minx=np.min(min_val)
    maxx=np.max(max_val)
    miny=np.min(min_val)
    maxy=np.max(max_val)
    min_plot=min_val
    max_plot=max_val


    x=np.linspace(0.1*min_plot,10*max_plot,100)
    #global x
    r=np.corrcoef(np.log(data_points[:,2]),np.log(simulated_points[:,0]))
    mean_error=jl.mean_error(np.log(data_points[:,2]),np.log(simulated_points[:,0]))
    mean_bias=jl.mean_bias(np.log(data_points[:,2]),np.log(simulated_points[:,0]))
    print param.title,r,mean_error,mean_bias
    add=0
    if param.title=='Meyers':
        add='a)Meyers et al. (1992)'
    if param.title=='DeMott_2010':
        add='b)DeMott et al. (2010)'
    if param.title=='Niemand_dust':
        add='c)Niemand et al. (2012)'
    if param.title=='Marine+Feldspar':
        add='d)Marine+Feldspar'
    if add:
        plt.title(add)
    else:
        plt.title(param.title)
    print '(log) R=%1.3f ERROR=%1.3f BIAS=%1.3f'%(r[0,1],mean_error,mean_bias)

    #rmsd=RMSD(data_points[:,2],simulated_points[:,0])
    #plt.title('R=%f RMSD=%f'%(r[0,1],rmsd))
    plt.plot(x,x,'k-')
    plt.plot(x,10*x,'k--')
    plt.plot(x,10**1.5*x,'k-.')
    plt.plot(x,0.1*x,'k--')
    plt.plot(x,10**(-1.5)*x,'k-.')
    plt.ylim(miny*0.1,maxy*10)
    plt.xlim(minx*0.1,maxx*10)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('/nfs/see-fs-01_users/eejvt/terrestial_marine/'+param.title+'.png',dpi=400)
    plt.savefig('/nfs/see-fs-01_users/eejvt/terrestial_marine/'+param.title+'.eps',format='eps')
    plt.show()
    bias=np.log10(simulated_points[:,0])-np.log10(data_points[:,2])
    bias_mason=np.log10(simulated_points_mason[:,0])-np.log10(data_points_mason[:,2])
    print param.title
    print 'datapoints',len(INPconc)+len(INPconc_mason)
    print 'low temp (-15) bias 1.5 fraction',(np.array(np.abs(bias[INPconc[:,1]<-15])<1.5).sum())/float(len(bias[INPconc[:,1]<-15]))
    print 'total marine+terrestrial bias 1',(np.array(np.abs(bias)<1).sum()+np.array(np.abs(bias_mason)<1).sum())/float(len(bias)+len(bias_mason))
    print (np.array(np.abs(bias_mason)<1).sum())/float(len(bias_mason))
    print 'total marine+terrestrial bias 1.5',(np.array(np.abs(bias)<1.5).sum()+np.array(np.abs(bias_mason)<1.5).sum())/float(len(bias)+len(bias_mason))
    print (np.array(np.abs(bias)<1.5).sum()+np.array(np.abs(bias_mason)<1.5).sum())/float(len(bias)+len(bias_mason))

#    plt.close()




#%%
'''

RATIO SURFACE_AREA

'''
s=jl.read_data('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2')
fel_modes=[2,3]#,5,6]
#def INP_feldspar_ext(s,T):
std=s.sigma[:]
#T=258
modes_vol=jl.volumes_of_modes(s)
feld_volfrac=(s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/jl.rhocomp[6])/modes_vol
dust_volfrac=(s.tot_mc_dust_mm_mode[:,:,:,:,:]/jl.rhocomp[6])/modes_vol
Nd=s.st_nd[:,:,:,:,:]*feld_volfrac
Nd_dust=s.st_nd[:,:,:,:,:]*dust_volfrac
rmean=s.rbardry[:,:,:,:,:]*1e2
surface_area_conc_feld=Nd*4*np.pi*rmean**2
surface_area_conc_dust=Nd_dust*4*np.pi*rmean**2
#%%
surface_area_conc_alldust=surface_area_conc_feld+surface_area_conc_dust
surface_area_conc_feld_sol=surface_area_conc_feld[2,]+surface_area_conc_feld[3,]#
surface_area_conc_alldust_sol=surface_area_conc_alldust[2,]+surface_area_conc_alldust[3,]#
percentage_feld_SA=surface_area_conc_feld_sol/surface_area_conc_alldust_sol
jl.plot(percentage_feld_SA[20,:,:,:].mean(axis=-1),title='Ratio Feld/Dust  Surface Area',cblabel='$[S_{feld}]/[S_{dust}]$',file_name=saving_folder+'Surface_area_ratio',saving_format='png')

#%%
'''
Plots by campaing
'''

class campaing():
    def __init__(self, name, location, m_or_t, color,values):
        self.name=name
        self.location=location
        self.m_or_t=m_or_t
        self.color=color
        self.values=values
header=1

data_marine=np.genfromtxt("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",delimiter="\t",skip_header=header,dtype=str)
data_terrestrial=np.genfromtxt("/nfs/a107/eejvt/INP_DATA/TERRESTIAL_INFLUENCED.dat",delimiter="\t",skip_header=header,dtype=str)

#for data_point in data_marine:
#    print data_point[0]
campaigns_dict={}
campaigns_dict['bigg 73']=campaing('Bigg1973','Southern Ocean','m','y',data_marine[:,0]=='bigg 73')
campaigns_dict['DeMott2015']=campaing('DeMott2015','Various marine locations','m','b',data_marine[:,0]=='DeMott2015')
campaigns_dict['R. H. Mason']=campaing('Mason2016','Various marine locations','m','aqua',data_marine[:,0]=='R. H. Mason')
campaigns_dict['rosisnky gulf']=campaing('Rosisnky','Gulf of Mexico','m','darkblue',data_marine[:,0]=='rosisnky gulf')


campaigns_dict['INSPECT-I']=campaing('INSPECT-I','check','t','sienna',data_terrestrial[:,0]=='INSPECT-I')
campaigns_dict['INSPECT-II']=campaing('INSPECT-II','check','t','darksalmon',data_terrestrial[:,0]=='INSPECT-II')
campaigns_dict['AMAZE-08']=campaing('AMAZE-08','check','t','green',data_terrestrial[:,0]=='AMAZE-08')
campaigns_dict['WISP94']=campaing('WISP94','check','t','red',data_terrestrial[:,0]=='WISP94')
campaigns_dict['ICE-L Ambient']=campaing('ICE-L Ambient','check','t','orange',data_terrestrial[:,0]=='ICE-L Ambient')
campaigns_dict['ICE-L CVI']=campaing('ICE-L CVI','check','t','peru',data_terrestrial[:,0]=='ICE-L CVI')
campaigns_dict['Bigg73']=campaing('Bigg73','check','t','brown',data_terrestrial[:,0]=='Bigg73')
campaigns_dict['CLEX']=campaing('CLEX','check','t','violet',data_terrestrial[:,0]=='CLEX')
campaigns_dict['Yin']=campaing('Yin','check','t','navy',data_terrestrial[:,0]=='Yin')
campaigns_dict['R. H. Mason']=campaing('Mason2016','Various terestrial locations','t','wheat',data_terrestrial[:,0]=='R. H. Mason')
campaigns_dict['Conen_JFJ']=campaing('Conen_JFJ','Joungfraiof (correct name)','t','grey',data_terrestrial[:,0]=='Conen_JFJ')
campaigns_dict['Conen_chaumont']=campaing('Conen_chaumont','chaumont','t','darkgrey',data_terrestrial[:,0]=='Conen_chaumont')
campaigns_dict['KAD_South_Pole']=campaing('KAD_South_Pole','South Pole','t','lime',data_terrestrial[:,0]=='KAD_South_Pole')
campaigns_dict['KAD_Israel']=campaing('KAD_Israel','Jerusalem','T','lightblue',data_terrestrial[:,0]=='KAD_Israel')

fig=plt.figure()
m = fig.add_subplot(1,1,1)
m = Basemap(projection='cyl',lon_0=0)
m.drawcoastlines()
m.drawcountries()
#m.bluemarble()
#m.drawmapboundary(fill_color='#99ffff')
#m.fillcontinents(color='#cc9966',lake_color='#99ffff')

#m = Basemap(resolution='c',projection='ortho',lat_0=60.,lon_0=-60.)
for camp_key in campaigns_dict.iterkeys():
    if campaigns_dict[camp_key].m_or_t =='m':
        data=data_marine
    else:
        data=data_terrestrial
    xx=[float(lon) for lon in data[campaigns_dict[camp_key].values,4]]
    yy=[float(lat) for lat in data[campaigns_dict[camp_key].values,3]]
    print xx
    m.scatter(xx,yy,c=campaigns_dict[camp_key].color,edgecolors='none')#,label=camp.name)


#%%
'''


PRUEBAS Y DEMAS

'''


#%%
jl.plot(surface_area_conc_feld[2,20,:,:,:].mean(axis=-1)/surface_area_conc_feld[3,20,:,:,:].mean(axis=-1),clevs=[0.1,1,10,100])

INP=Nd*4*np.pi*rmean**2*jl.feld_parametrization(265)/10000.

jl.plot(INP[2,20,:,:,:].mean(axis=-1)/INP[3,20,:,:,:].mean(axis=-1))
jl.plot(INP[2,20,:,:,:].mean(axis=-1)/INP[3,20,:,:,:].mean(axis=-1))
jl.plot(INP[3,20,:,:,:].mean(axis=-1))#/INP[3,20,:,:,:].mean(axis=-1))
#%%


def marine_org_parameterization(T):
    a=11.2186
    b=-0.4459
    INP=np.exp(a+b*T)#[per gram]
    return INP



archive_directory='/nfs/a201/eejvt/'
project='MARINE_PARAMETERIZATION/FOURTH_TRY'
project='MARINE_PARAMETERIZATION/DAILY'
os.chdir(archive_directory+project)

names=[#'tot_mc_oc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'tot_mc_bc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'tot_mc_su_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
'tot_mc_ss_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'tot_mc_feldspar_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'GLOMAP_mode_mol_no_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'mean_dry_r_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
#'sigmachained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
'hindcast3_temp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
'GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
]
names=[
'tot_mc_ss_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
'hindcast3_temp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
'GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
]
s={}
a=glob('*.sav')
for name in names:
    print name
    s=readsav(name,idict=s)
total_marine_mass=s.tot_mc_ss_mm_mode[2,]#+s.tot_mc_ss_mm_mode[3,]#ug/m3
total_marine_mass_year_mean=total_marine_mass.mean(axis=-1)
total_marine_mass_monthly_mean=jl.from_daily_to_monthly(total_marine_mass)
total_marine_mass_grams_OC=total_marine_mass*1e-6/1.9
total_marine_mass_grams_OC=total_marine_mass_monthly_mean*1e-6/1.9
total_marine_mass_grams_OC_daily=total_marine_mass*1e-6/1.9

INP_marine_alltemps=np.zeros((38,31,64,128,12))
INP_marine_alltemps_daily=np.zeros((38,31,64,128,365))
for i in range (38):
    INP_marine_alltemps[i,]=total_marine_mass_grams_OC*marine_org_parameterization(-i)
    INP_marine_alltemps_daily[i,]=total_marine_mass_grams_OC_daily*marine_org_parameterization(-i)
np.save('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy',INP_marine_alltemps)
np.save('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/DAILY/INP_marine_alltemps_daily.npy',INP_marine_alltemps_daily)
#INP_marine_alltemps_prueba=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')
#np.any([INP_marine_alltemps_prueba<0])



#%%



INP_feldext[15,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],[6,7,8]]*1e3
INP_marine_alltemps[15,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],[6,7,8]]*1e3
INP_marine_alltemps[15,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],[6,7,8]]*1e3
#%%
INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')*1e-3#l
INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e3 #l
#%%
for tem in [15,20,22,25,30]:

    print 'Temperature'
    print '-',tem,'\n'
    for imon in [6,7,8]:
        print jl.month_names[imon],'m3 \n'
        print 'feld %1.2e'%(INP_feldext[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],imon]*1e3)
        print 'marine %1.2e'%(INP_marine_alltemps[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],imon]*1e3)
        print 'total %1.2e \n'%(INP_marine_alltemps[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],imon]*1e3+(INP_feldext[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],imon]*1e3))
        print '% marine',(INP_marine_alltemps[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],imon]*1e3/(INP_feldext[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],imon]*1e3+INP_marine_alltemps[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],imon]*1e3))*100
        print '\n'
    print 'Campaing mean \n'
    print 'feld %1.2e'%(INP_feldext[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],[6,7,8]].mean(axis=-1)*1e3)
    print 'marine %1.2e'%(INP_marine_alltemps[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],[6,7,8]].mean(axis=-1)*1e3)
    print 'total %1.2e \n'%(INP_marine_alltemps[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],[6,7,8]].mean(axis=-1)*1e3+(INP_feldext[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],[6,7,8]].mean(axis=-1)*1e3))
    print '% marine',(INP_marine_alltemps[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],[6,7,8]].mean(axis=-1)*1e3/(INP_feldext[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],[6,7,8]].mean(axis=-1)*1e3+INP_marine_alltemps[tem,30,jl.mace_head_latlon_index[0],jl.mace_head_latlon_index[1],[6,7,8]].mean(axis=-1)*1e3))*100
    print '\n'

#%%
a=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/Conen_chaumont.dat',delimiter="\t",skip_header=1)
a=np.genfromtxt('/nfs/a107/eejvt/INP_DATA/Conen_JFJ.dat',delimiter="\t",skip_header=1)

temps=np.arange(a[:,1].min(),a[:,1].max()+1,1)
for t in temps:
    print t
    print '%1.2e'%np.mean(a[a[:,1]==t][:,2])






#%%
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

map = Basemap(llcrnrlon=-80.5,llcrnrlat=-88,urcrnrlon=40.,urcrnrlat=88,
             resolution='i', projection='cyl', lat_0 = 60, lon_0 = -3.25)

#map.drawmapboundary(fill_color='aqua')
#map.fillcontinents(color='coral',lake_color='aqua')
map.drawcoastlines()

plt.show()
#%%
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

map = Basemap(projection='aeqd',
              lon_0 = 0,
              lat_0 = 90,
              width = 10000000,
              height = 10000000)

map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='coral',lake_color='aqua')
map.drawcoastlines()

for i in range(0, 10000000, 1000000):
    map.plot(i, i, marker='o',color='k')

plt.show()
#%%
n=0
files=0
log_number=0
a=glob('/nfs/a86/shared/Mace Head 15/*')
for folder in a:
    b=glob(folder+'/*')
    print b
    for file_name in b:

        if 'Impinger' in file_name or 'impinger' in file_name:
           n=n+1
        #c=
        #if
        files=files+1
print n
print files
#%%
s=jl.read_data('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV')
#%%
cape_verde_indx=[26,8]
barbados_indx=[27,107]

radii_cp=s.rbarwet[:,30,cape_verde_indx[0],cape_verde_indx[1],:].mean(axis=-1)
radii_bb=s.rbarwet[:,30,barbados_indx[0],barbados_indx[1],:].mean(axis=-1)


n_cp=s.st_nd[:,30,cape_verde_indx[0],cape_verde_indx[1],:].mean(axis=-1)
n_bb=s.st_nd[:,30,barbados_indx[0],barbados_indx[1],:].mean(axis=-1)
sigma=[1.59,1.59,1.4,2.0,1.59,1.4,2.0]
rs=jl.logaritmic_steps(-8,-4,10000)
size_dist_cp=np.zeros(len(rs.mid_points))
size_dist_bb=np.zeros(len(rs.mid_points))
Ntot_bb=np.zeros(len(rs.mid_points))
Ntot_cp=np.zeros(len(rs.mid_points))
fig=plt.figure()

for i in range (7):
    Ntot_bb=Ntot_bb+n_bb[i]
    Ntot_cp=Ntot_cp+n_cp[i]
    size_dist_cp=size_dist_cp+n_cp[i]*jl.lognormal_PDF(radii_cp[i],rs.mid_points,sigma[i])
    size_dist_bb=size_dist_bb+n_bb[i]*jl.lognormal_PDF(radii_bb[i],rs.mid_points,sigma[i])
    #plt.plot()
#ax=plt.subplot(211)
#plt.title('Cape Verde')
plt.plot(rs.mid_points,size_dist_cp/Ntot_cp,label='Cape Verde')
plt.xscale('log')
plt.yscale('log')
#plt.title('Barbados')
plt.plot(rs.mid_points,size_dist_bb/Ntot_bb,label='Barbados')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r (m)')
plt.ylabel('$(dN/dr)/N_{tot}$')
plt.legend()


#%%
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
# setup of basemap ('lcc' = lambert conformal conic).
# use major and minor sphere radii from WGS84 ellipsoid.
m = Basemap(llcrnrlon=-120,llcrnrlat=1.,urcrnrlon=-50,urcrnrlat=46.352,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',area_thresh=1000.,projection='lcc',\
            lat_1=50.,lon_0=-107.,ax=ax)
# transform to nx x ny regularly spaced 5km native projection grid
nx = int((m.xmax-m.xmin)/5000.)+1; ny = int((m.ymax-m.ymin)/5000.)+1
#topodat = m.transform_scalar(topoin,lons,lats,nx,ny)
# plot image over map with imshow.
#im = m.imshow(topodat,cm.GMT_haxby)
# draw coastlines and political boundaries.
m.drawcoastlines()

#%%
fig=plt.figure(figsize=(20, 12))
m = fig.add_subplot(1,1,1)
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='merc',resolution='i',llcrnrlat=64,urcrnrlat=80,\
        llcrnrlon=-120,urcrnrlon=-50)#,lat_0=50.,lon_0=-107.)
m.drawcountries()
#m.drawstates()
m.drawcoastlines()
data=my_param.array_surface[:,:,7]
circles = np.arange(65,80,5).tolist()
m.drawparallels(circles,labels=[1,1,0,0])
lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
X,Y=np.meshgrid(lon.glon,lat.glat)
clevs=np.linspace(0,0.125,9).tolist()
cmap=plt.cm.Greens
plt.title('Surface Marine organic concentration',y=1.04)
cs=m.contourf(X,Y,data,clevs,latlon=True,cmap=cmap)#,norm= colors.BoundaryNorm(clevs, 256))
cb = m.colorbar(cs,format='%.2e',ticks=clevs)
cb.set_label('$\mu g/m^{-3}$',fontsize=15)
# draw meridians
meridians = np.arange(-120,-50,10)
m.drawmeridians(meridians,labels=[0,0,1,1])

#%%
from __future__ import print_function
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pickle, time

# create figure with aqua background (will be oceans)
fig = plt.figure()

# create Basemap instance. Use 'high' resolution coastlines.
t1 = time.clock()
#m = Basemap(llcrnrlon=-10.5,llcrnrlat=49.5,urcrnrlon=3.5,urcrnrlat=59.5,
#            resolution='h',projection='tmerc',lon_0=-4,lat_0=0)
m = Basemap(width=920000,height=1100000,
            resolution='i',projection='tmerc',lon_0=-85,lat_0=75)
# make sure countries and rivers are loaded
m.drawcountries()
m.drawrivers()
print(time.clock()-t1,' secs to create original Basemap instance')

# pickle the class instance.
pickle.dump(m,open('map.pickle','wb'),-1)

# clear the figure
plt.clf()
# read pickle back in and plot it again (should be much faster).
t1 = time.clock()
m2 = pickle.load(open('map.pickle','rb'))
# draw coastlines and fill continents.
m.drawcoastlines()
# fill continents and lakes
#m.fillcontinents(color='coral',lake_color='aqua')
# draw political boundaries.
m.drawcountries(linewidth=1)
# fill map projection region light blue (this will
# paint ocean areas same color as lakes).
#m.drawmapboundary(fill_color='aqua')
# draw major rivers.
m.drawrivers(color='b')
print(time.clock()-t1,' secs to plot using using a pickled Basemap instance')
# draw parallels
circles = np.arange(48,65,2).tolist()
m.drawparallels(circles,labels=[1,1,0,0])
# draw meridians
meridians = np.arange(-120,-50,10)
m.drawmeridians(meridians,labels=[0,0,1,1])
plt.title("High-Res British Isles",y=1.04)
plt.show()
