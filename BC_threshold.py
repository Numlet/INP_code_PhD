# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:43:37 2016

@author: eejvt
"""

import os
import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import Jesuslib as jl
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
from glob import glob
archive_directory='/nfs/a201/eejvt/'
project='BC_INP'
os.chdir(archive_directory+project)
from multiprocessing import Pool
from scipy.integrate import quad
#pool = Pool()
rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3

def murray(T):
    #T in C
    ns=np.exp(-0.0101*(T)**2-0.8525*(T)+0.7667)#cm-2
    return ns

def ulrich(T):
    #T in C
    ns=7.463*np.exp(-0.0101*(T)**2-0.8525*(T)+0.7667)#m-2
    ns=ns*1e-4#cm-2
    return ns

def BC_parametrization_tom(T):
    #A=-20.27
    #B=1.2
#    return 10**(-2.87-0.182*T)
    return np.exp((-6.608-0.419*T))
    
def volumes_of_modes(s):
    modes_vol=(s.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]
    +s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2]
    +s.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
    +s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]
    +s.tot_mc_oc_mm_mode[:,:,:,:,:]/rhocomp[5]
    +s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])
    return modes_vol
def feld_parametrization(T):
    ns=np.exp((-1.03802178356815*T)+275.263379304105)
    return ns


def area_lognormal_per_particle(rbar,sigma):
    #print isinstance(sigma,float)
    if isinstance(sigma,np.float32):
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        S=(2*rbar)**2*y
    else:
        S=np.zeros(rbar.shape)
        #y=np.zeros(len(rbar))
        #global S,y
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        for i in range (len(rbar)):
            S[i,]=(2*rbar[i,])**2*y[i]
    return S



    
#%%
path='/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2'


s=jl.read_data(path)
#%%

temperatures=np.load('/nfs/a107/eejvt/temperatures_daily.npy')
temperatures=temperatures+273.15
temperatures[temperatures<236]=100000
temperatures[temperatures<248]=248
temperatures[temperatures>268]=100000
temperatures=temperatures-273.15
temperatures_monthly=jl.from_daily_to_monthly(temperatures)

#%%
#INP_BC_ext_min=np.zeros((38,31,64,128,12))
#modes_vol=volumes_of_modes(s)
#BC_volfrac=(s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
#Nd=s.st_nd[:,:,:,:,:]*BC_volfrac
#rmean=s.rbardry[:,:,:,:,:]*1e2
#for itemp in range (0,38):
#    print itemp
#    ns=BC_parametrization_tom(-itemp)
#    for imode in range(7):
#        INP_BC_ext_min[itemp,:,:,:,:]=INP_BC_ext_min[itemp,:,:,:,:]+Nd[imode,:,:,:,:]*(1-np.exp(-ns*4*np.pi*rmean[imode,:,:,:,:]**2))

INP_BC_ext_min=np.zeros((38,31,64,128,12))
modes_vol=volumes_of_modes(s)
BC_volfrac=(s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
Nd=s.st_nd[:,:,:,:,:]*BC_volfrac
rmean=s.rbardry[:,:,:,:,:]*1e2
std=s.sigma[:]
for itemp in range (0,38):
    print itemp
    ns=BC_parametrization_tom(-itemp)
    for imode in range(7):
        area_particle=area_lognormal_per_particle(rmean[imode,:,:,:,:],std[imode])
        exponent=ns*area_particle
        ff=np.zeros_like(exponent)
        ff[exponent>=1e-5]=1-np.exp(-exponent[exponent>=1e-5])
        ff[exponent<1e-5]=exponent[exponent<1e-5]
        
        ff_fitted=jl.correct_ff(ff,std[imode])
#        INP=Nd*ff_fitted

        INP_BC_ext_min[itemp,:,:,:,:]=INP_BC_ext_min[itemp,:,:,:,:]+Nd[imode,:,:,:,:]*ff_fitted



        
INP_BC_ext_ambient_min=np.zeros((31,64,128,12))
modes_vol=volumes_of_modes(s)
BC_volfrac=(s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
Nd=s.st_nd[:,:,:,:,:]*BC_volfrac
rmean=s.rbardry[:,:,:,:,:]*1e2
ns=BC_parametrization_tom(temperatures_monthly)
for imode in range(7):
    print imode
    INP_BC_ext_ambient_min[:,:,:,:]=INP_BC_ext_ambient_min[:,:,:,:]+Nd[imode,:,:,:,:]*(1-np.exp(-ns*4*np.pi*rmean[imode,:,:,:,:]**2))


INP_BC_ext_murray=np.zeros((38,31,64,128,12))
modes_vol=volumes_of_modes(s)
BC_volfrac=(s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
Nd=s.st_nd[:,:,:,:,:]*BC_volfrac
rmean=s.rbardry[:,:,:,:,:]*1e2
std=s.sigma[:]
for itemp in range (0,38):
    print itemp
    ns=murray(-itemp)
    for imode in range(7):
        area_particle=area_lognormal_per_particle(rmean[imode,:,:,:,:],std[imode])
        exponent=ns*area_particle
        ff=np.zeros_like(exponent)
        ff[exponent>=1e-5]=1-np.exp(-exponent[exponent>=1e-5])
        ff[exponent<1e-5]=exponent[exponent<1e-5]
        
        ff_fitted=jl.correct_ff(ff,std[imode])
#        INP=Nd*ff_fitted

        INP_BC_ext_murray[itemp,:,:,:,:]=INP_BC_ext_murray[itemp,:,:,:,:]+Nd[imode,:,:,:,:]*ff_fitted

#INP_BC_ext_murray_old=np.copy(INP_BC_ext_murray)


INP_BC_ext_ambient_murray=np.zeros((31,64,128,12))
modes_vol=volumes_of_modes(s)
BC_volfrac=(s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
Nd=s.st_nd[:,:,:,:,:]*BC_volfrac
rmean=s.rbardry[:,:,:,:,:]*1e2
ns=murray(temperatures_monthly)
for imode in range(7):
    print imode
    INP_BC_ext_ambient_murray[:,:,:,:]=INP_BC_ext_ambient_murray[:,:,:,:]+Nd[imode,:,:,:,:]*(1-np.exp(-ns*4*np.pi*rmean[imode,:,:,:,:]**2))


INP_feld_ext=np.zeros((38,31,64,128,12))
modes_vol=volumes_of_modes(s)
feld_volfrac=(s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
Nd=s.st_nd[:,:,:,:,:]*feld_volfrac
rmean=s.rbardry[:,:,:,:,:]*1e2
for itemp in range (0,38):
    print itemp
    ns=feld_parametrization(-itemp+273.15)
    for imode in range(7):
        INP_feld_ext[itemp,:,:,:,:]=INP_feld_ext[itemp,:,:,:,:]+Nd[imode,:,:,:,:]*(1-np.exp(-ns*4*np.pi*rmean[imode,:,:,:,:]**2))
        

INP_feld_ext_ambient=np.zeros((31,64,128,12))
modes_vol=volumes_of_modes(s)
feld_volfrac=(s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])/modes_vol
Nd=s.st_nd[:,:,:,:,:]*feld_volfrac
rmean=s.rbardry[:,:,:,:,:]*1e2
ns=feld_parametrization(temperatures_monthly+273.15)
for imode in range(7):
    print imode
    INP_feld_ext_ambient[:,:,:,:]=INP_feld_ext_ambient[:,:,:,:]+Nd[imode,:,:,:,:]*(1-np.exp(-ns*4*np.pi*rmean[imode,:,:,:,:]**2))

#%%
INP_feld_ext_m3=INP_feld_ext*1e6
INP_feld_ext_ambient_m3=INP_feld_ext_ambient*1e6
INP_BC_ext_ambient_min_m3=INP_BC_ext_ambient_min*1e6
INP_BC_ext_min_m3=INP_BC_ext_min*1e6
INP_BC_ext_ambient_murray_m3=INP_BC_ext_ambient_murray*1e6
INP_BC_ext_murray_m3=INP_BC_ext_murray*1e6
#%%
#INP_feld_ext

INP_feldspar_alltemps_m3=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e6#m3
INP_marine_alltemps_m3=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')#m3
INP_total=INP_marine_alltemps_m3+INP_feldspar_alltemps_m3

#%%

def threshold(T,factor=0.01):
    #T in C
    ns=factor*np.exp(-0.0101*(T)**2-0.8525*(T)+0.7667)#cm-2
    return ns
def threshold(T,factor):
    ns=factor*np.exp((-1.03802178356815*(T+273.15)+275.263379304105))
    return ns

std=s.sigma
def calculate_BC(factor,T):
    INP_BC_ext_threshold=np.zeros((38,31,64,128,12))
    modes_vol=volumes_of_modes(s)
    BC_volfrac=(s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
    Nd=s.st_nd[:,:,:,:,:]*BC_volfrac
    rmean=s.rbardry[:,:,:,:,:]*1e2
#    for itemp in range (0,38):
    for itemp in [T]:
        print itemp
        ns=threshold(-itemp,factor=factor)
        for imode in range(7):
#            INP_BC_ext_threshold[itemp,:,:,:,:]=INP_BC_ext_threshold[itemp,:,:,:,:]+Nd[imode,:,:,:,:]*(1-np.exp(-ns*4*np.pi*rmean[imode,:,:,:,:]**2))
            area_particle=area_lognormal_per_particle(rmean[imode,:,:,:,:],std[imode])
            exponent=ns*area_particle
            ff=np.zeros_like(exponent)
            ff[exponent>=1e-5]=1-np.exp(-exponent[exponent>=1e-5])
            ff[exponent<1e-5]=exponent[exponent<1e-5]
            
            ff_fitted=jl.correct_ff(ff,std[imode])
    #        INP=Nd*ff_fitted
    
            INP_BC_ext_threshold[itemp,:,:,:,:]=INP_BC_ext_threshold[itemp,:,:,:,:]+Nd[imode,:,:,:,:]*ff_fitted
     
    INP_BC_ext_ambient_threshold=np.zeros((31,64,128,12))
#    modes_vol=volumes_of_modes(s)
#    BC_volfrac=(s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
#    Nd=s.st_nd[:,:,:,:,:]*BC_volfrac
#    rmean=s.rbardry[:,:,:,:,:]*1e2
#    ns=threshold(temperatures_monthly,factor=factor)
#    for imode in range(7):
#        print imode
#        INP_BC_ext_ambient_threshold[:,:,:,:]=INP_BC_ext_ambient_threshold[:,:,:,:]+Nd[imode,:,:,:,:]*(1-np.exp(-ns*4*np.pi*rmean[imode,:,:,:,:]**2))
    return INP_BC_ext_threshold,INP_BC_ext_ambient_threshold
#%%
INP_total_cm=INP_total*1e-6


def percentage_gridboxes(T,ns):
    INP_BC_ext_threshold=np.zeros((64,128,12))
    modes_vol=volumes_of_modes(s)
    BC_volfrac=(s.tot_mc_bc_mm_mode[:,30,:,:,:]/rhocomp[2])/modes_vol[:,30,:,:,:]
    Nd=s.st_nd[:,30,:,:,:]*BC_volfrac
    rmean=s.rbardry[:,30,:,:,:]*1e2
#    for itemp in range (0,38):
    for imode in range(7):
        area_particle=area_lognormal_per_particle(rmean[imode,:,:,:],std[imode])
        exponent=ns*area_particle
        ff=np.zeros_like(exponent)
        ff[exponent>=1e-5]=1-np.exp(-exponent[exponent>=1e-5])
        ff[exponent<1e-5]=exponent[exponent<1e-5]
        
        ff_fitted=jl.correct_ff(ff,std[imode])

        INP_BC_ext_threshold[:,:,:]=INP_BC_ext_threshold[:,:,:]+Nd[imode,:,:,:]*ff_fitted
#    print INP_total_cm.shape,INP_BC_ext_threshold.shape
    ratio=INP_BC_ext_threshold.mean(axis=-1)/INP_total_cm[-T,30,].mean(axis=-1)
    print ratio.shape
    fraction=np.array([ratio>1]).sum()/float(ratio.size)
    return fraction

a=percentage_gridboxes(-30,1e4)
print a

#%%
temps=np.linspace(-37,0,38)
ns_values=np.logspace(-5,10,120)
np.save(jl.bc_folder+'temps',temps)
np.save(jl.bc_folder+'ns_values',ns_values)

grid_of_fractions=np.zeros((len(temps),len(ns_values)))
for itemp in range(len(temps)):
    print itemp
    for ins in range(len(ns_values)):
        grid_of_fractions[itemp,ins]=percentage_gridboxes(temps[itemp],ns_values[ins])
        
np.save(jl.bc_folder+'grid_of_fractions',grid_of_fractions)

jl.send_email()

#%%


plt.figure()
temps=np.load(jl.bc_folder+'temps.npy')
grid_of_fractions=np.load(jl.bc_folder+'grid_of_fractions.npy')
#plt.yscale('log')
ns_values=np.load(jl.bc_folder+'ns_values.npy')
plt.contourf(temps,np.log10(ns_values),grid_of_fractions.T,cmap=plt.cm.RdBu_r)
plt.colorbar()


#%%

#
#factor_values=np.logspace(-4,0,10)
#percentage_gridboxes=[]
#ns=np.linspace
#for factor in factor_values:
#    INP_BC_ext_threshold,INP_BC_ext_ambient_threshold=calculate_BC(factor)
#    INP_BC_ext_ambient_threshold_m3=INP_BC_ext_ambient_threshold*1e6
#    INP_BC_ext_threshold_m3=INP_BC_ext_threshold*1e6
#
#    ratio=INP_BC_ext_threshold_m3[20,30,:,:,:].mean(axis=-1)/INP_total[20,30,:,:,:].mean(axis=-1)
#    fraction_grid=np.array([ratio>1]).sum()/float(ratio.size)
#    percentage_gridboxes.append(fraction_grid)
##%%
#plt.plot(factor_values,percentage_gridboxes)
#plt.xscale('log')
##%%
#plt.plot(ns)
#plt.yscale('log')

#%%
#plt.plot(ns)
#plt.close()
plt.close()
plt.close()
plt.figure()
yin=jl.read_INP_data('/nfs/a107/eejvt/INP_DATA/data_by_campaing/Yin.dat',header=0)
bigg=jl.read_INP_data('/nfs/a107/eejvt/INP_DATA/bigg73.dat',header=0)
print yin[:,1]
#jl.fitandplot_comparison(INP_BC_ext_min_m3.mean(axis=-1)*1e-6,yin)
#jl.fitandplot_comparison(INP_BC_ext_min_m3.mean(axis=-1)*1e-6,bigg)
#jl.fitandplot_comparison(INP_BC_ext_murray_m3.mean(axis=-1)*1e-6,bigg)
#jl.fitandplot_comparison(INP_BC_ext_threshold_m3.mean(axis=-1)*1e-6,yin)
#simulated_points_threshold=jl.obtain_points_from_data(INP_BC_ext_threshold_m3.mean(axis=-1)*1e-6,yin)

simulated_values_max=INP_BC_ext_min_m3.max(axis=-1)*1e-6
simulated_values_min=INP_BC_ext_min_m3.min(axis=-1)*1e-6

simulated_points_min=jl.obtain_points_from_data(INP_BC_ext_min_m3.mean(axis=-1)*1e-6,yin)
simulated_points_min_b=jl.obtain_points_from_data(INP_BC_ext_min_m3.mean(axis=-1)*1e-6,bigg)


simulated_points_murray_b=jl.obtain_points_from_data(INP_BC_ext_murray_m3.mean(axis=-1)*1e-6,bigg)


simulated_points_murray=jl.obtain_points_from_data(INP_BC_ext_murray_m3.mean(axis=-1)*1e-6,yin)

#simulated_points_threshold_b=jl.obtain_points_from_data(INP_BC_ext_threshold_m3.mean(axis=-1)*1e-6,bigg)
marker_size=20
#plt.scatter(yin[:,2],simulated_points_threshold[:,0],c='blue',marker='o',s=marker_size)
plt.scatter(yin[:,2],simulated_points_min[:,0],c='b',marker='o',s=marker_size,label='This work')
plt.scatter(yin[:,2],simulated_points_murray[:,0],c='r',marker='o',s=marker_size,label='Murray2012')

#
#if errors:
#    plt.errorbar(data_points_mason[:,2],simulated_points_mason[:,0],
#                 yerr=[simulated_points_mason[:,0]-simulated_points_mason_min[:,0],simulated_points_mason_max[:,0]-simulated_points_mason[:,0]],
#                 linestyle="None",c='k',zorder=0)
#    plt.errorbar(data_points[:,2],simulated_points[:,0],
#                 yerr=[simulated_points[:,0]-simulated_points_min[:,0],simulated_points_max[:,0]-simulated_points[:,0]],
#                linestyle="None",c='k',zorder=0)
#

#plt.scatter(bigg[:,2],simulated_points_threshold_b[:,0],c='blue',marker='^',s=marker_size)
plt.scatter(bigg[:,2],simulated_points_min_b[:,0],c='b',marker='^',s=marker_size)
plt.scatter(bigg[:,2],simulated_points_murray_b[:,0],c='r',marker='^',s=marker_size)
plt.scatter([],[],marker='^', label='bigg73')
plt.scatter([],[],marker='o', label='Yin')
plt.yscale('log')
plt.xscale('log')
min_val=1e-10
max_val=1e3

minx=np.min(min_val)
maxx=np.max(max_val)
miny=np.min(min_val)
maxy=np.max(max_val)
min_plot=min_val
max_plot=max_val

x=np.linspace(0.1*min_plot,10*max_plot,100)
plt.plot(x,x,'k-')
plt.plot(x,10*x,'k--')
plt.plot(x,10**1.5*x,'k-.')
plt.plot(x,0.1*x,'k--')
plt.plot(x,10**(-1.5)*x,'k-.')
plt.ylim(miny*0.1,maxy*10)
plt.xlim(minx*0.1,maxx*10)
plt.legend(loc='best')
plt.xlabel('Observed INP cm-3')
plt.ylabel('Simulated INP cm-3')
plt.savefig(jl.bc_folder+'one_to_one.png')
plt.show()
levels=(1e-9*np.logspace(-3,8,12)).tolist()

#jl.plot(INP_BC_ext_min_m3.mean(axis=-1)[20,30,:,:]*1e-6,clevs=levels)
#%%INP_BC_ext_min_m3.mean(axis=-1)*1e-6
ratio=INP_BC_ext_min_m3.mean(axis=-1)/INP_total.mean(axis=-1)
jl.plot(ratio[20,30,:,:],clevs=[0,0.0001,0.001,0.1,0.5,0.9,1,1.1,1.5,1.9,10,100,1000,10000])


print 'percentage of gridboxes:',np.sum(ratio[20,30,:,:]>1)/np.float(ratio[20,30,:,:].size)



#%%
levels=(1e-3*np.logspace(-3,8,12)).tolist()
contour_levels=np.logspace(-1,2,4).tolist()
contour_levels=np.logspace(0,3,4).tolist()
print levels
print contour_levels
cmap=plt.cm.CMRmap_r
#%%

jl.plot(s.tot_mc_feldspar_mm_mode[:,:,:,:,:].sum(axis=0).mean(axis=-1)[30,:,:],
        clevs=np.logspace(-3,3,13).tolist(),title='Feldspar mass surface concentrations',cmap=plt.cm.YlOrBr)


#%%
#jl.plot(INP_BC_ext_m3[30,15,:,:,:].mean(axis=-1)*1e-6,title='INP BC 600hpa T=-20C', cblabel='$m^{-3}$',cmap=plt.cm.OrRd)
fn=jl.bc_folder+'Surface_20_min'
jl.plot2(INP_BC_ext_min_m3[20,30,:,:,:].mean(axis=-1)*1e-3,clevs=levels,
        title='INP BC surface min T=-20C', cblabel='$L-1$',cmap=cmap,colorbar_format_sci=1,file_name=fn,saving_format='png',
        contour=INP_total[20,30,:,:,:].mean(axis=-1)*1e-3,contourlevs=contour_levels,line_color='k')
#%%
fn=jl.bc_folder+'Surface_20_murray'
jl.plot2(INP_BC_ext_murray_m3[20,30,:,:,:].mean(axis=-1)*1e-3,clevs=levels,
        title='INP BC surface high T=-20C', cblabel='$L-1$',cmap=cmap,colorbar_format_sci=1,file_name=fn,saving_format='png',
        contour=INP_feld_ext_m3[20,30,:,:,:].mean(axis=-1)*1e-3,contourlevs=contour_levels,line_color='w')
#%%
fn=jl.bc_folder+'Surface_20_threshold'
jl.plot2(INP_BC_ext_threshold_m3[20,30,:,:,:].mean(axis=-1)*1e-3,clevs=levels,
        title='INP BC surface threshold T=-20C', cblabel='$L-1$',cmap=cmap,colorbar_format_sci=1,file_name=fn,saving_format='png',
        contour=INP_feld_ext_m3[20,30,:,:,:].mean(axis=-1)*1e-3,contourlevs=contour_levels,line_color='w')
#
#jl.plot2(ratio[i,:,:],contour=INP_ambient_total[i,:,:,:].mean(axis=-1),
#         contourlevs=[1,10,100,1000],title='%i hpa'%((i+1)*50),cmap=plt.cm.OrRd,
#%%
fn=jl.bc_folder+'Surface_20_feldspar'
jl.plot(INP_feld_ext_m3[20,30,:,:,:].mean(axis=-1)*1e-3,clevs=levels,title='INP feld surface T=-20C',
        colorbar_format_sci=1,file_name=fn,saving_format='png',
        cblabel='$L-1$',cmap=cmap)
#                     file_name='days_BC-Feld_%i'%((i+1)*50),saving_format='svg')
#%%

jl.plot(INP_BC_ext_murray_m3[20,30,:,:,:].mean(axis=-1),clevs=levels,
        title='INP BC surface T=-20C', cblabel='$m^{-3}$',cmap=plt.cm.OrRd)

jl.plot(INP_BC_ext_threshold_m3[20,30,:,:,:].mean(axis=-1),clevs=levels,
        title='INP BC surface T=-20C', cblabel='$m^{-3}$',cmap=plt.cm.OrRd)


levels_ratio=np.logspace(-5,5,11).tolist()
print levels_ratio
jl.plot(INP_BC_ext_min_m3[20,30,:,:,:].mean(axis=-1)/INP_feld_ext_m3[20,30,:,:,:].mean(axis=-1),clevs=levels_ratio,title='Ratio BC/feld surface T=-20C', cblabel='$m^{-3}$',cmap=plt.cm.RdBu_r)
jl.plot(INP_BC_ext_murray_m3[20,30,:,:,:].mean(axis=-1)/INP_feld_ext_m3[20,30,:,:,:].mean(axis=-1),clevs=levels_ratio,title='Ratio BC/feld surface T=-20C', cblabel='$m^{-3}$',cmap=plt.cm.RdBu_r)
jl.plot(INP_BC_ext_threshold_m3[20,30,:,:,:].mean(axis=-1)/INP_feld_ext_m3[20,30,:,:,:].mean(axis=-1),clevs=levels_ratio,title='Ratio BC/feld surface T=-20C', cblabel='$m^{-3}$',cmap=plt.cm.RdBu_r)

jl.plot(INP_BC_ext_ambient_m3[15,:,:,:].mean(axis=-1))
jl.plot(INP_feld_ext_ambient_m3[15,:,:,:].mean(axis=-1))

#%%
for i in range(100):
    plt.close()
#%%
#jl.grid_earth_map(INP_BC_ext_ambient[15,:,:,:])
#jl.grid_earth_map(INP_feld_ext_ambient[15,:,:,:])
##%%
#fig=plt.figure()
#cx=plt.subplot(1,1,1)
#CF=cx.contourf(Xmo,Ymo,INP_BC_ext_ambient_m3[:,:,:,:].mean(axis=(-1,-2)),cmap=plt.cm.YlOrRd)
#CB=plt.colorbar(CF,drawedges=1,label='$m^{-3}$')
#
#cx.invert_yaxis()
#cx.set_ylim(ymax=200)
#plt.show()
#%%



#[0.1,0.5,1,10,50,100,500,1000,5000,10000]
levelsbc=np.logspace(-8,8,15).tolist()
levelsbc=[0.1,0.5,1,10,50,100,500,1000,5000,10000]
levelsfel=[100,1000]
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
fs=10
glolevs=jl.pressure
fig=plt.figure()
cx=plt.subplot(1,1,1)
Xmo,Ymo= np.meshgrid(lat.glat, glolevs)
Xfel,Yfel=np.meshgrid(lat.glat,glolevs)
cx.set_title('Anual and longitudinal mean')
CS=cx.contour(Xfel,Yfel,INP_feld_ext_ambient_m3[:,:,:,:].mean(axis=(-1,-2)),levelsfel,colors='k',hold='on')#,linewidths=np.linspace(2, 6, 4))#,

plt.clabel(CS, inline=1, fontsize=fs,fmt='%1.0f')
plt.setp(CS.collections )
CF=cx.contourf(Xmo,Ymo,INP_BC_ext_ambient_murray_m3[:,:,:,:].mean(axis=(-1,-2)),levelsbc,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsbc, 256))
CB=plt.colorbar(CF,ticks=levelsbc,drawedges=1,label='$m^{-3}$')

cx.invert_yaxis()
cx.set_ylim(ymax=200)
cx.tick_params(axis='both', which='major', labelsize=fs)
cx.set_ylabel('Pressure level $(hPa)$')
cx.set_xlabel('Latitude')
cx.xaxis.set_ticks(np.arange(-90,100,20))
#plt.savefig('jja_mean_lat_mean_NA.svg',dpi=300,format='svg')
#plt.savefig('jja_mean_lat_mean_NA.png',dpi=600,format='png')

plt.show()


#%%
def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
plt.figure()
feldspar=INP_feld_ext_ambient*1e6
BC=INP_BC_ext_ambient_murray*1e6
BC=INP_BC_ext_ambient_min*1e6
BC=INP_BC_ext_ambient_threshold*1e6

def plot_ratio_ambient(BC,feld,name='none'):
    plt.figure()
    xx=plt.subplot(1,1,1)
    
    BC=BC.mean(axis=(-1,-2))
    feld=feld.mean(axis=(-1,-2))
    BC[BC<1e-2]=0
    feld[feld<1e-2]=0
    BC[np.isnan(BC)]=0
    #ratiobcfel1=INP_BC_ext_ambient_m3_for_calc/INP_feld_ext_ambient_m3_for_calc
    ratiobcfel=BC/feld
    ratiobcfel[np.isnan(ratiobcfel)]=0
    #ratiobcfel1[np.isnan(ratiobcfel1)]=0
    levelsratio=np.logspace(-5,5,11).tolist()
    #levelsratio=np.linspace(-100,100,11).tolist()
    plt.title(name+ ' BC_ambient/Feld_ambient')
    CF=xx.contourf(Xfel,Yfel,ratiobcfel[:,:],levelsratio,cmap=plt.cm.RdBu_r,norm= colors.BoundaryNorm(levelsratio, 256))
    CB=plt.colorbar(CF,ticks=levelsratio,drawedges=1,label='$ratio$',format=ticker.FuncFormatter(fmt))
    xx.tick_params(axis='both', which='major')
    xx.invert_yaxis()    
    xx.set_ylim(ymax=200)
    plt.savefig(jl.bc_folder+name+'.png')

#ax=plt.subplot(1,3,1)
plot_ratio_ambient(INP_BC_ext_ambient_min*1e6,feldspar,name='min')#,ax)
#bx=plt.subplot(1,3,2)
plot_ratio_ambient(INP_BC_ext_ambient_threshold*1e6,feldspar,name='Threshold')#,bx)
#cx=plt.subplot(1,3,3)
plot_ratio_ambient(INP_BC_ext_ambient_murray*1e6,feldspar,name='High')#,cx)

'''
cx=plt.subplot(1,3,3)
CF=cx.contourf(Xfel,Yfel,ratiobcfel[:,:,:,5].mean(axis=(-1))/ratiobcfel1[:,:,:,5].mean(axis=(-1)),[0.9,1.1],cmap=plt.cm.jet,norm= colors.BoundaryNorm([0.9,1.1], 256))
CB=plt.colorbar(CF,ticks=[0.9,1.1],drawedges=1,label='$m^{-3}$',format=ticker.FuncFormatter(fmt))
cx.tick_params(axis='both', which='major', labelsize=10)
cx.invert_yaxis()    
cx.set_ylim(ymax=200)
'''
plt.show()

#%%

feld=INP_feld_ext_ambient*1e6
BC=INP_BC_ext_ambient_min*1e6

plt.figure()
xx=plt.subplot(1,1,1)

BC=BC.mean(axis=(-1,-2))
feld=feld.mean(axis=(-1,-2))
BC[BC<1e-2]=0
feld[feld<1e-2]=0
BC[np.isnan(BC)]=0
#ratiobcfel1=INP_BC_ext_ambient_m3_for_calc/INP_feld_ext_ambient_m3_for_calc
ratiobcfel=BC/feld
ratiobcfel[np.isnan(ratiobcfel)]=0
#ratiobcfel1[np.isnan(ratiobcfel1)]=0
levelsratio=np.logspace(-5,5,11).tolist()
#levelsratio=np.linspace(-100,100,11).tolist()
plt.title('BC/Feld')
CF=xx.contourf(Xfel,Yfel,ratiobcfel[:,:],levelsratio,cmap=plt.cm.RdBu_r,norm= colors.BoundaryNorm(levelsratio, 256))
CB=plt.colorbar(CF,ticks=levelsratio,drawedges=1,label='$ratio$',format=ticker.FuncFormatter(fmt))
xx.tick_params(axis='both', which='major')
xx.invert_yaxis()    
xx.set_ylim(ymax=200)

plt.show()
#%%
plt.imshow(ratiobcfel)
plt.colorbar()
plt.show()
#%%
for _ in range(1000):
    plt.close()
#%%

INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')*1e-3#l
INP_feldspar_alltemps=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e3#l



column_feldspar=INP_feldspar_alltemps[:,:,jl.cape_verde_latlon_index[0],jl.cape_verde_latlon_index[1],7]
column_marine=INP_marine_alltemps[:,:,jl.cape_verde_latlon_index[0],jl.cape_verde_latlon_index[1],7]
temps=np.arange(-37,1,1)
temps=temps[::-1]
#%%
plt.figure()
for i in range(len(column_marine[0,:])):
    if i <22:
        continue
    plt.plot(temps,column_marine[:,i],'g--',label='Marine organics')
    plt.plot(temps,column_feldspar[:,i],'r--',label='K-feldspar')
plt.yscale('log')
plt.ylabel('$[INP]/L$')
#%%
level=20
plt.plot(temps,column_marine[:,level],'g--')
plt.plot(temps,column_feldspar[:,level],'r--')
plt.yscale('log')
table=np.zeros((39,32))
ps=[(i+1)*1/31.*1000 for i in range(31)]
table[1:,0]=temps
table[0,1:]=ps
table[1:,1:]=column_feldspar
np.savetxt('marine_cape_verde.csv',table,delimiter=',')
np.savetxt('feldspar_cape_verde.csv',table,delimiter=',')
for i in range(len(column_marine[0,:])):


#plt.plot(ps)
#plt.plot(jl.pressure)
#'''