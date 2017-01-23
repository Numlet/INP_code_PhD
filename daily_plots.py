# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:04:05 2015

@author: eejvt
"""


import os
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
archive_directory='/nfs/a201/eejvt/'
project='DAILY_RUN/'
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
import glob as glob
os.chdir(archive_directory+project)
import matplotlib as mpl
mpl.use('Agg')

#%%
INP_marine_ambient_constant_press_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_ambient_constant_press.npy')
INP_feld_extamb_constant_press_m3=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_feld_extamb_constant_press.npy')*1e6

INP_BC_extamb_constant_press_m3=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_BC_extamb_constant_press.npy')*1e6
#%%
def gif(hpa):
    
    level=int(hpa/1000.*20)
    
    print 'Feldspar'
    archive_directory='/nfs/a201/eejvt/'
    project='DAILY_RUN/'
    os.chdir(archive_directory+project)
    INP_feld_extamb_constant_press=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_feld_extamb_constant_press.npy')
    path='GIF_FELDSPAR_%i'%hpa
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.ioff()
    os.chdir(path)
    iday=0
    for iday in range(365):
        print iday
        jl.plot(INP_feld_extamb_constant_press[level,:,:,iday]*1e3,cblabel='$L^{-1}$',clevs=np.logspace(-4,3,13).tolist()\
        ,cmap=plt.cm.OrRd,title='Feldspar INP ambient %ihpa day %i'%(hpa,iday+1),file_name='Feld_ambient_%i_%0.3i'%(hpa,iday),show=0,saving_format='ps',dpi=300,colorbar_format_sci=0)
        plt.close()
    os.system('cp /nfs/a201/eejvt/DAILY_RUN/animate_psfiles_fast %s%s%s'%(archive_directory,project,path))
    os.system('bash animate_psfiles_fast')


    print 'BC'
    archive_directory='/nfs/a201/eejvt/'
    project='DAILY_RUN/'
    os.chdir(archive_directory+project)
    INP_BC_extamb_constant_press=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_BC_extamb_constant_press.npy')
    path='GIF_BC_%i'%hpa
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.ioff()
    os.chdir(path)
    iday=0
    for iday in range(365):
        print iday
        jl.plot(INP_BC_extamb_constant_press[level,:,:,iday]*1e3,cblabel='$L^{-1}$',clevs=np.logspace(-4,3,13).tolist()\
        ,cmap=plt.cm.bone_r,title='BC INP ambient %ihpa day %i'%(hpa,iday+1),file_name='BC_ambient_%i_%0.3i'%(hpa,iday),show=0,saving_format='ps',dpi=300,colorbar_format_sci=0)
        plt.close()
    os.system('cp /nfs/a201/eejvt/DAILY_RUN/animate_psfiles_fast %s%s%s'%(archive_directory,project,path))
    os.system('bash animate_psfiles_fast')


    print 'MO'
    archive_directory='/nfs/a201/eejvt/'
    project='DAILY_RUN/'
    os.chdir(archive_directory+project)
    #INP_mo_extamb_constant_press=np.load('/nfs/a201/eejvt/COMPILATION_PAPER/DAILY_RUN/INP_marine_ambient_constant_press.npy')
    INP_mo_extamb_constant_press=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_ambient_constant_press.npy')
    path='GIF_MO_%i'%hpa
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.ioff()
    os.chdir(path)
    iday=0
    for iday in range(365):
        print iday
        jl.plot(INP_mo_extamb_constant_press[level,:,:,iday]*1e-3,cblabel='$L^{-1}$',clevs=np.logspace(-4,3,13).tolist()\
        ,cmap=plt.cm.Blues,title='Marine organic INP ambient %ihpa day %i'%(hpa,iday+1),file_name='MO_ambient_%i_%0.3i'%(hpa,iday),show=0,saving_format='ps',dpi=300,colorbar_format_sci=0)
        plt.close()
    os.system('cp /nfs/a201/eejvt/DAILY_RUN/animate_psfiles_fast %s%s%s'%(archive_directory,project,path))
    os.system('bash animate_psfiles_fast')




#%%
gif(500)

gif(600)

gif(800)
#%%
def gif_constant_t(hpa,t):
    
    level=int(hpa/1000.*20)
    
    print 'Feldspar'
    archive_directory='/nfs/a201/eejvt/'
    project='DAILY_RUN/'
    os.chdir(archive_directory+project)
    INP_feld_extamb_constant_press=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_feld_ext_daily_alltemps.npy')[t,]
    path='GIF_FELDSPAR_t%i_%i'%(t,hpa)
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.ioff()
    os.chdir(path)
    iday=0
    for iday in range(365):
        print iday
        jl.plot(INP_feld_extamb_constant_press[level,:,:,iday]*1e3,cblabel='$L^{-1}$',clevs=np.logspace(-3,2,13).tolist()\
        ,cmap=plt.cm.OrRd,title='Feldspar INP T=-%i %ihpa day %i'%(t,hpa,iday+1),file_name='Feld_ambient_%i_%0.3i'%(hpa,iday),show=0,saving_format='ps',dpi=150,colorbar_format_sci=0)
        plt.close()
    os.system('cp /nfs/a201/eejvt/DAILY_RUN/animate_psfiles_fast %s%s%s'%(archive_directory,project,path))
    os.system('bash animate_psfiles_fast')

    '''
    print 'BC'
    archive_directory='/nfs/a201/eejvt/'
    project='DAILY_RUN/'
    os.chdir(archive_directory+project)
    INP_BC_extamb_constant_press=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_BC_extamb_constant_press.npy')
    path='GIF_BC_%i'%hpa
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.ioff()
    os.chdir(path)
    iday=0
    for iday in range(365):
        print iday
        jl.plot(INP_BC_extamb_constant_press[level,:,:,iday]*1e3,cblabel='$L^{-1}$',clevs=np.logspace(-4,3,13).tolist()\
        ,cmap=plt.cm.bone_r,title='BC INP ambient %ihpa day %i'%(hpa,iday+1),file_name='BC_ambient_%i_%0.3i'%(hpa,iday),show=0,saving_format='ps',dpi=300,colorbar_format_sci=0)
        plt.close()
    os.system('cp /nfs/a201/eejvt/DAILY_RUN/animate_psfiles_fast %s%s%s'%(archive_directory,project,path))
    os.system('bash animate_psfiles_fast')
    '''
    '''
    print 'MO'
    archive_directory='/nfs/a201/eejvt/'
    project='DAILY_RUN/'
    os.chdir(archive_directory+project)
    #INP_mo_extamb_constant_press=np.load('/nfs/a201/eejvt/COMPILATION_PAPER/DAILY_RUN/INP_marine_ambient_constant_press.npy')
    INP_mo_extamb_constant_press=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_ambient_constant_press.npy')
    path='GIF_MO_%i'%hpa
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.ioff()
    os.chdir(path)
    iday=0
    for iday in range(365):
        print iday
        jl.plot(INP_mo_extamb_constant_press[level,:,:,iday]*1e-3,cblabel='$L^{-1}$',clevs=np.logspace(-4,3,13).tolist()\
        ,cmap=plt.cm.Blues,title='Marine organic INP ambient %ihpa day %i'%(hpa,iday+1),file_name='MO_ambient_%i_%0.3i'%(hpa,iday),show=0,saving_format='ps',dpi=300,colorbar_format_sci=0)
        plt.close()
    os.system('cp /nfs/a201/eejvt/DAILY_RUN/animate_psfiles_fast %s%s%s'%(archive_directory,project,path))
    os.system('bash animate_psfiles_fast')
    '''
#%%
gif_constant_t(700,20)

#%%
archive_directory='/nfs/a201/eejvt/'
project='DAILY_RUN/'
os.chdir(archive_directory+project)
INP_BC_extamb_constant_press=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_BC_extamb_constant_press.npy')

if not os.path.isdir('GIF_BC'):
    os.mkdir('GIF_BC')
plt.ioff()
os.chdir('GIF_BC')
iday=0
for iday in range(365):
    print iday
    jl.plot(INP_BC_extamb_constant_press[10,:,:,iday]*1e3,cblabel='$L^{-1}$',clevs=[0,0.1,0.5,1,2,5,10,20,50,100,200,500,1000]\
    ,cmap=plt.cm.bone_r,title='BC INP ambient 500hpa day %i'%(iday+1),file_name='BC_ambient_500_%0.3i'%iday,show=0,saving_format='png',dpi=300)
    plt.close()


#%%
archive_directory='/nfs/a201/eejvt/'
project='DAILY_RUN/'
os.chdir(archive_directory+project)
INP_mo_extamb_constant_press=np.load('/nfs/a201/eejvt/COMPILATION_PAPER/DAILY_RUN/INP_marine_ambient_constant_press.npy')

if not os.path.isdir('GIF_mo_'):
    os.mkdir('GIF_mo')
plt.ioff()
os.chdir('GIF_mo')
iday=0
for iday in range(365):
    print iday
    jl.plot(INP_mo_extamb_constant_press[10,:,:,iday]*1e-3,cblabel='$L^{-1}$',clevs=[0,0.1,0.5,1,2,5,10,20,50,100,200,500,1000]\
    ,cmap=plt.cm.Blues,title='mo INP ambient 500hpa day %i'%(iday+1),file_name='mo_ambient_500_%0.3i'%iday,show=0,saving_format='png',dpi=300)
    plt.close()




#%%
levelsmo=[1,10,20,30,40,50,60,70,80,90,100,500]
INP_mo_extamb_constant_press=np.load('/nfs/a201/eejvt/COMPILATION_PAPER/DAILY_RUN/INP_marine_ambient_constant_press.npy')
levarray=np.linspace(0,1000,21)
INP_mo_extamb_constant_press=INP_mo_extamb_constant_press.mean(axis=-1)
INP_mo_extamb_constant_press=INP_mo_extamb_constant_press.mean(axis=2)
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
Xmo,Ymo=np.meshgrid(lat.glat,levarray)
#plt.contourf(X,Y,INP_mo_extamb_constant_press)
CF=plt.contourf(Xmo,Ymo,INP_mo_extamb_constant_press,levelsmo,cmap=plt.cm.YlOrRd,norm= colors.BoundaryNorm(levelsmo, 256))
CB=plt.colorbar(CF,ticks=levelsmo,drawedges=1,label='$m^{-3}$')
plt.gca().invert_yaxis()
#plt.colorbar()
plt.show()


#%%


INP_marine_ambient_constant_press_daily=np.load('/nfs/a201/eejvt/MARINE_PARAMETERIZATION/DAILY/INP_marine_ambient_constant_press.npy')
INP_feld_extamb_constant_press_m3=np.load('/nfs/a201/eejvt/DAILY_RUN/INP_feld_extamb_constant_press.npy')*1e6
INP_marine_ambient_constant_press_monthly=jl.from_daily_to_monthly(INP_marine_ambient_constant_press_daily)
INP_feld_extamb_constant_press_m3_monthly=jl.from_daily_to_monthly(INP_feld_extamb_constant_press_m3)
'''
INP_marine_ambient_constant_press_daily[INP_marine_ambient_constant_press_daily<0.1]=0
ratio_MO_FELD=INP_marine_ambient_constant_press_daily/INP_feld_extamb_constant_press_m3
ratio_MO_FELD[ratio_MO_FELD==np.nan]=0
ratio_MO_FELD[ratio_MO_FELD>1]=1
ratio_MO_FELD[ratio_MO_FELD<1]=0
days_MO=ratio_MO_FELD.sum(axis=-1)
'''

INP_marine_ambient_constant_press_monthly[INP_marine_ambient_constant_press_monthly<0.1]=0
INP_feld_extamb_constant_press_m3_monthly[INP_feld_extamb_constant_press_m3_monthly<0.0001]=0.0001
ratio_MO_FELD=INP_marine_ambient_constant_press_monthly/INP_feld_extamb_constant_press_m3_monthly













