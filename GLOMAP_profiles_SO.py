# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import numpy as np


import numpy.ma as ma
import sys
import random
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
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
import Jesuslib as jl
archive_directory='/nfs/a107/eejvt/'
project='JB_TRAINING/'
import scipy as sc


ilat,ilon=50,3

os.chdir(archive_directory+project)

rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
rhocomp =rhocomp*1e+9#ug/m3
'''
;           Sulphate    Carbon    1.4*BC   NaCl      SiO2,      SOC   Feldspar                               
    mm_aer = [0.098,    0.012,   0.0168 , 0.05844,    0.10,    0.0168 ,0.1] ;SOC was 0.15  ; molar masses in Kg/mol
    rhocomp  = [1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.];masss density of components in Kg/m3
'''
#%%
s1=jl.read_data('WITH_ICE_SCAV2')

def volumes_of_modes(s):
    modes_vol=(s.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]
    +s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2]
    +s.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
    +s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]
    +s.tot_mc_oc_mm_mode[:,:,:,:,:]/rhocomp[5]
    +s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])
    return modes_vol

ss_vol=+s1.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
modes_vol=volumes_of_modes(s1)#m3
volfrac_ss=ss_vol/modes_vol
ss_particles_ext=volfrac_ss*s1.st_nd

ss_particles_05=ss_particles_ext[3,]+(ss_particles_ext[2,]-jl.lognormal_cummulative(ss_particles_ext[2,],250e-9,s1.rbardry[2,:,:,:,:],s1.sigma[2]))

partial_acc=s1.st_nd[2,:,:,:,:]-jl.lognormal_cummulative(s1.st_nd[2,:,:,:,:],250e-9,s1.rbardry[2,:,:,:,:],s1.sigma[2])
n05=s1.st_nd[3,:,:,:,:]+partial_acc-ss_particles_05#+s1.st_nd[6,:,:,:,:]#-ss_particles_05

#%%

#CCN=s1.st_nd[2,:,:,:,:]+s1.st_nd[3,:,:,:,:]+s1.st_nd[1,:,:,:,:]+s1.st_nd[0,:,:,:,:]#+s1.st_nd[5,:,:,:,:]+s1.st_nd[6,:,:,:,:]
#print CCN[:,50,3,0]

month=7
for month in range(12):
    print month
    for i in [0,1,2,3]:
        print 'nd mode:',i
        print s1.st_nd[i,30,50,3,month]
        print 'ccn1',s1.ccn_1[30,50,3,month]
        print 'ccn2',s1.ccn_2[30,50,3,month]
#        print s1.ccn_3[30,50,3,month]
#        print s1.ccn_4[30,50,3,month]
#%%

month=7
GLOMAP_pressures=s1.pl_m[:,ilat,ilon,month]

total_mass=s1.tot_mc_dust_mm_mode+s1.tot_mc_feldspar_mm_mode+\
s1.tot_mc_su_mm_mode+s1.tot_mc_ss_mm_mode+s1.tot_mc_oc_mm_mode+s1.tot_mc_bc_mm_mode


GLOMAP_pressures[30]=100000
GLOMAP_pressures[0]=0
casim_pressures=np.array([ 98733.04558553,  98529.33707583,  98244.65607866,  97879.51400373,
        97434.56722373,  96910.61832994,  96308.61579042,  95629.65841258,
        94875.00186532,  94046.06470376,  93144.43009919,  92171.84952957,
        91130.21628876,  90021.60619476,  88848.33813566,  87612.96270084,
        86318.66897059,  84970.63488176,  83576.06601026,  82138.06221477,
        80656.16031301,  79132.23795016,  77568.82437179,  75968.16773457,
        74332.79054707,  72665.31525054,  70968.38665514,  69244.85996438,
        67497.65537469,  65729.77395118,  63944.47493726,  62144.90483369,
        60333.86260163,  58513.94848776,  56687.96194254,  54859.11717473,
        53030.46602554,  51204.83570598,  49385.13643798,  47574.40091649,
        45775.68568889,  43991.90990913,  42225.08413896,  40475.68745269,
        38742.46450994,  37022.39591654,  35310.88571155,  33602.15434252,
        31888.67567921,  30161.28899659,  28410.31386492,  26626.37583869,
        24802.57011613,  22938.68146462,  21034.43666161,  19086.3360965 ,
        17098.61055948,  15083.27748278,  13058.29418384,  11058.41664482,
         9133.49832235,   7334.60859812,   5706.60877418,   4282.18620264,
         3084.74290968,   2126.49549991,   1400.65769967,    880.76683687,
          529.19944214,    304.91754213])

f_coarse_mass = sc.interpolate.interp1d(GLOMAP_pressures, total_mass[3,:,ilat,ilon,month])
coarse_mass_casim=f_coarse_mass(casim_pressures)

f_acc_mass = sc.interpolate.interp1d(GLOMAP_pressures, total_mass[2,:,ilat,ilon,month])
acc_mass_casim=f_acc_mass(casim_pressures)

f_aitken_mass = sc.interpolate.interp1d(GLOMAP_pressures, total_mass[1,:,ilat,ilon,month])
aitken_mass_casim=f_aitken_mass(casim_pressures)

f_coarse_number = sc.interpolate.interp1d(GLOMAP_pressures, s1.st_nd[3,:,ilat,ilon,month])
coarse_number_casim=f_coarse_number(casim_pressures)

f_acc_number = sc.interpolate.interp1d(GLOMAP_pressures, s1.st_nd[2,:,ilat,ilon,month])
acc_number_casim=f_acc_number(casim_pressures)

f_aitken_number = sc.interpolate.interp1d(GLOMAP_pressures, s1.st_nd[1,:,ilat,ilon,month])
aitken_number_casim=f_aitken_number(casim_pressures)

f_n05 = sc.interpolate.interp1d(GLOMAP_pressures, n05[:,ilat,ilon,month])
n05_casim=f_n05(casim_pressures)


#CCN_CASIM = f(casim_pressures)






total_mass[:,30,ilat,ilon,0]


aerosols_tracers=[33001,33002,33003,33004,33005,33006,33005,33006,33007,33008,33009,33010]
#33001   soluble Aitken mode mass
#33002   soluble Aitken mode number
#33003   soluble accumulation mode mass
#33004   soluble accumulation mode number
#33005   soluble coarse mode mass
#33006   soluble coarse mode number
#33007   insoluble coarse mode mass
#33008   insoluble coarse mode number
#33009   insoluble accumulation mode mass
#33010   insoluble accumulation mode number

with open("GLOMAP_profile_low_aerosol_v1.nml", "w") as text_file:
    for i in range(len(aerosols_tracers)):
        text_file.write("%i\n" % aerosols_tracers[i])
        for ilev in range(len(casim_pressures)):
            if aerosols_tracers[i]==33001:
                text_file.write('  %1.5e\n'%(aitken_mass_casim[ilev]*1e-9))
            elif aerosols_tracers[i]==33003:
                text_file.write('  %1.5e\n'%(acc_mass_casim[ilev]*1e-9))
            elif aerosols_tracers[i]==33005:
                text_file.write('  %1.5e\n'%(coarse_mass_casim[ilev]*1e-9))
            elif aerosols_tracers[i]==33002:
                text_file.write('  %1.5e\n'%(aitken_number_casim[ilev]*1e6))
            elif aerosols_tracers[i]==33004:
                text_file.write('  %1.5e\n'%(acc_number_casim[ilev]*1e6))
            elif aerosols_tracers[i]==33006:
                text_file.write('  %1.5e\n'%(coarse_number_casim[ilev]*1e6))
            elif aerosols_tracers[i]==33008:
#                print aerosols_tracers[i]
                text_file.write('  %1.5e\n'%(n05_casim[ilev]*1e6))
            elif aerosols_tracers[i]==33007:
                text_file.write('  5.88700e-07\n')
            else:
                text_file.write('      0.00000\n')




#

