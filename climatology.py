# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:51:32 2016

@author: eejvt
"""

import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import numpy as np
import Jesuslib as jl
import os
from scipy.io.idl import readsav
from glob import glob
from scipy.io import netcdf
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats.stats import pearsonr
from glob import glob
from scipy.io.idl import readsav
from scipy import stats
from scipy.optimize import curve_fit
import scipy


path='/nfs/a201/eejvt/CLIMATOLOGY/'


years=['2001','2002','2003']#,'2004']
year=years[0]
#%%
for year in years:
    os.chdir(path+year)

    names=['tot_mc_oc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_'+year+'.sav',
    'tot_mc_bc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_'+year+'.sav',
    'tot_mc_dust_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_'+year+'.sav',
    'tot_mc_su_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_'+year+'.sav',
    'tot_mc_ss_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_'+year+'.sav',
    'tot_mc_feldspar_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_'+year+'.sav',
    'GLOMAP_mode_mol_no_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_'+year+'.sav',
    'mean_dry_r_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_'+year+'.sav',
    'sigmachained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_'+year+'.sav',
    #'hindcast3_temp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
    #'GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_daily_2001.sav',
    'tot_mc_ss_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_'+year+'.sav',
    #'hindcast3_temp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
    #'GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_JVT_mo_2001.sav',
    ]
    s={}
    for name in names:
        print name
        s=readsav(name,idict=s)
    def calculate_INP_feld_ext_mean_area_fitted(T,fel_modes=[2,3]):
        std=s.sigma[:]
        #T=258
        modes_vol=jl.volumes_of_modes(s)
        kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/jl.rhocomp[6])/modes_vol
        Nd=s.st_nd[:,:,:,:,:]*kfeld_volfrac
        rmean=s.rbardry[:,:,:,:,:]*1e2#factor 1e2 because of cm in feldspar parameterization
        ns=jl.feld_parametrization(T)
        INP=np.zeros(Nd.shape)
        for i in fel_modes:
            print 'mode',i
            area_particle=jl.area_lognormal_per_particle(rmean,std[i])
            ff=1-np.exp(-ns*area_particle)
            ff_fitted=jl.correct_ff(ff,std[i])
            INP=Nd*ff_fitted
        return INP




    INP_feldext_alltemps=np.zeros((38,31,64,128,365))

    for i in range (38):
        INP_feldext_alltemps[i,]=calculate_INP_feld_ext_mean_area_fitted(-i+273.15).sum(axis=0)


    np.save('INP_feldext_alltemps_'+year+'.npy',INP_feldext_alltemps)

    total_marine_mass=s.tot_mc_ss_mm_mode[2,]
    total_marine_mass_grams_OC=total_marine_mass*1e-6/1.9#g/m3
    INP_marine_alltemps=np.zeros((38,31,64,128,365))
    for i in range (38):
        INP_marine_alltemps[i,]=total_marine_mass_grams_OC*jl.marine_org_parameterization(-i)

    np.save('INP_marine_alltemps_'+year+'.npy',INP_marine_alltemps)





print 'Code finished'

print 'runing climatology 2'

os.system('python climatology_2.py')

#%%
#class structured_year():
#    def __init__(self,data_values):



def mean_and_std(distribution):
    mean=distribution.mean()
    std=np.std(distribution)
    return mean, std
