
import os
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
archive_directory='/nfs/a107/eejvt/'
project='JB_TRAINING'
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
import glob as glob
os.chdir(archive_directory+project)

#%%

rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
#rhocomp =rhocomp*1e+3#ug/cm3
rhocomp =rhocomp*1e+9#ug/m3

s=jl.read_data('WITH_ICE_SCAV2')


def volumes_of_modes(s):
    modes_vol=(s.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]
    +s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2]
    +s.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
    +s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]
    +s.tot_mc_oc_mm_mode[:,:,:,:,:]/rhocomp[5]
    +s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])
    return modes_vol




modes_vol=volumes_of_modes(s)
dust_volfrac=((s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])+(s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]))/modes_vol
Nd=s.st_nd[:,:,:,:,:]*dust_volfrac

np.save(jl.home_dir+'GLOMAP_dust_Nd.npy',Nd)
np.save(jl.home_dir+'GLOMAP_rbar_dry.npy',s.rbardry)
