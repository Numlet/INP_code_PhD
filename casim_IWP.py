# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:23:55 2016

@author: eejvt
"""

import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib
import sys
import glob
dir_scripts='/nfs/see-fs-01_users/eejvt/UKCA_postproc'#Change this to the downloaded folder
sys.path.append(dir_scripts)
import UKCA_lib as ukl
import iris.quickplot as qp
import numpy as np
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import Jesuslib as jl
import matplotlib.pyplot as plt
import matplotlib.animation as animationlt
from scipy.io import netcdf
sys.path.append('/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/code')
#from amsre_daily_v7 import AMSREdaily
from pyhdf import SD
import datetime
import scipy as sc
from scipy.io import netcdf
import time
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE/Satellite_Comparison')
import satellite_comparison_suite as stc

from collections import OrderedDict

cube_3ord = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/L1/','IWP'))[0]
cube_2m = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_MORE/L1/','IWP'))[0]
cube = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/L1/','IWP'))[0]
cube_con = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/BASE_CONTACT_242/L1/','IWP'))[0]
#cube_oldm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/OLD_MICRO/L1/','LWP'))[0]
#cube_nh = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/L1/','LWP'))[0]
cube_ni = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NOICE/L1/','IWP'))[0]

cube_csb = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/CLOUD_SQUEME/BASE/L1/','IWP'))[0]
cube_csbm = iris.load(ukl.Obtain_name('/nfs/a201/eejvt/CASIM/SO_KALLI/CLOUD_SQUEME/MEYERS/L1/','IWP'))[0]

model_lons,model_lats=stc.unrotated_grid(cube)

X,Y=np.meshgrid(model_lons, model_lats)

runs_dict=OrderedDict()
runs_dict['ALL_ICE_PROC']=cube[12].data
runs_dict['BASE (CS)']=cube_csb[13].data
runs_dict['MEYERS']=cube_csbm[13].data
runs_dict['3_ORD_LESS']=cube_3ord[13].data
runs_dict['2_ORD_MORE']=cube_2m[13].data


levels=np.linspace(runs_dict['ALL_ICE_PROC'].min(),runs_dict['ALL_ICE_PROC'].max(),50)
same_bins=np.linspace(runs_dict['ALL_ICE_PROC'].min(),runs_dict['ALL_ICE_PROC'].max()*1.5,150)
#levels=np.linspace(runs_dict['Satellite (AMSR2)'].min(),runs_dict['Satellite (AMSR2)'].max(),15)
stc.plot_map(runs_dict,levels,lat=X,lon=Y,variable_name='IWP mm')
#stc.plot_PDF(runs_dict,same_bins, variable_name='IWP mm')





























