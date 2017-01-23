# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:53:35 2014

@author: eejvt
"""

from scipy.io.idl import readsav
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

#s=readsav('/nfs/see-fs-01_users/eejvt/GLOMAP/NO_ICE_SCAV/GLOMAPmode_hindcast3_sm_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_noICE_SCAV_jan_2001.sav')



names=os.listdir('/nfs/see-fs-01_users/eejvt/GLOMAP/NO_ICE_SCAV/')

a=glob(r'/nfs/see-fs-01_users/eejvt/GLOMAP/NO_ICE_SCAV/*.sav')
for i in range (len(a)):
    s=readsav(a[i])