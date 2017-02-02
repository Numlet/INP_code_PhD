#from IPython import embed  #for debugging - put embed() in code for breakpoint

import numpy as np
from Scientific.IO.NetCDF import NetCDFFile as Dataset

#Run the script to get the data
#execfile('open_L2_C6_MODIS_run.py')


from open_L2_C6_MODIS_file_func import *

#Jesus' MODIS file for SO
#path='/group_workspaces/jasmin/asci/dgrosv/MODIS/Jesus/'
path='/nfs/a201/eejvt/CASIM/SO_KALLI/SATELLITE/modis/'
file_hdf = 'MYD06_L2.A2014343.1325.006.2014344210847.hdf'

#Get the data
MODL2_C6_outputs = open_modis_L2(path,file_hdf)

#Will just write out N37
Nd_37 = MODL2_C6_outputs.get('N37')

nx=Nd_37.shape[0]
ny=Nd_37.shape[1]

#write the file
ncfile = Dataset(path + 'Nd_' + file_hdf + '.nc','w')
ncfile.createDimension('x',nx)
ncfile.createDimension('y',ny)

data = ncfile.createVariable('CDNC_37_MODIS',np.dtype('float64').char,('x','y'))
data[:] = Nd_37
ncfile.close()






#%%
