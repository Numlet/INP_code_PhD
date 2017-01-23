# =================================================================================
# Plotting UM output
# =================================================================================
# annette miltenberger             22.10.2014              ICAS University of Leeds

# importing libraries etc.
# ---------------------------------------------------------------------------------
import iris 					    # library for atmos data
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import netCDF4
import SkewT

# Directories, filenames, dates etc.
# ---------------------------------------------------------------------------------
data_path = '/group_workspaces/jasmin/asci/amilt/icmw/'  # directory of model output
models = ['um']
int_precip = [1]
dt_output = [30]                                         # min
factor = [2.]  # for conversion into mm/h
res_name = ['1km']                          # name of the UM run
aero_name = ['std_aerosol']
phys_name = ['allice']#['allice','nohm','warm']


fig_dir = '/home/users/amiltenberger/graphics/icmw_new/'

date = '0000'

# region considered for LAM statistics (based on Fig. 1 in Hanley and Lean 2014)
center_lat = np.array(50.65504)
center_lon = np.array(-4.61630)
center_rlon,center_rlat = iris.analysis.cartography.rotate_pole(center_lon,center_lat,175.3868,39.4375)
rlat = center_rlat+np.arange(-100,100)*0.009
rlon = center_rlon+np.arange(-100,200)*0.009
R_LON, R_LAT = np.meshgrid(rlon,rlat)
lon,lat = iris.analysis.cartography.unrotate_pole(R_LON.flatten(),R_LAT.flatten(),175.3868,39.4375)

m = 0 # index for models array
for i in range(0,1):#len(res_name)):
 input_file = data_path+models[0]+'/LMCONSTANTS_'+res_name[i]
 print input_file
 ncfile = netCDF4.Dataset(input_file,mode='r')
 lon_model = ncfile.variables['LON'][:,:]
 lat_model = ncfile.variables['LAT'][:,:]
 hl_cube = ncfile.variables['HL'][:,:,:]
 ncfile.close()
 mask_model = ((lon_model>=lon[0]) & (lon_model<=lon[-1]) & (lat_model>=lat[0]) & (lat_model<=lat[-1]))

 for j in range(0,len(aero_name)):
   k=j
  #for k in range(0,len(phys_name)):
   start_time = 8
   end_time = 13
   time_ind = 0
   hh = np.int(np.floor(start_time))
   mm = np.int((start_time-hh)*60)

   for t in np.arange(start_time*60,end_time*60-1,dt_output[m]):
    if t>start_time*60:
     mm = mm+dt_output[m]
    if mm>=60:
       mm = mm-60
       hh = hh+1
    if (hh < 10):
      if (mm<10):
       date = '0'+str(hh)+'0'+str(mm)
      else:
       date = '0'+str(hh)+str(mm)
    else:
      if (mm<10):
       date = str(hh)+'0'+str(mm)
      else:
       date = str(hh)+str(mm)

    # Loading data from file
    # ---------------------------------------------------------------------------------
    file_name = 'P20130800'+date
    input_file = data_path+models[m]+'/'+res_name[i]+'_'+aero_name[j]+'_'+phys_name[k]+'/'+file_name
    print input_file
    ncfile = netCDF4.Dataset(input_file,mode='r')
    p_cube = np.squeeze(ncfile.variables['P'][:,:,:]) # pressure
    t_cube = np.squeeze(ncfile.variables['T'][:,:,:]) # potential temperature
    ncfile.close()

    file_name = 'H20130800'+date
    input_file = data_path+models[m]+'/'+res_name[i]+'_'+aero_name[j]+'_'+phys_name[k]+'/'+file_name
    print input_file
    ncfile = netCDF4.Dataset(input_file,mode='r')
    qv_cube = np.squeeze(ncfile.variables['QV'][:,:,:])
    qc_cube = np.squeeze(ncfile.variables['QC'][:,:,:])
    ncfile.close()

    ind = np.where((np.abs(lat_model-50.64)+np.abs(lon_model+4.61))==(np.abs(lon_model+4.61)+np.abs(lat_model-50.64)).min())
    p_sonde = np.squeeze(p_cube[:,ind[0],ind[1]])*100.
    th_sonde = np.squeeze(t_cube[:,ind[0],ind[1]])
    qv_sonde = np.squeeze(qv_cube[:,ind[0],ind[1]])
    qc_sonde = np.squeeze(qc_cube[:,ind[0],ind[1]])
    z_sonde = np.squeeze(hl_cube[:,ind[0],ind[1]])
    th_sonde = 0.5*(th_sonde[1:]+th_sonde[0:-1])
    qv_sonde = 0.5*(qv_sonde[1:]+qv_sonde[0:-1])
    qc_sonde = 0.5*(qc_sonde[1:]+qc_sonde[0:-1])
    z_sonde = 0.5*(z_sonde[1:]+z_sonde[0:-1])

    print qc_sonde

    ind_z = np.where(z_sonde<10000)

    t_sonde = th_sonde*(p_sonde/10**5)**0.286-273.15 #deg C

    ew = qv_sonde*p_sonde/0.622
    gamma = np.log(ew/(6.112*10**2))
    tdp_sonde = 243.5*gamma/(17.67-gamma)
    p_sonde = p_sonde/100.

    sounding=dict(zip(('hght','pres','temp','dwpt'),(z_sonde[ind_z],p_sonde[ind_z],t_sonde[ind_z],tdp_sonde[ind_z])))
    S=SkewT.Sounding(soundingdata=sounding)
    S.make_skewt_axes(tmin=-20.,tmax=20.,pmin=300.,pmax=1050.)
#    parcel=S.get_parcel()
#    parcel=S.get_parcel()
    S.add_profile(bloc=0)

    print 'Save figure'
    fig_name = fig_dir + 'sonding_'+res_name[i]+'_'+aero_name[j]+'_'+phys_name[k]+'_'+date+'_um_obs.png'
    plt.savefig(fig_name)#, format='eps', dpi=300)
    plt.show()
    plt.close()
    

