#from get_hdf_data_dan import * #Put these '*' imports outside of the function to avoid an annoying warning 
from read_MODISL2_C6_funcs import get_hdf_data, getCDNC, maskread_1km
#from maskread_1km import *

import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
# Prevent warnings for doing e.g. i=x<5 when x is an array that contains NaNs

#from get_hdf_data_dan import get_hdf_data 
#from maskread_1km import maskread_1km

import numpy as np

def open_modis_L2(path,file_hdf):
#	from IPython import embed  #for debugging - put embed() in code for breakpoint

	from pyhdf.SD import SD,SDC
	from datetime import date





#	path='/group_workspaces/jasmin/asci/dgrosv/L2_C6/Hamish/'

#	#file_hdf = 'MOD06_L2.A2016222.1135.006.2016222210127.hdf'
#	file_hdf = 'MOD06_L2.A2016214.1040.006.2016215022430.hdf'

	filepath_hdf = path + file_hdf
	
	# Open the HDF object
	hdf = SD(filepath_hdf , SDC.READ)


	##N.B. MODIS files are hdf4 not hdf5, so need to use hdfinfo, etc rather
	##than hdf5info
	#
	#try #catch all errors later on - so that override_flags_openL2 flag can be
	#    #reset even if there is an error. The error is then rethrown
	#    #(re-issued).
	#
	#
	#
	#if ~exist('override_flags_openL2') | override_flags_openL2==0
	#    day_only=0;
	#end
	#
	#if exist('imodis_file_override') & imodis_file_override==1
	#    clear imodis_file_override  #don't set file_hdf and reset for next run
	#else
	#    filedir='/home/disk/eos1/d.grosvenor/';
	#    filedir='/home/disk/eos8/d.grosvenor/';   
	#
	#file_hdf='MOD_L2/C6/Hamish/MOD06_L2.A2016214.1040.006.2016215022430.hdf'; #Hamish CLARIFY test files
	#
	#end
	#



	# -- Find the date and time from the file name
	iday=file_hdf.find('.A')
	istart = iday+2; 
	nstr=4; iend=istart+nstr; modis_year_str=file_hdf[istart:iend];  istart=istart+nstr;
	nstr=3; iend=istart+nstr; modis_day_str=file_hdf[istart:iend];  istart=istart+nstr+1; #is a . in the name here
	nstr=4; iend=istart+nstr; modis_time_str=file_hdf[istart:iend];  istart=istart+nstr;
	nstr=4; iend=istart+nstr; modis_time_str=file_hdf[istart:iend];  istart=istart+nstr;
	aq_terr_str = file_hdf[iday-8:iday-3];

	modis_hour_str=modis_time_str[0:2]
	modis_min_str=modis_time_str[2:4]

	modis_date_time = date.toordinal(date(int(modis_year_str),1,1)) + int(modis_day_str) - 1
	#minus one since Python is zero index based. N.B. Python and Matlab use different basis years
	#Python uses 01-Jan-0001 whereas Matlab uses 00-Jan-0000 - check this
	# How to convert back to string?


	#date_str=datestr(datenum(['01-Jan-' modis_year_str])+str2num(modis_day_str)-1,1);
	#modis_date_time = datenum([date_str ' ' modis_hour_str ':' modis_min_str]);
	#
	#if day_only==1
	#    if (str2num(modis_time_str)>1800 | str2num(modis_time_str)<800)
	#        night_time=1;
	#        return
	#    end
	#end
	#
	
### ------  Start of read  -------		

	##typical attributes
	## 1) _FillValue
	## 2) long_name
	## 3) units
	## 4) scale_factor
	## 5) add_offset
	## 6) Parameter_Type
	## 7) Cell_Along_Swath_Sampling
	## 8) Cell_Across_Swath_Sampling
	## 9) Geolocation_Pointer
	## 10) Description (only for QA flags - useful info on how the bits work -see end of this script)

# call get_hdf_data to get selected variables - def get_hdf_data(SDS_NAME,hdf,disp_attr,fill_value) :
# SDS_NAME = variable_name; hdf=hdf file object; disp_att is an option (0 or 1) to display the attributes (need to implement..); fill_value is the value to use for fills - best to use 0 for the qa1, qa5, mask1 and mask5 arrays	

	(lat,dimsizes_lat,sampling_along,sampling_across) = get_hdf_data('Latitude',hdf,0,np.float64('Nan'))
	(lon,dimsizes_lon,tempA,tempB) = get_hdf_data('Longitude',hdf,0,np.float64('Nan'));
	(cf,dimsizes_cf,tempA,tempB) = get_hdf_data('Cloud_Fraction',hdf,0,np.float64('Nan')); #int8 (0...127)
	(t_top_1km,dimsizes_ctt,tempA,tempB) = get_hdf_data('cloud_top_temperature_1km',hdf,0,np.float64('Nan')); #int16 (>0)
	(p_top_1km,dimsizes_ctp,tempA,tempB) = get_hdf_data('cloud_top_pressure_1km',hdf,0,np.float64('Nan')); #int16 (>0)
	(cth_1km,dimsizes_ctp,tempA,tempB) = get_hdf_data('cloud_top_height_1km',hdf,0,np.float64('Nan')); #int16 (>0)
	(method_top_1km,dimsizes_ctp,tempA,tempB) = get_hdf_data('cloud_top_method_1km',hdf,0,np.float64('Nan')); #int16 (>0)
	(tau21,dimsizes_tau,tempA,tempB) = get_hdf_data('Cloud_Optical_Thickness',hdf,0,np.float64('Nan')); #int16 (>0)
	(tau16,dimsizes_tau,tempA,tempB) = get_hdf_data('Cloud_Optical_Thickness_16',hdf,0,np.float64('Nan')); #int16 (>0)
	(tau37,dimsizes_tau,tempA,tempB) = get_hdf_data('Cloud_Optical_Thickness_37',hdf,0,np.float64('Nan')); #int16 (>0)
	#(qa1,dimsizes_QA_1km,tempA,tempB) = get_hdf_data('Quality_Assurance_1km',hdf,0,0); #int8 (-128 min) unsigned
	#(qa5,dimsizes_qa5,tempA,tempB) = get_hdf_data('Quality_Assurance_5km',hdf,0,0); #int8 (-93 min) unsigned
	(mask1,dimsizes_mask1,tempA,tempB) = get_hdf_data('Cloud_Mask_1km',hdf,0,0); #int8 unsigned -- Using 0 as fill value for these integers for bit stripping
	#(mask5,dimsizes_mask5,tempA,tempB) = get_hdf_data('Cloud_Mask_5km',hdf,0,0); #int8 unsigned
	(re21,dimsizes_re,tempA,tempB) = get_hdf_data('Cloud_Effective_Radius',hdf,0,np.float64('Nan')); #int16, (401 min) sf=0.01 (microns)
	(re16,dimsizes_re,tempA,tempB) = get_hdf_data('Cloud_Effective_Radius_16',hdf,0,np.float64('Nan')); #int16, (401 min) sf=0.01 (microns)
	(re37,dimsizes_re,tempA,tempB) = get_hdf_data('Cloud_Effective_Radius_37',hdf,0,np.float64('Nan')); #int16, (401 min) sf=0.01 (microns)
	
	##NOTE - uncertainties are in % not absolute values! (check for C6)
	(re21_un,dimsizes_re_un,tempA,tempB) = get_hdf_data('Cloud_Effective_Radius_Uncertainty',hdf,0,np.float64('Nan'));
	(tau21_un,dimsizes_tau_un,tempA,tempB) = get_hdf_data('Cloud_Optical_Thickness_Uncertainty',hdf,0,np.float64('Nan')); #int16 (>0)
	(re16_un,dimsizes_re_un,tempA,tempB) = get_hdf_data('Cloud_Effective_Radius_Uncertainty_16',hdf,0,np.float64('Nan'));
	(tau16_un,dimsizes_tau_un,tempA,tempB) = get_hdf_data('Cloud_Optical_Thickness_Uncertainty_16',hdf,0,np.float64('Nan')); #int16 (>0)
	(re37_un,dimsizes_re_un,tempA,tempB) = get_hdf_data('Cloud_Effective_Radius_Uncertainty_37',hdf,0,np.float64('Nan'));
	(tau37_un,dimsizes_tau_un,tempA,tempB) = get_hdf_data('Cloud_Optical_Thickness_Uncertainty_37',hdf,0,np.float64('Nan')); #int16 (>0)
	
	##Cloud phase - need to remember to only look at liquid clouds for Nd!
	## This can replace the QA bit method, which is just legacy apparently
	
	(cloud_phase,dimsizes_cloud_phase,tempA,tempB) = get_hdf_data('Cloud_Phase_Optical_Properties',hdf,0,0); #int16 (>0)
	cloud_phase = cloud_phase.astype(int)
	#  # 0 = cloud mask unavailable, missing data,etc. (no phase result), 1= no
	#  # phase result due to clear sky, etc.; 2=liquid water; 3=ice;
	#  # 4=undetermined
	
	(scantime,dimsizes_scantime,tempA,tempB) = get_hdf_data('Scan_Start_Time',hdf,0,np.float64('Nan')); #double
	## (seconds since 1993-1-1 00:00:00.0) (!)
	
	(solar_zenith,dimsizes_solar_zenith,tempA,tempB) = get_hdf_data('Solar_Zenith',hdf,0,np.float64('Nan')); #int16, min=-32767
	(solar_azimuth,dimsizes_sensor_zenith,tempA,tempB) = get_hdf_data('Solar_Azimuth',hdf,0,np.float64('Nan')); #int16
	(sensor_zenith,dimsizes_sensor_zenith,tempA,tempB) = get_hdf_data('Sensor_Zenith',hdf,0,np.float64('Nan')); #int16
	(sensor_azimuth,dimsizes_sensor_zenith,tempA,tempB) = get_hdf_data('Sensor_Azimuth',hdf,0,np.float64('Nan')); #int16
	#strange minima with solar and sensor ZA at the end of the swath
	
	(surface_temp,dimsizes_temp,tempA,tempB) = get_hdf_data('Surface_Temperature',hdf,0,np.float64('Nan'));
	
	
	##[BTD,dimsizes_temp]=get_hdf_data_dan('Brightness_Temperature_Difference',SD_id,INFO);
	##[BT,dimsizes_temp]=get_hdf_data_dan('Brightness_Temperature',SD_id,INFO);
	##[rad_var,dimsizes_temp]=get_hdf_data_dan('Radiance_Variance',SD_id,INFO); #for same bands as BT (see below)
	##Bands 29,31,32,33,34,35,36
	##Wavelength 8.4-8.7 (29), 10.8-11.3 (31), 11.8-12.3 (32), 13.2-13.5 (33),
	##13.5-13.8 (34) 13.8-14.1 (35), 14.1-14.4 (36) um
	#
	##convert to Matlab date time (days since 01-Jan-0000)
	#scantime_matlab = scantime/3600/24 + datenum('01-Jan-1993');
	##then can do datestr(scantime_matlab(ilat,ilon),31) to get the date string
	#
	
	#mask_5km,qapq_5km,qapp_5km] = flagread_C6_5km_Dan(mask5,qa5);
	#qapq_1km,qapp_1km] = flagread_1km_Dan(qa1,1);
	##qapp_1km is a continuation of the 1km QA as documented in the
	##MODIS_Quality_Assurance_etc....pdf
	##plan document. Starts at the 3rd byte ("Primary Cloud Retrieval Phase
	##Flag").	
	
# ------ Decode the integer bytes in the mask1 array to get the flags -----
	(mask_1km) = maskread_1km(mask1)

# -------------------------------------------------------------------------

	## N.B. C6 doesn't use the water path (etc.) QA confidences any more - they should
	## all be set to very good. The idea is to use pixel level uncertainties I
	## think.

### ----------- Some filtering of data --------------------------------------
	#
	## ind will be all the pixels that are NOT confident cloudy - will set re,
	## etc. to NaN for these.
	ind = (mask_1km[:,:,0] < 1) | (mask_1km[:,:,1] > 0)
	  #N.B. in Python for comparing arrays use & and | rather than "and" and "or"

	#    
        ind_not_liq = (cloud_phase != 2) #indices for all non-liquid pixels
	#
	#    #mask_1km(2,:,:) has changed cf Rob's script - the (2,:,:) here is actually a
	#    #(1,:,:) one in Rob's script due to the zero-based indices quoted there
	#    #(checked in the .pdf below, 15th Feb, 2012).
	#    # Also, 0 now means good! See
	#    #http://modis-atmos.gsfc.nasa.gov/_docs/QA_Plan_2011_01_26.pdf -
	# 
	##  mask_1km(3)  :  SUN GLINT FLAG  - so is actually mask_1km(4,:,:)
	##			0: yes# 1: no
	#

	sun_glint_flag = mask_1km[:,:,3] #0: yes# 1: no
	##Might also want to filter for these :-
	heavy_aerosol_flag = mask_1km[:,:,6] #0: yes# 1: no
	thin_cirrus_flag = mask_1km[:,:,7] #0: yes# 1: no
	shadow_flag = mask_1km[:,:,8] #0: yes# 1: no
	
	
	## Also, select only low cloud in all cases (0.5km < cth <3.2km)
	ind_dubious = (cth_1km < 500.0) | (cth_1km > 3200.0) | (sun_glint_flag != 1) |  (heavy_aerosol_flag!= 1) | (thin_cirrus_flag!= 1)  | (shadow_flag!= 1) #other likely dubious pixels
	ind_dubious_ignore_aerosol = (cth_1km < 500.0) | (cth_1km > 3200.0) | (sun_glint_flag !=1) | (thin_cirrus_flag!= 1)  | (shadow_flag!= 1) #other likely dubious pixels
	##might want to keep heavy aerosol pixels for e.g. CLARIFY project...
	ind_dubious_just_cirrus_shadow = (cth_1km < 500.0) | (cth_1km > 3200.0) | (thin_cirrus_flag!= 1)  | (shadow_flag!= 1) #other likely dubious pixels
	##also ignorning sunglint since it can take out large chunks of the swath

	    
	#    #  lwp(ind)=0.00
	#    #if are not confident cloudy then we NaN the cloud variables like re
	#    #and tau
	re16[ind]=np.float64('Nan');   #
	re21[ind]=np.float64('Nan'); 
	re37[ind]=np.float64('Nan'); 
	tau16[ind]=np.float64('Nan');   #
	tau21[ind]=np.float64('Nan'); 
	tau37[ind]=np.float64('Nan');     
	re16_un[ind]=np.float64('Nan');   #
	re21_un[ind]=np.float64('Nan'); 
	re37_un[ind]=np.float64('Nan'); 
	tau16_un[ind]=np.float64('Nan');   #
	tau21_un[ind]=np.float64('Nan'); 
	tau37_un[ind]=np.float64('Nan'); 
	       



	#
	#catch openL2_error
	#   clear override_flags_openL2 #reset the flag if there was an error
	#   rethrow(openL2_error); #re-issue the error
	#end
	#
	#clear override_flags_openL2 #reset the flag if no errors

	
	#filtering_data_L2_C6
	#

# ------------------   Calculate droplet conc (Nd) ---------------------


# --- Have to be careful here :- previously used tau_cons = tau, but then this creates a pointer so that any changes to tau_cons also change tau...

	#Create new tau arrays for Nd calc in order to only use tau>5
	tau16_Nd = np.copy(tau16); tau16_Nd[tau16<5.0] = np.float64('Nan');
	tau21_Nd = np.copy(tau21); tau21_Nd[tau21<5.0] = np.float64('Nan');
	tau37_Nd = np.copy(tau37); tau37_Nd[tau37<5.0] = np.float64('Nan');
	# can just do np.nan - prob has the same effect

	# ----------- Run the Nd function -----------	
	(N16,H16,W16,Cw) = getCDNC(tau16_Nd,re16*1e-6,t_top_1km) 
	(N21,H21,W21,Cw) = getCDNC(tau21_Nd,re21*1e-6,t_top_1km)
	(N37,H37,W37,Cw) = getCDNC(tau37_Nd,re37*1e-6,t_top_1km)
	#W=LWP, H=cloud depth, Cw= dqL/dT
	#----------------------------------------------------------

	#Make sure we are using liquid pixels only	
	(N16,H16,W16,re16,tau16) = copy_and_filter(ind_not_liq,N16,H16,W16,re16,tau16_Nd)
	(N21,H21,W21,re21,tau21) = copy_and_filter(ind_not_liq,N21,H21,W21,re21,tau21_Nd)
	(N37,H37,W37,re37,tau37) = copy_and_filter(ind_not_liq,N37,H37,W37,re37,tau37_Nd)

# --- Make some more conservative Nd arrays where also filter for sunglint, heavy aerosol, thin

	# --- Have to be careful here :- previously used N16_cons = N16, but then this creates a pointer so that any changes to N16_cons also change N16...
# So use np.copy instead

	#Filter for sunglint, heavy aerosol, thin cirrus and shadow_flag	
	(N16_cons,H16_cons,W16_cons,re16_cons,tau16_cons) = copy_and_filter(ind_dubious,N16,H16,W16,re16,tau16_Nd)
	(N21_cons,H21_cons,W21_cons,re21_cons,tau21_cons) = copy_and_filter(ind_dubious,N21,H21,W21,re21,tau21_Nd)
	(N37_cons,H37_cons,W37_cons,re37_cons,tau37_cons) = copy_and_filter(ind_dubious,N37,H37,W37,re37,tau37_Nd)

	#Some less conervative filterings :-
	#Just filter for sunglint, thin cirrus and shadow_flag
	(N16_cons_ignore_aerosol,H16_cons_ignore_aerosol,W16_cons_ignore_aerosol,re16_cons_ignore_aerosol,tau16_cons_ignore_aerosol) = copy_and_filter(ind_dubious_ignore_aerosol,N16,H16,W16,re16,tau16_Nd)
	(N21_cons_ignore_aerosol,H21_cons_ignore_aerosol,W21_cons_ignore_aerosol,re21_cons_ignore_aerosol,tau21_cons_ignore_aerosol) = copy_and_filter(ind_dubious_ignore_aerosol,N21,H21,W21,re21,tau21_Nd)
	(N37_cons_ignore_aerosol,H37_cons_ignore_aerosol,W37_cons_ignore_aerosol,re37_cons_ignore_aerosol,tau37_cons_ignore_aerosol) = copy_and_filter(ind_dubious_ignore_aerosol,N37,H37,W37,re37,tau37_Nd)

	
#	#Just filter for thin cirrus and shadow_flag
	(N16_cons_just_cirrus_shadow,H16_cons_just_cirrus_shadow,W16_cons_just_cirrus_shadow,re16_cons_just_cirrus_shadow,tau16_cons_just_cirrus_shadow) = copy_and_filter(ind_dubious_just_cirrus_shadow,N16,H16,W16,re16,tau16_Nd)
	(N21_cons_just_cirrus_shadow,H21_cons_just_cirrus_shadow,W21_cons_just_cirrus_shadow,re21_cons_just_cirrus_shadow,tau21_cons_just_cirrus_shadow) = copy_and_filter(ind_dubious_just_cirrus_shadow,N21,H21,W21,re21,tau21_Nd)
	(N37_cons_just_cirrus_shadow,H37_cons_just_cirrus_shadow,W37_cons_just_cirrus_shadow,re37_cons_just_cirrus_shadow,tau37_cons_just_cirrus_shadow) = copy_and_filter(ind_dubious_just_cirrus_shadow,N37,H37,W37,re37,tau37_Nd)


#	embed()

	print 'Done read L2 MODIS'
	


	return {'N16':N16_cons_just_cirrus_shadow,'N21':N21_cons_just_cirrus_shadow,'N37':N37_cons_just_cirrus_shadow, 'tau16':tau16_cons_just_cirrus_shadow,'tau21':tau21_cons_just_cirrus_shadow,'tau37':tau37_cons_just_cirrus_shadow,'re16':re16_cons_just_cirrus_shadow,'re21':re21_cons_just_cirrus_shadow,'re37':re37_cons_just_cirrus_shadow}


def copy_and_filter(ifilter,N,H,W,re,tau):
	N_out = np.copy(N); N_out[ifilter] = np.nan
	H_out = np.copy(H); H_out[ifilter] = np.nan
	W_out = np.copy(W); W_out[ifilter] = np.nan
	re_out = np.copy(re); re_out[ifilter] = np.nan
	tau_out = np.copy(tau); tau_out[ifilter] = np.nan

	return (N_out,H_out,W_out,re_out,tau_out)
