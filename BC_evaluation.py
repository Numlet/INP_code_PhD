# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:41:44 2015

@author: eejvt
"""
import numpy as np
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
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
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import csv
import random
import datetime
from scipy.interpolate import interp1d
def read_ncdf(file_name):
    mb=netcdf.netcdf_file(file_name,'r') 
    return mb


rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
rhocomp =rhocomp*1e+3#ug/cm3
'''
;           Sulphate    Carbon    1.4*BC   NaCl      SiO2,      SOC   Feldspar                               
    mm_aer = [0.098,    0.012,   0.0168 , 0.05844,    0.10,    0.0168 ,0.1] ;SOC was 0.15  ; molar masses in Kg/mol
    rhocomp  = [1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.];masss density of components in Kg/m3
'''
   
#%%
mb=netcdf.netcdf_file('/nfs/a107/ear3clsr/GASSP/Processed_data/A-FORCE/SP2_A-FORCE_flt6_60s_v2.dat.nc','r')
mb=netcdf.netcdf_file('/nfs/a107/ear3clsr/GASSP/Processed_data/EUCAARI/SP2_EUCAARI_B379_FAAM_MAN_20080521_V1.nc','r')
'/nfs/a107/ear3clsr/GASSP/Processed_data/INTEX-B/SP2_mrg60_c130_20060308_R5.ict.nc'
#/nfs/a107/ear3clsr/GASSP/Processed_data/A-FORCE/SP2_A-FORCE_flt6_60s_v2.dat.nc
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/TEXAQS2006/SP2_mrg60_NP3_20060911_R1.ict.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/CAST/SP2_CAST_B841_FAAM_MAN_20140214_V1.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/SEAC4RS/HDSP2_mrg60-dc8_merge_20130821_R3.ict.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/COPE/SP2_COPE_B792_FAAM_MAN_20130803_V1.nc'


file1='/nfs/a107/ear3clsr/GASSP/Processed_data/ACCACIA/SP2_ACCACIA_B768_FAAM_MAN_20130403_V1.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/HIPPO/SP2_HIPPO_M5_9_NOAA_20110829_V1.nc' 
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/ARCPAC2008/SP2_mrg60_NP3_20080421_R6.ict.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/ACCACIA/SP2_ACCACIA_B762_FAAM_MAN_20130323_V1.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/HIPPO/SP2_HIPPO_M3_10_NOAA_20100415_V1.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/A-FORCE/SP2_A-FORCE_flt5_60s_v2.dat.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/DISCOVERAQ/SP2_mrg60-p3b_merge_20110716_R3.ict.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/CALNEX/SP2_mrg60_NP3_20100521_R0.ict.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/CALNEX/SP2_CalNexLA_20100515_R3.ict.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/INTEX-B/SP2_mrg60_c130_20060508_R5.ict.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/MAMM/SP2_MAMM_B796_FAAM_MAN_20130816_V1.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/EUCAARI/SP2_EUCAARI_B379_FAAM_MAN_20080521_V1.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/BORTAS/SP2_BORTAS_B627_FAAM_MAN_20110727_V1.nc'
file1='/nfs/a107/ear3clsr/GASSP/Processed_data/TRACEP/BC_final-v4-mrg60d15.trp.nc'
mb=read_ncdf(file1)
mb.variables
mb.Error_Relative
mb.Instrument
mb.Cutoff_High_Diameter
mb.Cutoff_Low_Diameter
mb.Additional_Data_Info
#%%
s={}
s=readsav('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2/tot_mc_bc_mm_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_withICE_SCAV_2001.sav',idict=s)
#%%
file_list_file='/nfs/see-fs-02_users/libclsr/GASSP/source_code/working_code/Search_results_BCdata_for_Jesus.txt'
with open(file_list_file) as f:
    file_list = f.readlines()
    

class time_params():
    def __init__(self,var_name,starting_year,starting_month=1,starting_day=1):
        self.var_name=var_name
        self.starting_year=starting_year
        self.starting_month=starting_month
        self.starting_day=starting_day





class campaign():
    def __init__(self, name,lat_name,lon_name,pressure_name,time=0,file_list_path='/nfs/see-fs-02_users/libclsr/GASSP/source_code/working_code/Search_results_BCdata_for_Jesus.txt'):
        self.name = name
        with open(file_list_path) as f:
            file_list = f.readlines()
        new_list=[name_file for name_file in file_list if self.name in name_file]
        self.file_list=new_list
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.pressure_name = pressure_name
        self.time = time
    
    def set_color(self,color='#%02X%02X%02X'%(random.randint(0,255),random.randint(0,255),random.randint(0,255))):
        self.color = color
        
    
    def set_BC_mass_values(self,bc_mass_name,bc_mass_units,conversor_to_ugm3=1,mass_mixing=0,mass_mixing_conversor_to_ug_kg=1):
        self.bc_mass_name = bc_mass_name
        self.bc_mass_units = bc_mass_units
        self.conversor_to_ugm3 = conversor_to_ugm3
        self.mass_mixing= mass_mixing
        self.mass_mixing_conversor_to_ug_kg= mass_mixing_conversor_to_ug_kg
        
    def set_BC_number_values(self,bc_number_name,bc_number_units,conversor_to_cm3=1):
        self.bc_number_name = bc_number_name
        self.bc_number_units = bc_number_units
        self.conversor_to_cm3 = conversor_to_cm3
        
    def P_and_T(self,p_name,t_name):
        self.p_name = p_name
        self.t_name = t_name


class comparison_data():
    def __init__(self,modelled,observed,legend,model_name='none',pressures=[]):
        self.modelled=modelled
        self.observed=observed
        self.legend=legend
        self.model_name=model_name
        self.pressures=pressures
    
    def plot121(self):
        fig=plt.figure()
        plt.scatter(np.array(self.observed),np.array(self.modelled),label=self.legend)
        plt.ylim(0.1*np.array(self.modelled).min(),10*np.array(self.modelled).max())
        plt.xlim(0.1*np.array(self.observed).min(),10*np.array(self.observed).max())
        x=np.logspace(-10,10,100)
        plt.plot(x,x,'k-')
        plt.ylabel('Modelled')
        plt.xlabel('Observed')
        plt.plot(x,x*10,'k--')
        plt.plot(x,x/10,'k--')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        #plt.show()
        plt.savefig('/nfs/see-fs-01_users/eejvt/BC_evaluation/'+self.legend+'_'+self.model_name+'.png',format='png')
        plt.close()
        print 'ploted'

class observed_data():
    def __init__(self,lats=0,lons=0,press=0,month=0,values=0):
        self.lats=lats
        self.lons=lons
        self.press=press
        self.month=month
        self.values=values
        
class modelled_data():
    def __init__(self,lats=0,lons=0,press=0,values=0):
        self.lats=lats
        self.lons=lons
        self.press=press
        self.values=values


    
#%%
'''
jl.convert_mass_mixing_ratio
t=datetime.datetime.fromtimestamp(0)#datetime object with the values of the epoch

diff_sec=(t-datetime.datetime(1904,1,1)).total_seconds()
time_arr=mb.variables['TIME_UTC'][:]-diff_sec
len_time=len(time_arr)
time_arr_struct=np.empty(len_time)
convert_time_vectorized=np.vectorize(datetime.datetime.fromtimestamp)

time_arr_struct=convert_time_vectorized(time_arr[:,])
'''
#%%
campaign_dict={}


TRACEP=campaign('TRACEP','LATITUDE','LONGITUDE','PRESSURE',time_params('UTC',1970))
TRACEP.set_BC_mass_values('Equivalent_BC_mass','ug m-3',1)
TRACEP.set_color('saddlebrown')
campaign_dict[TRACEP.name]=TRACEP

COPE=campaign('COPE','LAT','LON','PRES',time_params('TIME_UTC',1904))
COPE.set_BC_mass_values('INCAND_MASS','ng m-3',1e-3)
COPE.set_color('greenyellow')
campaign_dict[COPE.name]=COPE


BORTAS=campaign('BORTAS','LAT','LON','PRES',time_params('TIME_UTC',1904))
BORTAS.set_BC_mass_values('INCAND_MASS','ng m-3',1e-3)
BORTAS.set_color('lime')
campaign_dict[BORTAS.name]=BORTAS


A_FORCE=campaign('A-FORCE','LATITUDE','LONGITUDE','PRESSURE',time_params('TIME',2009))
A_FORCE.set_BC_mass_values('BCMCSTP','ng m-3',1e-3)
A_FORCE.set_color('coral')
campaign_dict[A_FORCE.name]=A_FORCE

HIPPO=campaign('HIPPO','LAT','LON','PRES',time_params('TIME',1904))
HIPPO.set_BC_mass_values('none','ng m-3',1,'BC_M',1e-3)
HIPPO.P_and_T('PRES','TEMP')
HIPPO.set_color('navy')
campaign_dict[HIPPO.name]=HIPPO

ACCACIA=campaign('ACCACIA','LAT','LON','PRES',time_params('TIME_UTC',1904))
ACCACIA.set_BC_mass_values('INCAND_MASS','ng m-3',1e-3)
ACCACIA.set_color('b')
CALNEX=campaign('CALNEX','GpsLat','GpsLon','StaticPrs',time_params('UTC_mid',2010,5,21))
CALNEX.set_BC_mass_values('BC_ng_m3','ng_m3',1e-3)
CALNEX.set_color('r')

INTEX_B=campaign('INTEX-B','LATITUDE','LONGITUDE','PRESSURE',time_params('UTC',2006,5,8))
INTEX_B.set_BC_mass_values('massInd','ng_m3',1e-3)#Not sure Parece que no hay datos de esta campa;a, solo ncdf files con missing values
INTEX_B.set_color('g')

DISCOVERAQ=campaign('DISCOVERAQ','LATITUDE','LONGITUDE','PRESSURE',time_params('UTC',2011,7,16))
DISCOVERAQ.set_BC_mass_values('BlackCarbonMassConcentration','ng_m3',1e-3)
DISCOVERAQ.set_color('lightblue')

TEXAQS2006=campaign('TEXAQS2006','GpsLat','GpsLon','StaticPrs',time_params('UTC_mid',2006,9,11))
TEXAQS2006.set_BC_mass_values('BC_ng_m3','ng_m3',1e-3)
TEXAQS2006.set_color('y')
campaign_dict[TEXAQS2006.name]=TEXAQS2006
#campaign_dict[INTEX_B.name]=INTEX_B
campaign_dict[CALNEX.name]=CALNEX
campaign_dict[DISCOVERAQ.name]=DISCOVERAQ
campaign_dict[ACCACIA.name]=ACCACIA


MAMM=campaign('MAMM','LAT','LON','PRES',time_params('TIME_UTC',1904,1,1))
MAMM.set_BC_mass_values('INCAND_MASS','ng m-3',1e-3)#carefull with units
MAMM.set_color('orange')
campaign_dict[MAMM.name]=MAMM

CAST=campaign('CAST','LAT','LON','PRES',time_params('TIME_UTC',1904,1,1))
CAST.set_BC_mass_values('INCAND_MASS','ng m-3',1e-3)#carefull with units
CAST.set_color('pink')
campaign_dict[CAST.name]=CAST

EUCAARI=campaign('EUCAARI','LAT','LON','PRES',time_params('TIME',1904,1,1))
EUCAARI.set_BC_mass_values('INCAND_MASS','ug m-3',1e-3)#carefull with units
EUCAARI.set_color('darkgreen')
campaign_dict[EUCAARI.name]=EUCAARI

ARCPAC2008=campaign('ARCPAC2008','GpsLat','GpsLon','StaticPrs',time_params('UTC_mid',2008,4,21))
ARCPAC2008.set_BC_mass_values('BC_ng_m3','ng_m3',1e-3)
ARCPAC2008.set_color('m')
campaign_dict[ARCPAC2008.name]=ARCPAC2008

SEAC4RS=campaign('SEAC4RS','LATITUDE','LONGITUDE','PRESSURE',time_params('UTC',2013,8,21))
SEAC4RS.set_BC_mass_values('BC_mass_90to550nm_HDSP2','ng_m3',1e-3)
SEAC4RS.set_color('paleturquoise')
campaign_dict[SEAC4RS.name]=SEAC4RS



#%%/nfs/a107/ear3clsr/GASSP/Processed_data/EUCAARI/SP2_EUCAARI_B379_FAAM_MAN_20080521_V1.nc
camp=INTEX_B
camp=CALNEX
camp=DISCOVERAQ
camp=ACCACIA

#%%
def get_months(camp,mb):
    t=datetime.datetime.fromtimestamp(0)
    diff_sec=(t-datetime.datetime(camp.time.starting_year,camp.time.starting_month,camp.time.starting_day)).total_seconds()
    time_arr=mb.variables[camp.time.var_name][:]-diff_sec
    convert_time_vectorized=np.vectorize(datetime.datetime.fromtimestamp)
    
    time_arr_struct=convert_time_vectorized(time_arr[:,])
    months=[time_arr_struct[i].month for i in range(len(time_arr_struct))]
    return months



#%%


s=jl.read_data('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2')
#%%
BC_modes=[1,2,3,4]
points=20
cut_low=0.090*1e-6
cut_up=0.5*1e-6
rs=np.linspace(cut_low,cut_up,points)
step=(cut_up-cut_low)/points
def lognormal_PDF(rmean,r,std):
   X=(1/(r*np.log(std)*np.sqrt(2*np.pi)))*np.exp(-(np.log(r)-np.log(rmean))**2/(2*np.log(std)**2))
   return X
pdf_frac=np.sum(lognormal_PDF(0.5,rs,1.5)*step)

def volumes_of_modes(s):
    modes_vol=(s.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]
    +s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2]
    +s.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
    +s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]
    +s.tot_mc_oc_mm_mode[:,:,:,:,:]/rhocomp[5]
    +s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])
    return modes_vol

modes_vol=volumes_of_modes(s)
BC_volfrac=(s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2])/modes_vol
rf=(BC_volfrac*s.rbardry**3)**(1/3.)
std=s.sigma
PDF_fraction=np.zeros(rf.shape)
for i in range(len(rf[:,0,0,0,0])):
    print i
    if rf[i,0,0,0,0]==0:
        print rf[i,0,0,0,0],i,'jumped'
        continue
    for ilev in range(len(rf[0,:,0,0,0])):
        print 'ilev',ilev
        for ilat in range(len(rf[0,0,:,0,0])):
            for ilon in range(len(rf[0,0,0,:,0])):
                for imon in range(len(rf[0,0,0,0,:])):
                    PDF_fraction[i,ilev,ilat,ilon,imon]=np.sum(lognormal_PDF(rf[i,ilev,ilat,ilon,imon],rs,std[i])*step)


#%%
model_data10=modelled_data()
model_data10.model_name='GLOMAP 10 modes'
model_data10.lons=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav').glon
model_data10.lats=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav').glat
model_data10.values=readsav('/nfs/a173/earjbr/mode10_setup/tot_mc_bc_mm_mode10_2001.sav').tot_mc_bc_mm
model_data10.press=readsav('/nfs/a173/earjbr/mode10_setup/GLOMAP_mode_pressure_mp_mode10_2001.sav').pl_m*1e-2
model_data7=modelled_data()
model_data7.model_name='GLOMAP 7 modes'
model_data7.lons=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav').glon
model_data7.lats=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav').glat
model_data7.values=readsav('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2/tot_mc_bc_mm_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_withICE_SCAV_2001.sav').tot_mc_bc_mm*PDF_fraction
model_data7.values=(readsav('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2/tot_mc_bc_mm_mode_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_withICE_SCAV_2001.sav').tot_mc_bc_mm_mode*PDF_fraction).sum(axis=0)
model_data7.press=readsav('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2/GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_withICE_SCAV_2001.sav').pl_m*1e-2
#%%
model_data=model_data10
model_data=model_data7
import random
r = lambda: random.randint(0,255)
file_name='/nfs/a107/ear3clsr/GASSP/Processed_data/ACCACIA/SP2_ACCACIA_B768_FAAM_MAN_20130403_V1.nc'
fig=plt.figure(2,figsize=(15,15))
file_name='/nfs/a107/ear3clsr/GASSP/Processed_data/CALNEX/SP2_mrg60_NP3_20100521_R0.ict.nc'
all_observed_values=np.array([])
all_modelled_values=np.array([])
all_pressures=np.array([])
thiscamp_observed_values=np.array([])
thiscamp_modelled_values=np.array([])
thiscamp_pressures=np.array([])
for camp in campaign_dict.itervalues():
    print camp.name
    for file_name in camp.file_list:
        print file_name
        mb=read_ncdf(file_name[:-1])
        if not camp.bc_mass_name in mb.variables:
            if isinstance(camp.mass_mixing,int):
                #print file_name
                print 'Has not the same structure as the whole campaign'
                print 'Jumped'        
                continue
            
        if not camp.pressure_name in mb.variables:
            #print file_name
            print 'Has not the same structure as the whole campaign'
            print 'Jumped'
            print 'Probably not aircraft'        
            continue
        
        camp_data=observed_data()
        camp_data.lons=(mb.variables[camp.lon_name][:])
        camp_data.lats=(mb.variables[camp.lat_name][:])
        camp_data.press=(mb.variables[camp.pressure_name][:])
        if isinstance(camp.mass_mixing,int):
            valids=((mb.variables[camp.bc_mass_name][:]!=mb.variables[camp.bc_mass_name].missing_value) & \
            (mb.variables[camp.pressure_name][:]!=mb.variables[camp.pressure_name].missing_value) & \
            (mb.variables[camp.lon_name][:]!=mb.variables[camp.lon_name].missing_value) & \
            (mb.variables[camp.lat_name][:]!=mb.variables[camp.lat_name].missing_value) &
            (mb.variables[camp.bc_mass_name][:]>1e-10)&
            (mb.variables[camp.bc_mass_name][:]<1e10)&
            (mb.variables[camp.lon_name][:]<360))
            camp_data.values=mb.variables[camp.bc_mass_name][valids]*camp.conversor_to_ugm3
        else:
            valids=((mb.variables[camp.mass_mixing][:]!=mb.variables[camp.mass_mixing].missing_value) & \
            (mb.variables[camp.pressure_name][:]!=mb.variables[camp.pressure_name].missing_value) & \
            (mb.variables[camp.lon_name][:]!=mb.variables[camp.lon_name].missing_value) & \
            (mb.variables[camp.lat_name][:]!=mb.variables[camp.lat_name].missing_value) &
            (mb.variables[camp.mass_mixing][:]>1e-10)&
            (mb.variables[camp.mass_mixing][:]<1e10)&
            (mb.variables[camp.lon_name][:]<360))
            camp_data.values=jl.convert_mass_mixing_ratio(mb.variables[camp.mass_mixing][:]*camp.mass_mixing_conversor_to_ug_kg,mb.variables[camp.pressure_name][:],mb.variables[camp.t_name][:])[valids]
    
    
        
        #camp_data.values=mb.variables[camp.bc_mass_name][valids]*camp.conversor_to_ugm3
        camp_data.lons=(mb.variables[camp.lon_name][valids])
        camp_data.lats=(mb.variables[camp.lat_name][valids])
        camp_data.press=(mb.variables[camp.pressure_name][valids])
        camp_data.lons[camp_data.lons<0]=camp_data.lons[camp_data.lons<0]+360
    
        months=get_months(camp,mb)
        months_arg=np.array(months)-1


   
        
        
        lat_arg=[int(jl.find_nearest_vector_index(model_data.lats, lat)) for lat in camp_data.lats]
        lon_arg=[int(jl.find_nearest_vector_index(model_data.lons, lon)) for lon in camp_data.lons]
        press_arg=[int(jl.find_nearest_vector_index(model_data.press[:,lat_arg[i],lon_arg[i],months_arg[i]], camp_data.press[i])) for i in range(len(camp_data.press[:]))]
        model_values=[]        
        pressures=[]        
        for i in range(len(press_arg)):
            f = interp1d(model_data.press[:,lat_arg[i],lon_arg[i],months_arg[i]],model_data.values[:,lat_arg[i],lon_arg[i],months_arg[i]])
            f_lat = interp1d(model_data.lats[:],model_data.values[press_arg[i],:,lon_arg[i],months_arg[i]])
            f_lon = interp1d(model_data.lons[:],model_data.values[press_arg[i],lat_arg[i],:,months_arg[i]])
            if camp_data.press[i]>model_data.press[-1,lat_arg[i],lon_arg[i],months_arg[i]]:
                value_p=f(model_data.press[-1,lat_arg[i],lon_arg[i],months_arg[i]])      
                value_lat=f_lat(camp_data.lats[i])
                value_lon=f_lon(camp_data.lons[i])
                value_final=np.mean([value_p,value_lat,value_lon])
                model_values.append(value_final)
                pressures.append(camp_data.press[i])
            else:
                value_p=f(model_data.press[-1,lat_arg[i],lon_arg[i],months_arg[i]])      
                value_lat=f_lat(camp_data.lats[i])
                value_lon=f_lon(camp_data.lons[i]) 
                value_final=np.mean([value_p,value_lat,value_lon])
                model_values.append(value_final)
                pressures.append(camp_data.press[i])
        '''
        model_values_next=[model_data.values[press_arg[i],lat_arg[i],lon_arg[i],months_arg[i]] for i in range(len(press_arg))]
        model_values_second_next=[model_data.values[press_arg_second[i],lat_arg[i],lon_arg[i],months_arg[i]] for i in range(len(press_arg))]
        dividend=np.abs([model_data.press[press_arg[i],lat_arg[i],lon_arg[i],months_arg[i]]-model_data.press[press_arg_second[i],lat_arg[i],lon_arg[i],months_arg[i]] for i in range(len(press_arg))] )
        factor1=np.abs([model_data.press[press_arg[i],lat_arg[i],lon_arg[i],months_arg[i]]-camp_data.press[i] for i in range(len(press_arg))])
        factor2=np.abs([model_data.press[press_arg_second[i],lat_arg[i],lon_arg[i],months_arg[i]]-camp_data.press[i] for i in range(len(press_arg))])
        model_values=model_values_next*factor1/dividend+model_values_second_next*factor2/dividend
        model_values=model_values.tolist()
        if np.any((factor2/dividend+factor1/dividend)>1):
            print factor2/dividend+factor1/dividend, 'ERRORR'
            break
        if np.any(factor2/dividend>factor1/dividend):
            print factor2/dividend,factor1/dividend, 'ERRORR second closest than first'
            break
        '''        
        
        comp_data=comparison_data([],[],camp.name)
        

            
            
            
        comp_data.modelled.append(model_values)
        comp_data.pressures.append(pressures)
        comp_data.observed.append(camp_data.values)

        all_modelled_values=np.concatenate((all_modelled_values,np.array(model_values)))     
        thiscamp_modelled_values=np.concatenate((thiscamp_modelled_values,np.array(model_values)))     
        

        all_observed_values=np.concatenate((all_observed_values,np.array(camp_data.values)))     
        thiscamp_observed_values=np.concatenate((thiscamp_observed_values,np.array(camp_data.values)))     

        all_pressures=np.concatenate((all_pressures,np.array(camp_data.press)))
        thiscamp_pressures=np.concatenate((thiscamp_pressures,np.array(camp_data.press)))
        
    rmse=jl.RMS_err(thiscamp_observed_values,thiscamp_modelled_values)
    nmb=jl.NMB(thiscamp_observed_values,thiscamp_modelled_values)
    r=np.corrcoef(thiscamp_observed_values,thiscamp_modelled_values)[0,1]
    
    with open('/nfs/see-fs-01_users/eejvt/BC_evaluation/'+camp.name+'_statistics.csv', 'w') as csvfile:
        fieldnames = ['NAME','R', 'RMSE','NMB']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'NAME': camp.name, 'R': r,'RMSE':rmse,'NMB':nmb})
    #comp_data.plot121()


    plt.figure(5)
    plt.plot(thiscamp_observed_values,thiscamp_pressures,'bo',label='observed')
    plt.plot(thiscamp_modelled_values,thiscamp_pressures,'ro',label='modelled')
    plt.xscale('log')
    plt.gca().invert_yaxis()
    plt.legend(loc='best')
    plt.title(camp.name)
    plt.savefig('/nfs/see-fs-01_users/eejvt/BC_evaluation/pressure_plot_'+camp.name+'_'+model_data.model_name+'.png',format='png')
    plt.close()

    thiscamp_observed_values=np.array([])
    thiscamp_modelled_values=np.array([])
    thiscamp_pressures=np.array([])
    plt.figure(2)
    plt.scatter(comp_data.observed,comp_data.modelled,label=comp_data.legend,c=camp.color,edgecolors='none')
rmse=jl.RMS_err(all_observed_values,all_modelled_values)
nmb=jl.NMB(all_observed_values,all_modelled_values)
r=np.corrcoef(all_observed_values,all_modelled_values)[0,1]
plt.ylim(1e-4,1e1)
plt.xlim(1e-4,1e1)
x=np.logspace(-10,10,100)
plt.plot(x,x,'k-')
plt.ylabel('Modelled')
plt.xlabel('Observed')
plt.plot(x,x*10,'k--')
plt.plot(x,x/10,'k--')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.title('R=%1.3f RMSE=%1.3f NMB=%1.3f'%(r,rmse,nmb))
plt.savefig('/nfs/see-fs-01_users/eejvt/BC_evaluation/ALL_CAMPS_'+model_data.model_name+'.png',format='png')
plt.show()
#%%






#%%
def plot_campaign_trajectories(camp,same_fig=0,fig=0):
    if not same_fig:
        fig=plt.figure(figsize=(20, 20))
    m = fig.add_subplot(1,1,1)
    m = Basemap(projection='cyl',lon_0=0)
    m.drawcoastlines()
    m.drawcountries()
    #m.bluemarble()  
    #m.drawmapboundary(fill_color='#99ffff')
    #m.fillcontinents(color='#cc9966',lake_color='#99ffff')
        
    #m = Basemap(resolution='c',projection='ortho',lat_0=60.,lon_0=-60.)
    lon_name=camp.lon_name
    lat_name=camp.lat_name
    for file_name in file_list:
        if camp.name in file_name:
            print file_name
            mb=read_ncdf(file_name[:-1])
            if not camp.bc_mass_name in mb.variables:
                if not camp.mass_mixing in mb.variables:
                    #print file_name
                    print 'Has not the same structure as the whole campaign'
                    print 'Jumped'        
                    continue
            lons=(mb.variables[lon_name][:])
            lats=(mb.variables[lat_name][:])
            if np.any((mb.variables[lon_name][:]<-180)):
                lons=np.copy(mb.variables[lon_name][:])
                lons[lons<-180]=lons[lons<-180]+360
                #lons=lons+360                 
            if np.any((mb.variables[lon_name][:]>180)):
                lons=np.copy(mb.variables[lon_name][:])
                lons[lons>180]=lons[lons>180]-360
                #lons=lons+360                 
                
            valid=[(lons[:]>-180)&(lons[:]<180)&(lats[:]>-90) & (lats[:]<90)]
            #xx,yy=lons[valid],lats[valid]
            xx,yy = m(lons[valid],lats[valid])
            print xx.max(),xx.min(),yy.max(),yy.min(),xx.shape,yy.shape
            print any(np.isnan(xx)),any(np.isnan(xx))
            m.scatter(xx,yy,c=camp.color,edgecolors='none')#,label=camp.name)
            #m.plot(xx,yy,linewidth=1.5,marker='.', markerfacecolor=camp.color,markeredgecolor=None)
    #xx,yy=np.linspace(-45,45,100),np.linspace(-45,45,100)
    #m.drawparallels(np.arange(10,70,20),labels=[1,1,0,0])
    #m.drawmeridians(np.arange(-100,0,20),labels=[0,0,0,1])
    if not same_fig:
        plt.savefig('/nfs/see-fs-01_users/eejvt/BC_evaluation/LOCATION_'+camp.name+'.png',format='png')
        plt.close()    
    #plt.show()
#%%#
fig=plt.figure(figsize=(20, 20))
for camp in campaign_dict.itervalues():
    plot_campaign_trajectories(camp,same_fig=1,fig=fig)#)
plt.legend(loc='best')
plt.savefig('/nfs/see-fs-01_users/eejvt/BC_evaluation/ALL_LOCATIONS'+'.png',format='png')
#plot_campaign_trajectories(HIPPO)
#plot_campaign_trajectories(ACCACIA)
#plot_campaign_trajectories(CALNEX)
#plot_campaign_trajectories(DISCOVERAQ)



#%%

    

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
model_data=modelled_data()
model_data.lons=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav').glon
model_data.lats=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav').glat
model_data.values=readsav('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2/tot_mc_bc_mm_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_withICE_SCAV_2001.sav').tot_mc_bc_mm
model_data.press=readsav('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV2/GLOMAP_mode_pressure_mp_chained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_withICE_SCAV_2001.sav').pl_m*1e-2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X,Y=np.meshgrid(model_data.lons, model_data.lats)
Lons=np.zeros((31,64,128)) 
Lats=np.zeros((31,64,128))
for i in range(31):
    Lons[i,:,:]=X
    Lats[i,:,:]=Y
#ax.scatter(Lons,Lats, model_data.press.mean(axis=-1), c='b')

from scipy.interpolate import RegularGridInterpolator
my_interpolating_function = RegularGridInterpolator((Lons,Lats,model_data.press.mean(axis=-1)),model_data.values.mean(axis=-1))




 grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
from scipy.interpolate import griddata

point=np.array(([800,60,70],[800,60,75]))
grid_z1 = griddata( model_data.values.mean(axis=-1),point, (Lons,Lats,model_data.press.mean(axis=-1)), method='linear')



#%%


def plot_one_campaign_trajectories(camp):
    fig=plt.figure(figsize=(20, 20))
    m = fig.add_subplot(1,1,1)
    m = Basemap(projection='cyl',lon_0=0)
    m.drawcoastlines()
    m.drawcountries()
    m.bluemarble()  
    #m.drawmapboundary(fill_color='#99ffff')
    #m.fillcontinents(color='#cc9966',lake_color='#99ffff')
        
    #m = Basemap(resolution='c',projection='ortho',lat_0=60.,lon_0=-60.)
    lon_name=camp.lon_name
    lat_name=camp.lat_name
    for file_name in file_list:
        if camp.name in file_name:
            print file_name
            mb=read_ncdf(file_name[:-1])
            if not camp.bc_mass_name in mb.variables:
                if not camp.mass_mixing in mb.variables:
                    #print file_name
                    print 'Has not the same structure as the whole campaign'
                    print 'Jumped'        
                    continue
            lons=(mb.variables[lon_name][:])
            lats=(mb.variables[lat_name][:])
            if np.any((mb.variables[lon_name][:]<-180)):
                lons=np.copy(mb.variables[lon_name][:])
                lons[lons<-180]=lons[lons<-180]+360
                #lons=lons+360                 
            if np.any((mb.variables[lon_name][:]>180)):
                lons=np.copy(mb.variables[lon_name][:])
                lons[lons>180]=lons[lons>180]-360
                #lons=lons+360                 
                
            valid=[(lons[:]>-180)&(lons[:]<180)&(lats[:]>-90) & (lats[:]<90)]
            #xx,yy=lons[valid],lats[valid]
            xx,yy = m(lons[valid],lats[valid])
            print xx.max(),xx.min(),yy.max(),yy.min(),xx.shape,yy.shape
            print any(np.isnan(xx)),any(np.isnan(xx))
            m.scatter(xx,yy,c=camp.color,edgecolors='none')#,label=camp.name)
            #m.plot(xx,yy,linewidth=1.5,marker='.', markerfacecolor=camp.color,markeredgecolor=None)
    #xx,yy=np.linspace(-45,45,100),np.linspace(-45,45,100)
    #m.drawparallels(np.arange(10,70,20),labels=[1,1,0,0])
    #m.drawmeridians(np.arange(-100,0,20),labels=[0,0,0,1])
    plt.show()


















#%%






fig=plt.figure(figsize=(20, 20))
m = fig.add_subplot(1,1,1)
m = Basemap(projection='moll',lon_0=0)
xx,yy=np.linspace(-0.0014600000000086766,0.0028,100),np.linspace(67.769302,80.342003,100)
m.plot(xx,yy,linewidth=1.5,latlon=1)
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='#99ffff')
plt.show()




#%%

'''
camp=ACCACIA
fig=plt.figure(figsize=(20, 20))
m = fig.add_subplot(1,1,1)
m = Basemap(projection='cyl',lon_0=180)
#m = Basemap(resolution='c',projection='ortho',lat_0=60.,lon_0=-60.)
lon_name=camp.lon_name
lat_name=camp.lat_name
file_name='/nfs/a107/ear3clsr/GASSP/Processed_data/ACCACIA/SP2_ACCACIA_B768_FAAM_MAN_20130403_V1.nc'
if camp.name in file_name:
    print file_name
    mb=read_ncdf(file_name[:])
    lons=(mb.variables[lon_name][:])
    lats=(mb.variables[lat_name][:])
    if np.any((mb.variables[lon_name][:]<-180)):
        lons=np.copy(mb.variables[lon_name][:])
        lons[lons<-180]=lons[lons<-180]+360                 
        
    valid=[(lons[:]>-180)&(lons[:]<360)&(lats[:]>-90) & (lats[:]<90)]
    xx,yy=lons[valid],lats[valid]
    xx,yy = m(lons[valid],lats[valid])
    
    print xx.max(),xx.min(),yy.max(),yy.min(),xx.shape,yy.shape
    print any(np.isnan(xx)),any(np.isnan(xx))
    m.plot(xx,yy,linewidth=1.5)
#xx,yy=np.linspace(-45,45,100),np.linspace(-45,45,100)
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966',lake_color='#99ffff')
#m.drawparallels(np.arange(10,70,20),labels=[1,1,0,0])
#m.drawmeridians(np.arange(-100,0,20),labels=[0,0,0,1])
plt.show()

#%%
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
ax.plot(xx,yy,linewidth=1.5)


def plot_campaign_trajectories(camp):
    
    fig=plt.figure(figsize=(20, 20))
    #m = fig.add_subplot(1,1,1)
    ax = plt.axes(projection=ccrs.PlateCarree())

    #m = Basemap(resolution='c',projection='ortho',lat_0=60.,lon_0=-60.)
    lon_name=camp.lon_name
    lat_name=camp.lat_name
    for file_name in file_list:
        if camp.name in file_name:
            print file_name
            mb=read_ncdf(file_name[:-1])
            lons=(mb.variables[lon_name][:])
            lats=(mb.variables[lat_name][:])
            if np.any((mb.variables[lon_name][:]<-180)):
                lons=np.copy(mb.variables[lon_name][:])
                lons[lons<-180]=lons[lons<-180]+360
                #lons=lons+360                 
                
            valid=[(lons[:]>-360)&(lons[:]<360)&(lats[:]>-90) & (lats[:]<90)]
            xx,yy=lons[valid],lats[valid]
            xx,yy = m(lons[valid],lats[valid])
            print xx.max(),xx.min(),yy.max(),yy.min(),xx.shape,yy.shape
            print any(np.isnan(xx)),any(np.isnan(xx))
            ax.plot(xx,yy,linewidth=1.5)
    #xx,yy=np.linspace(-45,45,100),np.linspace(-45,45,100)
    ax.stock_img()
    #m.drawparallels(np.arange(10,70,20),labels=[1,1,0,0])
    #m.drawmeridians(np.arange(-100,0,20),labels=[0,0,0,1])
    plt.show()

plot_campaign_trajectories(A_FORCE)#)
plot_campaign_trajectories(HIPPO)
plot_campaign_trajectories(ACCACIA)


'''
