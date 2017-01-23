
import numpy as np
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
import Jesuslib as jl
import os
from glob import glob
import matplotlib.pyplot as plt
import scipy as sc
from glob import glob
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
from scipy.io.idl import readsav
reload(jl)
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
saving_folder='/nfs/see-fs-01_users/eejvt/terrestial_marine/'
'''
Plots by campaing
'''
t_limit=-26
header=1
class campaing():
    def __init__(self, name, location, m_or_t, color,name_in_file):
        data_marine_str=np.genfromtxt("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",delimiter="\t",skip_header=1,dtype=str)
        data_terrestrial_str=np.genfromtxt("/nfs/a107/eejvt/INP_DATA/TERRESTIAL_INFLUENCED.dat",delimiter="\t",skip_header=1,dtype=str)
        data_marine=np.genfromtxt("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",delimiter="\t",skip_header=1)#,dtype=str)
        data_terrestrial=np.genfromtxt("/nfs/a107/eejvt/INP_DATA/TERRESTIAL_INFLUENCED.dat",delimiter="\t",skip_header=1)#,dtype=str)
        self.name=name
        self.location=location
        self.m_or_t=m_or_t
        self.color=color
        self.name_in_file=name_in_file
        if m_or_t=='m':
            data=np.copy(data_marine)
            data_str=np.copy(data_marine_str)
        else:
            data=np.copy(data_terrestrial)
            data_str=np.copy(data_terrestrial_str)
        self.values=data_str[:,0]==self.name_in_file
        self.data=data[self.values]
        #temps_in_range=[data[:,1]>t_limit]
        #data=data[temps_in_range and self.values,:]
        self.temperatures=self.data[:,1]
        self.points=len(self.temperatures)
        self.pressures=self.data[:,5]
        if np.any(self.pressures<600):
            self.altitude='Aircraft'
        else:
            self.altitude='Surface'
            
header=1

data_marine=np.genfromtxt("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",delimiter="\t",skip_header=header,dtype=str)
data_terrestrial=np.genfromtxt("/nfs/a107/eejvt/INP_DATA/TERRESTIAL_INFLUENCED.dat",delimiter="\t",skip_header=header,dtype=str)

#for data_point in data_marine:
#    print data_point[0]
campaigns_dict={}
campaigns_dict['bigg 73']=campaing('Bigg1973 marine','Southern Ocean','m','y','bigg 73')
campaigns_dict['DeMott2016']=campaing('DeMott2016','Various marine locations','m','b','DeMott2015')
campaigns_dict['R. H. Mason Marine']=campaing('Mason2016 Marine','Various marine locations','m','k','R. H. Mason')
campaigns_dict['rosisnky gulf']=campaing('Rosisnky','Gulf of Mexico','m','darkblue','rosisnky gulf')


#campaigns_dict['INSPECT-I']=campaing('INSPECT-I','check','t','darkorchid','INSPECT-I')
campaigns_dict['INSPECT-II']=campaing('INSPECT-II','check','t','darksalmon','INSPECT-II')
campaigns_dict['AMAZE-08']=campaing('AMAZE-08','check','t','green','AMAZE-08')
campaigns_dict['WISP94']=campaing('WISP94','check','t','red','WISP94')
campaigns_dict['ICE-L Ambient']=campaing('ICE-L Ambient','check','t','orange','ICE-L Ambient')
campaigns_dict['ICE-L CVI']=campaing('ICE-L CVI','check','t','peru','ICE-L CVI')
campaigns_dict['Bigg73']=campaing('Bigg1973 terrestrial','check','t','brown','Bigg73')
campaigns_dict['CLEX']=campaing('CLEX','check','t','violet','CLEX')
campaigns_dict['Yin']=campaing('Yin','check','t','navy','Yin')
campaigns_dict['R. H. Mason']=campaing('Mason2016 terrestrial   ','Various terestrial locations','t','wheat','R. H. Mason')
campaigns_dict['Conen_JFJ']=campaing('Conen_JFJ','Joungfraiof (correct name)','t','grey','Conen_JFJ')
campaigns_dict['Conen_chaumont']=campaing('Conen_chaumont','chaumont','t','darkviolet','Conen_chaumont')
campaigns_dict['KAD_South_Pole']=campaing('KAD_South_Pole','South Pole','t','lime','KAD_South_Pole')
campaigns_dict['KAD_Israel']=campaing('KAD_Israel','Jerusalem','t','lightblue','KAD_Israel')

header=['.,Campaign, location, Marine (m) or Terrestrial(t), data points']
table=[]
table.append(header)
for camp in campaigns_dict.itervalues():
    line=[]
    line.append(camp.name)
    line.append(camp.location)
    line.append(camp.m_or_t)
    line.append(camp.points)
    table.append(line)
import csv

with open(saving_folder+'Campaing_table.csv', "wb") as f:
    writer = csv.writer(f)
    writer.writerows(table)
#np.savetxt(saving_folder+'Campaing_table.csv',np.array(table))
#%%
fig=plt.figure(figsize=(20,15))
m = plt.subplot(1,1,1)
m = Basemap(projection='cyl',lon_0=0)
m.drawcoastlines()
m.drawcountries()
#m.bluemarble()
#m.drawmapboundary(fill_color='#99ffff')
#m.fillcontinents(color='#cc9966',lake_color='#99ffff')
import random
#m = Basemap(resolution='c',projection='ortho',lat_0=60.,lon_0=-60.)
for camp_key in campaigns_dict.iterkeys():
    marine=0
    if campaigns_dict[camp_key].m_or_t =='m':
        data=data_marine
        marine=1
    else:
        data=data_terrestrial
    factor=0
    if not 'bigg' in camp_key:
        factor=0
    if 'ICE-L Ambient' in camp_key:
        factor=factor+2
    if 'WISP94' in camp_key:
        factor=factor-4
    xx=[float(lon) for lon in data[campaigns_dict[camp_key].values,4]]
    yy=[float(lat)+factor for lat in data[campaigns_dict[camp_key].values,3]]
    print xx
    marker='o'
    if marine:
        marker='^'
    m.scatter(xx,yy,c=campaigns_dict[camp_key].color,edgecolors='none',label=campaigns_dict[camp_key].name,s=150,marker=marker,alpha=0.8)
#box = m.get_position()
#m.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.legend()
plt.title('a)')
lgd =plt.legend( loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.savefig(saving_folder+'map_campaings.png', dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
#%%
INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')#m3
INP_feldext=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e6 #m3
INP_total=INP_marine_alltemps+INP_feldext
INP_total_year_mean=INP_total.mean(axis=-1)*1e-6#cm-3
meyers=np.load('/nfs/a201/eejvt/meyers.npy')
demott=np.load('/nfs/a201/eejvt/demott.npy')

class INP_param():
    def __init__(self,title, simulated_values,errors=0,simulated_values_max=0,simulated_values_min=0):
        self.title=title
        self.simulated_values= simulated_values
        self.errors=errors
        self.simulated_values_max=simulated_values_max
        self.simulated_values_min=simulated_values_min


errors=0#
simulated_values=demott.mean(axis=-1)
simulated_values_max=demott.max(axis=-1)
simulated_values_min=demott.min(axis=-1)
errors=1#
title='DeMott'
param=INP_param(title,simulated_values,errors,simulated_values_max,simulated_values_min)
errors=0#
simulated_values=INP_total_year_mean
simulated_values_max=INP_total.max(axis=-1)*1e-6
simulated_values_min=INP_total.min(axis=-1)*1e-6
errors=1#
title='Marine+Feldspar'
param=INP_param(title,simulated_values,errors,simulated_values_max,simulated_values_min)



INP_obs_mason=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/MARINE_INFLUENCED.dat",header=1)
INP_obs=jl.read_INP_data("/nfs/a107/eejvt/INP_DATA/TERRESTIAL_INFLUENCED.dat",header=1)

#fig=plt.figure()
fig=plt.figure(figsize=(6.5, 6))
for camp_key in campaigns_dict.iterkeys():
    marine=0
    if campaigns_dict[camp_key].m_or_t =='m':
        data=INP_obs_mason[campaigns_dict[camp_key].values]
        marine=1
    else:
        data=INP_obs[campaigns_dict[camp_key].values]
    data=data[data[:,1]>t_limit]
    cmap=plt.cm.RdBu_r
    if marine:
        marker='^'
        marker_size=120
    else:
        marker='o'
        marker_size=50
    
    simulated_points=jl.obtain_points_from_data(param.simulated_values,data)#,surface_level_comparison_on=True)
    if param.errors:    
        simulated_points_max=jl.obtain_points_from_data(param.simulated_values_max,data)#,surface_level_comparison_on=True)
        simulated_points_min=jl.obtain_points_from_data(param.simulated_values_min,data)#,surface_level_comparison_on=True)
    data_points=data
    bias=np.log10(simulated_points[:,0])-np.log10(data_points[:,2])
    
#    if param.errors:
#        plt.errorbar(data_points[:,2],simulated_points[:,0],
#                     yerr=[simulated_points[:,0]-simulated_points_min[:,0],simulated_points_max[:,0]-simulated_points[:,0]],
#                    linestyle="None",c='k',zorder=0)
    plot=plt.scatter(data_points[:,2],simulated_points[:,0],c=campaigns_dict[camp_key].color,cmap=cmap,marker=marker,s=marker_size,label=campaigns_dict[camp_key].name)
        #plt.errorbar(data_points[:,2],simulated_points[:,0],yerr=[simulated_points_min[:,0],simulated_points_max[:,0]], linestyle="None",c='k')
        
    #plot=plt.scatter(data_points[:,2],simulated_points[:,0],c=bias,cmap=cmap,marker=marker,s=marker_size,vmin=-5, vmax=5)
    #plot=plt.scatter(data_points_mason[:,2],simulated_points_mason[:,0],c=bias_mason,cmap=cmap,marker=marker_mason,s=marker_size_mason,vmin=-5, vmax=5)
    
    
    #plt.colorbar(plot,label='Temperature $C$')
    
    plt.ylabel('Simulated ($cm^{-3}$)')
    plt.xlabel('Observed ($cm^{-3}$)')
    if np.min(simulated_points[:,0])>np.min(data_points[:,2]):
        min_plot=np.min(data_points[:,2])
    else:
        min_plot=np.min(simulated_points[:,0])
    
    if np.max(simulated_points[:,0])<np.max(data_points[:,2]):
        max_plot=np.max(data_points[:,2])
    else:
        max_plot=np.max(simulated_points[:,0])
        
    #minx=np.min(data_points[:,2])
    #maxx=np.max(data_points[:,2])
    #miny=np.min(simulated_points[:,0])
    #maxy=np.max(simulated_points[:,0])
    min_val=1e-9
    max_val=1e1
    minx=np.min(min_val)
    maxx=np.max(max_val)
    miny=np.min(min_val)
    maxy=np.max(max_val)
    min_plot=min_val
    max_plot=max_val
    
    x=np.linspace(0.1*min_plot,10*max_plot,100)
    #global x     
    r=np.corrcoef(np.log(data_points[:,2]),np.log(simulated_points[:,0]))
    mean_error=jl.mean_error(np.log(data_points[:,2]),np.log(simulated_points[:,0]))
    mean_bias=jl.mean_bias(np.log(data_points[:,2]),np.log(simulated_points[:,0]))
    print r,mean_error,mean_bias
    #rmsd=RMSD(data_points[:,2],simulated_points[:,0])
    #plt.title('R=%f RMSD=%f'%(r[0,1],rmsd))
    plt.plot(x,x,'k-')
    plt.plot(x,10*x,'k--')
    plt.plot(x,10**1.5*x,'k-.')
    plt.plot(x,0.1*x,'k--')
    plt.plot(x,10**(-1.5)*x,'k-.')
    plt.ylim(miny*0.1,maxy*10)
    plt.xlim(minx*0.1,maxx*10)
    plt.xscale('log')
    plt.yscale('log')
    #plt.savefig('/nfs/see-fs-01_users/eejvt/terrestial_marine/'+param.title+'_campaigns'+'.png')
    #plt.close()
#lgd =plt.legend( loc='center left', bbox_to_anchor=(1.0, 0.5))
#plt.colorbar()
plt.title('g)')
a=plt.plot(np.linspace(1e-20,2e-20,3),np.linspace(1e-20,2e-20,3),c=np.linspace(1e-20,2e-20,3))
#b=plt.plot(1e-80,1e-80,c='b')
#plt.colorbar(a)
#cb.remove()
#plt.draw()
#fig.savefig('/nfs/see-fs-01_users/eejvt/terrestial_marine/'+param.title+'_campaigns'+'.png', dpi=300, format='png')#, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()








#%%

















