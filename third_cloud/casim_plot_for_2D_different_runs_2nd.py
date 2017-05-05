# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:21:22 2016

@author: eejvt
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:33:56 2016

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
import matplotlib.pyplot as plt
import matplotlib.animation as animationlt
#from mayavi import mlab
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE/Satellite_Comparison')
import satellite_comparison_suite as stc

import variable_dict as vd
#from cube_browser import Contour, Browser, Contourf, Pcolormesh
#%%
word1='SW'
word2='shortwave'
word1='LW'
word2='longwave'
word1='CLOUD'
word2='cloud'
word2='TEMP'
word1='ICE'
word2='ice'
word=''
#path='/nfs/a201/eejvt/CASIM/SO_KALLI/NO_INITIAL_ICE/BASE_RUN/All_time_steps/'
#
path='/nfs/a201/eejvt/CASIM/SECOND_CLOUD/GP_HAM_DMDUST/All_time_steps/'
nc_files=glob.glob(path+'*.nc')
for inc in range(len(nc_files)):
    if word in nc_files[inc][len(path):] or word.upper() in nc_files[inc][len(path):]:
        print inc,nc_files[inc][len(path):]
stash_code='00i078'
stash_code='m01s00i272'
stash_code='m01s00i012'
stash_code='m01s00i075'
stash_code='m01s00i268'
stash_code='m01s00i271'
stash_code='m01s00i254'
stash_code='m01s01i235'
stash_code='m01s09i223'#TOTAL_CLOUD_TOP_HEIGHT_(KFT)
stash_code='m01s09i216'#TOTAL_CLOUD_AMOUNT_-_RANDOM_OVERLAP
stash_code='m01s01i208'#toa_outgoing_shortwave_flux.nc
stash_code='m01s02i205'#toa_outgoing_longwave_flux
stash_code='LWP'#LWP
#%%


class Experiment():
    def __init__(self,path,name):
        self.path=path
        self.name=name
    def Read_cube(self,string):
        try:
            self.cube = iris.load(ukl.Obtain_name(self.path,string))[0]
        except:
            self.cube = iris.load(ukl.Obtain_name(self.path[:-15]+'L1/',string))[0]
        
        
ordered_list=['BASE_RUN','3_ORD_LESS','NO_ICE']
ordered_list=['BASE_RUN','3_ORD_LESS']
#run_dict={}
#run_dict['DEMOTT']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/DEMOTT3ORD/All_time_steps/','DEMOTT')
#run_dict['BASE_RUN']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_INITIAL_ICE/BASE_RUN/All_time_steps/','BASE_RUN')
#run_dict['3_ORD_LESS']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_INITIAL_ICE/3_ORD_LESS/All_time_steps/','3_ORD_LESS')
#run_dict['NO_ICE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NO_ICE/All_time_steps/','NO_ICE')
ordered_list=['BASE_RUN','3_ORD_LESS','NO_ICE']
ordered_list=['BASE_RUN','3_ORD_LESS']
ordered_list=['CONTACT_RUN','NO_HALLET']
ordered_list=['ALL_ICE_PROC','NO_HALLET']
ordered_list=['ALL_ICE_PROC','3_ORD_LESS']
ordered_list=['ALL_ICE_PROC','BASE_CONTACT','OLD_BASE','3_ORD_LESS']
ordered_list=['ALL_ICE_PROC','BASE_CONTACT','NO_HALLET','3_ORD_LESS','NO_ICE']
ordered_list=['ALL_ICE_PROC','NO_HALLET','3_ORD_LESS','NO_ICE']
ordered_list=['NO_HALLET','3_ORD_LESS']
ordered_list=['ALL_ICE_PROC','3_ORD_LESS']
ordered_list=['BASE_CONTACT','ALL_ICE_PROC','NO_ICE']
ordered_list=['ALL_ICE_PROC','NO_HALLET','3_ORD_LESS']
ordered_list=['ALL_ICE_PROC','3_ORD_LESS','2_ORD_MORE']
ordered_list=['BASE_DM_DUST']
run_dict={}
#run_dict['DEMOTT']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/DEMOTT3ORD/All_time_steps/','DEMOTT')
#run_dict['ALL_ICE_PROC']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/','ALL_ICE_PROC')
#run_dict['BASE_CONTACT']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/BASE_CONTACT_242/All_time_steps/','BASE_CONTACT')
#run_dict['NO_HALLET']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/All_time_steps/','NO_HALLET')
run_dict['BASE_DM_DUST']=Experiment('/nfs/a201/eejvt/CASIM/SECOND_CLOUD/GP_HAM_DMDUST/All_time_steps/','BASE_DM_DUST')

#run_dict['NO_HALLET']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/All_time_steps/','NO_HALLET')
#run_dict['OLD_BASE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/BASE_RUN2/All_time_steps/','OLD_BASE')
#run_dict['3_ORD_LESS']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/All_time_steps/','3_ORD_LESS')
#run_dict['2_ORD_MORE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/2_ORD_MORE/All_time_steps/','2_ORD_MORE')
#run_dict['NO_ICE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NOICE/All_time_steps/','NO_ICE')
#run_dict['NO_ICE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NOICE/All_time_steps/','NO_ICE')
for run in ordered_list:
    run_dict[run].Read_cube(stash_code)
    print run_dict[run].name,run_dict[run].cube.long_name, run_dict[run].cube.shape
#%%
#ordered_list=['BASE_RUN','3_ORD_LESS','NO_ICE']
#ordered_list=['BASE_RUN','3_ORD_LESS']
#ordered_list=['CONTACT_RUN','NO_HALLET']
#ordered_list=['ALL_ICE_PROC','NO_HALLET']
#ordered_list=['BASE_RUN','CONTACT_RUN','NO_HALLET','3_ORD_LESS']
#run_dict={}
#
##'/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/BASE_RUN2'
#run_dict['BASE_RUN']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/BASE_RUN2/All_time_steps/','BASE_RUN')
##run_dict['DEMOTT']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/DEMOTT3ORD/All_time_steps/','DEMOTT')
##run_dict['BASE_RUN']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_INITIAL_ICE/BASE_RUN/All_time_steps/','BASE_RUN')
##run_dict['3_ORD_LESS']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/NO_INITIAL_ICE/3_ORD_LESS/All_time_steps/','3_ORD_LESS')
##run_dict['NO_ICE']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NO_ICE/All_time_steps/','NO_ICE')
#run_dict['CONTACT_RUN']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/BASE_CONTACT/All_time_steps/','CONTACT_RUN')
##run_dict['ALL_ICE_PROC']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/ALL_ICE_PROC/All_time_steps/','ALL_ICE_PROC')
#run_dict['NO_HALLET']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/NO_HALLET/All_time_steps/','NO_HALLET')
#run_dict['3_ORD_LESS']=Experiment('/nfs/a201/eejvt/CASIM/SO_KALLI/TRY2/3_ORD_LESS_762/All_time_steps/','3_ORD_LESS')

#%%
# This example uses a MovieWriter directly to grab individual frames and
# write them to a file. This avoids any event loop integration, but has
# the advantage of working with even the Agg backend. This is not recommended
# for use in an interactive setting.
# -*- noplot -*-
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
sample_cube=run_dict[ordered_list[0]].cube
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title=sample_cube.var_name, artist='CASIM',
                comment='UKCA nested high resolution')
writer = FFMpegWriter(fps=2, metadata=metadata)

fig = plt.figure(figsize=(15,15))
#fig = plt.figure(figsize=(len(ordered_list)*5, 12))

n_runs=len(ordered_list)


max_val=sample_cube[:,:,:].data.max()
min_val=sample_cube[:,:,:].data.min()
min_val = np.min(sample_cube[:,:,:].data[np.nonzero(sample_cube[:,:,:].data)])
vertical_levels=np.linspace(min_val,max_val,9).tolist()
diff_levels=np.linspace(-max_val,max_val,18).tolist()
cube_list=[]
#lats=sample_cube.coord('grid_latitude').points-52
#lons=sample_cube.coord('grid_longitude').points[:]-180
#lons=np.arange(-0.02*250,250*0.02,0.02)[0:]
lons,lats=stc.unrotated_grid(sample_cube)
#lats=np.arange(-0.02*250-52,250*0.02-52,0.02)
for run in ordered_list:
    cube=run_dict[run].cube
    try:
        cube.remove_coord('surface_altitude')
    except:
        a=8754784
    cube_list.append(cube)


import copy
sub_plots=[]
time_coord=copy.copy(sample_cube.coord('time'))
time_coord.convert_units('seconds since 1970-01-01 00:00:0.0')

first=[]
second=[]
third=[]
names='runs'
for name in ordered_list:
    names=names+'-'+name

with writer.saving(fig,'/nfs/a201/eejvt/SO_VIDEOS_2/2D_141116'+sample_cube.var_name+'_'+names+".mp4", 200):
    for it in range(len(sample_cube.coord('time').points)-1):
        try:
            first.append(cube_list[0][:,:,:].extract(iris.Constraint(time=sample_cube.coord('time').points[it])).data.mean())
            second.append(cube_list[1][:,:,:].extract(iris.Constraint(time=sample_cube.coord('time').points[it])).data.mean())
            third.append(cube_list[2][:,:,:].extract(iris.Constraint(time=sample_cube.coord('time').points[it])).data.mean())
        except:
#            first.append(np.nan)
#            second.append(np.nan)
#            third.append(np.nan)
            a='asdfasd'
        big_title=sample_cube.var_name.replace('_',' ')+' '+datetime.datetime.fromtimestamp(time_coord.points[it]).strftime('%D %H:%M:%S')
        print big_title        
        try:
            txt.remove()
        except:
            adsf=2
        txt=plt.figtext(0.2,0.95,big_title,fontsize=15)
        if it!=0:
            for x in sub_plots:
                x.cla()
        for ir in range(n_runs):
                        
            x=plt.subplot(2,(n_runs)/2+1,ir+1)
            x.set_title(ordered_list[ir])
            sub_plots.append(x)
            try:
                plot_cube=cube_list[ir][:,:,:].extract(iris.Constraint(time=sample_cube.coord('time').points[it])).data
            except:
                plot_cube=cube_list[ir][:,:,:].extract(iris.Constraint(time=sample_cube.coord('time').points[it-1])).data
#                a='asfdsasd'
            mapable=x.contourf(lons,lats, plot_cube,vertical_levels, cmap='viridis')
            if ir+1==n_runs:
                if it==0:
                    cb=plt.colorbar(mapable,label=sample_cube.units.origin)
            if ir==0:
                plt.ylabel('Latitude')
            plt.xlabel('Longitude')
#        print it
#        x1=plt.subplot(2,(n_runs)/2+1,4)
#        plt.plot(first,'b-',label=ordered_list[0])
#        plt.plot(second,'r-',label=ordered_list[1])
#        plt.plot(third,'y-',label=ordered_list[2])
##            plt.plot(np.array(second)-np.array(first),'g-',label='Diference')
##            plt.xlim(0,len(sample_cube.coord('time').points)-1)
##            plt.ylim(vertical_levels[0],620)
#        plt.ylabel('TOA outgoing shortwave radiation $W/m^{2}$')
        if it==0:
            plt.legend(loc='upper center', bbox_to_anchor=(0.2, 1.05),prop={'size':10})

#        try:
#            x=plt.subplot(2,(n_runs)/2+1,3)
#            plot_cube=cube_list[0][:,:,0:].extract(iris.Constraint(time=sample_cube.coord('time').points[it])).data-cube_list[1][:,:,0:].extract(iris.Constraint(time=sample_cube.coord('time').points[it])).data
#            print plot_cube.max(),plot_cube.min()
#            mapable=x.contourf(lons,lats, plot_cube,diff_levels, cmap='RdBu')
#            if it==0:
#                cb=plt.colorbar(mapable,label=sample_cube.units.origin)
#            
#            x1=plt.subplot(2,(n_runs)/2+1,4)
#            plt.plot(first,'b-',label=ordered_list[0])
#            plt.plot(second,'r-',label=ordered_list[1])
#            plt.plot(np.array(second)-np.array(first),'g-',label='Diference')
##            plt.xlim(0,len(sample_cube.coord('time').points)-1)
##            plt.ylim(vertical_levels[0],620)
##            plt.ylabel('TOA outgoing shortwave radiation $W/m^{2}$')
#            if it==0:
#                plt.legend(loc='upper center', bbox_to_anchor=(0.2, 1.05),prop={'size':10})
#        except:
#            a='adfas'
            
        writer.grab_frame()

#for cube in cube_list:
#%%
first=[]
second=[]
#plt.plot()
for it in range(len(sample_cube.coord('time').points)-1):
    first.append(cube_list[0][:,:,0:].extract(iris.Constraint(time=sample_cube.coord('time').points[it])).data.mean())
    second.append(cube_list[1][:,:,0:].extract(iris.Constraint(time=sample_cube.coord('time').points[it])).data.mean())

plt.plot(first,label=ordered_list[0])
plt.plot(second,label=ordered_list[1])
plt.legend()
plt.show()
print np.array(first).sum()
print np.array(second).sum()


#%%
sample_cube=run_dict[ordered_list[0]].cube
time_coord=sample_cube.coord('time')
time_coord.convert_units('seconds since 1970-01-01 00:00:0.0')

times1=[datetime.datetime.fromtimestamp(time_coord.points[it]).strftime('%D %H:%M:%S') for it in range(len(time_coord.points))]


sample_cube=run_dict[ordered_list[1]].cube
time_coord=sample_cube.coord('time')
time_coord.convert_units('seconds since 1970-01-01 00:00:0.0')

times2=[datetime.datetime.fromtimestamp(time_coord.points[it]).strftime('%D %H:%M:%S') for it in range(len(time_coord.points))]


for i in range(len(times2)):
    print times1[i]==times2[i]
#    if not times1[i]==times2[i]











