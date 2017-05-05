# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:49:38 2016

@author: eejvt
"""


import numpy as np
import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import Jesuslib as jl
import os
from glob import glob
import pylab
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
import numpy.ma as ma
import datetime
import matplotlib
import random


plt.figure()
ax=plt.subplot(121)
#plt.yscale('log')
bx=plt.subplot(122)
plt.yscale('log')

class particle():
    def __init__(self,radius):
        self.radius=radius
        self.surface_area=4*np.pi*self.radius**2
        self.volume=4/3.*np.pi*self.radius**3

for weight_percentage in [1e-2,1e-3,1e-5,1e-7,1e-9]:
    print weight_percentage
    r_mean=1.3e-6
    rs=jl.logaritmic_steps(-6.15,-4.8860566476931631,30)
    rs=jl.logaritmic_steps(-7.15,-4.8860566476931631,50)
    sigma=2
    total_surface_area_per_particle=np.array((4*np.pi*rs.mid_points**2*jl.lognormal_PDF(r_mean,rs.mid_points,sigma))*rs.grid_steps_width).sum()
    #weight_percentage=0.001#%
    droplet_volume=10e-3#g
    weight_feldspar=droplet_volume*1e-2*weight_percentage#g
    surface_area_density=0.89#g/m2
    total_surface_area_per_droplet=surface_area_density*weight_feldspar#m-2
    print total_surface_area_per_droplet
    droplets=20
    total_surface_area=total_surface_area_per_droplet*droplets
    N_particles=total_surface_area/total_surface_area_per_particle
    #print N_particles,N_particles/droplets
    
    
    
    #plt.plot(rs.mid_points,jl.lognormal_PDF(r_mean,rs.mid_points,sigma),'bo')
    #plt.plot(rs.mid_points,4*np.pi*rs.mid_points**2*jl.lognormal_PDF(r_mean,rs.mid_points,sigma)/total_surface_area_per_particle,'ro')
    #plt.xscale('log')
    #plt.yscale('log')
    
    
    probabilities=jl.lognormal_PDF(r_mean,rs.mid_points,sigma)*rs.grid_steps_width
    bins = np.add.accumulate(probabilities)
    radius_list=rs.mid_points[np.digitize(np.random.random_sample(N_particles), bins)]
    particle_list=np.array([particle(radii) for radii in radius_list])
    
    droplet_particle_falls=np.random.randint(droplets, size=N_particles)
    surface_area_per_droplet=[]
    for i in range(droplets):
        particles_in_droplet=particle_list[droplet_particle_falls==i]
        surface_area_in_droplet=np.array([particle_in_drop.surface_area for particle_in_drop in particles_in_droplet]).sum()
        surface_area_per_droplet.append(surface_area_in_droplet)
    surface_area_per_droplet=np.array(surface_area_per_droplet)
    print surface_area_per_droplet.max(),surface_area_per_droplet.min(),surface_area_per_droplet.min()/surface_area_per_droplet.max()
    #plt.figure()
    #plt.plot(surface_area_per_droplet)
    
    Ts=np.linspace(243,268,100)
    ff_list=[]
    
    for T in Ts:
        for _ in range(100):
            #print T
            ns=jl.feld_parametrization(T)
            active_sites=ns*(surface_area_per_droplet.sum())
            #print active_sites,active_sites/droplets
            if (active_sites-int(active_sites))>random.random():
                active_sites=float(int(active_sites)+1)
            else:
                active_sites=float(int(active_sites))
            
            if active_sites/droplets>2:
                ff=1.0
                ff_list.append(ff)
            elif active_sites==0:
                ff=0.0
                ff_list.append(ff)
            else:
                probabilities=surface_area_per_droplet/surface_area_per_droplet.sum()
                bins = np.add.accumulate(probabilities)
                pos_inp=np.digitize(np.random.random_sample(active_sites), bins)
                frezzed_droplets=0
                frezzed_droplets=len(set(pos_inp.tolist()))
                ff=frezzed_droplets/float(droplets)
                ff_list.append(ff)
            
            ns_sim=-np.log(1-ff)/(surface_area_per_droplet.sum()/droplets)
            
            ax.plot(T,ff,'bo')
            if ff!=1.0:
                if not ff==0:
                    bx.plot(T,ns_sim,'ro')
    
    
    bx.plot(Ts,jl.feld_parametrization(Ts),'k-')





