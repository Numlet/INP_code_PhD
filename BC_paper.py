# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:16:06 2016

@author: eejvt
"""

import numpy as np
import sys
sys.path.append('/nfs/see-fs-01_users/eejvt/PYTHON_CODE')
import Jesuslib as jl
import os
from scipy.io.idl import readsav
from glob import glob
from scipy.io import netcdf
import matplotlib.pyplot as plt
import scipy as sc


path='/nfs/a86/shared/Tom O soot data/Dissertation Figures Graphs and Data/'

book_name='All Data Final.xlsx'
import xlrd

book = xlrd.open_workbook(path+book_name)


print("The number of worksheets is {0}".format(book.nsheets))
print("Worksheet name(s): {0}".format(book.sheet_names()))
sh = book.sheet_by_index(0)
print("{0} {1} {2}".format(sh.name, sh.nrows, sh.ncols))
print("Cell D30 is {0}".format(sh.cell_value(rowx=29, colx=3)))
for rx in range(sh.nrows):
    print(sh.row(rx))

worksheet = book.sheet_by_name('ns_clean')

def BC_parametrization_tom_old(T):
    #A=-20.27
    #B=1.2
    return 10**(-2.87-0.182*T)

def BC_parametrization_tom(T):
    #A=-20.27
    #B=1.2
#    return 10**(-2.87-0.182*T)
    return np.exp((-6.608-0.419*T))

def feld_parametrization(T):
    ns=np.exp((-1.03802178356815*T)+275.263379304105)
    return ns

def ulrich(T):
    #T in C
    ns=7.463*np.exp(-0.0101*(T)**2-0.8525*(T)+0.7667)#m-2
    ns=ns*1e-4#cm-2
    return ns
def schill_in_ulrich(T):
    #T in C
    ns=np.exp(-0.0101*(T)**2-0.8525*(T)+0.7667)*(3.84e-4)#cm-2 
    #I think they changed the sign to negative in the exponential and multiply instead of dividing
    #but forgot to divide the main term
    return ns
def schill(T):
    #T in C
    ns=np.exp(-0.0101*(T)**2-0.8525*(T)+0.7667)/(3.84e4)#cm-2
    return ns
def murray(T):
    #T in C
    ns=np.exp(-0.0101*(T)**2-0.8525*(T)+0.7667)#cm-2
    return ns
def umo(T):
    A=-18.60
    B=1.10
    Ns=np.exp(A-B*T)
    return Ns

def threshold(T,factor=0.01):
    #T in C
    ns=factor*np.exp(-0.0101*(T)**2-0.8525*(T)+0.7667)#cm-2
    return ns

DM_dataset=np.genfromtxt(jl.home_dir+'BC/DeMott_from_danny.csv',delimiter=',',skip_header=2)
DM_temps=DM_dataset[:,0]
DM_values=DM_dataset[:,1]
Diehl_dataset=np.genfromtxt(jl.home_dir+'BC/Diehl_from_danny.csv',delimiter=',',skip_header=2)
Diehl_temps=Diehl_dataset[:,0]
Diehl_values=Diehl_dataset[:,1]
#%%
class run:
    def __init__(self):
        self.temps=[]
        self.ff=[]
        self.ns=[]
lw=3
runs_dict={}
jump_runs=['B4MB3.1', 'B3S1.2', 'B3S1.1']
for icol in range(200):
    try:
        head=worksheet.cell(0, icol).value
        if head=='':continue
        if head.split(" ", 1)[0] in jump_runs:continue
        if head[0]=='B':
            runs_dict[head]=run()
            for i in range(10000):
                try:
                    temp_value=worksheet.cell(i, icol).value
                except IndexError:
                    break
                temp_value=worksheet.cell(i, icol).value
                ff_value=worksheet.cell(i, icol+1).value
                ns_value=worksheet.cell(i, icol+2).value
                if isinstance(temp_value,float):
                    if ff_value!=1:
                        runs_dict[head].ff.append(ff_value)
                        runs_dict[head].temps.append(temp_value)
                        runs_dict[head].ns.append(ns_value)
                if temp_value=='':
                    break
    except:
        print icol

#plt.figure()
plt.figure(figsize=(15,7))
ax = plt.subplot(121)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
all_temps=[]
all_ns=[]
for run in runs_dict:
    if '10-2' in run:
        c='y'
        for T in runs_dict[run].temps:all_temps.append(T)
        for T in runs_dict[run].temps:all_temps.append(T)
    elif '10-3' in run:
        if 'cane' in run:
            c='orange'
        else:
            c='r'
    else:
        continue
        c='g'
    plt.plot(runs_dict[run].temps,runs_dict[run].ns,'o',c=c)#,label=run.split(" ", 1)[0])

    
plt.plot([],'o',c='r',label='Eugenol 10-3')
plt.plot([],'o',c='y',label='Eugenol 10-2')
plt.plot([],'o',c='orange',label='n-Decane 10-3')
plt.plot(DM_temps,DM_values,'o',c='darkgrey',label='DeMott')
plt.plot(Diehl_temps,Diehl_values,'o',c='k',label='Diehl')

Ts=np.linspace(-30,-10,100)
#plt.plot(Ts,BC_parametrization_tom_old(Ts),'o',label='This work ns upper limit')
plt.plot(Ts,BC_parametrization_tom(Ts),'--',lw=lw,label='This work ns upper limit')
Ts=np.linspace(-34,-18,100)
plt.plot(Ts,ulrich(Ts),'--',lw=lw,label='Ulrich ns upper limit')
Ts=np.linspace(-38,-18,100)
#plt.plot(Ts,schill_in_ulrich(Ts),'--',lw=lw,label='Schill in ulrich ns upper limit')
plt.plot(Ts,schill(Ts),'--',lw=lw,label='Schill ns upper limit')
plt.yscale('log')
Ts=np.linspace(-36,-15,100)
plt.plot(Ts,murray(Ts),'-',lw=lw,label='Murray')
#plt.plot(Ts,threshold(Ts),'-',lw=lw,label='Threshold')
#plt.plot(Ts,umo(Ts),'-',lw=lw,label='Umo contaminated')


plt.ylabel('ns (cm-2)')
plt.xlabel('Temperature $^oC$')
plt.yscale('log')
plt.legend()

Ts=np.linspace(-25,-5,100)
plt.plot(Ts,feld_parametrization(Ts+273.15),'-.',lw=5,label='Feldspar ns')
plt.yscale('log')
plt.legend()
ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
plt.savefig(jl.bc_folder+'all_params.png')
#%%

plt.figure()
lw=3

Ts=np.linspace(-30,-10,100)
#plt.plot(Ts,BC_parametrization_tom_old(Ts),'o',label='This work ns upper limit')
plt.plot(Ts,BC_parametrization_tom(Ts),'--',lw=lw,label='This work ns upper limit')
Ts=np.linspace(-34,-18,100)
plt.plot(Ts,ulrich(Ts),'--',lw=lw,label='Ulrich ns upper limit')
Ts=np.linspace(-38,-18,100)
#plt.plot(Ts,schill_in_ulrich(Ts),'--',lw=lw,label='Schill in ulrich ns upper limit')
plt.plot(Ts,schill(Ts),'y--',lw=lw,label='Schill ns upper limit')
plt.yscale('log')
Ts=np.linspace(-36,-15,100)
plt.plot(Ts,murray(Ts),'r-',lw=lw,label='Murray')

temps=np.load(jl.bc_folder+'temps.npy')
grid_of_fractions=np.load(jl.bc_folder+'grid_of_fractions.npy')
#plt.yscale('log')
ns_values=np.load(jl.bc_folder+'ns_values.npy')
levels=np.linspace(0,1,6).tolist()
plt.contourf(temps,ns_values,grid_of_fractions.T,levels,cmap=plt.cm.bone_r,alpha=0.7)
plt.colorbar(label='% surface gridboxes')
plt.legend(loc='lower left',fontsize=12)
plt.grid()
plt.ylabel('ns (cm-2)')
plt.xlabel('Temperature $^oC$')
plt.savefig(jl.bc_folder+'Threshold.png')
plt.show()


#%%
plt.figure()
for run in runs_dict:
    if '10-2' in run:
        c='y'
    elif '10-3' in run:
        if 'cane' in run:
            c='orange'
        else:
            c='r'
    else:
        c='g'
    plt.plot(runs_dict[run].temps,runs_dict[run].ff,'o',c=c)#,label=run.split(" ", 1)[0],c=c)
plt.plot([],'o',c='r',label='Eugenol 10-3')
plt.plot([],'o',c='y',label='Eugenol 10-2')
plt.plot([],'o',c='g',label='10-1')
plt.plot([],'o',c='orange',label='n-Decane 10-3')
plt.ylabel('Fraction frozen')
plt.xlabel('Temperature (C)')
plt.legend()
# time.sleep(1000)
#%%
#for i in range(20):
#    plt.close()

#%%

blank_temps=[-20.81,-21.25,-22.57,-23.33,-23.42,-23.63,-23.77,-24.02,-24.26,-24.60,-24.62,-24.69,-24.96,-25.29,-25.37,-25.41,-25.49,-25.49,-25.53,-25.81,-25.93,-25.93,-26.02,-26.03,-26.19,-26.32,-26.45,-26.46,-26.62,-26.64,-26.76,-26.77,-26.77,-26.83,-26.85,-26.92,-27.02,-27.10,-27.34,-27.36,-27.49,-27.53,-27.56,-27.72,-27.78,-27.83,-27.83,-27.84,-27.86,-27.88,-27.88,-27.92,-27.97,-27.97,-27.97,-28.00,-28.18,-28.23,-28.26,-28.45,-28.49,-28.51,-28.53,-28.53,-28.53,-28.55,-28.55,-28.61,-28.61,-28.70,-28.73,-28.75,-28.79,-28.80,-28.84,-28.87,-28.97,-29.00,-29.00,-29.05,-29.07,-29.09,-29.13,-29.15,-29.15,-29.15,-29.16,-29.18,-29.21,-29.21,-29.30,-29.30,-29.31,-29.34,-29.64,-29.65,-29.66,-29.74,-29.77,-29.79,-29.81,-29.94,-29.98,-30.01,-30.02,-30.02,-30.05,-30.14,-30.18,-30.19,-30.23,-30.29,-30.30,-30.32,-30.36,-30.36,-30.39,-30.44,-30.44,-30.48,-30.48,-30.52,-30.56,-30.72,-30.72,-30.78,-30.99,-31.01,-31.04,-31.11,-31.21,-31.45,-31.51,-31.64,-31.81,-31.83,-32.05,-32.45,-32.63,-33.66,-33.79,-33.93,-34.02,-34.46,-34.92]
blank_temps=[-21.25,-22.57,-23.33,-23.63,-23.77,-24.26,-24.69,-25.37,-25.41,-25.49,-26.03,-26.19,-26.46,-26.83,-26.85,-27.10,-27.53,-27.84,-27.92,-28.23,-28.26,-28.53,-28.53,-28.75,-28.87,-29.21,-29.21,-29.31,-29.65,-30.01,-30.32,-30.39,-30.44,-30.48,-32.63,-33.66,]
blank_ff=np.linspace(1,len(blank_temps),len(blank_temps))/float(len(blank_temps))

plt.plot(blank_temps,blank_ff,'o',c='b',label='blank')
plt.legend()

blank_temps=[-17.13,-19.21,-19.23,-19.78,-21.28,-24.01,-24.30,-24.63,-25.42,-25.77,-26.26,-26.62,-26.82,-27.03,-27.05,-27.95,-28.37,-28.43,-28.53,-28.73,-29.00,-29.08,-29.24,-29.61,-29.68,-29.92,-30.26,-30.46,-30.63,-31.20,-31.28,-31.61,-31.68,-31.71,-31.81,-32.08]
blank_ff=np.linspace(1,len(blank_temps),len(blank_temps))/float(len(blank_temps))
plt.plot(blank_temps,blank_ff,'o',c='b',label='blank')

plt.show()
import time

plt.savefig('last_fig.png')#,bbox_inches="tight",pad_inches=0.0)
