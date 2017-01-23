# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:52:11 2014

@author: eejvt
"""

import numpy as np
from random import random
from random import uniform
import matplotlib.pyplot as plt


tmin=-30
tmax=-5
nmax=100
nmin=1e-2
plt.figure()

a=(np.log(nmax)-np.log(nmin))/(tmin-tmax)
print 'a=',a

b=np.log(nmax)-tmin*a
print 'b=',b


def create_aerosol(tmin=-30,tmax=-5,nmax=100,nmin=1e2):
    A={}
    A['tmin']=-25
    A['tmax']=-5
    A['nmax']=10**8
    A['nmin']=1e-1
    A['a']=(np.log(A['nmax'])-np.log(A['nmin']))/(A['tmin']-A['tmax'])
    A['b']=np.log(A['nmax'])-A['tmin']*A['a']
    return A

A=create_aerosol()


def nINP_value(t,Aer=A):

    nINP=np.exp((t*Aer['a']+Aer['b']))
    return nINP




def t_value(ns_rand,Aer=A):

    return (np.log(ns_rand)-Aer['b'])/Aer['a']




def setup_values(time=30):
    area_fraction=1

    v_plane=2
    fiter_surface=0.0000000000001
    air_volume=v_plane*fiter_surface*time#using surface for feldspar
    #n=10
    air_volume=1.2566370614359174e-04*1e-3#cm**2

    particle_radio=1e-6

    particle_surface=4*np.pi*particle_radio**2

    inp=int(nINP_value(-37)*air_volume)
    print inp
    droplet_number=100

    box=np.linspace(0,1,droplet_number+1)
    INP=np.zeros((inp,3))
    #droplet_number=inp
    return INP, box,droplet_number,air_volume



def set_droplets(INP,box):
    inp=INP[:,0].size
    for i in xrange(inp):
        INP[i,0]=random()
        for j in xrange (len(box-1)):
            if INP[i,0]>box[j] and INP[i,0]<box[j+1]:
                INP[i,1]=int(j)
    return INP


#INP=set_droplets(INP,box)

def set_temperatures(INP):
    inp=INP[:,0].size
    for i in xrange(inp):
        INP[i,0]=uniform(nINP_value(-37),nINP_value(0))
        INP[i,2]=t_value(INP[i,0])
    print INP[:,2]
    return INP

#INP=set_temperatures(INP)

#ts=np.arange(-40,0,1)

def freezed_droplets(INP,box,t):
    drops=len(box-1)
    n=0
    f=0
    inp=INP[:,0].size
    for i in xrange (drops):
        for j in xrange(inp):
            if INP[j,1]==i and INP[j,2]>t:

                f=1
        n=n+f
        f=0


    return n
#s=len(ts)
#ns=np.zeros(len(ts))

def calculate_ff(time=30,plot_ff=1,plot_nv=1):
    INP,box,droplet_number,air_volume=setup_values(time)
    INP=set_droplets(INP,box)
    INP=set_temperatures(INP)
    ts=np.arange(-35,0,1)
    ns=np.zeros(len(ts))
    inp=len(INP[:,0])
    #ax=plt.subplot(121)
    #bx=plt.subplot(122)
    for i in xrange(len(ts)):

        ns[i]=freezed_droplets(INP,box,ts[i])

    print ns
    ff=ns/droplet_number
    print ff
    #nv=ff*len(INP[:,0])/air_volume
    #nv=ff*nINP_value(-35)*air_volume

    nv=-np.log(1-ff)*droplet_number/air_volume


    #for i in range(len(nv)):
    #    if nv[i]<=0:
     #       nv[i]=0
    print '-------------------------------------------'
    #print -np.log(1-ff)
    #print nINP_value(ts)
    #print (-np.log(1-ff))-nINP_value(ts)
    print'--------------INP---------------'
    print len(INP[:,0])
    #print 'nv             ',nv
    if plot_ff:
        ax.plot(ts,ff,label='t=%i'%time)
    if plot_nv:
        bx.plot(ts,nv,label=air_volume)
    #print ff
    return ff, ts,nv,air_volume


def ff_and_nv():
    #ff, ts= calculate_ff()
    #global ax, bx
    ax=plt.subplot(121)
    bx=plt.subplot(122)


    times=np.linspace(1,1000,5)
    for i in range (len(times)):
        ff, ts,nv,air_volume= calculate_ff(times[i])
        #ax.plot(ts,1-np.exp(-nINP_value(ts)*air_volume),'--',label='t=%i'%i)
        print times[i]

    bx.plot(np.arange(-35,0,1),nINP_value(np.arange(-35,0,1)),'bo')
    ax.legend(loc='best')
    bx.legend(loc='best')
    bx.set_yscale('log')
    plt.show()




#ff_and_nv()
bx=plt.subplot(111)
def multiple_nv(runs=10):

    times=np.linspace(1,1000,5)
    for i in range (runs):
        ff, ts,nv,air_volume= calculate_ff(30,plot_ff=0)
        print i
    bx.set_title('air_volume=%f'%air_volume)
    bx.plot(np.arange(-35,0,1),nINP_value(np.arange(-35,0,1)),'bo')
    bx.set_yscale('log')


multiple_nv(runs=100)

plt.show()
#nss=nINP_value(ts)



#particle_surface=4*np.pi*1e-4**2


#print particle_surface*nINP_value(-30)
