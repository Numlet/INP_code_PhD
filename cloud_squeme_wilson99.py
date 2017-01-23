# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:34:11 2016

@author: eejvt
"""


import numpy as np
import matplotlib.pyplot as plt
import math
import iris

qcf=0.1
qsatliq=1
rhcrit=1.01
ncf=qcf/((1-rhcrit)*qsatliq)
cfice=0

Mt=10*1e-12#kg
T_init=-10
def Nice(D,lamda_ice=0.1,T=T_init):
    No=2.0e6#m-4
    return No*np.exp(-0.1222*T)*np.exp(-lamda_ice*D)
'''
plt.plot(Ds,Nice(Ds,T=-20))
plt.xscale('log')
plt.yscale('log')
'''

def mice(D):
    return 0.069*D**2

def vice(D):
    return 25.2*mice(D)**0.473*D**0.527

Ds=np.logspace(-6,-3,1000)
Ds=np.logspace(-6,-3,10)
Ds_dist=Ds[1:]-Ds[:-1]


def solve_lamda(Mt,lambda_init=1,Ds=np.logspace(-9,-3,10000),T=T_init):
    diff=1
    lambda_ice=lambda_init
    Ds_dist=Ds[1:]-Ds[:-1]
    Ds_mid=np.exp(np.log(Ds[1:])-(np.log(Ds[1:])-np.log(Ds[:-1]))/2)
    iterac=1
    lambda_ice_old2=lambda_init*200
    lambda_ice_old=lambda_init*20
    factor=2        
    start_hs=0
    diff=(Nice(Ds_mid,lamda_ice=lambda_ice,T=T)*mice(Ds_mid)*Ds_dist).sum()-Mt
    while np.abs(diff)>1e-16:
        lambda_ice_old=lambda_ice
        diff_old=diff
        '''        
        factor=1.1
        if iterac>1000:
            factor=1.0001
        if iterac>10000:
            factor=1.000001
        if iterac>20000:
            factor=1.00000001
        if iterac>30000:
            factor=1.0000000001
        '''
        if diff>0:
            lambda_ice=lambda_ice_old*factor
        if diff<0:
            lambda_ice=lambda_ice_old/factor
            
        diff=(Nice(Ds_mid,lamda_ice=lambda_ice,T=T)*mice(Ds_mid)*Ds_dist).sum()-Mt
        if math.copysign(1,diff_old)!=math.copysign(1,diff):
            if diff_old>0:
                lambda_pos=lambda_ice_old
                lambda_neg=lambda_ice
            else:
                lambda_neg=lambda_ice_old
                lambda_pos=lambda_ice
            while np.abs(diff)>1e-18:
                lambda_test=(lambda_neg-lambda_pos)/2.+lambda_pos#+lambda_neg
                #print lambda_pos,lambda_neg
                #print lambda_test
                diff=(Nice(Ds_mid,lamda_ice=lambda_test,T=T)*mice(Ds_mid)*Ds_dist).sum()-Mt
                if diff>0:
                    lambda_pos=lambda_test
                if diff<0:
                    lambda_neg=lambda_test
                #print lambda_test
            lambda_ice=lambda_test
                
                        
            
                #print diff
        #print iterac
        iterac=iterac+1
    return lambda_ice


Ds=np.logspace(-9,-3,10000)
#plt.plot(Ds,Nice(Ds,lamda_ice=lam)*mice(Ds))
#plt.xscale('log')
#plt.yscale('log')

viscosity_air=1.6320527E-5#kgms
rho=0.8
Ls=2.838#J/kglatent_heat_sublimation_ice
R=461#JKkg REMEMBER KELVIN!!!!!!!!!!!!!!!gas_constant_water_vapor
ka=2.227e-2#W/mKtemal_cond_air
X=0.0000252#m2/s diff_water_air=

def reynolds(D):
    return vice(D)*rho*D/viscosity_air
def F(D):
    return (0.65+0.44*0.6**(0.333)*reynolds(D)**0.5)



def dmdt(lam,Ds=np.logspace(-9,-3,10000),Si=1.01,T=T_init,esat=1.01):
    Tk=T_init+273.15
    Ds_dist=Ds[1:]-Ds[:-1]
    Ds_mid=np.exp(np.log(Ds[1:])-(np.log(Ds[1:])-np.log(Ds[:-1]))/2)
    return (((4*np.pi*Ds_mid/2.*(Si-1)*F(Ds_mid))/((Ls/(R*Tk)-1)*Ls/(ka*Tk)+R*Tk/(X*esat))))

lam=solve_lamda(Mt,T=T_init)
#Mt=Mt+dmdt
#%%

Ds=np.logspace(-7,-4,10000)
Mt_init=1*1e-12#kg
T_init=-30
T_init=-20
plt.figure()

for T_init in (-1,-10,-20,-30):
    Mt=Mt_init
    lam=solve_lamda(Mt,T=T_init,Ds=Ds)
    print lam
    ax=plt.subplot((411))
    Ds_dist=Ds[1:]-Ds[:-1]
    Ds_mid=np.exp(np.log(Ds[1:])-(np.log(Ds[1:])-np.log(Ds[:-1]))/2)
    ax.plot(Ds_mid,Nice(Ds_mid,T=T_init,lamda_ice=lam),label='T=%1.1f '%T_init)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylabel('Nd')
    max_val=Nice(Ds_mid,T=T_init,lamda_ice=lam).max()
    ax.set_ylim(max_val*1e-5,max_val)
    plt.title('Total Mass (kg) %1.2e'%Mt)
    #ax.set_yscale('log')
    plt.legend()
    #plt.yscale('log')
    print (Nice(Ds_mid,T=T_init,lamda_ice=lam)*mice(Ds_mid)*Ds_dist).sum(),Mt_init,(Nice(Ds_mid,T=T_init,lamda_ice=lam)*mice(Ds_mid)*Ds_dist).sum()/Mt_init
    bx=plt.subplot((412))
    bx.plot(Ds_mid,Nice(Ds_mid,T=T_init,lamda_ice=lam)*mice(Ds_mid)*Ds_dist,label='T=%1.1f'%T_init)
    bx.set_xscale('log')
    plt.ylabel('Mass(D)')
    growth=dmdt(lam=lam,Ds=Ds,T=T_init,esat=1.3,Si=1.01)
    dx=plt.subplot(413)
    dx.plot(Ds_mid,vice(Ds_mid))
    plt.ylabel('Fall_speed (D)')
    dx.set_xscale('log')
    dx.set_yscale('log')
    cx=plt.subplot(414)
    cx.plot(Ds_mid,Nice(Ds_mid,T=T_init,lamda_ice=lam)*growth*Ds_dist,label=(Nice(Ds_mid,T=T_init,lamda_ice=lam)*growth*Ds_dist).sum())    
    #cx.set_xscale('log')
    plt.ylabel('Growth (D)')
    plt.legend()
    
#%%
'EULER'
plt.figure()
T_init=-20
seconds=60*60*3*8
tstep=1000
tsteps=[1000,60*60*3]
for tstep in tsteps:
    Ds=np.logspace(-7,-3,10000)
    Ds_dist=Ds[1:]-Ds[:-1]
    Ds_mid=np.exp(np.log(Ds[1:])-(np.log(Ds[1:])-np.log(Ds[:-1]))/2)
    lam=solve_lamda(Mt,T=T_init)
    Mt_init=0.01*1e-12#kg
    Mt=Mt_init
    times=np.arange(0,seconds,tstep)
    Mt_array=np.zeros(len(times))
    for isec in range(len(times)):
        print isec
        lam=solve_lamda(Mt,T=T_init)
        growth=dmdt(lam=lam,Ds=Ds,T=T_init,esat=1.2,Si=1.1)
        Mt=Mt+(Nice(Ds_mid,T=T_init,lamda_ice=lam)*growth*Ds_dist*tstep).sum()
        Mt_array[isec]=Mt
        
    print Mt/Mt_init
    plt.plot(times+tstep,Mt_array/Mt_init,'o')
#%%
'MIDPOINT'
def dmdt_per_D(M):
    lam=solve_lamda(M,T=T_init)
    growth=dmdt(lam=lam,Ds=Ds,T=T_init,esat=1.2,Si=0.8)
    growth_per_D=Nice(Ds_mid,T=T_init,lamda_ice=lam)*growth*Ds_dist
    
    return growth_per_D
plt.figure()
T_init=-20
seconds=60*60*3*8
tstep=1000
tsteps=[1000,60*60*3]
tsteps=[100,60*60*3]
tsteps=[100,500]
for tstep in tsteps:
    Ds=np.logspace(-7,-3,10000)
    Ds_dist=Ds[1:]-Ds[:-1]
    Ds_mid=np.exp(np.log(Ds[1:])-(np.log(Ds[1:])-np.log(Ds[:-1]))/2)
    lam=solve_lamda(Mt,T=T_init)
    Mt_init=100*1e-12#kg
    Mt=Mt_init
    times=np.arange(0,seconds,tstep)
    Mt_array=np.zeros(len(times))
    for isec in range(len(times)):
        print isec
        growth_per_D=dmdt_per_D(Mt)
        Mt=Mt+dmdt_per_D(Mt+(growth_per_D*tstep/2.).sum() if Mt+(growth_per_D*tstep/2.).sum()>0 else np.array(0)).sum()*tstep
        if Mt<0: Mt=0
        Mt_array[isec]=Mt
        
    print Mt/Mt_init
    plt.plot(times+tstep,Mt_array/Mt_init,'o')
#%%
mice(Ds)
Ds_dist=Ds[1:]-Ds[:-1]
Ds_mid=np.exp(np.log(Ds[1:])-(np.log(Ds[1:])-np.log(Ds[:-1]))/2)
(Nice(Ds_mid,lamda_ice=lam)*mice(Ds_mid)*Ds_dist).sum()
growth=dmdt(lam=lam,Ds=np.logspace(-9,-3,100000),T=T_init,esat=1.4,Si=1.8)

#%%
secs=600
for _ in range(secs): 
    lam=solve_lamda(Mt)
    
    
    
    
#%%

for d in Ds:
    plt.axhline(d)
for d in Ds_mid:
    plt.axhline(d,ls='--')
plt.yscale('log')
#%%
ts=np.linspace(-40,0,100)
ei = 6.112*np.exp(22.46*ts/(272.62 + ts)) #hpa
ew = 6.112*np.exp(17.62*ts/(243.12 + ts))
plt.plot(ts,ew/ei)     
plt.show()
