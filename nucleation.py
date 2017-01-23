# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:05:32 2014

@author: eejvt
"""
from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt


Y=90e-3 #J/m2

k=1.38065e-23 #J/K*molecule
T=104#K
m=18e-3/6.22e23#kg/molecule
rho=0.9167e3#kg/m3

Tc=647.15#K
print m/rho
v=m/rho#m3/molecule

def Ys(T):
    #Y=0.145-1.5e-4*T
    Y=(0.2358*(Tc-T)/Tc)**1.256*(1-0.625*(Tc-T)/Tc)
 
    return Y
    

    
    
    
def pw(T):
    pw=np.exp(54.842763-6763.22/T-4.210*np.log(T)+0.000367*T+np.tanh(0.0415*(T-218.8))*(53.878-1331.22/T-9.44523*np.log(T)+0.014025*T))
    return pw
    
    
    
def pic(T):
    pic=np.exp(9.550426-5723.265/T+3.53068*np.log(T)-0.00728332*T)
    return pic
    
    

#def rho(T):
 #   rho=-1.3102e-9*T**3+3.8109e-7
    

    
Ts=np.linspace(100,260,100000)

pws=pw(Ts)

pics=pic(Ts)

s=pws/pics

logs=np.log(s)
#plt.plot(Ts,s)

r=np.linspace(0,1e-9,10000)
#Y=Ys(T)
print Y

gvol=-4*np.pi*r**3*k*T*np.log(pw(T)/pic(T))/(3*v)
print np.log(pw(T)/pic(T))
gsur=4*np.pi*r**2*Y
G=gvol+gsur

def plotG():
    plt.figure()
    plt.plot(r,gvol,label='gvol')
    plt.plot(r,gsur,label='gsur')
    plt.plot(r,G,label='G')
    plt.legend(loc='best')
Y=Ys(Ts)
incG=16*np.pi*(Y**3)*(v**2)/(3*(k*Ts*logs)**2)
expfactor=16*np.pi*(Y**3)*(v**2)/(3*(k*Ts)**3*logs**2)
V=0.06
plt.plot(Ts,expfactor)
plt.show()
A=1
J=np.exp(-incG/(k*Ts)) #Terminar esto
Jc=np.sqrt(2*Y*v/np.pi/m)*v*(pics/k*Ts)**2*J
print J
plt.figure()
plt.plot(Ts,J)


theta=(36*np.pi)**(1/3.)*v**(2/3.)*Y/(k*Ts)
Jgc=np.exp(theta)/s*Jc

plt.figure()
plt.plot(Ts,Jgc)


c=10000
j=np.exp(-c/Ts)

vt=0.1
n0=1
nn=n0-n0*j*vt
#plt.plot(Ts,nn)
vt=vt*1e6
nn=n0-n0*j*vt
#plt.plot(Ts,nn)

