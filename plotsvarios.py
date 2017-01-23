# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:22:03 2015

@author: eejvt
"""

import numpy as np
import matplotlib.pyplot as plt



step=0.00001
D=np.arange(0.00001,0.10,step)

y=0.6
ax=2.9e7
b=2
v=y*ax*D**b

plt.plot(D,v)
plt.xscale('log')
plt.yscale('log')
plt.show()















'''
step=0.001
D=np.arange(0.001,10,step)
px=3.5
m=1
nx=m**(-(4+px))
A=2


fig=plt.figure()
y=nx*D**px*np.exp(-A*D)
ax=plt.subplot(1,2,1)
ax.plot(D,y)
bx=plt.subplot(1,2,2)

A=1
y=nx*D**px*np.exp(-A*D)
ax.plot(D,y)
ax.set_xscale('log')

px=2
m=1
nx=m**(-(4+px))
A=2
y=nx*D**px*np.exp(-A*D)
bx.plot(D,y)
A=1
y=nx*D**px*np.exp(-A*D)

bx.plot(D,y)
bx.set_xscale('log')
#plt.show()
ntot=np.sum(y*step)
pw=100
qx=pw*np.pi/6*np.sum(y*step)
plt.figure()
'''