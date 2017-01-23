# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:11:55 2014

@author: eejvt
"""

import numpy as np
import matplotlib.pyplot as plt

def convertCtoK(data):
    data[:,0]=data[:,0]+273.15
    return data


ilitedat = np.genfromtxt("ilite.dat",delimiter="\t")  
ilitedat=convertCtoK(ilitedat)

saharadustdat = np.genfromtxt("saharadust.dat",delimiter="\t")  
saharadustdat=convertCtoK(saharadustdat)

asiadustdat = np.genfromtxt("asiadust.dat",delimiter="\t")  
asiadustdat=convertCtoK(asiadustdat)

volcanicdat = np.genfromtxt("volcanic.dat",delimiter="\t")  
volcanicdat=convertCtoK(volcanicdat)

caolitedat = np.genfromtxt("caolite.dat",delimiter="\t")  
caolitedat=convertCtoK(caolitedat)

demottdat = np.genfromtxt("demott.dat",delimiter="\t")  
demottdat=convertCtoK(demottdat)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
line, = ax.plot(demottdat[:,0],demottdat[:,1],'ro')
ax.set_yscale('log')

#plt.plot(demottdat[:,0],demottdat[:,1],'ro')
#plt.ylabel('arbitrary')
#plt.yscale('log')