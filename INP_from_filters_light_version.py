# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:01:52 2016

@author: eejvt

Code developed by Jesus Vergara Temprado
Contact email eejvt@leeds.ac.uk
University of Leeds 2015

"""

'''
INP calculations light version for filters

Derivation:

ff=1-e**(-lambda)

lambda=[INP]*volume_of_air*Area_droplet/Area_filter=[INP]*volume_of_air*Rd**2/Rf**2

->

#####[INP]=-ln(1-ff)*(Rf**2/Rd**2)/Volume_of_air##########

-ln(1-ff)*(Rf**2/Rd**2) is dimenssionless  -> units of [INP] == units of 1/Volume_of air


'''

#importing modules many are not needed but still
import numpy as np
import sys
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np

#defining function for calculate the radious of a droplet in a filter as function of the contact angle and volume
def radius_of_droplet(conc_ang,vol=1e-9):
    #contact angle in degrees. it will convert it
    #volume in meters^3
    conc_ang=conc_ang*np.pi/180.
    #this derivation is much longer, ask me or hannah about that if you are interested. 
    radius=((3*vol)*np.sin(conc_ang)**3/(4*np.pi-np.pi*(2+3*np.cos(conc_ang)-np.cos(conc_ang)**3)))**(1/3.)
    return radius


#setting parameters
folder='\\server_example\\folder_example\\run_example\\'
file_temps='temps.csv'
folder='/nfs/a86/shared/Mace Head 15/MaceHeadFilters/Jesus/'
temperatures=np.genfromtxt(folder+file_temps,delimiter=',',dtype=str)


contact_angle=100#change this or check 

Rd=radius_of_droplet(contact_angle)
R_filter=0.0185892#m

volume=8000# change this or automate reading it
volume=9945.6# change this or automate reading it
events=len(temperatures)

#to calculate
ff=np.linspace(1,events,events)/float(events)
INP=-np.log(1-ff)*(R_filter**2/Rd**2)/volume


#to save
header='temperatures,ff,INP'# add units to INP if you like ex: 'temperatures,ff,INP (L^-3)'
data=np.zeros((events,3))
data[:,0]=temperatures
data[:,1]=ff
data[:,2]=INP

np.savetxt(folder+'INP.csv',data,header=header,delimiter=',')

#You can add some plotting here

plt.plot(temperatures,INP,'ro')

plt.xlabel('Temperature')
plt.ylabel('$INP (L^{-1})$')
plt.yscale('log')
plt.savefig(folder+'INP_plot.png')
plt.close()



