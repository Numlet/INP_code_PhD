# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:51:07 2015

@author: eejvt
"""

import pickle

class Company(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

with open('company_data.pkl', 'wb') as output:
    company1 = Company('banana', 40)
    pickle.dump(company1, output, pickle.HIGHEST_PROTOCOL)

    company2 = Company('spam', 42)
    pickle.dump(company2, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump('sdfasd', output, pickle.HIGHEST_PROTOCOL)

del company1
del company2

with open('company_data.pkl', 'rb') as input:
    company1 = pickle.load(input)
    print(company1.name)  # -> banana
    print(company1.value)  # -> 40

    company2 = pickle.load(input)
    company3 = pickle.load(input)
    print(company2.name) # -> spam
    print(company2.value)  # -> 42


#%%

x = np.zeros(3, dtype={'names':['col1', 'col2'], 'formats':['i4','f4']})

class Observation(object):
    def __init__(self,lat,lon,surf_pressure,temp,value,units):
        self.lat=lat
        self.lon=lon
        self.surf_pressure=surf_pressure
        self.temp=temp
        self.value=value
        self.units=units
    
def read_observational_data(path)
