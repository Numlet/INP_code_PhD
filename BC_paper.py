# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:16:06 2016

@author: eejvt
"""

import numpy as np
import sys
sys.path.append('/nfs/a107/eejvt/PYTHON_CODE')
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

#%%
class run:
    def __init__(self):
        self.temps=[]
        self.ff=[]
        self.ns=[]

runs_dict={}
jump_runs=['B4MB3.1', 'B3S1.2', 'B3S1.1']
for icol in range(100):
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


plt.figure()
all_temps=[]
all_ns=[]
for run in runs_dict:
    if '10-2' in run:
        c='b'
        for T in runs_dict[run].temps:all_temps.append(T)
        for T in runs_dict[run].temps:all_temps.append(T)
    elif '10-3' in run:
        c='r'
    else:
        c='g'
    plt.plot(runs_dict[run].temps,runs_dict[run].ns,'o',label=run.split(" ", 1)[0],c=c)
    
plt.yscale('log')
plt.legend()
    
plt.figure()
for run in runs_dict:
    if '10-2' in run:
        c='b'
    elif '10-3' in run:
        c='r'
    else:
        c='g'
    plt.plot(runs_dict[run].temps,runs_dict[run].ff,'o',label=run.split(" ", 1)[0],c=c)
    














