# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:10:51 2016

@author: eejvt
"""

import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib
import sys
import glob
dir_scripts='/nfs/see-fs-01_users/eejvt/UKCA_postproc'#Change this to the downloaded folder
sys.path.append(dir_scripts)
import UKCA_lib as ukl
import iris.quickplot as qp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animationlt
from mayavi import mlab
#from cube_browser import Contour, Browser, Contourf, Pcolormesh
#%%
word1='SW'
word2='shortwave'
word1='LW'
word2='longwave'
word1='CLOUD'
word2='cloud'
word2='TEMP'
word1='ICE'
word2='ice'

#%%
path='/home/numlet/Desktop/test_files/nc/'
path='/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/BASE_RUN/All_time_steps/'
path='/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/BASE_RUN2/All_time_steps/'
path='/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/NOICE/All_time_steps/'
nc_files=glob.glob(path+'*.nc')
#print nc_files


for inc in range(len(nc_files)):
#    if word1 in nc_files[inc][len(path):] or word2 in nc_files[inc][len(path):]:
#    if 'OPTIC' in nc_files[inc][len(path):] or 'cloud' in nc_files[inc][len(path):]:
    print inc,nc_files[inc][len(path):]
#%%
inc=31
print nc_files[inc]
cube1 = iris.load(nc_files[inc])[0]
#cube2 = iris.load_cube(iris.sample_data_path('GloSea4/ensemble_002.pp'))
print cube1
#print cube2
#%%
path='/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/DEMOTT3ORD/All_time_steps/'
nc_files=glob.glob(path+'*.nc')
for inc in range(len(nc_files)):
    if word1 in nc_files[inc][len(path):] or word2 in nc_files[inc][len(path):]:
#    if 'OPTIC' in nc_files[inc][len(path):] or 'cloud' in nc_files[inc][len(path):]:
        print inc,nc_files[inc][len(path):]
#%%
inc2=152
print nc_files[inc2]
cube2 = iris.load(nc_files[inc2])[0]
print cube2

potential_temperature=iris.load(ukl.Obtain_name(path,'m01s00i004'))[0]
air_pressure=iris.load(ukl.Obtain_name(path,'m01s00i408'))[0]
p0 = iris.coords.AuxCoord(1000.0,
                          long_name='reference_pressure',
                          units='hPa')
p0.convert_units(air_pressure.units)

Rd=287.05 # J/kg/K
cp=1005.46 # J/kg/K
Rd_cp=Rd/cp

temperature=potential_temperature*(air_pressure/p0)**(Rd_cp)
print temperature.data[0,0,0,0]
temperature._var_name='temperature'
R_specific=iris.coords.AuxCoord(287.058,
                          long_name='R_specific',
                          units='J-kilogram^-1-kelvin^-1')#J/(kg·K)

air_density=(air_pressure/(temperature*R_specific))





#%%

# This example uses a MovieWriter directly to grab individual frames and
# write them to a file. This avoids any event loop integration, but has
# the advantage of working with even the Agg backend. This is not recommended
# for use in an interactive setting.
# -*- noplot -*-
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=5, metadata=metadata)

fig = plt.figure()
#l, = plt.plot([], [], 'k-o')
ax=plt.subplot(221)
bx=plt.subplot(222)
cx=plt.subplot(223)
#plt.xlim(-5, 5)
#plt.ylim(-5, 5)
time_coord=cube1.coord('time')
time_coord.convert_units('seconds since 1970-01-01 00:00:0.0')
x0, y0 = 0, 0
lats=cube1.coord('grid_latitude').points
lons=cube1.coord('grid_longitude').points

pl=40
if cube1.ndim==4:
    temps=temperature[:,pl,:,:]
    cube1=cube1[:,pl,:,:]
    cube2=cube2[:,pl,:,:]
max_val=cube1[:,:,:].data.max()
min_val=cube1[:,:,:].data.min()
min_val = np.min(cube1[:,:,:].data[np.nonzero(cube1[:,:,:].data)])
#max_val=cube1[:,:,:].data.max()
#min_val=cube1[:,:,:].data.min()
levels=np.logspace(np.log(min_val),np.log(max_val),9).tolist()
levels=np.linspace(min_val,max_val,9).tolist()
with writer.saving(fig,'/nfs/a201/eejvt/casim_videos/'+cube1.var_name+".mp4", 300):
    for i in range(len(cube1.coord('time').points)):
        ax.cla()
        bx.cla()
        cx.cla()
        big_title=cube1.var_name.replace('_',' ')+' '+datetime.datetime.fromtimestamp(time_coord.points[i]).strftime('%D %H:%M:%S')
        plt.figtext(0.3,0.95,big_title,fontsize=10)
#        plt.title(cube1.var_name.replace('_',' ')+' '+datetime.datetime.fromtimestamp(time_coord.points[i]).strftime('%D %H:%M:%S'))
        mapable=ax.contourf(lons, lats, cube1[i,:,:].data,levels, cmap='viridis')
        mapable2=ax.contour(lons, lats, temps[i,:,:].data-273.15, cmap='RdBu_r')
        plt.clabel(mapable2, inline=1, fontsize=10)
        if i==0:
            cb=plt.colorbar(mapable,label=cube1.units.origin)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        mapable=bx.contourf(lons, lats, cube2[i,:,:].data,levels, cmap='viridis')
        mapable2=bx.contour(lons, lats, temps[i,:,:].data-273.15, cmap='RdBu_r')
        plt.clabel(mapable2, inline=1, fontsize=10)
#        if i==0:
#            cb=plt.colorbar(mapable,label=cube1.units.origin)
#        plt.title(cube1.var_name.replace('_',' ')+' '+datetime.datetime.fromtimestamp(time_coord.points[i]).strftime('%D %H:%M:%S'))
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
#        mapable=ax.contour(lons, lats, cube2[i,:,:].data,levels, linewidths=10,cmap='viridis')
        mapable=cx.contourf(lons, lats, cube1[i,:,:].data-cube2[i,:,:].data,levels, cmap='viridis')
        mapable2=cx.contour(lons, lats, temps[i,:,:].data-273.15, cmap='RdBu_r')
        plt.clabel(mapable2, inline=1, fontsize=10)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
#        if i==0:
#            cb=plt.colorbar(mapable,label=cube1.units.origin)
#        plt.title(cube2.var_name.replace('_',' ')+' '+datetime.datetime.fromtimestamp(time_coord.points[i]).strftime('%D %H:%M:%S'))
#        plt.xlabel('Longitude')
#        plt.ylabel('Latitude')
        #plt.yticks([len(lats)/15*j for j in range(15)],['%1.2f'%lats[len(lats)/15*j] for j in range(15)])
        #plt.xticks([len(lons)/8*j for j in range(8)],['%1.1f'%lons[len(lons)/8*j] for j in range(8)])
#        im = plt.imshow(cube1[i,level,:,:].data, cmap='viridis',levels=[1e-6,1e-5,1e-4])
            
        writer.grab_frame()

#%%

mlab.contour3d(cube2.data[15,:,:,:])

mlab.show()
#%%
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y, Z = axes3d.get_test_data(0.05)
lats=cube1.coord('grid_latitude').points
lons=cube1.coord('grid_longitude').points
X=lats
Y=lons
X,Y=np.meshgrid(X,Y)
Z=cube1
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.show()



#%%
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = cube1.coord('grid_latitude').points
    ys = randrange(n, 0, 100)
    zs = 
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#%%
iris.quickplot.contourf(cube1[5,35,])
plt.show()
#%%
plt.figure()
plt.plot(np.linspace(0,10,10),np.linspace(0,10,10))
plt.yticks([1,2,5],[5,6,7])
#%%
arr=np.ones((500,500))
arr[300:,6:80]=8
plt.contourf(lons, lats, arr, cmap='viridis')
plt.colorbar()
arr=arr*10000
plt.contourf(lons, lats, arr, cmap='viridis')
#plt.show()

#%%
import Tkinter
import tkMessageBox

top = Tkinter.Tk()

def helloCallBack():
   tkMessageBox.showinfo( "Hello Python lalaal", "Video plotting")

B = Tkinter.Button(top, text ="Hello", command = helloCallBack)

B.pack()
top.mainloop()
#%%
from tkinter import *

root = Tk()
var = StringVar()
var.set('hello')

l = Label(nc_files, textvariable = var)
l.pack()

t = Entry(root, textvariable = var)
t.pack()

root.mainloop() # the window is now displayed
#%%
from Tkinter import *  # from tkinter import *

lst = ['a', 'b', 'c', 'd']

root = Tk()
t = Text(root)
for x in nc_files:
    t.insert(END, x + '\n')
t.pack()
root.mainloop()
#%%
from Tkinter import Tk
from tkFileDialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)
#%%
path='/nfs/a201/eejvt/CASIM/SO_KALLI/SECOND_DOMAIN/DEMOTT3ORD/'
#%%

from Tkinter import *
import tkMessageBox
import Tkinter

top = Tk()

mb=  Menubutton ( top, text="condiments", relief=RAISED )
mb.grid()
mb.menu  =  Menu ( mb, tearoff = 0 )
mb["menu"]  =  mb.menu
    
mayoVar  = IntVar()
ketchVar = IntVar()

mb.menu.add_checkbutton ( label="mayo",
                          variable=mayoVar )
mb.menu.add_checkbutton ( label="ketchup",
                          variable=ketchVar )

mb.pack()
print mayoVar
top.mainloop()

#%%


from numpy import pi, sin, cos, mgrid
dphi, dtheta = pi/250.0, pi/250.0
[phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
x = r*sin(phi)*cos(theta)*4
y = r*cos(phi)
z = r*sin(phi)*sin(theta)

# View it.
from mayavi import mlab
s = mlab.mesh(x, y, z)
mlab.show()
