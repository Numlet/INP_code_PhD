# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:53:35 2014

@author: eejvt
"""

import numpy.ma as ma
#import cartopy.crs as ccrs
#import cartopy.feature as cfeat
from scipy.io.idl import readsav
from scipy.optimize import anneal
from scipy.optimize import minimize
import scipy as sp
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from glob import glob
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm

archive_directory='/nfs/a107/eejvt/'
project='JB_TRAINING/'


os.chdir(archive_directory+project)

rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
rhocomp =rhocomp*1e+9
'''
;           Sulphate    Carbon    1.4*BC   NaCl      SiO2,      SOC   Feldspar                               
    mm_aer = [0.098,    0.012,   0.0168 , 0.05844,    0.10,    0.0168 ,0.1] ;SOC was 0.15  ; molar masses in Kg/mol
    rhocomp  = [1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.];masss density of components in Kg/m3
'''
#python_dict=1
#latlon=True
#tri=True


#names=os.listdir('/nfs/a107/eejvt/JB_TRAINING/NO_ICE_SCAV/')

def read_data(simulation):
    global a
    s={}
    a=glob(simulation+'/*.sav')

    
    for i in range (len(a)):

        s=readsav(a[i],idict=s)
        
        keys=s.keys()
        print i, len(a)
        #np.save(a[i][:-4]+'python',s[keys[i]])
        print a[i]
    for j in range(len(keys)):
        print keys[j]
        print s[keys[j]].shape, s[keys[j]].ndim
    variable_list=s.keys()
    s=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav',idict=s,verbose=1)
    s=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav',idict=s,verbose=1)
    return s, variable_list
#s1,_=read_data('WITH_ICE_SCAV')
    
def logclevs(data,n=20):
    maximum=np.amax(data)
    minimum=np.amin(data)
    print maximum, minimum
    if minimum==0:
        clevs=np.array([0])
        return clevs
    else:
        logarray=np.logspace(minimum,maximum,n)
        print logarray
        
        #t=(data,)
        
        #logarray,dev=clevsopt(logarray,data)
        #print 'optimizado'
        #print logarray,dev
        return logarray

def plot(data,title='None',projection='robin',file_name='noname',show=0,cblabel='$\mu g/ m^3$',cmap=plt.cm.RdBu_r,clevs=np.zeros(1),return_fig=0,dpi=300):
    # lon_0 is central longitude of projection.

    #clevs=np.logspace(np.amax(data),np.amin(data),levels)
    #print np.amax(data),np.amin(data)
    fig=plt.figure()
    m = fig.add_subplot(1,1,1)
    # resolution = 'c' means use crude resolution coastlines.
    m = Basemap(projection=projection,lon_0=0)
    m.drawcoastlines()
    
    #m.fillcontinents(color='coral',lake_color='aqua')
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,120.,10.))
    m.drawmeridians(np.arange(0.,360.,60.))
    #m.drawmapboundary(fill_color='aqua')
    #if (np.log(np.amax(data))-np.log(np.amin(data)))!=0:
        #clevs=logscale(data)
        #s=m.contourf(X,Y,data,30,latlon=True,cmap=cmap,clevs=clevs)#locator=ticker.LogLocator(),
    #else:
    if clevs.all()==0:
        cs=m.contourf(X,Y,data,30,latlon=True,cmap=cmap)
        cb = m.colorbar(cs,"right")
       
        '''
        clevs=logclevs(data)
        print clevs
        if clevs.all()==0:
            cs=m.contourf(X,Y,data,30,latlon=True,cmap=cmap)
        else:
            cs=m.contourf(X,Y,data,30,latlon=True,cmap=cmap,levels=clevs)
            '''
    else:
        cs=m.contourf(X,Y,data,30,latlon=True,cmap=cmap,levels=clevs,norm=colors.LogNorm())
        cb = m.colorbar(cs,"right",ticks=clevs)
    '''
    if clevs.all==0:
        cs=m.contourf(X,Y,data,30,latlon=True,cmap=cmap)
        #m.bluemarble()
        cb = m.colorbar(cs,"right",size="5%", pad="2%")
        
    else:
        cs=m.contourf(X,Y,data,30,latlon=True,cmap=cmap,clevs=clevs)
        cb = m.colorbar(cs,"right",size="5%", pad="2%")
        '''
    
    #cb = m.colorbar(cs,"right",ticks=clevs)#,size="5%", pad="2%"
    cb.set_label(cblabel)
    
    plt.title(title)
    if os.path.isdir("PLOTS/"):
        plt.savefig('PLOTS/'+file_name+'.png',format='png',dpi=dpi)
    else:
        plt.savefig(file_name+'.png',format='png',dpi=dpi)
    if show:
        plt.show()
    #print clevs
    
    if return_fig:
        return fig
    else:
        plt.close()
    
    
    

#s,var_list=read_data('NO_ICE_SCAV')


lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
X,Y=np.meshgrid(lon.glon,lat.glat)



#lat.glat=-lat.glat
#clevs=np.linspace(10**-8,10**-6,10)

X,Y=np.meshgrid(lon.glon,lat.glat)
#X=lon.glon
#Y=lat.glat
#
#data=s.tot_mc_feldspar_mm[0,:,:,0]





#plot(data,file_name='funcionaaaaaaaaaaaaa')
variable='tot_mc_oc_mm'
#data=s[variable]

def compare(sim1,sim2,variable,month=0,level=31,allvar=0,year_mean=0):
    s1,_=read_data(sim1)
    s2,_=read_data(sim2)

    keys=s1.keys()
    diff=dic_difference(s1,s2)

    if allvar:
        for i in range(len(keys)):
            data=s1[keys[i]]-s2[keys[i]]
            print s1[keys[i]].ndim, keys[i], s1[keys[i]].shape
            if s1[keys[i]].ndim==4 and s1[keys[i]].shape==(31, 64, 128, 12):
                
                if year_mean:                
                    data_plot1=do_mean(s1[keys[i]][level,:,:,:])
                    data_plot2=do_mean(s2[keys[i]][level,:,:,:])
                    data=do_mean(data)
                    plot(data[level,:,:],title='Comparison between'+sim1+'and'+sim2+' for '+keys[i],file_name='Comparison_'+sim1+'_'+sim2+'_variable_'+keys[i])
                    
                    
                else:
                    data_plot1=s1[keys[i]][level,:,:,month]
                    data_plot2=s2[keys[i]][level,:,:,month]
                    plot(data[level,:,:,month],title='Comparison between'+sim1+'and'+sim2+' for '+keys[i],file_name='Comparison_'+sim1+'_'+sim2+'_variable_'+keys[i])
                plot(data_plot1,title='Distribution of '+keys[i]+' for '+sim1,file_name='Distribution_'+sim1+'_'+keys[i])
                plot(data_plot2,title='Distribution of '+keys[i]+' for '+sim2,file_name='Distribution_'+sim2+'_'+keys[i])
                
                #plot(s1[keys[i]][level,:,:,month],title='Distribution of '+keys[i]+' for '+sim1,file_name='Distribution_'+sim1+'_'+keys[i])
                #plot(s2[keys[i]][level,:,:,month],title='Distribution of '+keys[i]+' for '+sim2,file_name='Distribution_'+sim2+'_'+keys[i])
                print 'plotted'
    else:
        a=s1[variable]
        b=s2[variable]
        data=a-b
        plot(data[level,:,:,month],title='Comparison between'+sim1+'and'+sim2,file_name='Comparison_'+sim1+'_'+sim2)
    return s1, s2, data
    
    
    
    
    
    
def do_mean(data,year_mean=1,level_mean=0):
    dim=data.ndim
    print dim
    if dim==3:
        final_data=np.zeros((len(data[:,0,0]),len(data[0,:,0])))
        if year_mean:
            for ln in range(len(data[:,0,0])):
                for lt in range(len(data[0,:,0])):
                    final_data[ln,lt]=np.mean(data[ln,lt,:])
    elif dim==4:
        final_data=np.zeros((len(data[:,0,0,0]),len(data[0,:,0,0]),len(data[0,0,:,0])))
        if year_mean:
            for pl in range(len(data[:,0,0,0])):
                for ln in range(len(data[0,:,0,0])):
                    for lt in range(len(data[0,0,:,0])):
                        final_data[pl,ln,lt]=np.mean(data[pl,ln,lt,:])
    return final_data
        
        
        
def test():
    s,_=read_data('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV')
    plot(s.tot_mc_ss_mm[0,:,:,0],file_name='prueba',show=1)
   
   

def dic_difference(d1,d2):
    keys=d1.keys()
    diff={}
    for i in range (len(keys)):
        diff[keys[i]]=d1[keys[i]]-d2[keys[i]]
    return diff

def clevsdev(clevs,data):
    naverage=data.size/(len(clevs)-1)
    dev=0
    for i in range(len(clevs)-1):
        #if clevs[i]<clevs[i+1]:
        #    dev=1e20

        n=((clevs[i+1] < data) & (data < clevs[i])).sum()
        print (clevs[i+1] < data).sum(),'+',(data < clevs[i]).sum(),'=', n
        print 'naverage',naverage
        print len(data)
        print len(clevs)
        dev=dev+np.abs(naverage-n)
        print dev
        
    return dev




def clevsopt(clevs,data):
    naverage=data.size/(len(clevs)-1)
    dev=0
    incclevs=np.zeros((len(clevs-1)))
    for _ in range(200):
        dev=0
        for i in range(len(clevs)-1):
            incclevs[i]=clevs[i]-clevs[i+1]
            #if clevs[i]<clevs[i+1]:
            #    dev=1e20
    
            n=((clevs[i+1] < data) & (data < clevs[i])).sum()
            #n=np.where((data>clevs[i+1]) & (data<clevs[i]))
            '''print n
            #n=n.size()
            print ((clevs[i+1] < data)&(data < clevs[i])).sum(),'=', n
            print 'naverage',naverage
            print len(data)
            print len(clevs)'''
            dev=dev+np.abs(naverage-n)
            if n>naverage:
                incclevs[i]=incclevs[i]*0.98
            if n<naverage:
                incclevs[i]=incclevs[i]*1.02
            clevs[i+1]=clevs[i]-incclevs[i]
            
            #print dev
        
    return clevs,dev


#s1,s2, data=compare('WITH_ICE_SCAV2','NO_ICE_SCAV','tot_mc_dust_mm',allvar=True,level=30,year_mean=0)
s1,_=read_data('WITH_ICE_SCAV2')
print 'FINE'


def demott_parametrization(n05,T):
    
    a=0.0000594
    b=3.33
    c=0.0264
    d=0.0033
    nIN=a*(273.16-T)**b*n05**(c*(273.16-T)+d)
    return nIN
    
def tobo_parametrization(n05,T):
    a=-0.074
    b=3.8
    y=0.414
    d=-9.671
    nIN=n05**(a*(273.16-T)+b)*np.exp(y*(273.16-T)+d)
    return nIN
    
t=253
pl=15
datan250=do_mean(s1.n250nm[pl,:,:,:])
nIN=demott_parametrization(datan250,t)
tc=t-273

#plot(nIN*1e-3,title='DeMott IN parametrization t=%i pl=%i'%(tc,pl),file_name='deMott_WITH_ICE_SCAV_%i_%i'%(tc,pl),cblabel='$cm^{-3}$')

#plot(datan250,title='n05',file_name='n05',cblabel='$cm^{-3}$')






def lognormal_cummulative(N,r,rbar,sigma):
    total=(N/2)*(1+sp.special.erf(np.log(r/rbar)/np.sqrt(2)/np.log(sigma)))
    return total



def plot_resolution(res=2.8):
    fig=plt.figure()
    m = fig.add_subplot(1,1,1)
    # resolution = 'c' means use crude resolution coastlines.
    m = Basemap(projection='robin',lon_0=0)
    #m.drawcoastlines()

    if res!=2.8:    
        m.drawparallels(np.arange(-90.,90,res),linewidth=0.3)
        m.drawmeridians(np.arange(0.,360.,res),linewidth=0.3)
    else:
        m.drawparallels(np.linspace(-90.,90,64),linewidth=0.3)
        m.drawmeridians(np.linspace(0.,360.,128),linewidth=0.3)        
    m.bluemarble()
    plt.savefig(archive_directory+project+'PLOTS/'+'GLOMAP_res_%f'%res+'.png',format='png',dpi=600)
    plt.show()
    
#plot_resolution(1)


from scipy.stats import lognorm
stddev = 0.859455801705594
mean = 0.418749176686875
dist=lognorm([stddev],loc=mean)
pl=15

'''
partial_acc=do_mean(s1.st_nd[2,pl,:,:,:])-lognormal_cummulative(do_mean(s1.st_nd[2,pl,:,:,:]),250e-9,do_mean(s1.rbardry[2,pl,:,:,:]),s1.sigma[2])
+do_mean(s1.st_nd[5,pl,:,:,:])-lognormal_cummulative(do_mean(s1.st_nd[5,pl,:,:,:]),250e-9,do_mean(s1.rbardry[5,pl,:,:,:]),s1.sigma[5])
'''
#for pl in range(31):


partial_acc=s1.st_nd[2,:,:,:,:]-lognormal_cummulative(s1.st_nd[2,:,:,:,:],250e-9,s1.rbardry[2,:,:,:,:],s1.sigma[2])
+s1.st_nd[5,:,:,:,:]-lognormal_cummulative(s1.st_nd[5,:,:,:,:],250e-9,s1.rbardry[5,:,:,:,:],s1.sigma[5])
n05=s1.st_nd[3,:,:,:,:]+s1.st_nd[6,:,:,:,:]+partial_acc
    
demott=demott_parametrization(n05,t)*1e-3

#plot(demott_parametrization(n05,t)*1e-3,title='DeMott IN parametrization t=%i pl=%i'%(tc,pl),cblabel='$cm^{-3}$',file_name='deMott_%i_%i'%(tc,pl),show=0)#demott_parametrization(nmott,t)

#plot(n05,title='Particles larger than $0.5\mu m$ pl=%i'%pl,file_name='n05',cblabel='$cm^{-3}$',show=0)

'''
fig=plot(tobo_parametrization(n05,t),title='Tobo parametrization t=%i pl=%i'%(tc,pl),
cblabel='$L^{-1}$',file_name='Tobo_%i_%i_'%(tc,pl),show=1,clevs=np.array([0.1,1,5,10,50,100,200]),
return_fig=1)#clevs=[800,100,50,10,5,3,2,1,0.5,0.1,0.01]clevs=logclevs(tobo_parametrization(n05,t))
'''

def area_lognormal(rbar,sigma,Nd):
    
    y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
    S=Nd*(2*rbar)**2*y
    return S
    
def feld_parametrization(T):
    ns=np.exp((-1.03802178356815*T)+275.263379304105)
    return ns
    
def kfeld_frac(s,mode,T):
    mode_vol=(s.tot_mc_su_mm_mode[mode,:,:,:,:]/rhocomp[0]
    +s.tot_mc_bc_mm_mode[mode,:,:,:,:]/rhocomp[2]
    +s.tot_mc_ss_mm_mode[mode,:,:,:,:]/rhocomp[3]
    +s.tot_mc_dust_mm_mode[mode,:,:,:,:]/rhocomp[4]
    +s.tot_mc_oc_mm_mode[mode,:,:,:,:]/rhocomp[5]
    +s.tot_mc_feldspar_mm_mode[mode,:,:,:,:]/rhocomp[6])
    print 'modevol',mode,mode_vol[:,50,50,1]
    nd=s.st_nd[mode,:,:,:,:]
    
    kfeld_volfrac=(0.35*s.tot_mc_feldspar_mm_mode[mode,:,:,:,:]/rhocomp[6])/mode_vol
    kfeld_s_ext=area_lognormal(s.rbardry[mode,:,:,:,:],s.sigma[mode],nd*kfeld_volfrac)
    print kfeld_s_ext[:,50,50,1]
    kfeld_s_int=area_lognormal(s.rbardry[mode,:,:,:,:],s.sigma[mode],nd)*kfeld_volfrac
    ns=feld_parametrization(T)
    #kfeld_INP_ext=kfeld_s_ext*ns*1e-6
    #kfeld_INP_int=kfeld_s_int*ns*1e-6
    ff_ext=1.0-np.exp(-(ns*kfeld_s_ext))
    print ff_ext[:,50,50,1]
    ff_int=1.0-np.exp(-(ns*kfeld_s_int))
    kfeld_INP_ext=ff_ext*nd*kfeld_volfrac
    kfeld_INP_int=ff_int*nd
    return kfeld_INP_ext,kfeld_INP_int

kfeld_INP_ext=0
kfeld_INP_int=0
for i in range(7):
    k_e,k_i=kfeld_frac(s1,i,253)
    kfeld_INP_ext=kfeld_INP_ext+k_e
    kfeld_INP_int=kfeld_INP_int+k_i

#plot(kfeld_INP_ext[17,:,:,1],show=1)












def read_INP_data(path,header=0):
    data=np.genfromtxt(path,delimiter="\t",skip_header=header)
    return data

INPconc=read_INP_data("/nfs/a107/eejvt/INP_DATA/MURRAY.dat",header=1)


def lon_positive(data):
    for i in range(len(data)):
        if data[i]<0:
            data[i]=data[i]+360
    return data

def fit_pl_to_grid(data,pls=30):
    #plvs=np.linspace(0,1000,pls)
    for i in range(len(data)):
        data[i]=round(data[i]/1000.*pls)
    return data
    
    
    
def fit_lon_to_grid(data,grid_points=128):
    for i in range (len(data)):
        
        data[i]=round(data[i]/360.*grid_points)#*360./grid_points
    return data

def fit_lat_to_grid(data,grid_points=64):
    for i in range (len(data)):
        
        data[i]=data[i]+90
        data[i]=grid_points-round(data[i]/180.*grid_points)
    return data

def obtain_points_from_data(data_map,data_points,lat_points_index=3,lon_points_index=4,pl_points_index=5):
    ndata=len(data_points[:,0])    
    simulated_points=np.zeros((ndata,3))
    print data_points
    for i in range(ndata):
        
        simulated_points[i,0]=demott_parametrization(np.mean(data_map[data_points[i,pl_points_index],data_points[i,lat_points_index],data_points[i,lon_points_index]]),data_points[i,1]+273)
        print simulated_points[i,0]
        simulated_points[i,1]=lat.glat[data_points[i,lat_points_index]]
        simulated_points[i,2]=lon.glon[data_points[i,lon_points_index]]
    return simulated_points
    
def plot_comparison(simulated_points,data_points,lat_points_index=3,lon_points_index=4,inpconc_index=2,pl_points_index=5):
    plt.scatter(data_points[:,inpconc_index],simulated_points[:,0],c=data_points[:,1])
    plt.colorbar()
    plt.xlabel('simulated')
    plt.ylabel('measured')
    x=np.logspace(np.min(np.log(simulated_points[:,0]+data_points[:,inpconc_index])),np.max(np.log(simulated_points[:,0]+data_points[:,inpconc_index])),100)
        
    plt.plot(x,x,'k-')
    plt.plot(x,10*x,'k--')
    plt.plot(x,0.1*x,'k--')
    plt.xlim(x[0],x[-1])
    plt.xscale('log')
    plt.yscale('log')
    
INPconc=INPconc[INPconc[:,1]<-15]
INPconc[:,4]=lon_positive(INPconc[:,4])
INPconc[:,4]=fit_lon_to_grid(INPconc[:,4])
INPconc[:,3]=fit_lat_to_grid(INPconc[:,3])
print INPconc[:,5]
INPconc[:,5]=fit_pl_to_grid(INPconc[:,5])
print INPconc[:,5]

simulated_points=obtain_points_from_data(demott,INPconc)


plot_comparison(simulated_points,INPconc)


plt.show()
np.corrcoef(simulated_points[:,0],INPconc[:,2])