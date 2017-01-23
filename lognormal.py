# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:18:01 2015

@author: eejvt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav

from scipy import stats
from scipy.optimize import curve_fit#
import os
archive_directory='/nfs/a107/eejvt/'
project='LOGNORMAL'
os.chdir(archive_directory+project)


s=readsav('/nfs/a201/eejvt/BC_INP/PD/sigmachained_GLOMAP_mode_v7_MS8_OldDMS_ACCMIP_30traer_withICE_SCAV_2001.sav')
#def lognormal_PDF(rmean,r,std):
#   X=(1/(r*std*np.sqrt(2*np.pi)))*np.exp(-(np.log(r)-np.log(rmean))**2/(2*std**2))
#   return X
def lognormal_PDF(rmean,r,std):
   X=(1/(r*np.log(std)*np.sqrt(2*np.pi)))*np.exp(-(np.log(r)-np.log(rmean))**2/(2*np.log(std)**2))
   return X
   
def integral_norm_pdf(rmean,sigma,ns,nsigma=3):
    rmax=rmean*nsigma*sigma
    rmin=rmean/(nsigma*sigma)
    step=rmin
    rs=np.arange(rmin,rmax,step)
    inner_integral=(1-np.exp(-4*ns*np.pi*rs**2))*(1/(rs*np.log(sigma)*np.sqrt(2*np.pi))*np.exp((-(np.log(rs)-np.log(rmean))**2)/(2*np.log(sigma)**2)))
    result=(inner_integral*step).sum()
    return result
#%%
rs=np.logspace(1,5,10000)
rmean=1000
std=2
fig=plt.figure()
ax=plt.subplot(311)
bx=plt.subplot(312)
cx=plt.subplot(313)
conv=np.exp(3.0*np.log(std)*np.log(std))
ax.plot(rs,lognormal_PDF(rmean,rs,std))
ax.axvline(rmean)
ax.set_xscale('log')

bx.plot(rs,4./3.*np.pi*rs**3*lognormal_PDF(rmean,rs,std))
bx.axvline(rmean*conv)
bx.set_xscale('log')
cx.plot(rs,lognormal_PDF(rmean*conv,rs,std))
cx.axvline(4./3.*np.pi*rmean**3)
cx.set_xscale('log')




   
#%%   
def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

nss=np.logspace(1,7,7).tolist()

for ns in nss:
    #ns=10**10
    
    rmeans=np.logspace(-5,-3,21)#10^-5cm (0.1um) to 10^-3 cm (10um) covers acc and coarse
    sigma=1.3
    
    ffs=[integral_norm_pdf(i,sigma,ns) for i in rmeans]
    
    plt.plot(rmeans,ffs,label='%1.0E'%ns)

plt.xscale('log')
plt.legend(loc='upper left')
plt.show()


#%%

rmean=100.
std=2
rmin=rmean/3/std
rmax=rmean*3*std
step=rmin*0.1

rs=np.arange(rmin,rmax,step)
X=lognormal_PDF(rmean,rs,std)
print X.sum()*step
print (X*rs*step).sum()
#plt.plot(rs,X)
#plt.show()
#%%
def area_lognormal(rbar,sigma,Nd):
    #print isinstance(sigma,float)
    if isinstance(sigma,float):
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        S=Nd*(2*rbar)**2*y
    else:
        S=np.zeros(rbar.shape)
        #y=np.zeros(len(rbar))
        #global S,y
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        for i in range (len(rbar)):
            S[i,]=Nd[i,]*(2*rbar[i,])**2*y[i]
    return S
    
rmeans=np.linspace(0,2,1)
#    rs=np.arange(0.1,10,0.1)
    #X=lognormal_PDF(3,rs,np.sqrt(0.5))
    #plt.plot(rs,X)
    #plt.show()
    #integral=X*0.1*rs
    #print integral.sum()
#    rmean=0.1
#    rs=np.linspace(0.1,100,1000)

std=1.2
stds=np.linspace(1.1,3,3)
stds=np.array([1.39,1.59,2])
colors=['blue','red','green']
i=0
plt.figure()
ax=plt.subplot(111)
#bx=plt.subplot(122)
for std in stds:
    INPs=np.zeros(len(rmeans))
    INPs_mean_area=np.zeros(len(rmeans))
    for idx in range(len(rmeans)):
        N=100    
        rmin=rmeans[idx]/5/std
        rmax=rmeans[idx]*5*std
        step=rmin*0.001
        rs=np.arange(rmin,rmax,step)
        ns=0.3
        A=4*np.pi*rs**2
        ff=1-np.exp(-ns*A)
        PDF=lognormal_PDF(rmeans[idx],rs,std)
        print PDF.sum()*step        
        dINP=ff*PDF*N*step
        INPs[idx]=dINP.sum()
        A_mean_area=area_lognormal(rmeans[idx],std,N)/N
        ff_mean_area=1-np.exp(-ns*A_mean_area)
        INPs_mean_area[idx]=ff_mean_area*N
        diff=INPs_mean_area-INPs
        ratio=INPs/INPs_mean_area
        #ax.plot(rs,ff,'--',label='Diff std=%1.1f'%std)
    
    #np.savetxt('diff_%1.2f'%std,diff/100)
    #np.savetxt('ratio_%1.2f'%std,ratio)
    #np.savetxt('mean_area_%1.2f'%std,INPs_mean_area*0.01)
    #ax.plot(rmeans,INPs,'-',label='Integrated std=%1.1f'%std,color=colors[int(i)])
    #ax.plot(rmeans,INPs_mean_area,'--',label='Mean area std=%1.1f'%std,color=colors[int(i)])
    #ax.plot(rmeans,(INPs_mean_area-INPs)/INPs_mean_area,':',label='Diff std=%1.1f'%std,color=colors[int(i)])
    #ax.plot(INPs_mean_area,(INPs/INPs_mean_area),'--',label='Ratio std=%1.1f'%std,color=colors[int(i)])
    #plt.xscale('log')
    i=i+1    
    #plt.yscale('log')
plt.show()        
ax.set_title('INP 100 particles ns=%1.2f'%ns)
#ax.set_ylabel('INP')
#ax.set_xlabel('$r_m$')
ax.set_ylabel('Diff')
ax.set_xlabel('INP_mean area')

ax.legend(loc='best')
plt.show()

#%%
#for std in stds:
std=2
std=1.59
std=1.39
ffa=np.loadtxt('mean_area_%1.2f'%std)[1:]
diff=np.loadtxt('ratio_%1.2f'%std)[1:]

#dns=0
#def func(x, a, b):
#    return a * np.exp(-b * x)
def func(x,a,b):
    return 1-a*np.exp(-b*x)    
def func(x,a,b):
    return x**a*np.exp(b-x)    
def func(x,a,b):
    f=np.zeros(len(x))
    for i in range(len(x)):    
        if x[i]>0.95:
            m=-6.898345129999999
            n=-m
            #f[i]=a/(x[i]**2)+b
            #f[i]=a*x[i]**3+b*x[i]**2+e*x[i]+g
            f[i]=m*x[i]+n
        if x[i]<0.95:
            f[i]=a*x[i]+b
    return f    
def func(x,a,b,c,d,f,g):
    return (a*x**3+b*x**2+c*x+d)**(-f*x)
    

    
popt,pcov = curve_fit(func, ffa, diff)
#popt=np.array([-18.59897567,   1.10249526])
#pcov=np.array([[ 0.50795402, -0.02496729],[-0.02496729,  0.00123657]])
perr = np.sqrt(np.diag(pcov))
ci = 0.95
pp = (1. + ci) / 2.
nstd = stats.norm.ppf(pp)
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

#%%

ffax=np.linspace(0,1,99)
diff_fitted=func(ffax,*popt)
diff_fitted=func(ffa,*popt)
diff_low=func(ffax,*popt_dw)
diff_high=func(ffax,*popt_up)

f = plt.figure()
ax = f.add_subplot(111)

plt.plot(ffa,diff,'ro')#, capthick=2,ecolor='black')
#plt.errorbar(data_eug[:,0],data_eug[:,2],yerr=data_eug[:,3],fmt='bo',label='Eugenol')#, capthick=2,ecolor='black')
#plt.errorbar(data_ndec[:,0],data_ndec[:,2],yerr=data_ndec[:,3],fmt='ro',label='Ndecane')#, capthick=2,ecolor='black')
#plt.title('Black Carbon Parameterization')

plt.plot(ffa,diff_fitted,'k-',lw=3)

'''
plt.text(0.2, 0.95,'Function', ha='center', va='center', transform=ax.transAxes)
plt.text(0.2, 0.9,'$if:$ $ff<0.95$ $y=0.378*x-0.015$', ha='center', va='center', transform=ax.transAxes)
plt.text(0.2, 0.85,'$if:$ $ff>0.95$ $y=-6.89*x+6.89$', ha='center', va='center', transform=ax.transAxes)
'''

plt.text(0.5, 0.85,'$y=( %1.0f \cdot x^3  %1.0f \cdot x^2  %1.0f \cdot x +%1.0f)^{-%1.5f \cdot x}$'%(popt[0],popt[1],popt[2],popt[3],popt[4]), ha='center', va='center', transform=ax.transAxes,fontsize=15)
plt.text(0.5, 0.75,'Error max =$%1.3f $'%np.absolute(diff-diff_fitted).max(), ha='center', va='center', transform=ax.transAxes)
#plt.text(0.2, 0.85,'$y=( %1.0f \cdot x^3 + %1.0f \cdot x^2 + %1.0f \cdot x +%1.0f)^{-%1.0f \cdot x}$'%(popt[0],popt[1],popt[2],popt[3],popt[4]), ha='center', va='center', transform=ax.transAxes)

#plt.plot(ffax,diff_high,'k--')
#plt.plot(ffax,diff_low,'k--')
plt.title('std=%f'%std)
plt.ylabel('Ratio Integrated/Mean Area')
plt.xlabel('Fraction frozen')
plt.savefig('/nfs/see-fs-01_users/eejvt/Lognormal plots/Parameterization_std_%1.2f.png'%std)



#np.fft.fft

#%%

ffa=np.loadtxt('mean_area_%1.2f'%std)[1:]
diff=np.loadtxt('ratio_%1.2f'%std)[1:]

#dns=0
#def func(x, a, b):
#    return a * np.exp(-b * x)
def func(x,a,b):
    return 1-a*np.exp(-b*x)    
def func(x,a,b):
    return x**a*np.exp(b-x)    
def func(x,a,b):
    f=np.zeros(len(x))
    for i in range(len(x)):    
        if x[i]>0.95:
            m=-6.898345129999999
            n=-m
            #f[i]=a/(x[i]**2)+b
            #f[i]=a*x[i]**3+b*x[i]**2+e*x[i]+g
            f[i]=m*x[i]+n
        if x[i]<0.95:
            f[i]=a*x[i]+b
    return f    
def func(x,a,b,c,d,f):
    return (a*x**3+b*x**2+c*x+d)**(-f*x)
    

    
popt,pcov = curve_fit(func, ffa, diff)
#popt=np.array([-18.59897567,   1.10249526])
#pcov=np.array([[ 0.50795402, -0.02496729],[-0.02496729,  0.00123657]])
perr = np.sqrt(np.diag(pcov))
ci = 0.95
pp = (1. + ci) / 2.
nstd = stats.norm.ppf(pp)
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

#%%

ffax=np.linspace(0,1,500)
diff_fitted=func(ffax,*popt)
diff_low=func(ffax,*popt_dw)
diff_high=func(ffax,*popt_up)

f = plt.figure()
ax = f.add_subplot(111)

plt.plot(ffa,diff,'ro')#, capthick=2,ecolor='black')
#plt.errorbar(data_eug[:,0],data_eug[:,2],yerr=data_eug[:,3],fmt='bo',label='Eugenol')#, capthick=2,ecolor='black')
#plt.errorbar(data_ndec[:,0],data_ndec[:,2],yerr=data_ndec[:,3],fmt='ro',label='Ndecane')#, capthick=2,ecolor='black')
#plt.title('Black Carbon Parameterization')

plt.plot(ffax,diff_fitted,'k-',lw=3)

plt.text(0.5, 0.95,'Function', ha='center', va='center', transform=ax.transAxes)
plt.text(0.5, 0.9,'$(a \cdot x^3+b \cdot x^2+c \cdot x+d)^{-e \cdot x}$', ha='center', va='center', transform=ax.transAxes)
plt.text(0.5, 0.85,'$a=%1.2f,b=%1.2f,c=%1.2f,d=%1.2f,e=%1.4f$'%(popt[0],popt[1],popt[2],popt[3],popt[4]), ha='center', va='center', transform=ax.transAxes)

#plt.plot(ffax,diff_high,'k--')
#plt.plot(ffax,diff_low,'k--')
plt.title('std=%f'%std)
plt.ylabel('difference')
plt.xlabel('Fraction frozen')




#np.fft.fft




