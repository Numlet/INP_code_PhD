
import numpy as np
import matplotlib.pyplot as plt

def convertCtoK(data):
    data[:,0]=data[:,0]+273.15
    return data


ilitedat = np.genfromtxt("ilite.dat",delimiter="\t")  
#ilitedat=convertCtoK(ilitedat)

saharadustdat = np.genfromtxt("saharadust.dat",delimiter="\t")  
#saharadustdat=convertCtoK(saharadustdat)

#asiadustdat = np.genfromtxt("asiadust.dat",delimiter="\t")  
#asiadustdat=convertCtoK(asiadustdat)

dereje= np.genfromtxt("/nfs/see-fs-01_users/eejvt/Documents/dereje_data.dat",delimiter="\t")  
volcanicdat = np.genfromtxt("volcanic.dat",delimiter="\t")  
#volcanicdat=convertCtoK(volcanicdat)
np.sort(volcanicdat)
kaolinitedat = np.genfromtxt("kaolinite.dat",delimiter="\t")  
#kaolinitedat=convertCtoK(kaolinitedat)

demottdat = np.genfromtxt("demott.dat",delimiter="\t")  
#demottdat=convertCtoK(demottdat)
marinedat = np.genfromtxt("marine.dat",delimiter="\t")
funguralsdat=np.genfromtxt('funguralspores.dat', delimiter='\t')
psyringaedat=np.genfromtxt('psyringae.dat', delimiter='\t')
polendat=np.genfromtxt('polen.dat', delimiter='\t')

rpol=2e-3
nmaxpol=1e-6
nminpol=1e-9
rpsy=3.5e-4
nmaxpsy=1e-2
nminpsy=1e-5
rfs=1.6e-4
nmaxfs=1e-3
nminfs=1e-6
rsoot=6e-6
ril=5e-5
nmaxil=50
nminil=0.1
rkao=4E-5
nmaxkao=60
nminkao=0.1
rvol=4.5E-5
nmaxvol=150
nminvol=30


def Area(r):
    ar=4*3.1416*r**2
    return ar



def Ffunction(data,r):
   area=Area(r)
   data[:,1]=data[:,1]*area
   data[:,1]=1-np.exp(-data[:,1])
   return data 




def maxandmin(data,maximum,minimum):
    datamax=data[:,1]*maximum
    datamin=data[:,1]*minimum
    return datamax,datamin
    
    


def plotline(ax, data, maxdat,mindat,colorline,name):
    l1=ax.fill_between(data[:,0],mindat,maxdat, color=colorline, alpha=0.7,label=name)
    #plt.plot([], [], color=colorline,label=name, linewidth=10)


def prepare(data,r,nmax,nmin):
    data=Ffunction(data,r)
    maxvalues,minvalues=maxandmin(data,nmax,nmin)
    return data,maxvalues,minvalues
    
    
ilitedat,maxil,minil=prepare(ilitedat,ril,nmaxil,nminil)
volcanicdat,maxvol,minvol=prepare(volcanicdat,rvol,nmaxvol,nminvol)
kaolinitedat,maxkao,minkao=prepare(kaolinitedat,rkao,nmaxkao,nminkao)
funguralsdat,maxfs,minfs=prepare(funguralsdat,rfs,nmaxfs,nminfs)
psyringaedat,maxpsy,minpsy=prepare(psyringaedat,rpsy,nmaxpsy,nminpsy)
polendat,maxpol,minpol=prepare(polendat,rpol,nmaxpol,nminpol)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#l1=ax.fill_between(ilitedat[:,0],minil,maxil, color="green", alpha=0.5,label='Ilite')
l2=ax.fill_between(volcanicdat[:,0],minvol,maxvol,color='red',label='volcanic ash')
l3=ax.fill_between(kaolinitedat[:,0],minkao,maxkao,color='cyan',label='kaolinite')
l4=ax.fill_between(funguralsdat[:,0],minfs,maxfs,color='yellow',label='Fungural spores')
#plt.plot([], [], color='green',label='Ilite', linewidth=10)
#plt.plot([], [],color='red',label='volcanic ash', linewidth=10)
#plt.plot([], [],color='cyan',label='kaolinite', linewidth=10)
#plt.plot([], [],color='yellow',label='Fungural spores', linewidth=10)
line, = ax.plot(demottdat[:,0],demottdat[:,1],'ro',label='DeMott data')

plt.plot(dereje[:,0], dereje[:,1],'o',color='blue',label='Leeds Observations')

plotline(ax,ilitedat,maxil,minil,'green','ilite')  
plotline(ax,psyringaedat,maxpsy,minpsy,'orange','P.syringae')
plotline(ax,polendat,maxpol,minpol,'pink','Polen')
tfel=np.linspace(248,268,1000)
tsoot=np.linspace(-18,-35,1000)
l1=ax.fill_between(tfel-273.15,np.exp(-1.038*tfel+275.26)*0.01*0.1*Area(5e-5),np.exp(-1.038*tfel+275.26)*0.25*50*Area(5e-5),alpha=0.5, color='purple',label='Feldspar')
#plt.plot([], [],color='purple',label='Feldespar', linewidth=10)
#l5=ax.fill_between(tsoot,np.exp(-0.0101*tsoot**2-0.8525*tsoot+0.7667)*1*Area(5e-6)
#,np.exp(-0.0101*tsoot**2-0.8525*tsoot+0.7667)*100*Area(5e-6), color='grey',label='Soot')
#plt.plot([], [],color='grey',label='Soot', linewidth=10)

ax.set_yscale('log')
ax.invert_xaxis()
ax.set_ylabel('INP ($cm^3$)')
ax.set_xlabel('Temperature')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=4, fancybox=True, shadow=True)
plt.xlim(0,-38)

fig.savefig('IN',format='png')


#%%