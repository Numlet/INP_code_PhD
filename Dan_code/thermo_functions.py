#from IPython import embed  #for debugging - put embed() in code for breakpoint


import numpy as np

def qsatw(T, p): #Saturation vapour pressure. Authour: Hamish Gordon (U. Leeds)
    Tzero=273.16
    T0 = np.copy(T)
    T0.fill(273.16)
    #parentheses=line continuation in python
    try:
        log10esw = (10.79574*(1-np.divide(T0,T))-5.028*np.log10(np.divide(T,T0))
                    + 1.50475e-4*(1-np.power(10,(-8.2369*(np.divide(T,T0)-1))))
                    + 0.42873e-3*(np.power(10,(4.76955*(1-np.divide(T,T0))))-1) + 2.78614)
    except Exception as e:
        print str(e)
        print 'exception at T='+str(T)
    esw = 10**log10esw
    qsw = 0.62198*esw/(p-esw)

    return qsw

def dqdz(T, p): #Rate of change of qL with height. Authour: Hamish Gordon (U. Leeds). Based on formula derived by Daniel Grosvenor (U. Leeds) - see Ahmad et al. 2013 Tellus B
    g=9.8
    cp = 1004.67 # J/kgK
#    Hv = 2501000.0 #Latent heat of vap. - chagned to temp dependent one.
    Rsd = 287.04
#    Rsw = 461.5 #not used
    eps = 0.622
    #qsatw2 = np.vectorize(qsatw)
    r = qsatw(T, p) #r is the mixing ratio in kg/kg
    Lw = 1000.0*(2500.8-2.36*(T-273.15)+
                  0.0016*np.power((T-273.15),2) -
                  0.00006*np.power((T-273.15),3.0)) # J/kg wikipedia from Rogers & Yau, a Short Course in Cloud Physics; approx 2.6e6

    # Calculate the saturated moist adiabatic lapse rate using AMS glossary definition	
    Gamma_w_num = g*(1+(Lw/Rsd)*np.divide(r,T))
#    Gamma_w_den = (cp + ((Hv**2)/Rsd)*np.divide(r,np.power(T,2.0))) #K/m
    Gamma_w_den = (cp + (eps*(Lw**2)/Rsd)*np.divide(r,np.power(T,2.0))) #K/m
    Gamma_w  = np.divide(Gamma_w_num,Gamma_w_den)

    cpa = np.copy(T)
    Gamma_d = np.copy(T)
    cpa.fill(cp)
    Gamma_d.fill(g/cp)
    Cw1 = np.divide(cpa,Lw)*(Gamma_d-Gamma_w) #Ahmad 2013 Tellus B
    # units: m^-1, so need to multiply by rho_a (Grosvenor)
    Cw = (p/Rsd)*np.divide(Cw1,T) # units kgm^-4

    #embed()
    return Cw


