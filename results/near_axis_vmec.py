# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 18:56:29 2021

@author: kimsp
"""

#!/opt/local/bin/python3

import os
import math
import numpy as np
from qsc.qsc import Qsc
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.io import netcdf

## Point to the current location of the file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

pi = math.pi
mu0 = 4*pi*10**(-7)

## GS2 and GX input file parameters
nperiod  = 1
drhodpsi = 1.0
rmaj     = 1.0
kxfac    = 1.0

B0 = 1
etabar = 0.403
VMECfileIn = "C:\\Users\\kimsp\\Downloads\\wout_li383_1.4m_ns201.nc"

f = netcdf.netcdf_file(VMECfileIn,'r',mmap=False)
raxis = f.variables['raxis_cc'][()]
zaxis = f.variables['zaxis_cs'][()]
NFP   = f.variables['nfp'][()]
Aminor = abs(f.variables['Aminor_p'][()])
phiVMEC = f.variables['phi'][()]
phiEDGE = abs(phiVMEC[-1])/(2*pi)

#phiEDGE = 0.01 #0.1 # toroidal flux at the boundary divided by 2*pi
# etabar  = 0.632 #0.408 # # parametrization for all first order quasisymmetric fields
# B0      = 1.00#1.55 # # magnetic field strength on axis
# raxis   = [1.0,0.173,0.0168,0.00101]#[1.4757E+00,  9.7154E-02,  6.1231E-03,  2.3576E-04, -3.0796E-05, -1.7145E-05] # # # axis shape - radial coefficients
# zaxis   = [0.0,0.159,0.0165,0.000985]#[0.0000E+00, -6.0712E-02, -4.3777E-03, -2.3272E-04, -5.4560E-05,  2.2188E-05]# # # axis shape - vertical coefficients
# NFP     = 2 # number of field periods
# I2 = 0.0
#rVMEC   = 0.01

## Resolution
Nphi     = 250 # resolution along the axis
nlambda  = 10 # GS2 quantity: resolution in the pitch-angle variable
tgridmax = 3*pi # maximum (and -1*minumum) of the field line coordinate
ntgrid   = 48 # resolution along the field line  
zscale = 1.0

normalizedtorFlux = 0.001 # normalization for the toroidal flux
alpha = 0.0 # field line label
shat  = 10**(-6) # used in the definition of GX and GS2 quantities
#Aminor = np.sqrt((2*phiEDGE)/B0) # minor radius - GS2 normalization parameter
Amajor = 1.0 #major radius - GX normalization parameter

stel = Qsc(rc=raxis,zs=-zaxis, nfp=NFP, etabar=etabar, nphi=Nphi)
iota = abs(stel.iota)
print(iota)
sigmaSol = stel.sigma
Laxis = stel.axis_length
sprime = stel.d_l_d_phi
curvature = stel.curvature
phi = stel.phi
nNormal = stel.iotaN - stel.iota
varphi = stel.varphi



# ## Geometry and normalizations
# alpha = 0.0 # field line label
# shat  = 10**(-6) # used in the definition of GX and GS2 quantities
# Aminor = 1.0 # minor radius - GS2 normalization parameter
# Amajor = 1.0 #major radius - GX normalization parameter
# normalizedtorFlux = 0.001/pi # normalization for the toroidal flux

## Output files
gs2gridNA = "gs2input.out"
gxgridNA  = "gxinput.out"

#################################
############## RUN ##############
#################################

## Obtain near-axis configuration


## Find near-axis sigma and sprime functions for this specific theta grid
sigmaTemp  = interp1d(phi,sigmaSol, kind='cubic')
sprimeTemp = interp1d(phi,sprime, kind='cubic')
curvTemp   = interp1d(phi,curvature, kind='cubic')

period=2*pi*(1-1/Nphi)/NFP
def phiToNFP(phi):
	if phi==0:
		phiP=0
	else:
		phiP=-phi%period
	return phiP

def sigma(phi):      return sigmaTemp(phiToNFP(phi))
def sprimeFunc(phi): return sprimeTemp(phiToNFP(phi))
def curvFunc(phi):   return curvTemp(phiToNFP(phi))

## GS2 geometric quantities
# rVMEC 			        = -np.sqrt((2*phiEDGE*normalizedtorFlux)/B0)
# def Phi(theta):         return (theta - alpha)/(iota - nNormal)
# def bmagNew(theta):     return ((Aminor**2)*B0*(1+rVMEC*etabar*np.cos(theta)))/(2*phiEDGE)
# # def gradparNew(theta):  return  ((2*Aminor*pi*(1+rVMEC*etabar*np.cos(theta)))/Laxis)/(sprimeFunc((alpha-theta)/(iota-nNormal))*2*pi/Laxis)
# def gradparNew(theta):  return  ((iota*2*Aminor*pi*(1+rVMEC*etabar*np.cos(theta)))/Laxis)
# def gds2New(theta):     return (((Aminor**2)*B0)/(2*phiEDGE))*((etabar**2*np.cos(theta)**2)/curvFunc(Phi(theta))**2 + (curvFunc(Phi(theta))**2*(np.sin(theta)+np.cos(theta)*sigma(Phi(theta)))**2)/etabar**2)
# def gds21New(theta):    return -(1/(2*phiEDGE))*Aminor**2*shat*((B0*etabar**2*np.cos(theta)*np.sin(theta))/curvFunc(Phi(theta))**2+(1/etabar**2)*B0*(curvFunc(Phi(theta))**2)*(np.sin(theta)+np.cos(theta)*sigma(Phi(theta)))*(-np.cos(theta)+np.sin(theta)*sigma(Phi(theta))))
# def gds22New(theta):    return (Aminor**2*B0*(shat**2)*((etabar**4)*np.sin(theta)**2+(curvFunc(Phi(theta))**4)*(np.cos(theta)-np.sin(theta)*sigma(Phi(theta)))**2))/(2*phiEDGE*(etabar**2)*curvFunc(Phi(theta))**2)
# def gbdriftNew(theta):  return (2*np.sqrt(2)*etabar*np.cos(theta))/np.sqrt(B0/phiEDGE)*(1-0*2*rVMEC*etabar*np.cos(theta))
# def cvdriftNew(theta):  return gbdriftNew(theta)
# def gbdrift0New(theta): return -2*np.sqrt(2)*np.sqrt(phiEDGE/B0)*shat*etabar*np.sin(theta)*(1-0*2*rVMEC*etabar*np.cos(theta))
# def cvdrift0New(theta): return gbdrift0New(theta)

##GX geometric quantities
rVMEC 			        = -np.sqrt((2*phiEDGE*normalizedtorFlux)/B0)
def Phi(theta):         return (theta - alpha)/(iota - nNormal)
def bmagNew(theta):     return ((Aminor**2)*B0*(1+rVMEC*etabar*np.cos(theta)))/(2*phiEDGE)
# def gradparNew(theta):  return  ((2*Aminor*pi*(1+rVMEC*etabar*np.cos(theta)))/Laxis)/(sprimeFunc((alpha-theta)/(iota-nNormal))*2*pi/Laxis)
def gradparNew(theta):  return  (((iota-nNormal)*2*Aminor*pi*(1+rVMEC*etabar*np.cos(theta)))/Laxis)
def gds2New(theta):     return (((Aminor**2)*B0)/(2*phiEDGE))*((etabar**2*np.cos(theta)**2)/curvFunc(Phi(theta))**2 + (curvFunc(Phi(theta))**2*(np.sin(theta)+np.cos(theta)*sigma(Phi(theta)))**2)/etabar**2)
def gds21New(theta):    return -(1/(2*phiEDGE))*Aminor**2*shat*((B0*etabar**2*np.cos(theta)*np.sin(theta))/curvFunc(Phi(theta))**2+(1/etabar**2)*B0*(curvFunc(Phi(theta))**2)*(np.sin(theta)+np.cos(theta)*sigma(Phi(theta)))*(-np.cos(theta)+np.sin(theta)*sigma(Phi(theta))))
def gds22New(theta):    return (Aminor**2*B0*(shat**2)*((etabar**4)*np.sin(theta)**2+(curvFunc(Phi(theta))**4)*(np.cos(theta)-np.sin(theta)*sigma(Phi(theta)))**2))/(2*phiEDGE*(etabar**2)*curvFunc(Phi(theta))**2)
def gbdriftNew(theta):  return (2*np.sqrt(2)*etabar*np.cos(theta))/np.sqrt(B0/phiEDGE)*(1-0*2*rVMEC*etabar*np.cos(theta))
def cvdriftNew(theta):  return gbdriftNew(theta)
def gbdrift0New(theta): return -2*np.sqrt(2)*np.sqrt(phiEDGE/B0)*shat*etabar*np.sin(theta)*(1-0*2*rVMEC*etabar*np.cos(theta))
def cvdrift0New(theta): return gbdrift0New(theta)


lambdamin=((2*phiEDGE)/((Aminor**2)*B0))/(1+abs(rVMEC*etabar))
lambdamax=((2*phiEDGE)/((Aminor**2)*B0))/(1-abs(rVMEC*etabar))

# lambdamin = 1/(1+abs(rVMEC*etabar))
# lambdamax = 1/(1-abs(rVMEC*etabar))
lambdavec=np.linspace(lambdamin,lambdamax,nlambda)

## GX geometric quantities
phiMax=tgridmax
phiMin=-tgridmax
pMax=(iota-nNormal)*phiMax
pMin=(iota-nNormal)*phiMin
zMax= zscale * pi
zMin=-zscale * pi
denMin=phiMin-rVMEC*etabar*np.sin(pMin)
denMax=phiMax-rVMEC*etabar*np.sin(pMax)
iotaN=iota-nNormal

thetamax = tgridmax
thetamin = -tgridmax

# gradparGX = 2*pi*(2*iotaN*pi)/(Laxis*(iotaN*(phiMax - phiMin) + etabar*rVMEC*(-np.sin(iotaN*phiMax) + np.sin(iotaN*phiMin))))
# z0 = pi - 2*pi*(iotaN*phiMax-rVMEC*etabar*np.sin(iotaN*phiMax))/(iotaN*(phiMax - phiMin) + etabar*rVMEC*(-np.sin(iotaN*phiMax) + np.sin(iotaN*phiMin)))
gradparGX = zscale * 4*pi**2*iotaN*Aminor/(Laxis*(thetamax-thetamin-rVMEC*etabar*(np.sin(thetamax)-np.sin(thetamin))))
z0 = zscale * pi*(thetamax + thetamin - rVMEC*etabar*(np.sin(thetamax)+np.sin(thetamin)))/((-thetamax+thetamin+rVMEC*etabar*(np.sin(thetamax)-np.sin(thetamin))))

def tgridGX(theta,zz):
# 	phi=Phi(theta)
# 	return zz-(z0+Laxis*gradparGX/(2*pi)*phi-rVMEC*Laxis*gradparGX*etabar/(2*pi*iotaN)*np.sin(iotaN*phi))
    return zz - (z0 - gradparGX*Laxis*(-theta+rVMEC*etabar*np.sin(theta))/(2*pi*iotaN*Aminor))

def thetaGXgrid(zz):
	sol = fsolve(tgridGX, 0.9*zz*iotaN, args=zz)
	return sol[0]
zGXgrid=np.linspace(zMin, zMax, 2*ntgrid+1)
paramThetaGX = [thetaGXgrid(zz) for zz in zGXgrid]

#Output to GS2 grid (bottleneck, thing that takes longer to do, needs to be more pythy)
nz=ntgrid*2
paramTheta = np.multiply(np.linspace(-tgridmax,tgridmax,nz+1),(iota-nNormal))

open(gs2gridNA, 'w').close()
f = open(gs2gridNA, "w")
f.write("nlambda\n"+str(nlambda)+"\nlambda")
for item in lambdavec:
	f.write("\n%s" % item)
f.write("\nntgrid nperiod ntheta drhodpsi rmaj shat kxfac q")
f.write("\n"+str(ntgrid)+" "+str(nperiod)+" "+str(nz)+" "+str(drhodpsi)+" "+str(rmaj)+" "+str(shat)+" "+str(kxfac)+" "+str(1/iota))
f.write("\ngbdrift gradpar grho tgrid")
for zz in paramTheta:
	f.write("\n"+str(gbdriftNew(zz/(iota-nNormal)))+" "+str(gradparNew(zz/(iota-nNormal)))+" 1.0 "+str(zz/(iota-nNormal)))
f.write("\ncvdrift gds2 bmag tgrid")
for zz in paramTheta:
	f.write("\n"+str(cvdriftNew(zz/(iota-nNormal)))+" "+str(gds2New(zz/(iota-nNormal)))+" "+str(bmagNew(zz/(iota-nNormal)))+" "+str(zz/(iota-nNormal)))
f.write("\ngds21 gds22 tgrid")
for zz in paramTheta:
	f.write("\n"+str(gds21New(zz/(iota-nNormal)))+" "+str(gds22New(zz/(iota-nNormal)))+" "+str(zz/(iota-nNormal)))
f.write("\ncvdrift0 gbdrift0 tgrid")
for zz in paramTheta:
	f.write("\n"+str(cvdrift0New(zz/(iota-nNormal)))+" "+str(gbdrift0New(zz/(iota-nNormal)))+" "+str(zz/(iota-nNormal)))
f.write("\nRplot Rprime tgrid")
for zz in paramTheta:
	f.write("\n0.0 0.0 "+str(zz/(iota-nNormal)))
f.write("\nZplot Rprime tgrid")
for zz in paramTheta:
	f.write("\n0.0 0.0 "+str(zz/(iota-nNormal)))
f.write("\naplot Rprime tgrid")
for zz in paramTheta:
	f.write("\n0.0 0.0 "+str(zz/(iota-nNormal)))
f.write("\n")
f.close()

#Output to GX grid (bottleneck, thing that takes longer to do, needs to be more pythy)
open(gxgridNA, 'w').close()
f = open(gxgridNA, "w")
f.write("ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale")
f.write("\n"+str(ntgrid)+" "+str(nperiod)+" "+str(nz)+" "+str(drhodpsi)+" "+str(rmaj)+" "+str(shat)+" "+str(kxfac)+" "+str(1/iota) + " " + str(thetamax/zMax))
f.write("\ngbdrift gradpar grho tgrid")
for count,zz in enumerate(paramThetaGX):
	f.write("\n"+str(gbdriftNew(zz))+" "+str(gradparGX)+" 1.0 "+str(zGXgrid[count]))
f.write("\ncvdrift gds2 bmag tgrid")
for count,zz in enumerate(paramThetaGX):
	f.write("\n"+str(cvdriftNew(zz))+" "+str(gds2New(zz))+" "+str(bmagNew(zz))+" "+str(zGXgrid[count]))
f.write("\ngds21 gds22 tgrid")
for count,zz in enumerate(paramThetaGX):
	f.write("\n"+str(gds21New(zz))+" "+str(gds22New(zz))+" "+str(zGXgrid[count]))
f.write("\ncvdrift0 gbdrift0 tgrid")
for count,zz in enumerate(paramThetaGX):
	f.write("\n"+str(cvdrift0New(zz))+" "+str(gbdrift0New(zz))+" "+str(zGXgrid[count]))
f.close()