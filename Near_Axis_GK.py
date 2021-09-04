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

class Near_Axis_GK:
    def __init__(self,raxis=[0],zaxis=[0],NFP=1,etabar=1,Nphi=250,phiEDGE=1.0,B0=1.0,normalizedtorFlux=0.001,alpha=0.0,shat=10^(-6),stel=""):
        self.Nphi = Nphi
        self.normalizedtorFlux = normalizedtorFlux
        self.alpha = alpha
        self.shat = 10**(-6)
        
        self.pi = math.pi
        self.mu0 = 4*self.pi*10**(-7)
        
        ## GS2 and GX input file parameters
        self.nperiod  = 1
        self.drhodpsi = 1.0
        self.rmaj     = 1.0
        self.kxfac    = 1.0
        
        ## Resolution
        self.nlambda  = 10 # GS2 quantity: resolution in the pitch-angle variable
        self.tgridmax = 3*self.pi # maximum (and -1*minumum) of the field line coordinate
        self.ntgrid   = 48 # resolution along the field line  
        self.zscale = 1.0
        
        if stel=="":
            self.raxis = raxis
            self.zaxis = zaxis
            self.NFP = NFP
            self.etabar = etabar
            self.phiEDGE = phiEDGE
            self.Aminor = np.sqrt((2*phiEDGE)/B0)
            self.B0 = B0
            
            self.rVMEC = -np.sqrt((2*self.phiEDGE*self.normalizedtorFlux)/self.B0)
            self.drho = 1/(self.B0*self.rVMEC)
            self.etabar = etabar
            self.make_stel()
            
        else:
            self.read_stel_file(stel=stel)
            self.make_stel()
        
        self.make_stel()
    def read_stel_file(self,stel):
        # abspath = os.path.abspath(__file__)
        # dname = os.path.dirname(abspath)
        # os.chdir(dname)
        
        VMECfileIn = "/home/pk123/Documents/GK_Files/test_cases/VMEC/" + stel + "/wout_" + stel + ".nc"
        boozmnFile = "/home/pk123/Documents/GK_Files/test_cases/VMEC/" + stel + "/boozmn_" + stel + ".nc"
        f = netcdf.netcdf_file(VMECfileIn,'r',mmap=False)
        bf = netcdf.netcdf_file(boozmnFile,'r',mmap=False)
        self.raxis = f.variables['raxis_cc'][()]
        self.zaxis = f.variables['zaxis_cs'][()]
        self.NFP   = f.variables['nfp'][()]
        self.Aminor = abs(f.variables['Aminor_p'][()])
        self.phiVMEC = f.variables['phi'][()]
        self.phiEDGE = abs(self.phiVMEC[-1])/(2*self.pi)
        self.B0 = bf.variables['bmnc_b'][(0)][0]
        
        self.rVMEC = -np.sqrt((2*self.phiEDGE*self.normalizedtorFlux)/self.B0)
        self.drho = 1/(self.B0*self.rVMEC)
        
        self.etabar = self.get_NearAxis(boozFile=boozmnFile,vmecFile = VMECfileIn)
        
        # sqrt_s_over_r = np.sqrt(np.pi*B0/self.phiEDGE)
        # print(sqrt_s_over_r/)
        
        self.make_stel()
    def make_stel(self):
        stel = Qsc(rc=self.raxis,zs=-self.zaxis, nfp=self.NFP, etabar=self.etabar, nphi=self.Nphi)
        self.iota = abs(stel.iota)
        sigmaSol = stel.sigma
        self.Laxis = stel.axis_length
        sprime = stel.d_l_d_phi
        curvature = stel.curvature
        phi = stel.phi
        self.nNormal = stel.iotaN - stel.iota
        
        self.sigmaTemp  = interp1d(phi,sigmaSol, kind='cubic')
        self.sprimeTemp = interp1d(phi,sprime, kind='cubic')
        self.curvTemp   = interp1d(phi,curvature, kind='cubic')
    
    def phiToNFP(self,phi):
        period=2*self.pi*(1-1/self.Nphi)/self.NFP
        if phi==0:
            phiP=0
        else:
            phiP=-phi%period
        return phiP

    def sigma(self,phi):      return self.sigmaTemp(self.phiToNFP(phi))
    def sprimeFunc(self,phi): return self.sprimeTemp(self.phiToNFP(phi))
    def curvFunc(self,phi):   return self.curvTemp(self.phiToNFP(phi))
    

    def Phi(self,theta):         return (theta - self.alpha)/(self.iota - self.nNormal)
    def bmagNew(self,theta):     return ((self.Aminor**2)*self.B0*(1+self.rVMEC*self.etabar*np.cos(theta)))/(2*self.phiEDGE)
    # def gradparNew(theta):  return  ((2*Aminor*pi*(1+rVMEC*etabar*np.cos(theta)))/Laxis)/(sprimeFunc((alpha-theta)/(iota-nNormal))*2*pi/Laxis)
    def gradparNew(self,theta):  return  (((self.iota-self.nNormal)*2*self.Aminor*self.pi*(1+self.rVMEC*self.etabar*np.cos(theta)))/self.Laxis)
    def gds2New(self,theta):     return (((self.Aminor**2)*self.B0)/(2*self.phiEDGE))*((self.etabar**2*np.cos(theta)**2)/self.curvFunc(self.Phi(theta))**2 + (self.curvFunc(self.Phi(theta))**2*(np.sin(theta)+np.cos(theta)*self.sigma(self.Phi(theta)))**2)/self.etabar**2)
    def gds21New(self,theta):    return -(1/(2*self.phiEDGE))*self.Aminor**2*self.shat*((self.B0*self.etabar**2*np.cos(theta)*np.sin(theta))/self.curvFunc(self.Phi(theta))**2+(1/self.etabar**2)*self.B0*(self.curvFunc(self.Phi(theta))**2)*(np.sin(theta)+np.cos(theta)*self.sigma(self.Phi(theta)))*(-np.cos(theta)+np.sin(theta)*self.sigma(self.Phi(theta))))
    def gds22New(self,theta):    return (self.Aminor**2*self.B0*(self.shat**2)*((self.etabar**4)*np.sin(theta)**2+(self.curvFunc(self.Phi(theta))**4)*(np.cos(theta)-np.sin(theta)*self.sigma(self.Phi(theta)))**2))/(2*self.phiEDGE*(self.etabar**2)*self.curvFunc(self.Phi(theta))**2)
    def gbdriftNew(self,theta):  return (2*np.sqrt(2)*self.etabar*np.cos(theta))/np.sqrt(self.B0/self.phiEDGE)*(1-0*2*self.rVMEC*self.etabar*np.cos(theta))
    def cvdriftNew(self,theta):  return self.gbdriftNew(theta)
    def gbdrift0New(self,theta): return -2*np.sqrt(2)*np.sqrt(self.phiEDGE/self.B0)*self.shat*self.etabar*np.sin(theta)*(1-0*2*self.rVMEC*self.etabar*np.cos(theta))
    def cvdrift0New(self,theta): return self.gbdrift0New(theta)
    def gradpsisq(self,theta):   return self.rVMEC^2*self.B0^2/(self.etabar^2*self.curvFunc(self.Phi(theta))^2)*(self.etabar^4*np.sin(theta)^2 + self.curvFunc(self.Phi(theta))^4*(np.cos(theta)-self.sigma(self.Phi(theta))*np.sin(theta))^2)
    def grho(self,theta):        return self.drho*np.sqrt(self.gradpsisq(theta))
    
    def make_gs2(self):
        
        lambdamin=((2*self.phiEDGE)/((self.Aminor**2)*self.B0))/(1+abs(self.rVMEC*self.etabar))
        lambdamax=((2*self.phiEDGE)/((self.Aminor**2)*self.B0))/(1-abs(self.rVMEC*self.etabar))
        lambdavec=np.linspace(lambdamin,lambdamax,self.nlambda)

        #Output to GS2 grid (bottleneck, thing that takes longer to do, needs to be more pythy)
        gs2gridNA = "gs2input.out"
        nz=self.ntgrid*2
        paramTheta = np.multiply(np.linspace(-self.tgridmax,self.tgridmax,nz+1),(self.iota-self.nNormal))
        
        open(gs2gridNA, 'w').close()
        f = open(gs2gridNA, "w")
        f.write("nlambda\n"+str(self.nlambda)+"\nlambda")
        for item in lambdavec:
        	f.write("\n%s" % item)
        f.write("\nntgrid nperiod ntheta drhodpsi rmaj shat kxfac q")
        f.write("\n"+str(self.ntgrid)+" "+str(self.nperiod)+" "+str(nz)+" "+str(self.drhodpsi)+" "+str(self.rmaj)+" "+str(self.shat)+" "+str(self.kxfac)+" "+str(1/self.iota))
        f.write("\ngbdrift gradpar grho tgrid")
        for zz in paramTheta:
        	f.write("\n"+str(self.gbdriftNew(zz/(self.iota-self.nNormal)))+" "+str(self.gradparNew(zz/(self.iota-self.nNormal)))+" 1.0 "+str(zz/(self.iota-self.nNormal)))
        f.write("\ncvdrift gds2 bmag tgrid")
        for zz in paramTheta:
        	f.write("\n"+str(self.cvdriftNew(zz/(self.iota-self.nNormal)))+" "+str(self.gds2New(zz/(self.iota-self.nNormal)))+" "+str(self.bmagNew(zz/(self.iota-self.nNormal)))+" "+str(zz/(self.iota-self.nNormal)))
        f.write("\ngds21 gds22 tgrid")
        for zz in paramTheta:
        	f.write("\n"+str(self.gds21New(zz/(self.iota-self.nNormal)))+" "+str(self.gds22New(zz/(self.iota-self.nNormal)))+" "+str(zz/(self.iota-self.nNormal)))
        f.write("\ncvdrift0 gbdrift0 tgrid")
        for zz in paramTheta:
        	f.write("\n"+str(self.cvdrift0New(zz/(self.iota-self.nNormal)))+" "+str(self.gbdrift0New(zz/(self.iota-self.nNormal)))+" "+str(zz/(self.iota-self.nNormal)))
        f.write("\nRplot Rprime tgrid")
        for zz in paramTheta:
        	f.write("\n0.0 0.0 "+str(zz/(self.iota-self.nNormal)))
        f.write("\nZplot Rprime tgrid")
        for zz in paramTheta:
        	f.write("\n0.0 0.0 "+str(zz/(self.iota-self.nNormal)))
        f.write("\naplot Rprime tgrid")
        for zz in paramTheta:
        	f.write("\n0.0 0.0 "+str(zz/(self.iota-self.nNormal)))
        f.write("\n")
        f.close()
        
    def tgridGX(self,theta,zz):
        # 	phi=Phi(theta)
        # 	return zz-(z0+Laxis*gradparGX/(2*pi)*phi-rVMEC*Laxis*gradparGX*etabar/(2*pi*iotaN)*np.sin(iotaN*phi))
        return zz - (self.z0 - self.gradparGX*self.Laxis*(-theta+self.rVMEC*self.etabar*np.sin(theta))/(2*self.pi*(self.iota-self.nNormal)*self.Aminor))

    def thetaGXgrid(self,zz):
        sol = fsolve(self.tgridGX, 0.9*zz*(self.iota-self.nNormal), args=zz)
        return sol[0]

    def make_gx(self):
        gxgridNA  = "gxinput.out"
        
        ## GX geometric quantities
        # phiMax=self.tgridmax
        # phiMin=-self.tgridmax
        # pMax=(self.iota-self.nNormal)*phiMax
        # pMin=(self.iota-self.nNormal)*phiMin
        zMax= self.zscale * self.pi
        zMin=-self.zscale * self.pi
        # denMin=phiMin-self.rVMEC*self.etabar*np.sin(pMin)
        # denMax=phiMax-self.rVMEC*self.etabar*np.sin(pMax)
        iotaN=self.iota-self.nNormal
        nz=self.ntgrid*2

        thetamax = self.tgridmax
        thetamin = -self.tgridmax
        
        # gradparGX = 2*pi*(2*iotaN*pi)/(Laxis*(iotaN*(phiMax - phiMin) + etabar*rVMEC*(-np.sin(iotaN*phiMax) + np.sin(iotaN*phiMin))))
        # z0 = pi - 2*pi*(iotaN*phiMax-rVMEC*etabar*np.sin(iotaN*phiMax))/(iotaN*(phiMax - phiMin) + etabar*rVMEC*(-np.sin(iotaN*phiMax) + np.sin(iotaN*phiMin)))
        self.gradparGX = self.zscale * 4*self.pi**2*iotaN*self.Aminor/(self.Laxis*(thetamax-thetamin-self.rVMEC*self.etabar*(np.sin(thetamax)-np.sin(thetamin))))
        self.z0 = self.zscale * self.pi*(thetamax + thetamin - self.rVMEC*self.etabar*(np.sin(thetamax)+np.sin(thetamin)))/((-thetamax+thetamin+self.rVMEC*self.etabar*(np.sin(thetamax)-np.sin(thetamin))))
        
        zGXgrid=np.linspace(zMin, zMax, 2*self.ntgrid+1)
        paramThetaGX = [self.thetaGXgrid(zz) for zz in zGXgrid]
        
        #Output to GX grid (bottleneck, thing that takes longer to do, needs to be more pythy)
        open(gxgridNA, 'w').close()
        f = open(gxgridNA, "w")
        f.write("ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale")
        f.write("\n"+str(self.ntgrid)+" "+str(self.nperiod)+" "+str(nz)+" "+str(self.drhodpsi)+" "+str(self.rmaj)+" "+str(self.shat)+" "+str(self.kxfac)+" "+str(1/self.iota) + " " + str(thetamax/zMax))
        f.write("\ngbdrift gradpar grho tgrid")
        for count,zz in enumerate(paramThetaGX):
        	f.write("\n"+str(self.gbdriftNew(zz))+" "+str(self.gradparGX)+" 1.0 "+str(zGXgrid[count]))
        f.write("\ncvdrift gds2 bmag tgrid")
        for count,zz in enumerate(paramThetaGX):
        	f.write("\n"+str(self.cvdriftNew(zz))+" "+str(self.gds2New(zz))+" "+str(self.bmagNew(zz))+" "+str(zGXgrid[count]))
        f.write("\ngds21 gds22 tgrid")
        for count,zz in enumerate(paramThetaGX):
        	f.write("\n"+str(self.gds21New(zz))+" "+str(self.gds22New(zz))+" "+str(zGXgrid[count]))
        f.write("\ncvdrift0 gbdrift0 tgrid")
        for count,zz in enumerate(paramThetaGX):
        	f.write("\n"+str(self.cvdrift0New(zz))+" "+str(self.gbdrift0New(zz))+" "+str(zGXgrid[count]))
        f.close()




    def get_NearAxis(self,boozFile, vmecFile, max_s_for_fit = 0.5, N_phi = 500):
        # Read properties of BOOZ_XFORM output file
        f = netcdf.netcdf_file(boozFile,'r',mmap=False)
        bmnc = f.variables['bmnc_b'][()]
        ixm = f.variables['ixm_b'][()]
        ixn = f.variables['ixn_b'][()]
        jlist = f.variables['jlist'][()]
        ns = f.variables['ns_b'][()]
        nfp = f.variables['nfp_b'][()]
        Psi = f.variables['phi_b'][()]
        Psi_a = np.abs(Psi[-1])
        iotaVMECt=f.variables['iota_b'][()][1]
        f.close()
    
        # Read properties of VMEC output file
        f = netcdf.netcdf_file(vmecFile,'r',mmap=False)
        rc = f.variables['raxis_cc'][()]
        zs = f.variables['zaxis_cs'][()]
        f.close()
    
        # Calculate nNormal
        stel = Qsc(rc=rc,zs=zs)
        nNormal = stel.iota - stel.iotaN
    
        # Prepare coordinates for fit
        s_full = np.linspace(0,1,ns)
        ds = s_full[1] - s_full[0]
        #s_half = s_full[1:] - 0.5*ds
        s_half = s_full[jlist-1] - 0.5*ds
        mask = s_half < max_s_for_fit
        s_fine = np.linspace(0,1,400)
        sqrts_fine = s_fine
        phi = np.linspace(0,2*np.pi / nfp, N_phi)
        B0  = np.zeros(N_phi)
        B1s = np.zeros(N_phi)
        B1c = np.zeros(N_phi)

    
        # Perform fit
        for jmn in range(len(ixm)):
            m = ixm[jmn]
            n = ixn[jmn] / nfp
            if m>2:
                continue
            if m==0:
                # For m=0, fit a polynomial in s (not sqrt(s)) that does not need to go through the origin.
                degree = 4
                p = np.polyfit(s_half[mask], bmnc[mask,jmn], degree)
                B0 += p[-1] * np.cos(n*nfp*phi)
            if m==1:
                # For m=1, fit a polynomial in sqrt(s) to an odd function
                x1 = np.sqrt(s_half[mask])
                y1 = bmnc[mask,jmn]
                x2 = np.concatenate((-x1,x1))
                y2 = np.concatenate((-y1,y1))
                degree = 5
                p = np.polyfit(x2,y2, degree)
                B1c += p[-2] * (np.sin(n*nfp*phi) * np.sin(nNormal*phi) + np.cos(n*nfp*phi) * np.cos(nNormal*phi))
                B1s += p[-2] * (np.sin(n*nfp*phi) * np.cos(nNormal*phi) - np.cos(n*nfp*phi) * np.sin(nNormal*phi))
                #B1c += p[-2] * np.cos(n*nfp*phi)
                #B1s += p[-2] * np.sin(n*nfp*phi)

    
        # Convert expansion in sqrt(s) to an expansion in r
        BBar = max(B0)
        sqrt_s_over_r = np.sqrt(np.pi * BBar / Psi_a)
        B1s *= sqrt_s_over_r
        B1c *= sqrt_s_over_r
        eta_bar = np.mean(max(B1c)) / BBar
        stel = Qsc(rc=rc,zs=zs,etabar=eta_bar,nphi=N_phi,nfp=nfp,B0=BBar)
    
        # Return results
        return eta_bar
    
    


