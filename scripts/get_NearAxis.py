from qsc import Qsc
import numpy as np
from scipy.io import netcdf

def get_NearAxis(boozFile, vmecFile, max_s_for_fit = 0.5, N_phi = 500):
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
    B20 = np.zeros(N_phi)
    B2s = np.zeros(N_phi)
    B2c = np.zeros(N_phi)

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
            B20 += p[-2] * np.cos(n*nfp*phi)
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
        if m==2:
            # For m=2, fit a polynomial in s (not sqrt(s)) that does need to go through the origin.
            x1 = s_half[mask]
            y1 = bmnc[mask,jmn]
            x2=x1
            y2=y1
            degree = 4
            p = np.polyfit(x2,y2, degree)
            B2c += p[-2] * (np.sin(n*nfp*phi) * np.sin(nNormal*phi) + np.cos(n*nfp*phi) * np.cos(nNormal*phi))
            B2s += p[-2] * (np.sin(n*nfp*phi) * np.cos(nNormal*phi) - np.cos(n*nfp*phi) * np.sin(nNormal*phi))
            #B2c += p[-2] * np.cos(n*nfp*phi)
            #B2s += p[-2] * np.sin(n*nfp*phi)

    # Convert expansion in sqrt(s) to an expansion in r
    BBar = max(B0)
    sqrt_s_over_r = np.sqrt(np.pi * BBar / Psi_a)
    B1s *= sqrt_s_over_r
    B1c *= sqrt_s_over_r
    B20 *= sqrt_s_over_r*sqrt_s_over_r
    B2c *= sqrt_s_over_r*sqrt_s_over_r
    B2s *= sqrt_s_over_r*sqrt_s_over_r
    eta_bar = np.mean(max(B1c)) / BBar
    print(eta_bar)
    stel = Qsc(rc=rc,zs=zs,etabar=eta_bar,nphi=N_phi,nfp=nfp,B0=BBar)

    # Return results
    return [BBar,eta_bar,B0,B1s,B1c,B20,B2c,B2s,phi,Psi_a,iotaVMECt,nfp,stel]

if __name__ == '__main__':
    boozFile = "/home/pk123/Documents/GK_Files/test_cases/VMEC/HSX/boozmn_HSX.nc"
    vmecFile = "/home/pk123/Documents/GK_Files/test_cases/VMEC/HSX/wout_HSX.nc"
    [BBar,eta_bar,B0,B1s,B1c,B20,B2c,B2s,phi,Psi_a,iotaVMECt,nfp,stel] = get_NearAxis(boozFile, vmecFile)
    print("eta_bar =",stel.etabar)
    print("Bbar =",BBar)
    print("True iota =",iotaVMECt)
    print("Near-axis iota =",stel.iota)

