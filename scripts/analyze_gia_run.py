from amrfile import io as amrio
import os, sys

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from intersect import intersection

import matplotlib.pyplot as plt

os.environ['PROJ_LIB'] = '/home/skachuck/anaconda2/share/proj/'

LEV = 3

# SETUP
x_lo = -1838250
y_lo = -880250
xh = x_lo + 4000./2**LEV*(0.5+np.arange(224*2**LEV))
yh = y_lo + 4000./2**LEV*(0.5+np.arange(256*2**LEV))

aseint = lambda x: np.trapz(np.trapz(x, x=xh, axis=1), x=yh)

# INTERPOLATION POINTS
kay_pt = (-1506387.2409297135, -576740.1143521154)
murphy_pt = (-1498577.1203334671, -567251.6795597454)
cross_cl = np.loadtxt('/global/cscratch1/sd/skachuck/ismip6results/data/crosson_centerline.txt')
pig_cl = np.loadtxt('/global/cscratch1/sd/skachuck/ismip6results/data/pig_centerline.txt')

def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:0.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def get_fnames(outpath, basename='plot', cut=-3, return_steps=False):
    # List of all output files, following basename
    fnames = [i for i in os.listdir(outpath) 
                if basename in i 
                    and not 'stats' in i
                    and not '__errorr__' in i]
    # Extract the timestep associated, for sorting
    steps = np.array([int(i.split('.')[cut]) for i in fnames])
    # Sort the fnames by timestep
    fnames = [fnames[i] for i in np.argsort(steps)]
    steps = [steps[i] for i in np.argsort(steps)]
    
    if return_steps:
        return fnames, steps
    else:
        return fnames
    
def read_input_rheology(outpath):
    f = open(outpath+[i for i in os.listdir(outpath) if 'inputs' in i][0])
    rheology = {}
    for l in f.readlines():
        if 'topographyFlux.' in l:
            parts = l.split('=')
            rheology[parts[0].strip()[len('topographyFlux.'):]] = parts[1].strip()
    if rheology != {}:
        rheology['nlayers'] = int(rheology['nlayers'])
        rheology['flex'] = float(rheology['flex'])
        if rheology['nlayers'] == 1:
            rheology['visc'] = float(rheology['visc'])
        elif rheology['nlayers'] > 1:
            rheology['visc'] = [float(visc) for visc in rheology['visc'].split(' ')]
            rheology['thk'] = float(rheology['thk'])
    return rheology

def compute_relaxation_and_filter(freq, rheology):
    SECSPERYEAR=31556926
    g = 9.8
    mu = 26.6 # GPa
    rho = 3313

    if rheology['nlayers'] == 2:
        u1 = rheology['visc'][1]
        u2 = rheology['visc'][0]
        h = rheology['thk']

        c = np.cosh(freq*h)
        s = np.sinh(freq*h)
        u = u2/u1
        ui = 1./u
        r = ((2*c*s*u + (1-u**2)*(freq*h)**2 + ((u*s)**2 + c**2))/
            ((u+ui)*s*c + freq*h*(u-ui) + (s**2 + c**2)))

        c = np.cosh(freq*h)
        s = np.sinh(freq*h)
        r = ((2*c*s*u + (1-u**2)*(freq*h)**2 + ((u*s)**2 + c**2))/
            ((u+ui)*s*c + freq*h*(u-ui) + (s**2 + c**2)))

        if isinstance(r, float):
            if r<(1 + u)/(1+ui): r=(1 + u)/(1+ui)
            elif np.isnan(r): r=(1 + u)/(1+ui)
        else:
            r[r<(1 + u)/(1+ui)] = (1 + u)/(1+ui)
            r[np.isnan(r)] = (1 + u)/(1+ui)
    else:
        u1 = rheology['visc']
        r = 1

    taus = 2*u1*np.abs(freq)/g/rho / SECSPERYEAR*r
    alphas = 1 + freq**4*rheology['flex']/g/rho

    return taus, alphas

def compute_load(thk, bas, rho_i=910, rho_w=1028, include_ocean=True, mask=None):
    if mask is None: mask = np.ones_like(thk)
    taf = np.maximum(thk + np.minimum(bas, 0)*rho_w/rho_i, 0)*mask
    if include_ocean:
        return -np.minimum(bas, 0)*rho_w + taf*rho_i
    else:
        return taf*rho_i

def intersect_grounding_line(mask,centerline):
    # Get the grounding line contour
    pc = plt.contour(-xh, -yh, mask==1)
    glx, gly = pc.allsegs[0][np.argmax([len(cont) for cont in pc.allsegs[0]])].T
    plt.close()

    xint, yint = intersection(glx, gly, centerline[0], centerline[1])

    return np.array([xint, yint])

def analyze_gia_run(outpath, Ao=362e9):
    fnames = get_fnames(outpath,)[1:]

    rho_i = 917.
    rho_w = 1027.
    riw = rho_i/rho_w

    # Volume above flotation
    vafs = np.zeros_like(fnames, dtype=float)
    # Volume below sea level
    vbss = np.zeros_like(fnames, dtype=float)
    # Volume of ice-free below sea level
    #vcbs = np.zeros_like(fnames, dtype=float)
    # Volume of groudnded below sea level
    #vgbs = np.zeros_like(fnames, dtype=float)
    # Volume of floating below sea level
    #vfbs = np.zeros_like(fnames, dtype=float)
    # Volume of floating ice
    vfis = np.zeros_like(fnames, dtype=float)
    # Volume of grounded ice
    vgis = np.zeros_like(fnames, dtype=float)
    # Volume of water
    vwas = np.zeros_like(fnames, dtype=float)
    # Volume of ocean basin uplift
    vups = np.zeros_like(fnames, dtype=float)
    # Volume of total uplift
    vtus = np.zeros_like(fnames, dtype=float)
    # Basal melt-rate volume
    vbms = np.zeros_like(fnames, dtype=float)
    # Mean frequency of load
    mfqs = np.zeros_like(fnames, dtype=float)
    # Mean wavelength of load
    mwvs = np.zeros_like(fnames, dtype=float)
    # Mean relaxation time of load
    mtas = np.zeros_like(fnames, dtype=float)
    # Times
    ts = np.zeros_like(fnames, dtype=float)


    # Area of ice-free below sea level
    #acbs = np.zeros_like(fnames, dtype=float)
    # Area of groudnded below sea level
    #agbs = np.zeros_like(fnames, dtype=float)
    # Area of floating below sea level
    #afbs = np.zeros_like(fnames, dtype=float)

    # Area of grounded ice
    aigs = np.zeros_like(fnames, dtype=float)
    # Area of floating ice
    #aifs = np.zeros_like(fnames, dtype=float)   
    # Area of ice free
    #aics = np.zeros_like(fnames, dtype=float)

    # Point-wise measurements
    kay_peak_thk = np.zeros_like(fnames, dtype=float)
    kay_peak_upl = np.zeros_like(fnames, dtype=float)
    mt_murph_thk = np.zeros_like(fnames, dtype=float)
    mt_murph_upl = np.zeros_like(fnames, dtype=float)

    # Grounding line positions
    cross_gl = np.zeros_like(fnames, dtype=float)
    pig_gl = np.zeros_like(fnames, dtype=float)

    amrID = amrio.load(outpath+fnames[0])
    lo,hi = amrio.queryDomainCorners(amrID, LEV)
    xh,yh,bas = amrio.readBox2D(amrID, LEV, lo, hi, "Z_base", 0)
    xh,yh,bot = amrio.readBox2D(amrID, LEV, lo, hi, "Z_bottom", 0)    
    xh,yh,thk = amrio.readBox2D(amrID, LEV, lo, hi, "thickness", 0)    
    xh,yh,mask = amrio.readBox2D(amrID, LEV, lo, hi, "mask", 0)    

    amrio.free(amrID)
    
    dx = xh[1]-xh[0]
    freqx = np.fft.fftfreq(len(xh), dx)
    freqy = np.fft.fftfreq(len(yh), dx)
    freqs = 2*np.pi*np.sqrt(freqx[None,:]**2 + freqy[:,None]**2).flatten()
    


    rheology = read_input_rheology(outpath)
    if rheology != {}:
        print(rheology)
        taus, alphas = compute_relaxation_and_filter(freqs, rheology)
        
    for i,f in enumerate(fnames):
        update_progress(i/float(len(fnames)))
        try:
            amrID = amrio.load(outpath+f)
        except:
            print("Couldn't open {}".format(outpath+f))
            raise
            
        lo,hi = amrio.queryDomainCorners(amrID, LEV)
        xh,yh,bas = amrio.readBox2D(amrID, LEV, lo, hi, "Z_base", 0)
        xh,yh,bot = amrio.readBox2D(amrID, LEV, lo, hi, "Z_bottom", 0)    
        xh,yh,thk = amrio.readBox2D(amrID, LEV, lo, hi, "thickness", 0)    
        xh,yh,mask = amrio.readBox2D(amrID, LEV, lo, hi, "mask", 0)    
        xh,yh,bml = amrio.readBox2D(amrID, LEV, lo, hi, "activeBasalThicknessSource", 0)    
        ts[i] = amrio.queryTime(amrID)
        amrio.free(amrID)

        bas_interp = RectBivariateSpline(xh, yh, bas.T)
        thk_interp = RectBivariateSpline(xh, yh, thk.T)
        taf = np.maximum(thk + np.minimum(bas, 0)*rho_w/rho_i, 0)
        
        if i == 0:
            bas0 = bas.copy()
            load0 = compute_load(thk, bas)
            aig0 = aseint((thk>0)*(taf>=0))

            cross_glpt0 = intersect_grounding_line(mask, cross_cl)
            pig_glpt0 = intersect_grounding_line(mask, pig_cl)


        vafs[i] = aseint(taf)/Ao
        vbss[i] = aseint(-np.minimum(bas, 0))/Ao
        vfis[i] = aseint(thk*(taf==0))/Ao
        vgis[i] = aseint(thk*(taf>0))/Ao
        vwas[i] = aseint(np.maximum(bot - bas, 0))/Ao
        #vcbs[i] = aseint(-np.minimum(bas, 0)*(thk==0))    
        #vgbs[i] = aseint(-np.minimum(bas, 0)*(thk>0)*(taf>0))    
        #vfbs[i] = aseint(-np.minimum(bas, 0)*(thk>0)*(taf==0))
        vups[i] = aseint(-np.minimum(bas, 0) - (-np.minimum(bas0, 0)))/Ao
        vtus[i] = -aseint(bas-bas0)/Ao
        vbms[i] = aseint(bml)
        
        #acbs[i] = aseint((thk<=0)*(bas<0))    
        #agbs[i] = aseint((thk>0)*(taf>0)*(bas<0))    
        #afbs[i] = aseint((thk>0)*(taf==0)*(bas<0))   
        aigs[i] = aseint((thk>0)*(taf>=0))-aig0
        #aifs[i] = aseint((thk>0)*(taf==0))        
        #aics[i] = aseint((thk<=0))


        kay_peak_thk[i]=float(thk_interp.ev(*kay_pt))
        kay_peak_upl[i]=float(bas_interp.ev(*kay_pt))
        mt_murph_thk[i]=float(thk_interp.ev(*murphy_pt))
        mt_murph_upl[i]=float(bas_interp.ev(*murphy_pt))


        cross_glpt = intersect_grounding_line(mask, cross_cl)
        pig_glpt = intersect_grounding_line(mask, pig_cl)

        cross_gl[i]=np.sum((cross_glpt-cross_glpt0)**2)
        pig_gl[i]=np.sum((pig_glpt-pig_glpt0)**2)
        
        if rheology != {}:
            load = compute_load(thk, bas)
            dload_hat = np.abs(np.fft.fft2(load-load0)).flatten()/alphas
            dload_hat_norm = dload_hat / dload_hat.max()
            meanfreq = np.sum(dload_hat_norm*freqs) / np.sum(dload_hat_norm)
            
            
            mfqs[i] = meanfreq
            mwvs[i] = (2*np.pi/meanfreq/1000)
            tau, alpha = compute_relaxation_and_filter(meanfreq,rheology)
            mtas[i] = tau/alpha


    df = pd.DataFrame(index=ts)
    df.index.name = 'year'
    df['Vol_above_flotation'] = vafs*riw
    df['Vol_ocean_basin'] = vbss
    #df['Vol_ocean_basin_nogrounded'] = vcbs
    #df['Vol_ocean_basin_yesgrounded'] = vgbs
    #df['Vol_ocean_basin_yesfloating'] = vfbs
    df['Vol_floating_ice'] = vfis*riw
    df['Vol_grounded_ice'] = vgis*riw
    df['Vol_ice'] = vgis*riw + vfis*riw
    df['Vol_ocean'] = vwas
    df['Vol_ocean_uplift'] = vups
    df['Vol_total_uplift'] = vtus
    df['Vol_water'] = vwas + vgis*riw + vfis*riw
    df['DArea_grounded_ice'] = aigs
    df['Basal_melt_rate'] = vbms
    
    df['DVol_above_flotation'] = (vafs-vafs[0])*riw
    df['DVol_ocean_basin'] = vbss-vbss[0]
    df['DVol_ice'] = (vgis-vgis[0])*riw + (vfis-vfis[0])*riw
    df['DVol_grounded_ice'] = (vgis-vgis[0])*riw
    df['DVol_floating_ice'] = (vfis-vfis[0])*riw
    df['DVol_ocean'] = vwas-vwas[0]
    df['DVol_water'] = vwas-vwas[0] + ((vgis-vgis[0])*riw + (vfis-vfis[0])*riw)

    df['KayPeak_thickness'] = kay_peak_thk
    df['KayPeak_uplift'] = kay_peak_upl-kay_peak_upl[0]
    df['MtMurphy_thickness'] = mt_murph_thk
    df['MtMurphy_uplift'] = mt_murph_upl-mt_murph_upl[0]

    df['CrossonGL_retreat'] = cross_gl
    df['PIGGL_retreat'] = pig_gl
    
    df['MeanFrequency'] = mfqs
    df['MeanWavelength'] = mwvs
    df['MeanRelaxationTime'] = mtas
    
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute the offline GIA-ice dynamics coupling')
    parser.add_argument('rundir', metavar='outfile', type=str, nargs='?', 
                       help='the directory of the run')
    parser.add_argument('outname', metavar='outbase', type=str, nargs='?', 
                       help='the name of the plotfiles')
    
    args = parser.parse_args()
    
    print(args.rundir)
    df = analyze_gia_run(args.rundir)
    if args.outname is None: 
        args.outname = args.rundir+'pandas_stats.csv'
    df.to_csv(args.outname)
