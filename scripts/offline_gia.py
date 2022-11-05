
from amrfile import io as amrio
import os

import numpy as np

import shapefile
import pyproj
from matplotlib import path

os.environ['PROJ_LIB'] = '/home/skachuck/anaconda2/share/proj/'

_SECSPERYEAR = 31536000.
mask_glaciers = {'thwaites': ['Thwaites'],
                 'pig': ['Pine_Island'],
                 'east': ['Kamb', 'Bindschadler', 'MacAyeal'],
                 'west': ['Cosgrove', 'King', 'Lucchitta_Velasco', 'Abbot', 
                          'Venable', 'Walgreen_Coast'],
                 'north': ['Fox', 'Ferrigno', 'Rutford', 'Evans', 'Minnesota',
                           'Union', 'Hercules', 'Institute'],
                 'south': ['Haynes', 'Kohler', 'Smith', 'Pope', 'Haynes',
                           'Kohler', 'Getz', 'Philbin']}

def form_masks(xh, yh):
    p = pyproj.Proj(proj='stere', lat_0=-90, lat_ts=-71, ellps='WGS84',lon_0=180)
    basins = shapefile.Reader('/home/skachuck/work/giabisicles/basindefs/Basins_Antarctica_v02_WGS84.shp')
    # Mask out regions for offline GIA
    masks = {}
    nx, ny = len(xh), len(yh)
    xh = -1838250 + xh
    yh = -880250 + yh
    for maskname, glaclist in mask_glaciers.iteritems():
        agg_mask = np.zeros((ny,nx), dtype=bool)
        for basin in basins.iterShapeRecords():
            if basin.record[0] in glaclist:
                lon,lat = np.array(basin.shape.points).T

                x, y = p(lon, lat)
                outline = path.Path(zip(x,y))
                mask = outline.contains_points(np.r_[np.meshgrid(-xh, -yh)].reshape(2,len(xh)*len(yh)).T)
                mask = mask.reshape(len(yh),len(xh))
                agg_mask += mask
        masks[maskname] = agg_mask

    agg_mask = np.ones((len(yh),len(xh)), dtype=bool)
    for maskname, maskval in masks.iteritems():
        agg_mask = np.logical_xor(maskval, agg_mask)
    masks['remainder'] = agg_mask

    agg_mask = np.zeros((len(yh),len(xh)), dtype=bool)
    for maskname, maskval in masks.iteritems():
        agg_mask += maskval
    assert np.all(agg_mask)
    
    masks['total'] = np.ones((len(yh),len(xh)), dtype=bool)
    return masks

def get_fnames(outpath, basename='plot', cut=-3, return_steps=False):
    # List of all output files, following basename
    fnames = [i for i in os.listdir(outpath) if basename in i and not 'stats'
    in i]
    # Extract the timestep associated, for sorting
    steps = np.array([int(i.split('.')[cut]) for i in fnames])
    # Sort the fnames by timestep
    fnames = [fnames[i] for i in np.argsort(steps)]
    
    if return_steps:
        return fnames, steps
    else:
        return fnames

    
def compute_load(thk, bas, rho_i=910, rho_w=1028, include_ocean=True, mask=None):
    if mask is None: mask = np.ones_like(thk)
    taf = np.maximum(thk + np.minimum(bas, 0)*rho_w/rho_i, 0)*mask
    if include_ocean:
        return -np.minimum(bas, 0)*rho_w + taf*rho_i
    else:
        return taf*rho_i
    
def fft2andpad(arr, nx, ny, fac):
    shape = (ny*fac, nx*fac)
    return np.fft.fft2(arr, shape)

def ifft2andcrop(arr, nx, ny, fac):
    return np.real(np.fft.ifft2(arr))[..., :ny, :nx]
    
def offline_gia(outpath, plotbase='plot', outbase='giaoffline',
                fac=2, 
                u=1e18, g=9.81, D=1e23,
                u2=None, u1=None, h=None,
                mu=26.6e9, lam=34.2666e9, 
                rho_r=3313., rho_i=910., rho_w=1027.,
                include_ocean=True, include_elastic=True,
                compute_bas=False,
                mask=None, fullfac=False, **kwargs):
    
    
    elra = bool(kwargs.get('elra', False))
    if elra:
        tau = float(kwargs.get('tau', 10))

    fnames, steps = get_fnames(outpath,plotbase,return_steps=True)
    
    amrID = amrio.load(outpath+fnames[0])
    lo,hi = amrio.queryDomainCorners(amrID, 0)
    xh,yh,bas0 = amrio.readBox2D(amrID, 0, lo, hi, "Z_base", 0)
    xh,yh,thk = amrio.readBox2D(amrID, 0, lo, hi, "thickness", 0)    
    t0 = amrio.queryTime(amrID)
    amrio.free(amrID)
    
    if mask is None:
        masks = {'total':np.ones_like(bas0)}
    elif mask == 'all':
        masks = form_masks(xh,yh)
    elif mask == 'grl2020':
        x = 1453000, 1707000, 1707000, 1453000
        y = 2000, 2000, 384000, 384000
        outline = path.Path(zip(x,y))
        mask = outline.contains_points(np.r_[np.meshgrid(1838250-xh, 880250-yh)].reshape(2,len(xh)*len(yh)).T)
        masks = {'grl2020': mask.reshape(len(yh),len(xh))}
    else:
        masks = {mask:form_masks(xh,yh)[mask]}
    
    # Grid and FFT properties
    nx, ny = len(xh), len(yh)
    dx = xh[1]-xh[0]                           # m
    dy = yh[1]-yh[0]                           # m
    kx = np.fft.fftfreq(nx*fac, dx)            # m^-1
    ky = np.fft.fftfreq(ny*fac, dy)            # m^-1
    k = 2*np.pi*np.sqrt(kx[None,:]**2 + ky[:,None]**2) 
    
    warn = 'Two layer model must have u1 and u2 set'
    assert (u1 is not None) == (u2 is not None) == (h is not None), warn
    if u1 is not None:
        # Cathles (1975) III-21
        # Bueler, et al. (2007) (15)
        c = np.cosh(k*h)
        s = np.sinh(k*h)
        ur = u2/u1
        ui = 1./ur
        r = 2*c*s*ur + (1-ur**2)*(k*h)**2 + ((ur*s)**2+c**2)
        r = r/((ur+ui)*s*c + k*h*(ur-ui) + (s**2+c**2))
        # Some hackery to correct overflow for large wavenumbers
        if ur < 1:
            r[r<ur] = ur
        else:
            r[r>ur] = ur
        r[np.isnan(r)] = ur
        u = u1
    else:
        r = 1

    # Lithospher filter, Cathles (1975) III-34 (foornote 3)
    alpha_l = 1 + k**4*D/g/rho_r
    # Relaxation time, Cathles
    taus = 2*u*k/g/rho_r/alpha_l*r  # s
    if elra: taus = tau*np.ones_like(taus)
    # Elastic halfspace response, Cathles (1975) III-46
    ue = -1/(2*k)*(1/mu + 1/(mu+lam))  # m/Pa
    ue[0,0] = 0
    # with litosphere filter.
    ue *= (1-alpha_l**(-1))                           # m/Pa
    
    
    beta = rho_r*g+D*k**4    # Pa / m 

    for maskid, mask in masks.iteritems():
        
        outname= '.'.join([outbase, maskid]+fnames[0].split('.')[1:-3]+['{n:06}', 'npy'])
        
        amrID = amrio.load(outpath+fnames[0])
        lo,hi = amrio.queryDomainCorners(amrID, 0)
        xh,yh,bas0 = amrio.readBox2D(amrID, 0, lo, hi, "Z_base", 0)
        xh,yh,thk = amrio.readBox2D(amrID, 0, lo, hi, "thickness", 0)    
        t0 = amrio.queryTime(amrID)
        amrio.free(amrID)
        
        load = compute_load(thk, bas0, rho_i, rho_w, include_ocean, mask)
        load0hat = fft2andpad(load, nx, ny, fac)
        Uhatn = np.zeros((ny*fac,nx*fac), dtype=np.complex128)  
        uen = np.zeros((ny*fac,nx*fac), dtype=np.complex128)
        dLhat = np.zeros((ny*fac,nx*fac), dtype=np.complex128)    
        dLhatold = 0.
        print(outname.format(n=0))
        if fullfac:
            np.save(outname.format(n=steps[0]), ifft2andcrop(Uhatn+uen, fac*nx,
        fac*ny, fac))
        else:
            np.save(outname.format(n=steps[0]), ifft2andcrop(Uhatn+uen, nx, ny, fac))

        for step, f in zip(steps[1:], fnames[1:]):
            amrID = amrio.load(outpath+f)
            lo,hi = amrio.queryDomainCorners(amrID, 0)
            xh,yh,bas = amrio.readBox2D(amrID, 0, lo, hi, "Z_base", 0)
            xh,yh,thk = amrio.readBox2D(amrID, 0, lo, hi, "thickness", 0)    
            t1 = amrio.queryTime(amrID)
            amrio.free(amrID)

            if compute_bas:
                bas = bas0 + ifft2andcrop(Uhatn+uen, nx, ny, fac)

            load = compute_load(thk, bas, rho_i, rho_w, include_ocean, mask)
            dLhat = (fft2andpad(load, nx, ny, fac) - load0hat)*g

            dt = (t1-t0)
            gamma = (beta*(taus + 0.5*dt*_SECSPERYEAR))**(-1) # m/yr/Pa 
            gamma[0,0] = 0.
            # Bueler, et al. 2007 eq 11
            Uhatdot = -gamma*(dLhat + beta*Uhatn)*_SECSPERYEAR    # m / yr
            # Update uplift field prior to including elastic effect, so that fluid
            # equilibrium is corrct.
            Uhatn += Uhatdot*dt
            # Now include the elastic effect if requested.
            if include_elastic:
                uen = ue*(dLhat-dLhatold)

            dLhatold = dLhat

            t0 = t1
            print(outname.format(n=step))
            if fullfac:
                np.save(outpath+outname.format(n=step), ifft2andcrop(Uhatn+uen,
                    fac*nx, fac*ny, fac))
            else:
                np.save(outpath+outname.format(n=step), ifft2andcrop(Uhatn+uen, nx, ny, fac))


            
def offline_elra(outpath, plotbase='plot', outbase='elraoffline',
                fac=2, 
                tau=1e18, g=9.81, D=1e23,
                mu=26.6e9, lam=34.2666e9, 
                rho_r=3313., rho_i=910., rho_w=1027.,
                include_ocean=True, include_elastic=True,
                compute_bas=False,
                mask=None, **kwargs):
    

    fnames, steps = get_fnames(outpath,plotbase,return_steps=True)
    
    amrID = amrio.load(outpath+fnames[0])
    lo,hi = amrio.queryDomainCorners(amrID, 0)
    xh,yh,bas0 = amrio.readBox2D(amrID, 0, lo, hi, "Z_base", 0)
    xh,yh,thk = amrio.readBox2D(amrID, 0, lo, hi, "thickness", 0)    
    t0 = amrio.queryTime(amrID)
    amrio.free(amrID)
    
 #   tau *= _SECSPERYEAR
    tau2 = 2*tau
    
    if mask is None:
        masks = {'total':np.ones_like(bas0)}
    elif mask == 'all':
        masks = form_masks(xh,yh)
    else:
        masks = {mask:form_masks(xh,yh)[mask]}
    
    # Grid and FFT properties
    nx, ny = len(xh), len(yh)
    dx = xh[1]-xh[0]                           # m
    dy = yh[1]-yh[0]                           # m
    kx = np.fft.fftfreq(nx*fac, dx)            # m^-1
    ky = np.fft.fftfreq(ny*fac, dy)            # m^-1
    k = 2*np.pi*np.sqrt(kx[None,:]**2 + ky[:,None]**2) 

    # Lithospher filter, Cathles (1975) III-34 (foornote 3)
    alpha_l = 1 + k**4*D/g/rho_r
    # Elastic halfspace response, Cathles (1975) III-46
    ue = -1/(2*k)*(1/mu + 1/(mu+lam))  # m/Pa
    ue[0,0] = 0
    # with litosphere filter.
    ue *= (1-alpha_l**(-1))                           # m/Pa
    
    
    beta = rho_r*g+D*k**4    # Pa / m 
    beta_inv = beta**(-1)

    for maskid, mask in masks.iteritems():
        
        outname= '.'.join([outbase, maskid]+fnames[0].split('.')[1:-3]+['{n:06}', 'npy'])
        
        amrID = amrio.load(outpath+fnames[0])
        lo,hi = amrio.queryDomainCorners(amrID, 0)
        xh,yh,bas0 = amrio.readBox2D(amrID, 0, lo, hi, "Z_base", 0)
        xh,yh,thk = amrio.readBox2D(amrID, 0, lo, hi, "thickness", 0)    
        t0 = amrio.queryTime(amrID)
        amrio.free(amrID)
        
        load = compute_load(thk, bas0, rho_i, rho_w, include_ocean, mask)
        load0hat = fft2andpad(load, nx, ny, fac)
        Uhatn = np.zeros((ny*fac,nx*fac), dtype=np.complex128)
        uen = np.zeros((ny*fac,nx*fac), dtype=np.complex128)
        dLhat = np.zeros((ny*fac,nx*fac), dtype=np.complex128)    
        #dLhatold = 0.
        print(outname.format(n=0))
        np.save(outname.format(n=steps[0]), ifft2andcrop(Uhatn+uen, nx, ny, fac))

        for step, f in zip(steps[1:], fnames[1:]):
            amrID = amrio.load(outpath+f)
            lo,hi = amrio.queryDomainCorners(amrID, 0)
            xh,yh,bas = amrio.readBox2D(amrID, 0, lo, hi, "Z_base", 0)
            xh,yh,thk = amrio.readBox2D(amrID, 0, lo, hi, "thickness", 0)    
            t1 = amrio.queryTime(amrID)
            amrio.free(amrID)

            if compute_bas:
                bas = bas0 + ifft2andcrop(Uhatn+uen, nx, ny, fac)

            load = compute_load(thk, bas, rho_i, rho_w, include_ocean, mask)
            dLhat = (fft2andpad(load, nx, ny, fac) - load0hat)*g

            dt = (t1-t0)
            # Update uplift field prior to including elastic effect, so that fluid
            # equilibrium is corrct.
            Uhatn = (Uhatn*(1-dt/tau2) - beta_inv*dt/tau*dLhat)/(1+dt/tau2)
            # Now include the elastic effect if requested.
            if include_elastic:
                uen = ue*(dLhat)

            t0 = t1
            print(outname.format(n=step))
            np.save(outpath+outname.format(n=step), ifft2andcrop(Uhatn+uen, nx, ny, fac))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute the offline GIA-ice dynamics coupling')
    parser.add_argument('outdir', metavar='outfile', type=str, nargs='?', 
                       help='the directory of the plotfiles')
    parser.add_argument('outbase', metavar='outbase', type=str, nargs='?', 
                       help='the base name for output')
    parser.add_argument('--no-elas', dest='include_elastic', action='store_const', const=False, default=True,
                       help='omit the elast component')
    parser.add_argument('--no-ocean', dest='include_ocean', action='store_const', const=False, default=True,
                       help='omit the ocean load')
    parser.add_argument('--compute-bas', dest='compute_bas', action='store_const', const=True, default=False,
                       help='compute load from uplift')
    parser.add_argument('--mask')
    parser.add_argument('--model', default='ub')
    parser.add_argument('--fac', default=5)
    parser.add_argument('--elra', dest='elra', action='store_const', const=True, default=False,)
    parser.add_argument('--importing', dest='importing', action='store_const', const=True, default=False,)
    parser.add_argument('--fullfac', dest='fullfac', action='store_const', const=True, default=False,)
    parser.add_argument('--tau')
    
    args = parser.parse_args()

    if not args.importing:
    
        if args.model == 'ub':
            model = {'u':1e18, 'D':1e23}
        elif args.model == 'best2':
            model = {'u1':2e19, 'u2':4e18, 'D':13e23, 'h':200000}
        elif args.tau is not None:
            model = {'tau':args.tau}
            
        #offline_gia(args.outdir, outbase=args.outbase, 
        #    include_ocean=args.include_ocean,
        #    include_elastic=args.include_elastic,
        #    compute_bas=args.compute_bas,
        #    mask=args.mask, elra=args.elra, tau=args.tau, 
        #    **model)
        
        if not args.elra:
            offline_gia(args.outdir, outbase=args.outbase, 
                        fac=int(args.fac),
                        include_ocean=args.include_ocean,
                        include_elastic=args.include_elastic,
                        compute_bas=args.compute_bas,
                        mask=args.mask, fullfac=args.fullfac, **model)
        else:
            offline_elra(args.outdir, outbase=args.outbase, 
                include_ocean=args.include_ocean,
                include_elastic=args.include_elastic,
                compute_bas=args.compute_bas,
                mask=args.mask, tau=float(args.tau))
