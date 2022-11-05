from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import shapefile
import pyproj

from netCDF4 import Dataset

#import giapy.plot_tools.interp_path
from amrfile import io as amrio
#from giapy.giaflat import thickness_above_floating
from intersection import intersection
from scipy.interpolate import interp2d

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

def gen_basemap():
    """Generate the standard projection."""
    m = Basemap(resolution='i',projection='spstere',\
                lat_ts=-71,lon_0=180, boundinglat=-64, ellps='WGS84')
    return m

pstere = pyproj.Proj(proj='stere', lat_0=-90, lat_ts=-71, ellps='WGS84',
lon_0=180)
basins = shapefile.Reader('/home/skachuck/work/giabisicles/basindefs/Basins_Antarctica_v02_WGS84.shp')

coastline = shapefile.Reader('/mnt/c/Users/skachuck/Downloads/ne_10m_coastline/ne_10m_coastline.shp')
recs = coastline.records()
shapes = coastline.shapes()

def pig2proj(x, y, xmin=-5905823.470038407, xmax=0,
                    ymin=-5905823.470038407,ymax=0, 
                    xoff=1707000, yoff=384000):
    """Transform coordinates for Pine Island Glacier example to the standard
    projection coordinates."""
    return -x-0.5*(xmax-xmin)+xoff, -y-0.5*(ymax-ymin)+yoff

def pig2proj(x,y):
    return 1707000-x, 384000-y

def ase2proj(x,y):
    return 3071500-x, 3072500-y

def pignetcdf(arr, name, fname):
    """Generate a netcdf file for the PIG geometry"""
    rootgrp = Dataset(fname, "w", format="NETCDF4")
    xd = rootgrp.createDimension("xdim", 128)
    yd = rootgrp.createDimension("ydim", 192)
    # nctoamr needs x defined
    x = rootgrp.createVariable("x", "f8", ("xdim",))
    y = rootgrp.createVariable("y", "f8", ("ydim",))
    x[:] = np.arange(1000, 256000, 2000)                        # PIG at 2km
    y[:] = np.arange(1000, 384000, 2000)                        # PIG at 2km
    # copy array into dataset
    if isinstance(arr, list):
        assert len(arr) == len(name), "arr and name must match"
        for a, n in zip(arr, name):
            tmp = rootgrp.createVariable(n, "f8", ("ydim", "xdim"))
            tmp[:,:] = a
    tmp = rootgrp.createVariable(name, "f8", ("ydim", "xdim"))
    tmp[:,:] = arr
    rootgrp.close()

def add_grid(ax, proj, lats=[], lons=[], 
            latmin=-90, latmax=90, lonmin=-180,
            lonmax=180, **kwargs):
    for lat in lats:
        xs, ys = proj(np.linspace(lonmin, lonmax, 50), np.ones(50)*lat)
        ax.plot(xs, ys, **kwargs)
    for lon in lons:
        xs, ys = proj(np.ones(50)*lon,np.linspace(latmin, latmax, 50))
        ax.plot(xs, ys, **kwargs)
    return plt.gca()
                                        

def genpigplot(ax=None, figsize=(6,8)):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    ax.set_xlim([1707000, 1707000-254000])
    ax.set_ylim([384000, 384000-382000])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_aspect('equal')

    return plt.gca()

def genaseplot(ax=None, figsize=(6,8), draw_coasts=True, draw_grid=True,
                draw_basins=True, basin_list=['Pine_Island', 'Thwaites']):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    if draw_coasts:
        for blah in shapes[:]:
            lon,lat = np.array(blah.points).T
            x,y = pstere(lon,lat)
            if any(np.logical_and(np.logical_and(np.logical_and(x<1836250, x>944250),y<878250.),y>-141750.)):
                ax.plot(x,y,c='k', lw=0.1)

    if draw_grid:
        add_grid(ax, pstere, np.arange(-70,-85,-1), np.arange(-80,-160,-5), -70, -85, -80, -160, c='w', ls=':')

    if draw_basins:
        plotbasins(ax=ax, basin_list=basin_list)

    ax.set_xlim([1836250,944250])
    #ax.set_xlim([1900000,944250])
    ax.set_ylim([878250,-141750])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_aspect('equal')

    return plt.gca()

def plotbasins(ax=None, figsize=(8,6), basin_list=['Pine_Island', 'Thwaites']):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)
    for basin in basins.iterShapeRecords():
        if basin.record[0] in basin_list:
            lon,lat = np.array(basin.shape.points).T
            x,y = pstere(lon,lat)
            ax.plot(x,y,marker=None, color='k',lw=1)
    return plt.gca()

def get_fnames(outpath, basename='plot', cut=-3, return_steps=False):
    # List of all output files, following basename
    fnames = [i for i in os.listdir(outpath) if basename in i and not 'stats'
    in i]
    # Extract the timestep associated, for sorting
    steps = np.array([int(i.split('.')[cut]) for i in fnames])
    # Sort the fnames by timestep
    fnames = [fnames[i] for i in np.argsort(steps)]
    
    if return_steps:
        return fnames, np.sort(steps)
    else:
        return fnames

def extract_ts_and_vols(outpath,skip=1,taf=False, subdomain=None,
                        grounded_area=False):
    fnames = get_fnames(outpath)[::skip]

    if subdomain is None:
        sl = np.s_[:,:]
        xsl = np.s_[:]
        ysl = np.s_[:]
    else:
        sl = np.s_[subdomain[1][0]:subdomain[1][1],
                   subdomain[0][0]:subdomain[0][1]]
        xsl = np.s_[subdomain[0][0]:subdomain[0][1]]
        ysl = np.s_[subdomain[1][0]:subdomain[1][1]]

    vols = []
    ts = []

    for f in fnames:
        amrID = amrio.load(outpath+f)
        lo,hi = amrio.queryDomainCorners(amrID, 0)
        xh,yh,bas1 = amrio.readBox2D(amrID, 0, lo, hi, "Z_base", 0)
        xh,yh,thk1 = amrio.readBox2D(amrID, 0, lo, hi, "thickness", 0)
        ts.append(amrio.queryTime(amrID))
        amrio.freeAll()

        # Extract the subdomain
        thk1, bas1 = thk1[sl], bas1[sl]
        
        if taf:
            thk1 = thickness_above_floating(thk1, bas1)

        if grounded_area:
            thk1 = (thk1>0)
        tmp = np.trapz(thk1, axis=1, x=xh[xsl])
        vol = np.trapz(tmp, x=yh[ysl])

        vols.append(vol)

    vols = np.array(vols)
    
    return ts, vols

def load_model_vol_by_t(modellist, skiplist=[], resultpath='./', taf=True,
                            grounded_area=False,modeldict={}, subdomain=None):
    if skiplist == []:
        skiplist = [1]*len(modellist)
    elif len(skiplist)==1:
        skiplist = skiplist*len(modellist)
    elif len(skiplist)!=len(modellist):
        raise ValueError('modellist and skiplist of different lengths')
    for key, skip in zip(modellist, skiplist):
        if key in modeldict:
            continue
        ts, vols = extract_ts_and_vols(resultpath+key+'/',skip=skip,taf=taf,
                                        subdomain=subdomain,
                                        grounded_area=grounded_area)
        modeldict[key] = ts, vols
    return modeldict

def plot_model_vol_by_t(modeldict, tmax=None, keylist=None, stylelist=None, axs=None):
    vol_fac = 997 / 1e12
    gt_to_mm = -1e12 / 1000 / 360e12 * 1000

    if axs is None:
        fig, axs = plt.subplots(2,1,figsize=(4.5,5), sharex=True)

    if keylist is None: keylist = modeldict.keys()
    if stylelist is not None:
        assert len(stylelist) == len(keylist), 'not right number of styles'

    for i, key in enumerate(keylist):
        tv = modeldict[key]
        ts = np.array(tv[0])
        vols = (tv[1] - tv[1][0])*vol_fac
        if tmax is not None:
            tind = ts<=tmax
            ts = ts[tind]
            vols = vols[tind]
        if stylelist is not None:
            label = stylelist[i].get('label', key)
            axs[0].plot(ts[:-1], np.diff(vols)/np.diff(ts),
                            **stylelist[i])
            axs[1].plot(ts, vols, **stylelist[i])
        else:
            axs[0].plot(ts[:-1], np.diff(vols)/np.diff(ts), label=key)
            axs[1].plot(ts, vols)
        print('Initial specs of {}: lost {} final rate {}'.format(key, vols[0],
        (np.diff(vols)/np.diff(ts))[0]))

        print('Final specs of {}: lost {} final rate {}'.format(key, vols[-1],
        (np.diff(vols)/np.diff(ts))[-1]*gt_to_mm))

    axs[0].set_ylabel('VAF Rate (Gt/yr)')
    yticks = np.array([-200., -150., -100.,  -50.])
    yticks = axs[0].get_yticks()
    axs[0].set_yticks(yticks)
    y1, y2=axs[0].get_ylim()
    x1, x2=axs[0].get_xlim()
    ax2=axs[0].twinx()
    ax2.set_ylim(y1*gt_to_mm, y2*gt_to_mm)
    ax2.set_yticks( yticks*gt_to_mm)
    ax2.set_yticklabels( np.round(yticks*gt_to_mm, 2))
    ax2.set_ylabel('SLE (mm/yr)')
    ax2.set_xlim(x1, x2)
    
    axs[1].legend(frameon=False)
    
    axs[1].set_ylabel('$\Delta$VAF ($10^3$ Gt)')
    
    y1, y2=axs[1].get_ylim()
    x1, x2=axs[1].get_xlim()
    ax2=axs[1].twinx()
    ax2.set_ylim(y1*gt_to_mm, y2*gt_to_mm)
    ax2.set_yticks( axs[1].get_yticks()[1:-1]*gt_to_mm)
    ax2.set_yticklabels( np.round(axs[1].get_yticks()[1:-1]*gt_to_mm, 0))
    ax2.set_ylabel('SLE (mm)')
    ax2.set_xlim(x1, x2)
    
#    plt.gcf().subplots_adjust(left=0.17)
#    plt.gcf().subplots_adjust(right=0.85)
    axs[-1].set_xlabel('Time (years)')
    for ax in axs: ax.set_xlim(x1,x2)
    return plt.gcf()


def collect_fields(fnames, field_names, outpath='./', return_ts=False):
    flist = []
    ts = []
    if not isinstance(field_names, list):
        field_names = [field_names]
    returndict = {}
    for field in field_names:
        returndict[field] = []
    for f in fnames:
        amrID = amrio.load(outpath+f)
        lo,hi = amrio.queryDomainCorners(amrID, 0)
        for field in field_names:
            xh,yh,z = amrio.readBox2D(amrID, 0, lo, hi, field, 0)
            returndict[field].append(z)
        ts.append(amrio.queryTime(amrID))
        amrio.free(amrID)
    for field in field_names:
        returndict[field] = np.array(returndict[field])
    returnset= returndict,
    if return_ts: returnset+= np.array(ts),
    return returnset

def collect_field(fnames, field_name, outpath='./', return_ts=False):
    flist = []
    ts = []
    for f in fnames:
        amrID = amrio.load(outpath+f)
        lo,hi = amrio.queryDomainCorners(amrID, 0)
        xh,yh,z = amrio.readBox2D(amrID, 0, lo, hi, field_name, 0)
        ts.append(amrio.queryTime(amrID))

        flist.append(z)
        amrio.free(amrID)
    returnset= np.array(flist),
    if return_ts: returnset+= np.array(ts),
    return returnset

def locate_grounding_line(xh,yh,thk,bas):
    """Locates the grounding line by finding the longest contour where
    transition from floating occurs"""
    p = plt.contour(xh, yh, thickness_above_floating(thk,bas), levels=[0]);
    ilongestcontour = np.argmax([len(cont) for cont in p.allsegs[0]])
    glx, gly = p.allsegs[0][ilongestcontour].T

    return glx, gly

def intersect_grounding_and_center(fnames, centerline, outpath='./',
                                    return_depths=False, return_thks=False):
    """
    Finds the intersection between the grounding lines in fnames and a
    centerline, using the intersection tool from Sukhbinder Singh.


    """

    xints, yints, depths, thks, ts, slps = [], [], [], [], [], []

    thetas = np.arctan(np.diff(centerline.xs)/np.diff(centerline.ys))

    for f in fnames:
        amrID = amrio.load(outpath+f)
        lo,hi = amrio.queryDomainCorners(amrID, 0)
        xh,yh,bas = amrio.readBox2D(amrID, 0, lo, hi, "Z_base", 0)
        xh,yh,bot = amrio.readBox2D(amrID, 0, lo, hi, "Z_bottom", 0)
        xh,yh,thk = amrio.readBox2D(amrID, 0, lo, hi, "thickness", 0)

        glx, gly = locate_grounding_line(xh,yh,thk,bas)
        xint, yint = intersection.intersection(glx, gly, centerline.xs, centerline.ys)
        xint = np.mean(xint)
        yint = np.mean(yint)
        xints.append(xint)
        yints.append(yint)
        ts.append(amrio.queryTime(amrID))
        depths.append(interp2d(xh, yh, bas)(xint, yint))
        thks.append(interp2d(xh, yh, thk)(xint, yint))

        xi = np.argmin(np.abs(xh-xint))
        yi = np.argmin(np.abs(yh-yint))
        slpx = bas[yi, xi+1]-bas[yi, xi-1]/(2*(xh[1]-xh[0]))
        slpy = bas[yi+1, xi]-bas[yi-1, xi]/(2*(yh[1]-yh[0]))

    

        thi = np.argmin((centerline.xs-xint)**2 + (centerline.ys-yint)**2)
        theta = thetas[thi]
      
        slp = slpx*np.cos(theta) + slpy*np.sin(theta)

        slp = np.sqrt(slpx**2 + slpy**2)
        slps.append(slp)

        amrio.free(amrID)

    xints = np.array(xints)
    yints = np.array(yints)
    depths = np.array(depths)
    ts = np.array(ts)
    plt.close()
    
    return_set = ts, xints, yints
    if return_depths: return_set+=depths,
    if return_thks: return_set+=thks,
    return_set += slps,
    return_set = [np.asarray(rs) for rs in return_set]
    return return_set

def find_centerline():
    xh,yh,xvel = amrio.readBox2D(amrID, 0, lo, hi, "xVel", 0)
    xh,yh,yvel = amrio.readBox2D(amrID, 0, lo, hi, "yVel", 0)
    points = np.array([[0,0]])

    for l in p.lines.get_paths():
        i = 0
        for s in l.iter_segments():
            if i == 0: 
                if s[0] not in points:
                    points = np.vstack([points, s[0]])
            i += 1
 
    center_x, center_y = points.T[0,sl], points.T[1,sl]
    center_d = np.r_[0,np.cumsum(np.sqrt((center_x[1:] - center_x[:-1])**2 + (center_y[1:] - center_y[:-1])**2))]/1000.

    centerline = giapy.plot_tools.interp_path.TransectPath(center_x, center_y, center_d)

    return centerline

def computeGeoid():
    fac = 16
    freqx = np.fft.fftfreq(len(xh)*fac, 2.)
    freqy = np.fft.fftfreq(len(yh)*fac, 2.)
    freq = np.sqrt(freqx[None,:]**2 + freqy[:,None]**2)
    
    dLoad = (thickness_above_floating(thks_ub[1],
    bass_ub[1])-thickness_above_floating(thks_ub[0], bass_ub[0]))
    dLoad_fft = np.fft.fft2(dLoad, (len(yh)*fac, len(xh)*fac))
    ls = 6371*2*np.pi*freq - 0.5
    ls[0,0] = 6371000/min(xh.max(), yh.max())/fac
    geoid_fft = 4*dLoad_fft*1000*np.pi*6.67e-11*6371e3 / (2*ls+1) / 9.8
    #geoid_fft[0,0] = 0
    geoid_ub = np.real(np.fft.ifft2(geoid_fft))[:len(yh),:len(xh)]
    
    dU_fft = np.fft.fft2(bass_ub[1] - bass_ub[0], (len(yh)*fac, len(xh)*fac))
    geoid_dyn_fft = 4*dU_fft*3313*np.pi*6.67e-11*6371e3 / (2*ls+1) / 9.8
    #geoid_dyn_fft[0,0] = 0
    geoid_dyn_ub = np.real(np.fft.ifft2(geoid_dyn_fft))[:len(yh),:len(xh)]


def plotGeoid(fielddict):
    fig, axs = plt.subplots(2,3,figsize=(6,8))
    
    ax = axs[0,0]
    q=ax.pcolormesh(1707000-xh, 384000-yh, dLoad);
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(q, cax=cax, label=r'$\Delta$TAF (m)')
    
    ax = axs[0,1]
    q=ax.pcolormesh(1707000-xh, 384000-yh, fielddict['Z_base'][1] -
                                fielddict['Z_base'][0], vmin=-20, vmax=20);
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(q, cax=cax, label=r'$\Delta U$ (m)')
    
    ax = axs[1,0]
    q=ax.pcolormesh(1707000-xh, 384000-yh, -geoid_ub, vmin=-2, vmax=2,
    cmap='RdBu_r');
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(q, cax=cax, label=r'$\Delta\Phi_L$ (m)')
    
    ax = axs[1,1]
    q=ax.pcolormesh(1707000-xh, 384000-yh, -geoid_dyn_ub, vmin=-2, vmax=2,
    cmap='RdBu_r'); 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(q, cax=cax, label=r'$\Delta\Phi_U$ (m)')
    
    ax = axs[1,2]
    q=ax.pcolormesh(1707000-xh, 384000-yh, -geoid_ub-geoid_dyn_ub, vmin=-2, vmax=2,
    cmap='RdBu_r');
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(q, cax=cax, label=r'$\Delta\Phi_L+\Delta\Phi_U$ (m)')
    
    for ax in axs.flatten():
        genpigplot(ax)
        ax.contour(thickness_above_floating(fielddict['thickness'][1],
            fielddict['Z_base'][1]), levels=[0], colors='k', linewidths=0.5)
        ax.contour(thickness_above_floating(fielddict['thickness'][0],
                fielddict['Z_base'][0]), levels=[0], colors='k', linewidths=0.5,
                linestyles=':')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.5)
