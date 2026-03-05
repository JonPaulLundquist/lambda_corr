#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:26:55 2024

@author: Jon Paul Lundquist
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
import new_cm
from matplotlib.colors import Normalize, LinearSegmentedColormap, ListedColormap, to_rgba
import colorsys

import healpy as hp
from matplotlib.collections import PolyCollection

def map_wrap(lon, center=180*u.deg):
    """
    lon : astropy Angle (scalar or array) of your longitudes (e.g. RA or SGL)
    center : astropy Quantity (angle) you want at the center of your map
    returns : radians in [-pi, pi], with increasing lon -> leftward
    """
    lon_deg = np.asarray(lon.to_value(u.deg))
    cen = center.to_value(u.deg)

    # shift relative to center, wrap into [0,360)
    delta = (lon_deg - cen) % 360.0
    # move into [–180,180]
    delta = np.where(delta > 180.0, delta - 360.0, delta)

    # flip so that +delta (east) goes left -> negative x
    return np.deg2rad(-delta)

def healpix_polygons(ax, nside, ipix, values, frame='tete', proj='supergalactic',
                         cmap='viridis', step=1, center=180*u.deg):
    if cmap=='tau':
        hues = {'yellow':60/360, 'cyan':180/360, 'red':0/360, 'blue':240/360}
        L_high, L_low, S = 0.85, 0.35, 1.0
        yellow = colorsys.hls_to_rgb(hues['yellow'], L_high, S)
        cyan   = colorsys.hls_to_rgb(hues['cyan'],   L_high, S)
        red    = colorsys.hls_to_rgb(hues['red'],    L_low,  S)
        blue   = colorsys.hls_to_rgb(hues['blue'],   L_low,  S)
        
        vmin, vmax = -1,  1
        zero    = -vmin/(vmax-vmin)
        neg_mid = zero/2
        pos_mid = (zero+1)/2
        
        cmap = LinearSegmentedColormap.from_list(
            "C–B–K–R–Y",
            [
                (0.0,     yellow),
                (neg_mid, red),
                (zero,    (0,0,0)),
                (pos_mid, blue),
                (1.0,     cyan),
            ]
        )
        
    elif cmap=='sigma':
        cmap = new_cm.extended_cmap
        
    ipix = np.asarray(ipix, dtype=np.int64)
    npix = ipix.size
    if npix == 0:
        raise ValueError("ipix is empty")

    values = np.asarray(values)
    npix_total = hp.nside2npix(nside)
    if values.size == npix_total:
        vsel = values[ipix]
    elif values.size == npix:
        vsel = values
    else:
        raise ValueError("values must be full-sky (hp.nside2npix) or aligned with ipix")

    # healpy (your version): vec.shape == (npix, 3, nverts)
    vec = hp.boundaries(nside, ipix, step=step, nest=False)
    if vec.ndim != 3 or vec.shape[0] != npix or vec.shape[1] != 3:
        raise ValueError(f"Unexpected boundaries shape: {vec.shape}")

    nverts = vec.shape[2]

    # ---- IMPORTANT: vec2ang wants (N, 3) in your build ----
    v = np.transpose(vec, (0, 2, 1)).reshape(-1, 3)   # (npix*nverts, 3)

    # Drop any bad boundary vertices BEFORE vec2ang to kill warnings
    d = np.linalg.norm(v, axis=1)
    bad_v = (~np.isfinite(d)) | (d == 0)
    if np.any(bad_v):
        # drop whole pixels that have any bad vertex
        bad_pix = bad_v.reshape(npix, nverts).any(axis=1)
        keep = ~bad_pix
        ipix = ipix[keep]
        vsel = vsel[keep]
        vec  = vec[keep]
        npix = ipix.size
        if npix == 0:
            raise ValueError("All pixels had invalid boundary vectors.")

        v = np.transpose(vec, (0, 2, 1)).reshape(-1, 3)

    theta, phi = hp.vec2ang(v)  # 1D arrays length npix*nverts

    theta = theta.reshape(npix, nverts).T  # (nverts, npix)
    phi   = phi.reshape(npix, nverts).T

    ra  = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)

    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame=frame)
    if proj != frame:
        sc = sc.transform_to(proj)

    if proj == 'supergalactic':
        lon, lat = sc.sgl, sc.sgb
    elif proj == 'galactic':
        lon, lat = sc.l, sc.b
    else:
        lon, lat = sc.ra, sc.dec

    x = map_wrap(lon, center=center)      # (nverts, npix)
    y = lat.to_value(u.rad)               # (nverts, npix)

    polys = []
    cols  = []

    for j in range(npix):
        xj = np.unwrap(x[:, j].astype(float))
        yj = y[:, j].astype(float)

        # Shift polygon into [-pi, pi] without duplicating
        k = np.round(np.median(xj) / (2*np.pi))
        xj = xj - k*(2*np.pi)
        if xj.max() > np.pi:
            xj -= 2*np.pi
        elif xj.min() < -np.pi:
            xj += 2*np.pi

        polys.append(np.c_[xj, yj])
        cols.append(vsel[j])
    
    mag = np.ceil(np.abs(np.log10(max(abs(vsel)))))
    vmin = np.floor(min(vsel)*10**mag)/10**mag
    vmax = np.ceil(max(vsel)*10**mag)/10**mag
    if (vmin < 0) & (vmax >0):
        vmin = -max((np.abs(vmin),vmax))
        vmax = max((np.abs(vmin),vmax))
    
    norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
    
    pc = PolyCollection(
    polys,
    array=np.asarray(cols),
    cmap=cmap,
    norm=norm,
    edgecolors='face',   # or 'face'
    linewidths=0.0,
    antialiaseds=False,  # key for “hairlines”
    zorder=2,
    rasterized=True
    )
        
    ax.add_collection(pc)
    return pc

def set_fig():
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection="hammer") #theta 90 to -90, phi -180 to 180
    #ax.set_facecolor('#373737')
    ax.set_facecolor('black')
    
    plt.tight_layout()
    fig.subplots_adjust(top=1.0, bottom=0.0, left=0.055, right=1.0, hspace=0.2, wspace=0.2)
    
    return fig, ax

def shade_outside_dec_cut(ax, dec_lim_deg, proj='supergalactic',
                          outside_color='#1f1f1f',
                          center=180*u.deg,
                          nlon=720, nlat=360,
                          zorder=0.4):
    """
    Shades region with (equatorial) dec < dec_lim_deg on a Hammer axis.
    Works in any displayed proj (supergalactic/galactic/tete) by transforming
    grid points back to 'tete' and applying the cut.
    """
    # grid *edges* in displayed lon/lat (Hammer expects radians)
    x_edges = np.linspace(-np.pi, np.pi, nlon + 1)
    y_edges = np.linspace(-np.pi/2, np.pi/2, nlat + 1)

    # grid *centers* for classification
    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Invert your map_wrap: x = deg2rad(-delta), delta = lon - center wrapped to [-180,180]
    cen = center.to_value(u.deg)
    lon_deg = (cen - np.rad2deg(x_cent)) % 360.0
    lat_deg = np.rad2deg(y_cent)

    Lon, Lat = np.meshgrid(lon_deg, lat_deg)  # (nlat, nlon)

    # Build SkyCoord in the DISPLAYED frame
    if proj == 'supergalactic':
        sc = SkyCoord(sgl=Lon*u.deg, sgb=Lat*u.deg, frame='supergalactic')
    elif proj == 'galactic':
        sc = SkyCoord(l=Lon*u.deg, b=Lat*u.deg, frame='galactic')
    elif proj == 'tete':
        sc = SkyCoord(ra=Lon*u.deg, dec=Lat*u.deg, frame='tete')
    else:
        # if you ever pass something else, treat it like RA/Dec
        sc = SkyCoord(ra=Lon*u.deg, dec=Lat*u.deg, frame=proj)

    # Apply the *equatorial* declination cut
    sc_eq = sc.transform_to('tete')
    outside = (sc_eq.dec.to_value(u.deg) < dec_lim_deg).astype(np.uint8)

    # 0 -> transparent, 1 -> outside_color
    cmap = ListedColormap([(0, 0, 0, 0), to_rgba(outside_color, 1.0)])

    ax.pcolormesh(
        x_edges, y_edges, outside,
        cmap=cmap, vmin=0, vmax=1,
        shading='auto',
        edgecolors='none', linewidth=0,
        antialiased=False,
        rasterized=True,
        zorder=zorder
    )
    
#def map_scatter(x, y, title=None, c=None, color=None, cmap='viridis'):
def map_scatter(x, y, x0=None, y0=None, c_title=None, c=None, s=0.5, marker='o', 
                color=None, cmap='viridis', arrows=False, multiplet=False, dirs=None, 
                B=None, arrow_len=0.02):
    
    if cmap=='tau':
        hues = {'yellow':60/360, 'cyan':180/360, 'red':0/360, 'blue':240/360}
        L_high, L_low, S = 0.85, 0.35, 1.0
        yellow = colorsys.hls_to_rgb(hues['yellow'], L_high, S)
        cyan   = colorsys.hls_to_rgb(hues['cyan'],   L_high, S)
        red    = colorsys.hls_to_rgb(hues['red'],    L_low,  S)
        blue   = colorsys.hls_to_rgb(hues['blue'],   L_low,  S)
        
        vmin, vmax = -1,  1
        zero    = -vmin/(vmax-vmin)
        neg_mid = zero/2
        pos_mid = (zero+1)/2
        
        cmap = LinearSegmentedColormap.from_list(
            "C–B–K–R–Y",
            [
                (0.0,     yellow),
                (neg_mid, red),
                (zero,    (0,0,0)),
                (pos_mid, blue),
                (1.0,     cyan),
            ]
        )

    x = map_wrap(x)
    fig, ax = set_fig()
    
    if not arrows:
        # —— standard scatter mode ——
        if c is None:
            s = ax.scatter(x, y.rad, marker=marker, color=color, zorder=2)
        else:
            ind = np.argsort(np.abs(c))
            x, y, c = x[ind], y[ind], c[ind]
            mag = np.ceil(np.abs(np.log10(max(abs(c)))))
            vmin = np.floor(min(c)*10**mag)/10**mag
            vmax = np.ceil(max(c)*10**mag)/10**mag
            if (vmin < 0) & (vmax >0):
                vmin = -max((np.abs(vmin),vmax))
                vmax = max((np.abs(vmin),vmax))
            s = ax.scatter(x, y.rad, marker=marker,
                           c=c, cmap=cmap,
                           vmin=vmin,
                           vmax=vmax,
                           s=s, zorder=5)
            map_cbar(fig, ax, s, c_title)

        if multiplet:
            if dirs is None:
                raise ValueError("multiplet=True but direction is None")
            
            if (x0 is None) or (y0 is None):
                raise ValueError("multiplet=True but location not given")
            
            arrow_len = abs(arrow_len)
            x0 = map_wrap(x0)
            
            theta = np.mod(dirs-90,360)
            # compute vector displacements
            dx    = -arrow_len * np.sin(theta*np.pi/180)
            dy    = arrow_len * np.cos(theta*np.pi/180)
            
            if color is None:
                #color = 'mediumvioletred'
                color = 'xkcd:hot purple'
                
            s = ax.quiver(x0, y0.rad, dx, dy,
                          color=color,
                          angles='xy', scale_units='inches', scale=5,
                          width=0.008,
                          zorder=4)

            if B is not None:
                pos = ax.get_position()
                x_center_of_ax = 0.5*(pos.x0 + pos.x1)
    
                fig.text(x_center_of_ax, 0.1, rf'B$\cdot$S$\cdot$Z = {B:.1f} nG$\cdot$Mpc',
                         ha="center", va="bottom",
                         fontsize=14, fontweight="semibold", color="black")

    else:
        arrow_len = abs(arrow_len) #+0.5

        # —— arrow mode ——
        if dirs is None:
            raise ValueError("arrows=True but dirs is None")

        # sort everything if colouring
        if c is not None:
            ind    = np.argsort(np.abs(c))
            x, y, c, dirs, arrow_len = x[ind], y[ind], c[ind], dirs[ind], arrow_len[ind]
            norm = Normalize(vmin=np.floor(c.min()*10)/10,
                             vmax=np.ceil (c.max()*10)/10)
        else:
            x, y, dirs = x, y, dirs

        theta = np.mod(dirs-90,360)
        
        # compute vector displacements
        dx    = -arrow_len * np.sin(theta*np.pi/180)
        dy    = arrow_len * np.cos(theta*np.pi/180)

        if c is None:            
            s = ax.quiver(x, y.rad, dx, dy,
                          color=color,
                          angles='xy', scale_units='inches', scale=5,
                          width=0.004,
                          zorder=2)
        else:
            s = ax.quiver(x, y.rad, dx, dy, c,
                          norm=norm,
                          cmap=cmap,
                          angles='xy', scale_units='inches', scale=5,
                          width=0.004,
                          zorder=2)
            map_cbar(fig, ax, s, c_title)
    
    return fig, ax, s

def map_decor(fig, ax, dec_lim, proj='tete'):
    ax.set_xticks(np.radians([-180, -120, -60, 0, 60, 120, 180]))
    ax.set_xticklabels(['','','','', '','', ''])
    ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(['-60','-30','','30','60'], fontweight='semibold', size=12, 
                       ha='right')
 
    # Manually add the vertical grid lines
    for lon in np.arange(-180, 181, 60):
        ax.plot(np.full(1000, np.radians(lon)), np.radians(np.linspace(-90, 90, 1000)), 
                color='darkgrey', linestyle='-', linewidth=1)
    
    for lat in np.arange(-90, 91, 30):
        ax.plot(np.radians(np.linspace(-180, 180, 1000)), np.full(1000, np.radians(lat)), 
                color='darkgrey', linestyle='-', linewidth=1)

    max_line, super_line, gal_line = map_lines(dec_lim, proj=proj)
    
    if proj=='tete':
        xname = 'R.A. [deg]'
        yname = 'Dec. [deg]'
        ax.text(np.radians(-165), np.deg2rad(-7), '360', fontweight='semibold', 
                color='white', size=12, ha='center')
        ax.text(0, np.deg2rad(-9.5), '180', fontweight='semibold', color='white', 
                size=12, ha='center')    
        ax.text(np.radians(172), np.deg2rad(-7), '0', fontweight='semibold', 
                color='white', size=12, ha='center')
        ax.text(np.radians(0), np.radians(-20), xname, fontweight='semibold', 
                color='white', size=11, rotation='horizontal', ha='center')
    elif proj=='supergalactic':
        xname = 'SGL [deg]'
        yname = 'SGB [deg]'
        
        x0   = map_wrap(Angle(0.001,   u.deg))   # radians x-location for “0”
        x180 = map_wrap(Angle(180, u.deg))
        x360 = map_wrap(Angle(360, u.deg))

        if x0 > np.pi/4:
            x0  = x0 - np.radians(6)
        elif x0 < -np.pi/4:
            x0  = x0 + np.radians(6)

        if x360 > np.pi/4:
            x360  = x360 - np.radians(14)
        elif x360 < -np.pi/4:
            x360  = x360 + np.radians(14)

        if x180 > np.pi/4:
            x180  = x180 - np.radians(14)
        elif x180 < -np.pi/4:
            x180  = x180 + np.radians(14)
                    
        ax.text(x180, np.deg2rad(-7), '180', fontweight='semibold', 
                color='white', size=12, ha='center')
        ax.text(x0, np.deg2rad(-7), '0', fontweight='semibold', color='white', 
                size=12, ha='center')
        ax.text(x360, np.deg2rad(-7), '360', fontweight='semibold', color='white', 
                size=12, ha='center')
        
        ax.text(np.radians(123), np.radians(4), xname, fontweight='semibold', 
                color='white', size=11, rotation='horizontal', ha='center')
        
    elif proj=='galactic':
        xname = 'l  [deg]'
        yname = 'b  [deg]'
        
    ax.text(np.radians(-225), np.radians(40), yname, fontweight='semibold', 
            color='black', size=11, rotation='horizontal', ha='center')
    
    return max_line, super_line, gal_line

def map_cbar(fig, ax, s, title):
    cbar = fig.colorbar(s, ax=ax, orientation='vertical', shrink=0.7, aspect=15)
    if title is not None:
        cbar.set_label(title, fontsize=12, fontweight='semibold', color='black')
    
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('semibold')
    
    return

def map_lines(dec_lim, proj='tete'):
    #FOV limit

    max_line = SkyCoord(ra=np.linspace(0, 360, 1000) * u.deg, dec=1000*[dec_lim]*u.deg, 
                       frame='tete').transform_to(proj)
    
    super_line = SkyCoord(sgl=np.linspace(0, 360, 1000) * u.deg, 
                          sgb=np.zeros(1000) * u.deg, 
                          frame='supergalactic').transform_to(proj)
    
    gal_line = SkyCoord(l=np.linspace(0, 360, 1000) * u.deg, 
                          b=np.zeros(1000) * u.deg, 
                          frame='galactic').transform_to(proj)
    
    return max_line, super_line, gal_line

def make_Angle(ra,dec):
    if not isinstance(ra, Angle) and not isinstance(ra, SkyCoord):
        ra = np.asarray(ra)
        dec = np.asarray(dec)
        if any(ra>2*np.pi):
            ra = Angle(ra, unit='deg')
            dec = Angle(dec, unit='deg')
        else:
            ra = Angle(ra, unit='rad')
            dec = Angle(dec, unit='rad')
    elif isinstance(ra, SkyCoord):
            ra = Angle(ra.deg, unit='deg')
            dec = Angle(dec.deg, unit='deg')
    return ra, dec

def plot_great_circle(line_coord, ax, **plot_kw):
    """
    line_coord: a SkyCoord with either .ra/.dec or .sgl/.sgb
    ax: Hammer projection axis
    plot_kw: e.g. color='red', linestyle='-'
    """
    # pick fields
    try:
        lon_rad = map_wrap(line_coord.ra)
        lat = line_coord.dec
    except AttributeError:
        lon_rad = map_wrap(Angle(line_coord.sgl, unit=u.deg))
        lat = Angle(line_coord.sgb, unit=u.deg)
    
    # wrap to [–π, +π]
    #lon_rad = lon.wrap_at('180d').rad
    lat_rad = lat.rad

    # find where jumps > π (i.e. the wrap)
    dlon = np.abs(np.diff(lon_rad))
    split_idx = np.where(dlon > np.pi)[0] + 1

    # split into continuous segments
    segs = np.split(np.arange(len(lon_rad)), split_idx)
    for seg in segs:
        ax.plot(lon_rad[seg], lat_rad[seg], **plot_kw)

def map_supergalactic(sgl, sgb, x0=None, y0=None, ipix=None, nside=None, c_title=None, title=None, 
                      c=None, s=0.5, marker='o', color=None, cmap='viridis', file=None, 
                      proj='tete', savefig=1, arrows=False, multiplet=False, dirs=None, B=None, arrow_len=0.02):
    
    if cmap=='sigma':
        cmap = new_cm.extended_cmap

    sgl, sgb = make_Angle(sgl,sgb)

    if proj !='supergalactic':        
        coords = SkyCoord(ra=sgl.deg*u.deg, dec=sgb.deg*u.deg, frame=proj).transform_to('supergalactic')
        sgl = coords.sgl
        sgb = coords.sgb
    
    if ipix is not None:
        fig, ax = set_fig()
    
        pc = healpix_polygons(ax, nside=nside, ipix=ipix, values=c, frame='tete',
            proj='supergalactic', cmap=cmap)
        
        map_cbar(fig, ax, pc, c_title)

    else:
        fig, ax, s = map_scatter(sgl, sgb, x0=x0, y0=y0, c_title=c_title, c=c, s=s, 
                                 marker=marker, color=color, cmap=cmap, arrows=arrows, 
                                 dirs=dirs, arrow_len=arrow_len, multiplet=multiplet, B=B)
    
    pos = ax.get_position()          # Bbox in figure coords
    x_center_of_ax = 0.5*(pos.x0 + pos.x1)
    
    fig.suptitle(title,
                 x=x_center_of_ax, y=0.92,
                 fontsize=16, fontweight="semibold", color="black")
    
    dec_lim = 25
    
    max_line, super_line, gal_line = map_decor(fig, ax, dec_lim, proj='supergalactic')
    max_x = map_wrap(max_line.sgl)
    max_y = max_line.sgb.rad
    max_ind = np.argsort(max_x)
    max_x = max_x[max_ind]
    max_y = max_y[max_ind]
    
    super_x = map_wrap(super_line.sgl)
    super_y = super_line.sgb.rad
    super_ind = np.argsort(super_x)
    super_x = super_x[super_ind]
    super_y = super_y[super_ind]

    gal_x = map_wrap(gal_line.sgl)
    gal_y = gal_line.sgb.rad
    gal_ind = np.argsort(gal_x)
    gal_x = gal_x[gal_ind]
    gal_y = gal_y[gal_ind]

    # find where the Hammer projection jumps (i.e. wrap-around)
    dlon    = np.abs(np.diff(max_x))
    wrap_ix = np.where(dlon > np.pi)[0] + 1

    # # split into continuous segments
    # segs = np.split(np.arange(len(max_x)), wrap_ix)

    # # fill each piece down to the bottom edge (-π/2)
    # for seg in segs:
    #     ax.fill_between(max_x[seg],
    #                     max_y[seg],
    #                     -np.pi/2,
    #                     color='#1f1f1f',
    #                     zorder=0.5)
        
    # mask outside FOV (dec cut) — robust in any rotated frame
    shade_outside_dec_cut(ax, dec_lim, proj='supergalactic', outside_color='#1f1f1f', zorder=0.5)
    # —————————————————————————————————————————————————————
    
    plot_great_circle(max_line, ax, color='lightgrey', linestyle='--', linewidth=1)
    # supergalactic plane
    plot_great_circle(super_line, ax, color='red', linewidth=1)
    # galactic plane
    plot_great_circle(gal_line, ax, color='blue', linewidth=1)
    
    if savefig and (file is not None):
        plt.savefig(file, dpi=600, format='png')
    
    plt.show()
    
    return