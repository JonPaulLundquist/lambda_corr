#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 17:04:57 2025

@author: Jon Paul Lundquist
"""
import healpy as hp
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from numba import njit
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
from scipy.stats import norm, siegelslopes
from map_fig import map_supergalactic
from iso_MC_Auger import iso_MC_Auger
import gc
from parabola_fig import parabola_fig
from lambda_corr import lambda_corr_nb
from pathlib import Path

def grid_equal(sep):
    Nside = np.array((2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9))
    seps = hp.nside2resol(Nside, arcmin=True) / 60
    ind = np.argmin(abs(seps - sep))
    nside = int(Nside[ind])

    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)

    theta, phi = hp.pix2ang(nside, ipix, nest=False)  # radians
    dec = np.degrees(0.5*np.pi - theta)
    ra  = np.degrees(phi)

    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='tete')
    return coords, ipix, nside

def wedge(events, energies, center, Dir, W, E_cut, Dist):
    separation = center.separation(events).deg                
    azimuthal = center.position_angle(events).deg
    width = (azimuthal - Dir) % 360
    width = np.where(width > 180, 360 - width, width)
    inside = ((energies>=E_cut) & (width<=W) & (separation<=Dist))
    
    return inside, separation[inside]
     
def rotate_ra(angle_degrees):
    adjust = np.ones(angle_degrees.size)*np.pi
    adjust[(angle_degrees >= 0) & (angle_degrees <= 180)] = -np.pi
    return adjust

def parabola(x, a, x0, y0):
    return a*10e-4 * (x - x0)**2 + y0

def residuals(params, x, y, y_err):
    return (y - parabola(x, params)) / y_err

@njit(nogil=True, fastmath=True)
def quickdup(arr, lo, hi):
    # Base case: no or one element means no duplicates.
    if lo >= hi:
        return False

    # Use the first element as the pivot.
    pivot = arr[lo]
    i = lo + 1
    j = hi

    # Partition the array.
    while i <= j:
        # Move i to the right as long as elements are less than pivot.
        while i <= hi and arr[i] < pivot:
            i += 1
        # Move j to the left as long as elements are greater than pivot.
        while j >= lo and arr[j] > pivot:
            j -= 1
        if i <= j:
            # If we find two equal elements, report duplicate.
            if arr[i] == arr[j]:
                return True
            # Swap arr[i] and arr[j]
            tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp
            i += 1
            j -= 1

    # Now, j is the last index of the left partition.
    # Check if pivot equals the element at j (if j is in range).
    if j > lo and arr[j] == pivot:
        return True

    # Recurse on the left and right partitions.
    if quickdup(arr, lo, j):
        return True
    if quickdup(arr, i, hi):
        return True
    return False

@njit(nogil=True, fastmath=True, inline='always')
def get_indices_l(n, x, cut):
    ind = np.empty(n, dtype=np.bool_)
    N = 0
    for i in range(n):
        if x[i] <= cut:
            ind[i] = True
            N += 1
        else:
            ind[i] = False
    return ind, N

@njit(nogil=True, fastmath=True, inline='always')
def get_indices_g(n, x, cut):
    ind = np.empty(n, dtype=np.bool_)
    N = 0
    for i in range(n):
        if x[i] >= cut:
            ind[i] = True
            N += 1
        else:
            ind[i] = False
    return ind, N

@njit(nogil=True, fastmath=True, inline='always')
def select_indices(arr, ind):
    N = ind.size
    n = 0
    for i in range(N):
        if ind[i]:
            n += 1
    result = np.empty(n, dtype=arr.dtype)
    j = 0
    for i in range(N):
        if ind[i]:
            result[j] = arr[i]
            j += 1
    return result

def process_center_unpack(args):
    return process_center(*args)

@njit(nogil=True, fastmath=True)
def process_center(energies, separation, azimuthal,
                   directions, Ecuts, widths, distances, minN):
    
    def energy_duplicates(arr,n):
        for i in range(1, n):
            if arr[i] == arr[i - 1]:
                return True
        return False
    
    def dist_duplicates(arr,n):
        arr = np.sort(arr)
        for i in range(1, n):
            if arr[i] == arr[i - 1]:
                return True
        return False
     
    # Initialize tracking variables for the best (lowest) p-value
    best_lambda = 0
    best_p = 1
    best_E = np.nan
    best_dir = np.nan
    best_distance = np.nan
    best_width = np.nan
    scan_count = 0
    
    neg_lambda = 0
    neg_p = 1
    neg_E = np.nan
    neg_dir = np.nan
    neg_distance = np.nan
    neg_width = np.nan
    
    energies1 = energies
    separation1 = separation
    energies2 = energies
    separation2 = separation
    energies3 = energies
    separation3 = separation
    energies4 = energies
    separation4 = separation
    
    N = energies.size
    width1 = np.empty(N, dtype=np.float32)
    
    for direction in directions:
        width1 = (azimuthal - direction) % 360
        width1 = np.where(width1 > 180, 360 - width1, width1)
                
        ind1, Ndir = get_indices_l(N, width1, max(widths))
                
        if (Ndir < minN):
            continue
        
        energies1 = select_indices(energies, ind1)
        separation1 = select_indices(separation, ind1)
        if (np.array_equal(energies2, energies1) and 
            np.array_equal(separation2, separation1)):
            continue

        energies2 = energies1
        separation2 = separation1
        width2 = select_indices(width1, ind1)
                
        NE_prev = np.inf
        for Ecut in Ecuts:                 
            ind2, NE = get_indices_g(Ndir, energies2, Ecut)
            
            if (NE < minN):
                break         
            
            if (NE == NE_prev):
                continue

            energies3 = select_indices(energies2, ind2)
            separation3 = select_indices(separation2, ind2)
            width3 = select_indices(width2, ind2)
            
            NE_prev = NE
            
            NW_prev = np.inf
            for width in widths:
                ind3, NW = get_indices_l(NE, width3, width)
                
                if (NW < minN):
                    break
                
                if (NW == NW_prev):
                    continue
                
                energies4 = select_indices(energies3, ind3)
                separation4 = select_indices(separation3, ind3)
                    
                NW_prev = NW
                
                Ndist_prev = np.inf
                for distance in distances:                    
                    ind4, Ndist = get_indices_l(NW, separation4, distance)
                        
                    if (Ndist < minN):
                        break
                    
                    if (Ndist == Ndist_prev):
                        continue    
                        
                    Ndist_prev = Ndist
                    
                    E_final = select_indices(energies4, ind4)
                    D_final = select_indices(separation4, ind4)
                    
                    t_lambda, t_p, _, _, _, _, _ = lambda_corr_nb(E_final, D_final, Ndist, ptype='approx')
                    #t_lambda2 = t_lambda
                    #t_p2 = np.inf   # sentinel meaning "no permutation result"
                    
                    # if Ndist < 30 and abs(t_lambda) >= 0.5:
                    #     if (t_p <= 1e-5) and (t_p > 1e-6):
                    #         t_lambda, t_p, _, _, _, _, _ = lambda_corr_nb(
                    #             E_final, D_final, Ndist,
                    #             ptype='perm', p_tol=1e-6, n_perm=int(1e6)
                    #         )
                    #     elif t_p <= 1e-6:
                    #         t_lambda, t_p, _, _, _, _, _ = lambda_corr_nb(
                    #             E_final, D_final, Ndist,
                    #             ptype='perm', p_tol=1e-8, n_perm=int(1e8)
                    #         )
                    
                    #if t_p2 < t_p:
                    #    t_lambda, t_p = t_lambda2, t_p2
    
                    scan_count += 1

                    if ((t_p < neg_p) and (t_lambda < 0)) or \
                        ((t_p == neg_p) and (t_lambda <= neg_lambda)):
                        neg_p = t_p
                        neg_lambda = t_lambda
                        neg_E = Ecut
                        neg_dir = direction
                        neg_distance = distance
                        neg_width = width
                        neg_N = Ndist
                    
                    if (t_p < best_p) or \
                       ((t_p == best_p) and (abs(t_lambda) >= abs(best_lambda))):
                        best_p = t_p
                        best_lambda = t_lambda
                        best_E = Ecut
                        best_dir = direction
                        best_distance = distance
                        best_width = width
                        best_N = Ndist

    return best_lambda, best_p, best_E, best_dir, best_distance, best_width, best_N, scan_count,\
           neg_lambda, neg_p, neg_E, neg_dir, neg_distance, neg_width, neg_N

def process_center_noscan(events, energies, center, Dir1, W1, E1, Dist1):                   
    inside, sep = wedge(events, energies, center, Dir1, W1, E1, Dist1)
    N = sep.size
    
    if N >=6:
        Lambda, p, _, _, _, _, _ = lambda_corr_nb(select_indices(energies, inside), sep, N)

    else:
        Lambda = np.nan
        p = np.nan
    
    return Lambda, p, N

def super_corr(input_type='data'):
    #global distances, widths, directions, Ecuts, minN
    
    # Determine the number of workers (example calculation)
    num_workers = int(np.round(os.cpu_count()))
    
    minN = 6 #Minimum number of events in a wedge
    
    grid, ipix, nside = grid_equal(1) 
    mask = grid.dec.deg <= 25
    grid = grid[mask]
    ipix = ipix[mask]
    grid = grid.transform_to('supergalactic')
    
    Ecut = 16 #13 Lower bound energy cut
    
    #Scan parameters
    # distances = np.arange(15, 95, 5)[::-1] #np.linspace(15, 90, 16)[::-1]
    # widths = np.arange(5, 95, 5)[::-1] / 2 #np.arange(10, 95, 5)[::-1] / 2 #np.linspace(10, 90, 9)[::-1] / 2
    # directions = np.arange(0,360,5) #np.linspace(0, 355, 72)
    # Ecuts = np.arange(15,75,5) #np.linspace(10, 80, 15)
    
    distances = np.arange(14, 92, 2)[::-1] #np.linspace(15, 90, 16)[::-1]
    widths = np.arange(4, 92, 2)[::-1] / 2 #np.arange(10, 95, 5)[::-1] / 2 #np.linspace(10, 90, 9)[::-1] / 2
    directions = np.arange(0,360,2) #np.linspace(0, 355, 72)
    Ecuts = np.arange(15,75,5) #np.linspace(10, 80, 15)
    
    distances = distances.astype(np.float32, order='C')
    widths = widths.astype(np.float32, order='C')
    directions = directions.astype(np.float32, order='C')
    Ecuts = Ecuts.astype(np.float32, order='C')
    
    ROOT = Path(__file__).resolve().parent.parent   # adjust if this file isn't in src/
    data_path = ROOT / "data" / "dataSummarySD1500.csv"
    result_path = str(ROOT / "results") + "/lambda/"
    
    data_Auger = np.genfromtxt(data_path, delimiter=',', names=True, filling_values=np.nan)
    
    data_Auger = data_Auger[~np.isnan(data_Auger['sd_ra']) & ~np.isnan(data_Auger['sd_dec']) 
                            & ~np.isnan(data_Auger['sd_energy'])]
        
    if input_type == 'data':

        ra = data_Auger['sd_ra'].astype(np.float32)
        dec = data_Auger['sd_dec'].astype(np.float32)
        energy = data_Auger['sd_energy']
        events = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='tete').transform_to('supergalactic')
        
    elif input_type == 'random':
        
        mc_coords, mc_energy, _ = iso_MC_Auger(data_Auger)
        
        events = mc_coords.transform_to('supergalactic')
        energy = mc_energy.astype(np.float32)

        
    events = events[energy >= Ecut]
    energies = energy[energy >= Ecut].astype(np.float32)
    ind = np.argsort(energies)
    events = events[ind]
    energies = energies[ind]
    
    center_seps = grid[:, None].separation(events)
    event_mask = (center_seps <= max(distances)*u.deg)
    # Precompute azimuthal angles (deg) for each args element (outside Numba function)
    azimuthal_list = [center.position_angle(events[event_mask[i]]).deg.astype(np.float32, 
                                                                              order='C') 
                      for i, center in enumerate(grid)]
    
    energies = energies.astype(np.float32, order='C')
    separation = center_seps.deg.astype(np.float32, order='C')
    
    def iter_args():
        for i in range(grid.size):
            m = event_mask[i]
            yield (energies[m], separation[i, m], azimuthal_list[i],
                  directions, Ecuts, widths, distances, minN)
    
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        results = list(tqdm(ex.map(process_center_unpack, iter_args()),
                            total=grid.size, smoothing=0.2))
    
    # Unpack the results into arrays.
    Lambda = np.empty(grid.size)
    p_value = np.empty(grid.size)
    E = np.empty(grid.size)
    Dir = np.empty(grid.size)
    Dist = np.empty(grid.size)
    W = np.empty(grid.size)
    N = np.empty(grid.size)
    scans = np.empty(grid.size)
    neg_lambda = np.empty(grid.size)
    neg_p = np.empty(grid.size)
    neg_E = np.empty(grid.size)
    neg_Dir = np.empty(grid.size)
    neg_Dist = np.empty(grid.size)
    neg_W = np.empty(grid.size)
    neg_N = np.empty(grid.size)
    
    for i, res in enumerate(results):
        Lambda[i], p_value[i], E[i], Dir[i], Dist[i], W[i], N[i], scans[i], \
        neg_lambda[i], neg_p[i], neg_E[i], neg_Dir[i], neg_Dist[i], neg_W[i], neg_N[i] = res

    #Non-threaded loop for debugging
    # for i in range(grid.size):
    #     print(i)
    #     m = event_mask[i]
    #     Lambda[i], p_value[i], E[i], Dir[i], Dist[i], W[i], N[i], scans[i], \
    #     neg_lambda[i], neg_p[i], neg_E[i], neg_Dir[i], neg_Dist[i], neg_W[i], neg_N[i] = \
    #         process_center(energies[m], separation[i, m], azimuthal_list[i],
    #                            directions, Ecuts, widths, distances, minN)
            
    sigma = norm.isf(p_value/2)
    neg_sigma = norm.isf(neg_p/2)
    
    map_supergalactic(grid.sgl, grid.sgb, ipix=ipix, nside=nside, c_title=r"Lundquist's $\mathbf{\Lambda}$", 
                      c=Lambda, cmap='tau', file=result_path+'SuperCorr_lambdaMap.png', proj='supergalactic', savefig=1,
                      title=r"$\mathbf{\Lambda}$ from Max. $\mathbf{\sigma}$ Scan ($\mathbf{|\Lambda| > 0}$)")

    popt, score, reduced_chi2, bin_edges = parabola_fig(grid.sgb.deg, Lambda, result_path+'SuperCorr_MeanLambda', ylim=(-0.5,0.9), varname=r'$\mathbf{\Lambda}$',
                                                        title=r"$\mathbf{\langle \Lambda \rangle}$ from Max. $\mathbf{\sigma}$ Scan ($\mathbf{|\Lambda| > 0}$)")
    
    a, x0, y0 = popt  #a is the test statistic of supergalactic structure of correlations
    print(f"Supergalactic Curvature Test Statistic a={a*(180/np.pi)**2:.3f}") #Convert to rad^-2 for a larger number
    print(f"χ²/dof: {reduced_chi2:3f}")
    
    SS = np.zeros(len(grid))
    for i in range(len(grid)):
        inside, sep = wedge(events, energies, grid[i], Dir[i], W[i], 
                            E[i], Dist[i])

        res = siegelslopes(sep, 1/(energies[inside])) #seigelslopes(y,x) for some reason
        
        slope = res.slope
        SS[i] = slope/(0.5*10**2)
    
    map_supergalactic(grid.sgl, grid.sgb, ipix=ipix, nside=nside,
                      c_title=r'$\mathbf{\sigma}$ (Local Significance)', c=sigma, cmap='sigma', 
                      file=result_path+'SuperCorr_SigmaMap.png', proj='supergalactic', savefig=1, 
                      title=r"$\mathbf{\sigma}$ from Max. $\mathbf{\sigma}$ Scan ($\mathbf{|\Lambda| > 0}$)")
    
    map_supergalactic(grid.sgl[Lambda<0], grid.sgb[Lambda<0], ipix=ipix[Lambda<0], nside=nside,
                      c_title=r'$\mathbf{\sigma}$ (Local Significance)', c=sigma[Lambda<0], cmap='sigma', 
                      file=result_path+'SuperCorr_SigmaMap_NegLambda.png', proj='supergalactic', savefig=1, 
                      title=r"$\mathbf{\sigma}$ from Max. $\mathbf{\sigma}$ Scan ($\mathbf{|\Lambda| > 0}$) for $\mathbf{\Lambda < 0}$")
    
    popt, score, reduced_chi2, _ = parabola_fig(grid.sgb.deg[Lambda<0], sigma[Lambda<0], result_path+'SuperCorr_MeanSigma', 
                                             varname=r'$\mathbf{\sigma}$', ylim=(3.4,4.6),
                                             bin_edges=bin_edges,
                                             title=r"$\mathbf{\langle \sigma \rangle}$ from Max. $\mathbf{\sigma}$ Scan ($\mathbf{|\Lambda| > 0}$) for $\mathbf{\Lambda < 0}$")
    
    map_supergalactic(grid.sgl, grid.sgb, ipix=ipix, nside=nside,
                      c_title=r'$\mathbf{\sigma}$ (Local Significance)', 
                      c=neg_sigma, cmap='sigma', file=result_path+'SuperCorr_NegSigmaMap.png', 
                      proj='supergalactic', savefig=1, 
                      title=r"$\mathbf{\sigma}$ from Max. $\mathbf{\sigma}$ Scan ($\mathbf{\Lambda < 0}$)")
    
    popt, score, reduced_chi2, _ = parabola_fig(grid.sgb.deg, neg_sigma, result_path+'SuperCorr_MeanNegSigma', 
                                             varname=r'$\mathbf{\sigma}$', ylim=(3.2,4.3), 
                                             bin_edges=bin_edges,
                                             title=r"$\mathbf{\langle \sigma \rangle}$ from Max. $\mathbf{\sigma}$ Scan ($\mathbf{\Lambda < 0}$)")

    popt, score, reduced_chi2, _ = parabola_fig(grid.sgb.deg, SS, result_path+'SuperCorr_MeanSiegel', 
                                                stat='Mean', varname='Seigel Slope', ylim=(-30,25),
                                                bin_edges=bin_edges,
                                                title=r"Mean Siegel Slopes: Distance vs 1/E ($\mathbf{\sigma}$ Scan $\mathbf{|\Lambda| > 0}$)")
    
    map_supergalactic(grid.sgl, grid.sgb, ipix=ipix, nside=nside,
                      c_title=r'Siegel Slope', c=SS, cmap='tau', 
                      file=result_path+'SuperCorr_SiegelMap.png', proj='supergalactic', savefig=1,
                      title=r"Siegel Slopes: Distance vs 1/E ($\mathbf{\sigma}$ Scan $\mathbf{|\Lambda| > 0}$)")
    
    SS_ind = (SS>0) & (Lambda<0)
    map_supergalactic(grid.sgl[SS_ind], grid.sgb[SS_ind],
                      c_title=r'B$\cdot$S$\cdot$Z  [nG$\cdot$Mpc]', c=SS[SS_ind], 
                      cmap='plasma', file=result_path+'SuperCorr_FieldMap.png', 
                      proj='supergalactic', savefig=1, arrows=True, 
                      dirs=Dir[SS_ind], arrow_len=np.exp(sigma[SS_ind]-min(sigma[SS_ind]))/1.5,
                      title=r"Magnetic Field Map from Multiplets ($\mathbf{\sigma}$ Scan $\mathbf{|\Lambda| > 0}$)")
    
    neg_SS = np.zeros(len(grid))
    for i in range(len(grid)):
        inside, sep = wedge(events, energies, grid[i], neg_Dir[i], neg_W[i], 
                            neg_E[i], neg_Dist[i])
        x = 1/(energies[inside])
        y = sep

        res = siegelslopes(y, x)
        
        slope = res.slope
        neg_SS[i] = slope/(0.5*10**2) 
    
    map_supergalactic(grid.sgl, grid.sgb, ipix=ipix, nside=nside,
                      c_title=r'Siegel Slope', c=neg_SS, cmap='plasma', 
                      file=result_path+'SuperCorr_NegSiegelMap.png', proj='supergalactic', savefig=1,
                      title=r"Siegel Slopes: Distance vs 1/E ($\mathbf{\sigma}$ Scan $\mathbf{\Lambda < 0}$)")
    
    map_supergalactic(grid.sgl, grid.sgb,
                      c_title=r'B$\cdot$S$\cdot$Z  [nG$\cdot$Mpc]', c=neg_SS, 
                      cmap='plasma', file=result_path+'SuperCorr_NegFieldMap.png', 
                      proj='supergalactic', savefig=1, arrows=True, 
                      dirs=neg_Dir, arrow_len=np.exp(neg_sigma-min(neg_sigma))/1.5,
                      title=r"Magnetic Field Map from Multiplets ($\mathbf{\sigma}$ Scan $\mathbf{\Lambda < 0}$)")
    
    mask = (Lambda < 0) & np.isfinite(Lambda) & np.isfinite(sigma)
    idx = np.where(mask)[0]

    smax = np.max(sigma[idx])
    close = idx[np.abs(sigma[idx] - smax) <= 1e-4]

    best = close[np.argmin(Lambda[close])]

    inside, sep = wedge(events, energies, grid[best], Dir[best], W[best], E[best], Dist[best])
    
    inside_sgl = events[inside].sgl.rad
    inside_sgb = events[inside].sgb.rad
    inside_energy = energies[inside]
    
    map_supergalactic(inside_sgl, inside_sgb, x0=grid.sgl[best], y0=grid.sgb[best],
                      c_title=r'Energy  [EeV]', multiplet=True, dirs=Dir[best], arrow_len=3.5,
                      marker='o', c=inside_energy, s=25, cmap='Reds_r', B=SS[best],
                      file=result_path+'SuperCorr_HighestSigma.png', proj='supergalactic', savefig=1,
                      title = rf"Most Significant Multiplet: $\mathbf{{\sigma={sigma[best]:.2f},\ \Lambda={Lambda[best]:.2f}}}$")
    
    popt, score, reduced_chi2, bin_edges = parabola_fig(grid.sgb.deg, Lambda, result_path+'SuperCorr_MeanLambda_Galactic', grid=grid, 
                                                        proj='Galactic', ylim=(-0.6,0.5), varname=r'$\mathbf{\Lambda}$',
                                                        title=r"$\mathbf{\langle \Lambda \rangle}$ from Max. $\mathbf{\sigma}$ Scan ($\mathbf{|\Lambda| > 0}$)")
    
    
    # SB = np.zeros(len(grid))
    # r2 = np.zeros(len(grid))
    # rmse = np.zeros(len(grid))
    # for i in range(len(grid)):
    #     inside, sep = wedge(events, energies, grid[i], Dir[i], W[i], 
    #                         E[i], Dist[i])
    #     x = 1/(energies[inside])

    #     X = sm.add_constant(x) 
    #     y = sep
    #     # Fit robust linear model using Tukey's biweight
    #     rlm_model = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight())
    #     rlm_results = rlm_model.fit()
        
    #     slope = rlm_results.params[1]
    #     SB[i] = slope/(0.5*10**2)

    #     yhat = rlm_results.fittedvalues

    #     r2[i] = r2_score(y, yhat)
    #     rmse[i] = np.sqrt(mean_squared_error(y, yhat))

    # SB_ind = (Lambda<0) & (r2>0) & (SB>0)
    # map_supergalactic(grid.sgl[SB_ind], grid.sgb[SB_ind],
    #                   c_title=r'B$\cdot$S$\cdot$Z  [nG$\cdot$Mpc]', c=SB[SB_ind], 
    #                   cmap='plasma', file='SuperCorr_Field.png', 
    #                   proj='supergalactic', savefig=1, arrows=True, 
    #                   dirs=Dir[SB_ind], arrow_len=np.exp(sigma[SB_ind]-min(sigma[SB_ind]))/1.5)
        
    # neg_SB = np.zeros(len(grid))
    # for i in range(len(grid)):
    #     inside, sep = wedge(events, energies, grid[i], neg_Dir[i], neg_W[i], 
    #                         neg_E[i], neg_Dist[i])
    #     x = 1/(energies[inside])
    #     X = sm.add_constant(x) 
    #     y = sep
    #     # Fit robust linear model using Tukey's biweight
    #     rlm_model = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight())
    #     rlm_results = rlm_model.fit()
        
    #     slope = rlm_results.params[1]
    #     neg_SB[i] = slope/(0.5*10**2)
        
    #     yhat = rlm_results.fittedvalues

    #     r2[i] = r2_score(y, yhat)
    #     rmse[i] = np.sqrt(mean_squared_error(y, yhat))
    
    # neg_SB_ind = (neg_SB>0) & (r2>0)
    # map_supergalactic(grid.sgl[neg_SB_ind], grid.sgb[neg_SB_ind],
    #                   c_title=r'B$\cdot$S$\cdot$Z  [nG$\cdot$Mpc]', c=neg_SB[neg_SB_ind], 
    #                   cmap='plasma', file='SuperCorr_negField.png', 
    #                   proj='supergalactic', savefig=1, arrows=True, 
    #                   dirs=neg_Dir[neg_SB_ind], arrow_len=np.exp(neg_sigma[neg_SB_ind]-min(neg_sigma[neg_SB_ind]))/1.5)
    
    np.savez(result_path+'super_corr.npz',
         Lambda=Lambda,
         p_value=p_value,
         E=E,
         Dir=Dir,
         Dist=Dist,
         W=W,
         N=N,
         sigma=sigma,
         scans=scans,
         neg_lambda=neg_lambda,
         neg_p=neg_p,
         neg_E=neg_E,
         neg_Dir=neg_Dir,
         neg_Dist=neg_Dist,
         neg_W=neg_W,
         neg_N=neg_N,
         neg_sigma=neg_sigma,
         grid=grid,
         ipix=ipix,
         nside=nside,
         SS=SS,
         neg_SS=neg_SS,
         a=a,
         x0=x0,
         y0=y0,
         events=events,
         energies=energies)

    # res = np.load('super_corr.npz', allow_pickle=True)
    
    for _ in range(3):
        gc.collect()
    
    return (Lambda, p_value, E, Dir, Dist, W, N, sigma, scans, SS, a, x0, y0, neg_lambda, neg_p,
           neg_E, neg_Dir, neg_Dist, neg_W, neg_N, neg_sigma, neg_SS, grid, ipix, nside, events, energies)
           
# --- Main block ---
if __name__ == "__main__":
    super_corr()