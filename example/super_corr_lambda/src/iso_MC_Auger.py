#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:33:09 2023

@author: Jon Paul Lundquist
"""

import numpy as np
import utm
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
import pickle
#from scipy.interpolate import interp1d, griddata
import numpy.random as nr

# def rand_ecdf(data, num_samples):
#     """
#     Generate random numbers based on an empirical distribution using
#     the inverse CDF method with interpolation.

#     Parameters:
#     data (array-like): The observed data to mimic.
#     num_samples (int): The number of random samples to generate.

#     Returns:
#     numpy.ndarray: Array of random samples from the empirical distribution.
#     """
#     # Sort the data
#     sorted_data = np.sort(data)

#     # Create the empirical CDF
#     cdf_y = np.linspace(0, 1, len(sorted_data), endpoint=False)

#     # Create an interpolation function for the inverse CDF
#     inverse_cdf = interp1d(cdf_y, sorted_data, fill_value="extrapolate",kind='cubic')

#     # Generate uniform random samples
#     uniform_samples = np.random.uniform(0, 1, num_samples)

#     # Use the inverse CDF to get samples from the empirical distribution
#     random_samples = inverse_cdf(uniform_samples)

#     return random_samples

def local_to_equatorial(easting, northing, altitude, time, zenith, azimuth):

    latitude, longitude = utm.to_latlon(easting.copy(), northing.copy(), 19, northern=False)

    location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=altitude * u.m)
    obs_time = Time(time, format='gps')
    altaz = AltAz(alt=(90 - zenith) * u.deg, az=(360 - azimuth + 90) % 360 * u.deg, location=location, obstime=obs_time)
    coords = SkyCoord(altaz).transform_to('tete')
    #ra = coords.ra
    #dec = coords.dec
    
    return coords #, ra, dec

def permutation_within_time_blocks(d_time, block_size):
    n = d_time.size
    order = np.argsort(d_time)
    idx_arr = np.arange(n)

    for start in range(0, n, block_size):
        block = order[start:start + block_size]
        if block.size > 1:
            idx_arr[block] = block[nr.permutation(block.size)]
    return idx_arr

def iso_MC_Auger(data, E_cut=None, N_mc=None, save=False):
    if E_cut == None:
        Ecut = min(data['sd_energy'])

    data = data[data['sd_energy']>=Ecut]
    
    if N_mc == None:
        N_mc = data.size
        
    d_energy = data['sd_energy']
    d_easting = data['sd_easting']
    d_northing = data['sd_northing'] #Eastward-,northward-coordinate and altitude of the shower core (UTM coordinates system)
    d_altitude = data['sd_altitude']
    d_zenith = data['sd_theta']
    d_azimuth = data['sd_phi']
    d_time = data['gpstime'] + data['sd_gpsnanotime']*10**-9
    
    #d_coords, d_ra, d_dec = local_to_equatorial(d_easting, d_northing, d_altitude, d_time, d_zenith, d_azimuth)
    
    idx_arr = permutation_within_time_blocks(d_time, block_size=25) #about 4.5 months
    
    # mc_time = np.random.choice(d_time, N_mc, replace=True)
    # mc_zenith = rand_ecdf(d_zenith, N_mc)
    # mc_azimuth = rand_ecdf(d_azimuth, N_mc)
    # mc_easting = rand_ecdf(d_easting, N_mc)
    # mc_northing = rand_ecdf(d_northing, N_mc)
    # mc_altitude = griddata((d_easting, d_northing), d_altitude, (mc_easting, mc_northing), method='nearest')
    
    mc_time     = d_time           # detector state fixed
    mc_easting  = d_easting
    mc_northing = d_northing
    mc_altitude = d_altitude

    #keep detector correlations intact
    mc_zenith   = d_zenith[idx_arr]
    mc_azimuth  = d_azimuth[idx_arr]
    mc_energy   = d_energy[idx_arr]
    mc_coords = local_to_equatorial(mc_easting, mc_northing, mc_altitude, mc_time, mc_zenith, mc_azimuth)
    #mc_energy = rand_ecdf(d_energy, N_mc)
    #mc_energy.sort()
    
    ra_deg  = mc_coords.ra.deg
    dec_deg = mc_coords.dec.deg

    dtype = [("time", "f8"), ("easting", "f8"), ("northing", "f8"), ("altitude", "f8"),
             ("zenith", "f8"), ("azimuth", "f8"), ("energy", "f8"),("ra", "f8"),
             ("dec", "f8")]
    
    mc_data = np.empty(N_mc, dtype=dtype)
    mc_data["time"]     = mc_time
    mc_data["easting"]  = mc_easting
    mc_data["northing"] = mc_northing
    mc_data["altitude"] = mc_altitude
    mc_data["zenith"]   = mc_zenith
    mc_data["azimuth"]  = mc_azimuth
    mc_data["energy"]   = mc_energy
    mc_data["ra"]       = ra_deg
    mc_data["dec"]      = dec_deg

    if save:
        with open('iso_MC_Auger.pkl', 'wb') as file:
            pickle.dump((mc_coords, mc_energy), file)
    
    return mc_coords, mc_energy, mc_data