#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Sun Oct 19 13:00:16 2025

@author: Jon Paul Lundquist
"""

import os
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy.random import SeedSequence, default_rng
#from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
from lambda_corr import lambda_corr
from math import erf, sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm 
from numba import njit, prange, get_num_threads

@njit(nogil=True, parallel=True)
def online_moments(x):
    """
    Compute mean, sample variance (ddof=1), skew, and excess kurtosis of a 1D array
    using a low-memory parallel scheme.

    Returns
    -------
    mean      : float64
    var       : float64   (sample variance, ddof=1)
    skew      : float64   (bias-corrected, like scipy.stats.skew(..., bias=False))
    kurtosis  : float64   (excess kurtosis, bias-corrected, like scipy.stats.kurtosis(..., fisher=True, bias=False))
    """
    n = x.shape[0]
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan

    num_threads = get_num_threads()
    MAX_CHUNKS = num_threads * 2
    n_chunks = MAX_CHUNKS if n >= MAX_CHUNKS else n

    S1 = np.zeros(n_chunks, np.float64)
    S2 = np.zeros(n_chunks, np.float64)
    S3 = np.zeros(n_chunks, np.float64)
    S4 = np.zeros(n_chunks, np.float64)
    counts = np.zeros(n_chunks, np.int64)

    # Parallel accumulation
    for c in prange(n_chunks):
        start = c * n // n_chunks
        end = (c + 1) * n // n_chunks

        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        cnt = 0

        for i in range(start, end):
            v = x[i]
            cnt += 1
            s1 += v
            v2 = v * v
            s2 += v2
            v3 = v2 * v
            s3 += v3
            s4 += v3 * v

        S1[c] = s1
        S2[c] = s2
        S3[c] = s3
        S4[c] = s4
        counts[c] = cnt

    # Combine chunks
    n_total = 0
    S1_total = 0.0
    S2_total = 0.0
    S3_total = 0.0
    S4_total = 0.0

    for c in range(n_chunks):
        n_total += counts[c]
        S1_total += S1[c]
        S2_total += S2[c]
        S3_total += S3[c]
        S4_total += S4[c]

    n = n_total
    if n <= 1:
        return np.nan, np.nan, np.nan, np.nan

    # Raw moment averages
    m1 = S1_total / n
    m2_raw = S2_total / n
    m3_raw = S3_total / n
    m4_raw = S4_total / n

    # Central moments
    mu = m1
    mu2 = m2_raw - mu * mu
    if mu2 <= 0.0:
        return mu, 0.0, np.nan, np.nan

    mu3 = m3_raw - 3.0 * mu * m2_raw + 2.0 * mu * mu * mu
    mu4 = (
        m4_raw
        - 4.0 * mu * m3_raw
        + 6.0 * mu * mu * m2_raw
        - 3.0 * mu ** 4
    )

    # SAMPLE variance (ddof=1)
    var = (n * mu2) / (n - 1)

    # Population-style g1, g2
    g1 = mu3 / (mu2 ** 1.5)
    g2 = mu4 / (mu2 ** 2) - 3.0

    # Bias-corrected skew (SciPy)
    if n > 2:
        skew = np.sqrt(n * (n - 1)) / (n - 2) * g1
    else:
        skew = np.nan

    # Bias-corrected excess kurtosis (SciPy)
    if n > 3:
        kurt = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)
    else:
        kurt = np.nan

    return mu, var, skew, kurt

# @njit(cache=True, nogil=True, inline='always')
# def draw_bivariate_normal(n, rho, rng):
#     x = rng.standard_normal(n)
#     z = rng.standard_normal(n)
#     y = rho * x + math.sqrt(max(1e-12, 1 - rho**2)) * z
#     return x, y

# #@njit(cache=True, nogil=True, inline='always')
# def _worker_batch(N, n_per_task, rho, seed):
#     """Do n_per_task draws in one process and return an array of L0 values."""
#     rng = default_rng(seed)
#     out = np.empty(n_per_task, dtype=np.float64)
#     for i in range(n_per_task):
#         x, y = draw_bivariate_normal(N, rho, rng)
#         out[i] = lambda_corr(x, y, pvals=False)[0]
#     return out

# def compute_lambda0_parallel(N, total =1_000_000, rho=0.0,
#                              n_per_task=1_000, max_workers=None):
#     """
#     Compute total samples of L0 in parallel using process batches.

#     total:        total number of L0 samples to generate
#     rho:          correlation for draw_bivariate_normal
#     n_per_task:   batch size per process task (tune: 500–5,000 works well)
#     max_workers:  processes to use (default: os.cpu_count())
#     """
#     if max_workers is None:
#         max_workers = os.cpu_count() or 1

#     n_tasks = math.ceil(total / n_per_task)
#     # Split last batch if total not divisible
#     batch_sizes = [n_per_task] * (n_tasks - 1) + [total - n_per_task * (n_tasks - 1)]

#     # Independent RNG seeds per task (reproducible)
#     ss = SeedSequence(0xC0FFEE)  # put your master seed here if you like
#     child_seeds = ss.spawn(n_tasks)

#     out = np.empty(total, dtype=np.float64)

#     # warm up Numba in the main process (helps if not using cache=True)
#     x0, y0 = draw_bivariate_normal(10, rho, default_rng(123))
#     _ = lambda_corr(x0, y0, pvals=False)[0]

#     with ProcessPoolExecutor(max_workers=max_workers) as ex:
#         futures = []
#         for bs, sseq in zip(batch_sizes, child_seeds):
#             seed = sseq.generate_state(1, np.uint32)[0].item()
#             fut = ex.submit(_worker_batch, N, bs, rho, int(seed))
#             futures.append(fut)

#         pos = 0
#         # Progress bar over *tasks*
#         with tqdm(total=n_tasks, desc=f"Computing L0 (N={N})", 
#                   smoothing=0.2, mininterval=1) as pbar:
#             for fut in as_completed(futures):
#                 block = fut.result()      # if worker crashed, you’ll see an exception here
#                 bs = block.size
#                 out[pos:pos + bs] = block
#                 pos += bs
#                 pbar.update(1)            # <- 1 per completed task
    
#     print('\a', end='', flush=True)  # makes the terminal beep (if enabled)
    
#     return out

@njit(cache=True, nogil=True, inline='always')
def draw_bivariate_normal_numba(n, rho):
    """
    Numba-friendly version: draws n samples of a bivariate normal
    with correlation rho using Numba's RNG.
    """
    x = np.random.standard_normal(n)
    z = np.random.standard_normal(n)
    c = math.sqrt(max(1e-12, 1.0 - rho * rho))
    y = rho * x + c * z
    return x, y

@njit(cache=True, nogil=True, parallel=True)
def compute_lambda0_numba_block(N, batch, rho, seed):
    """
    Numba-parallel version: compute 'total' samples of L0.

    Parameters
    ----------
    N      : int
        Sample size for each lambda_corr evaluation.
    batch  : int
        Number of L0 samples to generate.
    rho    : float
        Correlation for the bivariate normal draws.
    seed   : int
        Global seed for Numba's RNG (reproducible in practice, though
        per-thread order is implementation-dependent).

    Returns
    -------
    out : 1D np.ndarray (float64)
        Array of Lambda_s values (L0 samples).
    out = np.empty(batch, dtype=np.float64)
    """
    out = np.empty(batch, dtype=np.float64)
    # Seed (Numba global RNG)
    np.random.seed(seed)

    for i in prange(batch):
        x, y = draw_bivariate_normal_numba(N, rho)
        out[i] = lambda_corr(x, y, False)[0] # index 2 or 4 for asymmetrical lambda

    return out

def compute_lambda0_numba(N, total, rho=0.0, batch=100_000, seed=12345):
    """
    Computes total L0 samples in chunks with tqdm progress bar.
    """
    out = np.empty(total, dtype=np.float64)

    # How many batches?
    n_batches = total // batch
    rem = total % batch

    pos = 0
    cur_seed = seed

    with tqdm(total=total, unit="samples", desc=f"Λ₀ (N={N})") as bar:
        # Full batches
        for _ in range(n_batches):
            block = compute_lambda0_numba_block(N, batch, rho, cur_seed)
            out[pos:pos+batch] = block
            pos += batch
            cur_seed += 1  # optional: vary seed per batch
            bar.update(batch)

        # Remainder batch
        if rem > 0:
            block = compute_lambda0_numba_block(N, rem, rho, cur_seed)
            out[pos:pos+rem] = block
            bar.update(rem)

    return out

#Or for even less memory use run_simulation_streamed thought it appears to be slower:
def run_simulation_streamed(N, total_samples, rho=0.0, batch_size=200_000, seed=12345):
    """
    Stream Λ0 samples in blocks using compute_lambda0_numba_block,
    keep only sums of powers in memory (O(1) in total_samples).
    """
    n_batches = math.ceil(total_samples / batch_size)

    total_n = 0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0

    cur_seed = seed

    with tqdm(total=total_samples, desc=f"Simulating N={N}") as pbar:
        for _ in range(n_batches):
            current_batch = min(batch_size, total_samples - total_n)
            if current_batch <= 0:
                break

            # this is your already-working Numba function
            block = compute_lambda0_numba_block(N, current_batch, rho, cur_seed)
            cur_seed += 1

            # accumulate moments
            total_n += current_batch
            # s1: sum x
            s1 += block.sum()
            # s2: sum x^2
            s2 += np.dot(block, block)
            # s3: sum x^3
            s3 += np.power(block, 3).sum()
            # s4: sum x^4
            s4 += np.power(block, 4).sum()

            pbar.update(current_batch)

    # --- finalize moments ---
    n = float(total_n)
    m1 = s1 / n
    m2_raw = s2 / n
    m3_raw = s3 / n
    m4_raw = s4 / n

    mu = m1
    mu2 = m2_raw - mu * mu
    mu3 = m3_raw - 3.0 * mu * m2_raw + 2.0 * mu**3
    mu4 = m4_raw - 4.0 * mu * m3_raw + 6.0 * mu**2 * m2_raw - 3.0 * mu**4

    # sample variance (ddof=1)
    var = (n * mu2) / (n - 1)
    
    std = math.sqrt(N*var)
    
    # population-style g1, g2
    if mu2 > 0.0:
        g1 = mu3 / (mu2 ** 1.5)
        g2 = mu4 / (mu2 ** 2) - 3.0
    else:
        g1 = 0.0
        g2 = 0.0

    # bias-corrected skew and kurtosis (SciPy-style, bias=False)
    if n > 2:
        skew_val = math.sqrt(n * (n - 1)) / (n - 2) * g1
    else:
        skew_val = float("nan")

    if n > 3:
        kurt_val = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)
    else:
        kurt_val = float("nan")

    return mu, std, skew_val, kurt_val

def lambda_p_normal(Lambda_S, n, alt="two-sided",
                    mu0=0, sig0=1.1149479707810142):
    def Phi(t): return 0.5*(1.0 + erf(t / sqrt(2.0)))
    z = (Lambda_S - mu0) * (n**0.5) / sig0
    
    if alt == "greater": return 1.0 - Phi(z)
    if alt == "less":    return Phi(z)
    return 2.0 * (1.0 - Phi(abs(z)))

def sigma_model(n, L_inf, a, alpha):
    return L_inf + a * n**(-alpha)

def sigma_model2(n, L_inf, a, b, alpha, beta):
    return L_inf + a * n**(-alpha) + b * n**(-beta)

def kurt_model(n, K, B):
    return -K / n - B / n**2
 
def kurt_model2(n, A, B, alpha, beta):
    return -A*n**(-alpha) - B*n**(-beta)
       
def lambda_p_edgeworth(Lambda_S, n, alt="two-sided"):
    #https://mathworld.wolfram.com/Cornish-FisherAsymptoticExpansion.html
    def Phi(t): return 0.5*(1.0 + erf(t / sqrt(2.0)))
    
    sig0 = sigma_model(n, 1.1143107462, 0.7608778059, 0.819902)
    kurt0=kurt_model(n, 10.0680789272, -31.3478710947)
    
    z = (Lambda_S - 0) * (n**0.5) / sig0

    # Cornish–Fisher adjusted z kurtosis only terms up to k^2
    
    z_cf = z + (kurt0/24.0)*(z**3 - 3.0*z) - (kurt0**2/384)*(3*z**5-24*z**3+29*z)

    if alt == "greater": return 1.0 - Phi(z_cf)
    if alt == "less":    return Phi(z_cf)
    return 2.0 * (1.0 - Phi(abs(z_cf)))

if __name__ == "__main__":
    # Precompute at rho=0
    N = 5
    L0 = compute_lambda0_numba(N, 2_000_000_000, rho=0.0)
    L0_mean, L0_var, L0_skew, L0_kurt = online_moments(L0) #uses far less memory for large arrays
    L0_sigma = math.sqrt(L0_var * N)   # ≈ constant ≈ sd*sqrt(n)
    
    # L0_mean = np.mean(L0)
    # L0_var = np.var(L0, ddof=1)
    # L0_sigma = math.sqrt(L0_var * N)   # ≈ constant ≈ sd*sqrt(n)
    # L0_skew = skew(L0, bias=False)
    # L0_kurtosis = kurtosis(L0, fisher=True, bias=False) #excess
    
    # # Precalibrate (rho=0 runs) to get:
    # #N = 5, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.2427194283692038
    # Lambda0_kurt = -0.9796857497961158
    
    # #N = 10, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.2342323619229914
    # Lambda0_kurt = -0.7083543193745968
    
    # #N = 15, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1889036078950401
    # Lambda0_kurt = -0.49670399170232676
    
    # #N = 20, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1807605532665546
    # Lambda0_kurt = -0.41832360858561307
    
    # #N = 25, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1659075501991003
    # Lambda0_kurt = -0.34574164798571777
    
    # #N = 30, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1621354397316412
    # Lambda0_kurt = -0.30604350843190414
    
    # #N = 35, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.154722647611056
    # Lambda0_kurt = -0.26764197001977047
    
    # #N = 40, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1525032817803291
    # Lambda0_kurt = -0.24262846358079862
    
    # #N = 45, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1479968498985975
    # Lambda0_kurt = -0.21830158248623785
    
    # #N = 50, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1463994400388124
    # Lambda0_kurt = -0.20077525573267818
    
    # #N = 75, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.137281722390685
    # Lambda0_kurt = -0.14010689166101917
    
    # #N = 100, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1328454970729835
    # Lambda0_kurt = -0.10706262494013322
    
    # #N = 125, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1296146049248483
    # Lambda0_kurt = -0.08615383922806277
    
    # #N = 150
    # Lambda0_sigma = 1.1275554759227395
    # Lambda0_kurt = -0.07178280591525248
    
    # #N = 175
    # Lambda0_sigma = 1.1257979604619563
    # Lambda0_kurt = -0.0613159300011852
    
    # #N = 200
    # Lambda0_sigma = 1.1246179048668103
    # Lambda0_kurt = -0.053012648518502514
    
    # #N = 250
    # Lambda0_sigma = 1.1226668462557217
    # Lambda0_kurt = -0.04194448835739059
    
    # #N = 300
    # Lambda0_sigma = 1.121333655683788
    # Lambda0_kurt = -0.033908776638267454
    
    # #N = 400
    # Lambda0_sigma = 1.1196200397312195
    # Lambda0_kurt = -0.023804557413529
    
    # #N = 500
    # Lambda0_sigma = 1.1185922603932983
    # Lambda0_kurt = -0.01712070971892745
    
    # #N = 1000
    # Lambda0_sigma = 1.1161654384753121
    # Lambda0_kurt = -0.005366571715119776
    
    # #N = 1500
    # Lambda0_sigma = 1.1146550586952357
    # Lambda0_kurt = -0.0018975099707952952
    
    # #N = 2000
    # Lambda0_sigma = 1.114661271818217
    # Lambda0_kurt = -0.0006393247494535884
    
    # #N = 2500
    # Lambda0_sigma = 1.1145159864293124
    # Lambda0_kurt = -0.0004680049568111806
    
    # #N = 3000
    # Lambda0_sigma = 1.113773641119334
    # Lambda0_kurt = 0
    
    # #N = 4000
    # Lambda0_sigma = 1.1139471520173059
    # Lambda0_kurt = 0
    
    # #N = 5000    
    # Lambda0_sigma = 1.1125819190785213     # stabilized sd*sqrt(n) at rho=0
    # Lambda0_kurt = 0

    #For all N
    #Lambda0_mean = 0  
    #Lambda0_skew = 0
    
    # data
    n = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
    75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000
    ])

    Lambda0_sigma = np.array([
        1.2427194283692038, 1.2342323619229914, 1.1889036078950401, 1.1807605532665546, 
        1.1659075501991003, 1.1621354397316412, 1.154722647611056, 1.1525032817803291, 
        1.1479968498985975, 1.1463994400388124, 1.137281722390685, 1.1328454970729835, 
        1.1296146049248483, 1.1275554759227395, 1.1257979604619563, 1.1246179048668103, 
        1.1226668462557217, 1.121333655683788, 1.1196200397312195, 1.1185922603932983, 
        1.1161654384753121, 1.1146550586952357, 1.114661271818217, 1.1145159864293124, 
        1.113665033592611, 1.1139471520173059, 1.1125819190785213])
    
    Lambda0_kurt = np.array([
        -0.9796857497961158, -0.7083543193745968, -0.49670399170232676, -0.41832360858561307, 
        -0.34574164798571777, -0.30604350843190414, -0.26764197001977047, -0.24262846358079862, 
        -0.21830158248623785, -0.20077525573267818, -0.14010689166101917, -0.10706262494013322, 
        -0.08615383922806277, -0.07178280591525248, -0.0613159300011852, -0.053012648518502514, 
        -0.04194448835739059, -0.033908776638267454, -0.023804557413529, -0.01712070971892745, 
        -0.005366571715119776, -0.0018975099707952952, -0.0006393247494535884, 
        -0.0004680049568111806, 0.0, 0.0, 0.0
    ])
    
    # optional: weights (downweight small-n or the extreme n=5000, tweak as you like)
    # sigma_y = np.full_like(n, 0.002)   # example constant uncertainty
    # popt, pcov = curve_fit(model, n, Lambda0_sigma, sigma=sigma_y, absolute_sigma=True,
    #                        p0=(1.11, 0.7, 0.8), bounds=([1.0, 0.0, 0.1], [1.2, 5.0, 2.0]))
    
    popt, pcov = curve_fit(sigma_model, n[4:], Lambda0_sigma[4:], bounds=([1.0,0.0,0.0],[Lambda0_sigma[-1],5,5]), 
                           max_nfev=10**6)
    
    L_inf, a, alpha = popt
    perr = np.sqrt(np.diag(pcov))  # 1σ uncertainties from covariance
    
    print(f"L_inf  = {L_inf:.10f} ± {perr[0]:.3g}")
    print(f"a      = {a:.10f} ± {perr[1]:.3g}")
    print(f"alpha  = {alpha:.6f}   ± {perr[2]:.3g}")
    yhat = sigma_model(n, L_inf, a, alpha)
    res  = Lambda0_sigma[4:] - yhat[4:]
    RSS  = float(np.dot(res, res))
    TSS  = float(np.dot(Lambda0_sigma[4:] - Lambda0_sigma[4:].mean(),
                        Lambda0_sigma[4:] - Lambda0_sigma[4:].mean()))
    R2   = 1.0 - RSS/TSS
    N    = n.size
    k    = 3
    df   = N - k
    RMSE = np.sqrt(RSS/N)
    sigma_hat = np.sqrt(RSS/df)   # “standard error of regression”
    AIC = N*np.log(RSS/N) + 2*k
    BIC = N*np.log(RSS/N) + k*np.log(N)
    
    print(R2, RMSE, sigma_hat, AIC, BIC)
    
    fig, ax = plt.subplots(figsize=(10,7))
    ax.loglog(n,Lambda0_sigma, label='STD', marker='d')
    ax.loglog(n, sigma_model(n, L_inf, a, alpha), label='Fit', marker='s')
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    #sigma_model2 does no better than sigma_model1
    popt, pcov = curve_fit(sigma_model2, n[4:], Lambda0_sigma[4:], bounds=([1.0,0.0,0.0,0.0,0.0],[1.113,1,1,1,1]), 
                           max_nfev=10**6)
    
    L_inf, a, b, alpha, beta = popt
    perr = np.sqrt(np.diag(pcov))  # 1σ uncertainties from covariance
    
    print(f"L_inf  = {L_inf:.10f} ± {perr[0]:.3g}")
    print(f"a      = {a:.6f} ± {perr[1]:.3g}")
    print(f"b = {b:.6f}   ± {perr[2]:.3g}")
    print(f"alpha      = {alpha:.10f} ± {perr[3]:.3g}")
    print(f"beta = {beta:.10f}   ± {perr[4]:.3g}")
    yhat = sigma_model2(n, L_inf, a, b, alpha, beta)
    res  = Lambda0_sigma[4:] - yhat[4:]
    RSS  = float(np.dot(res, res))
    TSS  = float(np.dot(Lambda0_sigma[4:] - Lambda0_sigma[4:].mean(),
                        Lambda0_sigma[4:] - Lambda0_sigma[4:].mean()))
    R2   = 1.0 - RSS/TSS
    N    = n.size
    k    = 3
    df   = N - k
    RMSE = np.sqrt(RSS/N)
    sigma_hat = np.sqrt(RSS/df)   # “standard error of regression”
    AIC = N*np.log(RSS/N) + 2*k
    BIC = N*np.log(RSS/N) + k*np.log(N)
    
    print(R2, RMSE, sigma_hat, AIC, BIC)
    
    fig, ax = plt.subplots(figsize=(10,7))
    ax.loglog(n,Lambda0_sigma, label='STD', marker='d')
    ax.loglog(n, sigma_model2(n, L_inf, a, b, alpha, beta), label='Fit', marker='s')
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    popt, pcov = curve_fit(kurt_model, n[4:], Lambda0_kurt[4:], maxfev=10**6)
    
    K, B = popt
    perr = np.sqrt(np.diag(pcov))  # 1σ uncertainties from covariance
    
    print(f"K  = {K:.10f} ± {perr[0]:.3g}")
    print(f"B      = {B:.10f} ± {perr[1]:.3g}")
    yhat = kurt_model(n, K, B)
    res  = Lambda0_kurt[4:] - yhat[4:]
    RSS  = float(np.dot(res, res))
    TSS  = float(np.dot(Lambda0_kurt[4:] - Lambda0_kurt[4:].mean(),
                        Lambda0_kurt[4:] - Lambda0_kurt[4:].mean()))
    R2   = 1.0 - RSS/TSS
    N    = n.size
    k    = 3
    df   = N - k
    RMSE = np.sqrt(RSS/N)
    kurt_hat = np.sqrt(RSS/df)   # “standard error of regression”
    AIC = N*np.log(RSS/N) + 2*k
    BIC = N*np.log(RSS/N) + k*np.log(N)
    
    print(R2, RMSE, kurt_hat, AIC, BIC)
    
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(n, Lambda0_kurt, label='Kurtosis', marker='d')
    ax.plot(n, kurt_model(n, K, B), label='Fit', marker='s')
    ax.set_xscale('log')
    ax.set_yscale('symlog')
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    