#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Sun Oct 19 13:00:16 2025

@author: Jon Paul Lundquist
"""

#import os
import math
import numpy as np
#from concurrent.futures import ProcessPoolExecutor, as_completed
#from numpy.random import SeedSequence, default_rng
#from scipy.stats import skew, kurtosis
#from scipy.optimize import curve_fit
from lambda_corr import lambda_corr_nb
from math import erf, sqrt
#import matplotlib.pyplot as plt
from tqdm import tqdm 
from numba import njit, prange, get_num_threads
import itertools
from scipy.stats import beta
from concurrent.futures import ThreadPoolExecutor
import numba as nb

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
    Compute batch samples of (Lambda_s, Lambda_xy, Lambda_yx).
    """
    out_s  = np.empty(batch, dtype=np.float64)
    out_xy = np.empty(batch, dtype=np.float64)
    out_yx = np.empty(batch, dtype=np.float64)

    # Seed global RNG (Numba-compatible)
    np.random.seed(seed)

    for i in prange(batch):
        x, y = draw_bivariate_normal_numba(N, rho)

        # lambda_corr returns a tuple
        Lam_s, _, Lam_xy, _, Lam_yx, _, _ = lambda_corr_nb(x, y, N, pvals=False)

        out_s[i]  = Lam_s
        out_xy[i] = Lam_xy
        out_yx[i] = Lam_yx

    return out_s, out_xy, out_yx

def compute_lambda0_numba(N, total, rho=0.0, batch=100_000, seed=12345):
    """
    Computes total L0 samples for Lambda_s, Lambda_xy, Lambda_yx.
    """
    out_s  = np.empty(total, dtype=np.float64)
    out_xy = np.empty(total, dtype=np.float64)
    out_yx = np.empty(total, dtype=np.float64)

    n_batches = total // batch
    rem = total % batch

    pos = 0
    cur_seed = seed

    with tqdm(total=total, unit="samples", desc=f"Λ₀ (N={N})") as bar:

        for _ in range(n_batches):
            bs, bxy, byx = compute_lambda0_numba_block(
                N, batch, rho, cur_seed
            )

            out_s[pos:pos+batch]  = bs
            out_xy[pos:pos+batch] = bxy
            out_yx[pos:pos+batch] = byx

            pos += batch
            cur_seed += 1
            bar.update(batch)

        if rem > 0:
            bs, bxy, byx = compute_lambda0_numba_block(
                N, rem, rho, cur_seed
            )

            out_s[pos:pos+rem]  = bs
            out_xy[pos:pos+rem] = bxy
            out_yx[pos:pos+rem] = byx

            bar.update(rem)

    print("\a", end="", flush=True)
    return out_s, out_xy, out_yx

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

#Generation of all possible permutations
def enumerate_lambda(n: int) -> np.ndarray:
    """Full enumeration of Lambda_s under permutation null (x fixed as 1..n, y permuted)."""
    x = np.arange(1, n + 1, dtype=np.float64)
    out = np.empty(math.factorial(n), dtype=np.float64)

    k = 0
    for y_perm in itertools.permutations(range(1, n + 1)):
        y = np.array(y_perm, dtype=np.float64)
        L, *_ = lambda_corr_nb(x, y, n, pvals=False)
        out[k] = L
        k += 1

    return out

def _block_worker(args):
    x, n, prefix = args
    # build the rest list
    used = set(prefix)
    rest = [v for v in range(1, n+1) if v not in used]
    m = len(prefix)

    out = np.empty(math.factorial(n - m), dtype=np.float64)
    k = 0
    for tail in itertools.permutations(rest):
        y = np.empty(n, dtype=np.float64)
        # prefix
        for i, v in enumerate(prefix):
            y[i] = v
        # tail
        for j, v in enumerate(tail):
            y[m + j] = v
        L, *_ = lambda_corr_nb(x, y, n, pvals=False)
        out[k] = L
        k += 1
    return prefix, out  # return a block

def enumerate_lambda_threaded(n: int, m: int = 2, num_workers: int = 8) -> np.ndarray:
    """
    Enumerate all permutations by fixing a prefix of length m and threading over prefixes.
    For n=10, m=2 => 90 tasks of size 8! each (good granularity).
    """
    x = np.arange(1, n+1, dtype=np.float64)

    prefixes = list(itertools.permutations(range(1, n+1), m))
    block_size = math.factorial(n - m)
    total = math.factorial(n)

    out = np.empty(total, dtype=np.float64)

    def iter_args():
        for prefix in prefixes:
            yield (x, n, prefix)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for t, (prefix, block) in enumerate(tqdm(ex.map(_block_worker, iter_args()),
                                                 total=len(prefixes), smoothing=0.1)):
            out[t*block_size:(t+1)*block_size] = block

    return out

def _block_worker_count(args):
    x, n, prefix = args

    used = set(prefix)
    rest = [v for v in range(1, n + 1) if v not in used]
    m = len(prefix)

    c = 0
    for tail in itertools.permutations(rest):
        y = np.empty(n, dtype=np.float64)

        for i, v in enumerate(prefix):
            y[i] = v
        for j, v in enumerate(tail):
            y[m + j] = v

        L, *_ = lambda_corr_nb(x, y, n, pvals=False)

        # If you truly expect exact ±1 for some permutations, exact compare is fine.
        # If there is any chance of float fuzz, use a tolerance.
        if L == 1.0 or L == -1.0:
            c += 1
            # or: if abs(L) == 1.0: c += 1

    return c


def count_abs_lambda_eq1_threaded(n: int, m: int = 2, num_workers: int = 8) -> int:
    """
    Count how many permutations yield abs(Lambda_s)==1
    by fixing a prefix of length m and threading over prefixes.
    """
    x = np.arange(1, n + 1, dtype=np.float64)

    prefixes = list(itertools.permutations(range(1, n + 1), m))

    def iter_args():
        for prefix in prefixes:
            yield (x, n, prefix)

    total_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for c in tqdm(ex.map(_block_worker_count, iter_args()),
                      total=len(prefixes), smoothing=0.1, desc=f"|Λ|=1 count (n={n})"):
            total_count += c

    return total_count

def _factorials_u64(n: int) -> np.ndarray:
    fac = np.empty(n + 1, dtype=np.uint64)
    fac[0] = 1
    for i in range(1, n + 1):
        fac[i] = fac[i - 1] * np.uint64(i)
    return fac

@njit(cache=True, inline='always')
def _unrank_lex_u64(rank, n, fac, out_perm):
    avail = np.empty(n, dtype=np.int32)
    for i in range(n):
        avail[i] = i + 1

    r = rank
    m = n
    for pos in range(n):
        f = fac[m - 1]                  # (m-1)!
        idx = np.int64(r // f)          # 0..m-1
        r = r - np.uint64(idx) * f

        out_perm[pos] = avail[idx]

        for j in range(idx, m - 1):
            avail[j] = avail[j + 1]
        m -= 1

@njit(cache=True, inline='always')
def _next_permutation(a):
    n = a.size
    i = n - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        return False

    j = n - 1
    while a[j] <= a[i]:
        j -= 1

    tmp = a[i]
    a[i] = a[j]
    a[j] = tmp

    lo = i + 1
    hi = n - 1
    while lo < hi:
        tmp = a[lo]
        a[lo] = a[hi]
        a[hi] = tmp
        lo += 1
        hi -= 1
    return True

@njit(cache=True, inline='always')
def _perm_to_float_y(perm, y):
    for i in range(perm.size):
        y[i] = float(perm[i])

@njit(cache=True, inline='always')
def _lambda_only(x, y, n):
    # Avoid UNPACK_EX: index the tuple result
    res = lambda_corr_nb(x, y, n, pvals=False)
    return res[0]

@njit(cache=True, parallel=True)
def count_abs_lambda_eq1_all_perms(n, fac):
    total = fac[n]  # n!

    x = np.empty(n, dtype=np.float64)
    for i in range(n):
        x[i] = float(i + 1)

    nthreads = nb.get_num_threads()
    nchunks = nthreads * 8
    if nchunks < 1:
        nchunks = 1

    base = total // np.uint64(nchunks)
    rem  = total - base * np.uint64(nchunks)

    counts = np.zeros(nchunks, dtype=np.uint64)
    counts2 = np.zeros(nchunks, dtype=np.uint64)
    for c in prange(nchunks):
        c_u = np.uint64(c)

        # start = c*base + min(c, rem)
        add = c_u if c_u < rem else rem
        start = c_u * base + add

        length = base + (np.uint64(1) if c_u < rem else np.uint64(0))
        if length == 0:
            continue

        perm = np.empty(n, dtype=np.int32)
        _unrank_lex_u64(start, n, fac, perm)

        y = np.empty(n, dtype=np.float64)
        local = np.uint64(0)
        local2 = np.uint64(0)
        for k in range(length):
            _perm_to_float_y(perm, y)
            L = _lambda_only(x, y, n)

            if L == 1.0 or L == -1.0:
                local += np.uint64(1)
            if L == 0.0:
                local2 += np.uint64(1)

            if k + 1 < length:
                _next_permutation(perm)

        counts[c] = local
        counts2[c] = local2

    s = np.uint64(0)
    s2 = np.uint64(0)
    for c in range(nchunks):
        s += counts[c]
        s2 += counts2[c]
    return s, s2

def count_abs_eq1_exact(n, num_threads= 0):
    fac = _factorials_u64(n)
    if num_threads and num_threads > 0:
        nb.set_num_threads(num_threads)
    c1, c2 = count_abs_lambda_eq1_all_perms(n, fac)
    return int(c1), int(c2)

if __name__ == "__main__":
    # Precompute at rho=0
    # N = 5
    # L0, L0_yx, L0_xy = compute_lambda0_numba(N, 2_000_000_000, rho=0.0)
    # L0_mean, L0_var, L0_skew, L0_kurt = online_moments(L0) #uses far less memory for large arrays
    # L0_sigma = math.sqrt(L0_var * N)   # ≈ constant ≈ sd*sqrt(n)
    # print(np.sum(L0==0)/L0.size)
    # print(np.sum(np.abs(L0)==1)/L0.size)
    # L0 = L0[:500_000_000]
    # beta.fit(np.abs(L0[(L0!=0) & (np.abs(L0)!=1)]), floc=0.0, fscale=1.0)
    
    # N = 18
    # L0, _, _ = compute_lambda0_numba(N, 1_000_000_000, rho=0.0)
    # L0_mean, L0_var, L0_skew, L0_kurt = online_moments(L0) #uses far less memory for large arrays
    # L0_sigma = math.sqrt(L0_var * N)   # ≈ constant ≈ sd*sqrt(n)
    # print(np.sum(L0==0)/L0.size)
    # print(np.sum(np.abs(L0)==1)/L0.size)
    # alpha, beta, loc, scale = beta.fit(np.abs(L0[(L0!=0) & (np.abs(L0)!=1)]), floc=0.0, fscale=1.0)
    
    N = 225
    L0, _, _ = compute_lambda0_numba(N, 1_000_000_000, rho=0.0)
    #L0_mean, L0_var, L0_skew, L0_kurt = online_moments(L0) #uses far less memory for large arrays
    #L0_sigma = math.sqrt(L0_var * N)   # ≈ constant ≈ sd*sqrt(n)
    print(repr(np.sum(L0==0)/L0.size))
    print(repr(np.sum(np.abs(L0)==1)/L0.size))
    alpha, beta, loc, scale = beta.fit(np.abs(L0[(L0!=0) & (np.abs(L0)!=1)]), floc=0.0, fscale=1.0)
    print(repr(alpha))
    print(repr(beta))

    # L0_mean = np.mean(L0)
    # L0_var = np.var(L0, ddof=1)
    # L0_sigma = math.sqrt(L0_var * N)   # ≈ constant ≈ sd*sqrt(n)
    # L0_skew = skew(L0, bias=False)
    # L0_kurtosis = kurtosis(L0, fisher=True, bias=False) #excess
    
    # # Precalibrate (rho=0 runs) to get:
    # #N = 5, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.2427194283692038
    # Lambda0_kurt = -0.9796857497961158
    # p0 = 0.18333333333333332
    # p1 = 0.03333333333333333     5! = 120, N_perm = +/-1 = 4
    
    # N = 6
    # p0 = 0.03194444444444444
    # p1 = 0.011111111111111112    6! = 720, N_perm = +/-1 = 8
    
    # N = 7
    # p0 = 0.08611111111111111
    # p1 = 0.0015873015873015873   7! = 5040, N_perm = +/-1 = 8
    
    # N = 8
    # p0 = 0.06483134920634921
    # p1 = 0.0009424603174603175   8! = 40320, N_perm = +/-1 = 38
    
    # N = 9
    # p0 = 0.07830687830687831
    # p1 = 9.92063492063492e-05    9! = 362880, N_perm = +/-1 = 36
    
    #BETA DISTRIBUTIONS ARE FIT TO ABSOLUTE VALUES FOR MORE STATISTICS AND NULL IS SYMMETRIC
    # #N = 10, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.2342323619229914
    # Lambda0_kurt = -0.7083543193745968
    #Beta distribution with 0 and 1 probabilities for EXACT abs(L) distribution
    # p0 = 0.06776014109347443
    # p1 = 3.031305114638448e-05   10! = 3628800, N_perm = +/-1 = 110
    # alpha = 1.4031180011417934
    # beta = 2.6579089754819853
    
    # #N = 11
    #Beta distribution with 0 and 1 probabilities for EXACT abs(L) distribution
    # p0 = 0.0708556798140131
    # p1 = 2.0041686708353374e-06  11! = 39916800, N_perm = +/-1 = 80
    # alpha = 1.3964874007690553
    # beta = 2.9759841690111215
    
    # #N = 12
    #Beta distribution with 0 and 1 probabilities for EXACT abs(L) distribution
    # p0 = 0.06714529554807333
    # p1 = 6.513548180214847e-07  12! = 479001600, N_perm = +/-1 = 312
    # alpha = 1.4378036180809
    # beta = 3.2336691391510604

    # #N = 13,  
    #EXACT 0 and 1 probabilities
    # p0 = 0.06600454715038048
    # p1 = 3.789934345489901e-08     13! = 6227020800,  N_perm = +/-1 = 236
    #Beta distribution N_MC = 500,000,000
    # alpha = 1.4274144119465353
    # beta = 3.518822685429833

    # #N = 14
    #EXACT 0 and 1 probabilities
    # p0 = 0.06496984457983962
    # p1 = 1.0553085949911346e-08   14! = 87178291200,  N_perm = +/-1 = 920
    #Beta distribution N_MC = 500,000,000
    # alpha = 1.4552545703790116
    # beta = 3.7485744560979195
    
    # #N = 15, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1889036078950401
    # Lambda0_kurt = -0.49670399170232676
    #Beta distribution N_MC = 500,000,000 with 0 probability for N_MC = 6,000,000,000, 1 searched with find_abs1.py
    # p0 = 0.06238369283333333
    # p1 = 4.1753513975736197e-10    N_perm = +/-1 = 546
    # alpha = 1.4447437136748829
    # beta = 4.00762594005941
    
    # #N = 16, #Beta distribution N_MC = 500,000,000 with 0 probability for N_MC = 6,000,000,000, 1 searched with find_abs1.py
    # p0 = 0.062640989
    # p1 = 1.4663436455764497e-10     N_perm = +/-1 = 3068
    # alpha = 1.4671825739472821
    # beta = 4.223033938955358
    
    # #N = 17,  #Beta distribution N_MC = 500,000,000 with 0 probability for N_MC = 6,000,000,000, 1 searched with find_abs1.py
    # p0 = 0.059537863333333337
    # p1 = 5.263047980134815e-12     N_perm = +/-1 = 1872
    # alpha = 1.45608952243845
    # beta = 4.45823438758898
    
    # #N = 18, #Beta distribution N_MC = 500,000,000 with 0 probability for N_MC = 6,000,000,000, 1 searched with find_abs1.py
    # p0 = 0.0604504015
    # p1 = 1.4366546569705612e-12     N_perm = +/-1 = 9198
    # alpha = 1.4748300243343895
    # beta = 4.661919097481938
    
    # #N = 19, #Beta distribution N_MC = 500,000,000 with 0 probability for N_MC = 6,000,000,000, 1 searched with find_abs1.py
    # p0 = 0.057216893
    # p1 = 4.1251147667560884e-14   N_perm = +/-1 = 5018
    # alpha = 1.4644181779078853
    # beta = 4.879683679160764
    
    # #N = 20, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1807605532665546
    # Lambda0_kurt = -0.41832360858561307
    #Beta distribution N_MC = 500,000,000 with 0 probability for N_MC = 6,000,000,000, 1 searched with find_abs1.py
    # p0 = 0.05846732416666667
    # p1 = 1.069422439233359e-14  N_perm = +/-1 = 26018
    # alpha = 1.4805092620739457
    # beta = 5.0736720165789455
    
    # #N = 21
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 6,000,000,000
    # p0 = 0.05526443416666667
    # p1 = 2.675229584544318e-16  N_perm = +/-1 = 13668
    # alpha = 1.4702557385709127
    # beta = 5.275428822561992
    
    # #N = 22
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.056689878
    # p1 = 0.0
    # alpha = 1.4845023191449047
    # beta = 5.461907884453
    
    # #N = 23
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.05361363
    # p1 = 0.0
    # alpha = 1.4751097546757268
    # beta = 5.651686059204632
    
    # #N = 24
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.055135552
    # p1 = 0.0
    # alpha = 1.4876743353571005
    # beta = 5.830734762026783
    
    # #N = 25, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1659075501991003
    # Lambda0_kurt = -0.34574164798571777
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.052150197
    # p1 = 0.0
    # alpha = 1.47888092293881
    # beta = 6.0102024965515985

    # #N = 26
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.053692919
    # p1 = 0.0
    # alpha = 1.4902126712917316
    # beta = 6.183505864613792
    
    # #N = 27
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.050890785
    # p1 = 0.0
    # alpha = 1.481845624216896
    # beta = 6.352736078846976
    
    # #N = 28
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.052419307
    # p1 = 0.0
    # alpha = 1.4922291390943292
    # beta = 6.5203056989847
    
    # #N = 29
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.049762158
    # p1 = 0.0
    # alpha = 1.4845170545731274
    # beta = 6.683121593579365
    
    # #N = 30, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1621354397316412
    # Lambda0_kurt = -0.30604350843190414
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.051266391
    # p1 = 0.0
    # alpha = 1.4939833744022095
    # beta = 6.845129043097732
    
    # #N = 35, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.154722647611056
    # Lambda0_kurt = -0.26764197001977047
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.047058179
    # p1 = 0.0
    # alpha = 1.4898825870424968
    # beta = 7.602912186121273
    
    # #N = 40, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1525032817803291
    # Lambda0_kurt = -0.24262846358079862
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.046875418
    # p1 = 0.0
    # alpha = 1.499302409292932
    # beta = 8.315653246239936
    
    # #N = 45, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1479968498985975
    # Lambda0_kurt = -0.21830158248623785
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.043886314
    # p1 = 0.0
    # alpha = 1.4955990847250638
    # beta = 8.969373209567118
    
    # #N = 50, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1463994400388124
    # Lambda0_kurt = -0.20077525573267818
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.043944742
    # p1 = 0.0
    # alpha = 1.502465594762055
    # beta = 9.602286385340967
    
    # #N = 55
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.041731068
    # p1 = 0.0
    # alpha = 1.4992361161278511
    # beta = 10.185169129999263

    # #N = 60
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.041834139
    # p1 = 0.0
    # alpha = 1.5044662234906618
    # beta = 10.761719585499176
    
    # #N = 65
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.040133117
    # p1 = 0.0
    # alpha = 1.5022242332542404
    # beta = 11.29663472972513
    
    # #N = 70
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.040235615
    # p1 = 0.0
    # alpha = 1.5062253835425243
    # beta = 11.824783962631374
    
    # #N = 75, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.137281722390685
    # Lambda0_kurt = -0.14010689166101917
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.038882284
    # p1 = 0.0
    # alpha = 1.5042723097595319
    # beta = 12.319867078793214
    
    # #N = 80
    #Beta distribution and 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.039020723
    # p1 = 0.0
    # alpha = 1.5079053750994416
    # beta = 12.81532905149589

    # #N = 85
    #Beta distribution and 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.037892605
    # p1 = 0.0
    # alpha = 1.50612090780228
    # beta = 13.278011578898136

    # #N = 90
    #Beta distribution and 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.038032011
    # p1 = 0.0
    # alpha = 1.5089013366844017
    # beta = 13.7419513511831
    
    # #N = 95
    #Beta distribution and 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.03708275
    # p1 = 0.0
    # alpha = 1.5075254654395471
    # beta = 14.17886825457339

    # #N = 100, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1328454970729835
    # Lambda0_kurt = -0.10706262494013322
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.037203714
    # p1 = 0.0
    # alpha = 1.5102234802411307
    # beta = 14.620807506130546
    
    # #N = 125, N_MC = 2,000,000,000
    # Lambda0_sigma = 1.1296146049248483
    # Lambda0_kurt = -0.08615383922806277
    #Beta distribution N_MC = 500,000,000 with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.035357447
    # p1 = 0.0
    # alpha = 1.5113284065657007
    # beta = 16.631275961748585
    
    # #N = 150
    # Lambda0_sigma = 1.1275554759227395
    # Lambda0_kurt = -0.07178280591525248
    #Beta distribution with 0 and 1 probabilities for N_MC = 500,000,000
    # p0 = 0.034638402
    # p1 = 0.0
    # alpha = 1.514566218290616
    # beta = 18.46135081454609
    
    # #N = 175
    # Lambda0_sigma = 1.1257979604619563
    # Lambda0_kurt = -0.0613159300011852
    #Beta distribution with 0 and 1 probabilities for N_MC = 500,000,000
    # p0 = 0.033672204
    # p1 = 0.0
    # alpha = 1.51565912355074
    # beta = 20.13302497097813
    
    # #N = 200
    # Lambda0_sigma = 1.1246179048668103
    # Lambda0_kurt = -0.053012648518502514
    #Beta distribution with 0 and 1 probabilities for N_MC = 500,000,000
    # p0 = 0.033266614
    # p1 = 0.0
    # alpha = 1.5181813633855017
    # beta = 21.701969321294357
    
    # #N = 225
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000  #Started using HPC
    # p0 = 0.032681559
    # p1 = 0.0
    # alpha = 1.518913532842462
    # beta = 23.163118691724332
    
    # #N = 250
    # Lambda0_sigma = 1.1226668462557217
    # Lambda0_kurt = -0.04194448835739059
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.032427826
    # p1 = 0.0
    # alpha = 1.5205505305248728
    # beta = 24.552094234696717

    # #N = 275
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.032034477
    # p1 = 0.0
    # alpha = 1.5214446822928052
    # beta = 25.870278062401255

    # #N = 300
    # Lambda0_sigma = 1.121333655683788
    # Lambda0_kurt = -0.033908776638267454
    # p0 = 0.031861522
    # p1 = 0.0
    # alpha = 1.5229574230485505
    # beta = 27.136947451508497

    # #N = 325
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.031575871
    # p1 = 0.0
    # alpha = 1.5236559811756816
    # beta = 28.342616246520397

    # #N = 350
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.031441454
    # p1 = 0.0
    # alpha = 1.5247232196563312
    # beta = 29.507296214303995

    # #N = 375
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.031220126
    # p1 = 0.0
    # alpha = 1.5254575297765316
    # beta = 30.631514126811364

    # #N = 400
    # Lambda0_sigma = 1.1196200397312195
    # Lambda0_kurt = -0.023804557413529
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.031163257
    # p1 = 0.0
    # alpha = 1.526318019911779
    # beta = 31.7192585034286
    
    # #N = 425
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030977032
    # p1 = 0.0
    # alpha = 1.526845897280529
    # beta = 32.768332687149446

    # #N = 450
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030888612
    # p1 = 0.0
    # alpha = 1.5277077843012459
    # beta = 33.79554550614419
    
    # #N = 475
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030751585
    # p1 = 0.0
    # alpha = 1.528195807582924
    # beta = 34.78506303697264

    # #N = 500
    # Lambda0_sigma = 1.1185922603932983
    # Lambda0_kurt = -0.01712070971892745
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030699806
    # p1 = 0.0
    # alpha = 1.5288456636813765
    # beta = 35.754911133449475
    
    # #N = 525
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030585179
    # p1 = 0.0
    # alpha = 1.5292162131170872
    # beta = 36.698065581167064

    # #N = 550
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030540247
    # p1 = 0.0
    # alpha = 1.5300201255867647
    # beta = 37.623570262759436
    
    # #N = 575
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030446717
    # p1 = 0.0
    # alpha = 1.530457233851254
    # beta = 38.52795788132418

    # #N = 600
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030377763
    # p1 = 0.0
    # alpha = 1.5310201246651984
    # beta = 39.411441943365325
    
    # #N = 625
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030312046
    # p1 = 0.0
    # alpha = 1.5317665585255542
    # beta = 40.28258751199046
    
    # #N = 650
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030270575
    # p1 = 0.0
    # alpha = 1.5317090284693289
    # beta = 41.11695548772955

    # #N = 675
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030216366
    # p1 = 0.0
    # alpha = 1.5322985276327077
    # beta = 41.95506071921157

    # #N = 700
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030182576
    # p1 = 0.0
    # alpha = 1.5326493081879098
    # beta = 42.768932424746424

    # #N = 725
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030133224
    # p1 = 0.0
    # alpha = 1.533454878326082
    # beta = 43.58453326711583

    # #N = 750
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.03011434
    # p1 = 0.0
    # alpha = 1.533373190827337
    # beta = 44.35884279549922

    # #N = 775
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030066833
    # p1 = 0.0
    # alpha = 1.5337987163954845
    # beta = 45.137388992535854
    
    # #N = 800
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.030032078
    # p1 = 0.0
    # alpha = 1.5341584699361657
    # beta = 45.898482432935275

    # #N = 825
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.029965395
    # p1 = 0.0
    # alpha = 1.5344860867391108
    # beta = 46.65003909704631

    # #N = 850
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.029968917
    # p1 = 0.0
    # alpha = 1.5349810276918976
    # beta = 47.39322880772411

    # #N = 875
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.029930739
    # p1 = 0.0
    # alpha = 1.5354628204406753
    # beta = 48.12850686411679
    
    # #N = 900
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.029911014
    # p1 = 0.0
    # alpha = 1.5353554431348562
    # beta = 48.8360834115559

    # #N = 925
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.029861657
    # p1 = 0.0
    # alpha = 1.5356060363460318
    # beta = 49.542915424547346

    # #N = 950
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.029838691
    # p1 = 0.0
    # alpha = 1.5359283689755698
    # beta = 50.24456877336037

    # #N = 975
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.029816765
    # p1 = 0.0
    # alpha = 1.5362005077273875
    # beta = 50.932650989089765

    # #N = 1000
    # Lambda0_sigma = 1.1161654384753121
    # Lambda0_kurt = -0.005366571715119776
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.029800397
    # p1 = 0.0
    # alpha = 1.5365810229894095
    # beta = 51.61939556620012

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
    # n = np.array([
    # 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
    # 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000
    # ])

    # Lambda0_sigma = np.array([
    #     1.2427194283692038, 1.2342323619229914, 1.1889036078950401, 1.1807605532665546, 
    #     1.1659075501991003, 1.1621354397316412, 1.154722647611056, 1.1525032817803291, 
    #     1.1479968498985975, 1.1463994400388124, 1.137281722390685, 1.1328454970729835, 
    #     1.1296146049248483, 1.1275554759227395, 1.1257979604619563, 1.1246179048668103, 
    #     1.1226668462557217, 1.121333655683788, 1.1196200397312195, 1.1185922603932983, 
    #     1.1161654384753121, 1.1146550586952357, 1.114661271818217, 1.1145159864293124, 
    #     1.113665033592611, 1.1139471520173059, 1.1125819190785213])
    
    # Lambda0_kurt = np.array([
    #     -0.9796857497961158, -0.7083543193745968, -0.49670399170232676, -0.41832360858561307, 
    #     -0.34574164798571777, -0.30604350843190414, -0.26764197001977047, -0.24262846358079862, 
    #     -0.21830158248623785, -0.20077525573267818, -0.14010689166101917, -0.10706262494013322, 
    #     -0.08615383922806277, -0.07178280591525248, -0.0613159300011852, -0.053012648518502514, 
    #     -0.04194448835739059, -0.033908776638267454, -0.023804557413529, -0.01712070971892745, 
    #     -0.005366571715119776, -0.0018975099707952952, -0.0006393247494535884, 
    #     -0.0004680049568111806, 0.0, 0.0, 0.0
    # ])
    
    # # optional: weights (downweight small-n or the extreme n=5000, tweak as you like)
    # # sigma_y = np.full_like(n, 0.002)   # example constant uncertainty
    # # popt, pcov = curve_fit(model, n, Lambda0_sigma, sigma=sigma_y, absolute_sigma=True,
    # #                        p0=(1.11, 0.7, 0.8), bounds=([1.0, 0.0, 0.1], [1.2, 5.0, 2.0]))
    
    # popt, pcov = curve_fit(sigma_model, n[4:], Lambda0_sigma[4:], bounds=([1.0,0.0,0.0],[Lambda0_sigma[-1],5,5]), 
    #                        max_nfev=10**6)
    
    # L_inf, a, alpha = popt
    # perr = np.sqrt(np.diag(pcov))  # 1σ uncertainties from covariance
    
    # print(f"L_inf  = {L_inf:.10f} ± {perr[0]:.3g}")
    # print(f"a      = {a:.10f} ± {perr[1]:.3g}")
    # print(f"alpha  = {alpha:.6f}   ± {perr[2]:.3g}")
    # yhat = sigma_model(n, L_inf, a, alpha)
    # res  = Lambda0_sigma[4:] - yhat[4:]
    # RSS  = float(np.dot(res, res))
    # TSS  = float(np.dot(Lambda0_sigma[4:] - Lambda0_sigma[4:].mean(),
    #                     Lambda0_sigma[4:] - Lambda0_sigma[4:].mean()))
    # R2   = 1.0 - RSS/TSS
    # N    = n.size
    # k    = 3
    # df   = N - k
    # RMSE = np.sqrt(RSS/N)
    # sigma_hat = np.sqrt(RSS/df)   # “standard error of regression”
    # AIC = N*np.log(RSS/N) + 2*k
    # BIC = N*np.log(RSS/N) + k*np.log(N)
    
    # print(R2, RMSE, sigma_hat, AIC, BIC)
    
    # fig, ax = plt.subplots(figsize=(10,7))
    # ax.loglog(n,Lambda0_sigma, label='STD', marker='d')
    # ax.loglog(n, sigma_model(n, L_inf, a, alpha), label='Fit', marker='s')
    # ax.legend()
    # fig.tight_layout()
    # plt.show()
    
    # #sigma_model2 does no better than sigma_model1
    # popt, pcov = curve_fit(sigma_model2, n[4:], Lambda0_sigma[4:], bounds=([1.0,0.0,0.0,0.0,0.0],[1.113,1,1,1,1]), 
    #                        max_nfev=10**6)
    
    # L_inf, a, b, alpha, beta = popt
    # perr = np.sqrt(np.diag(pcov))  # 1σ uncertainties from covariance
    
    # print(f"L_inf  = {L_inf:.10f} ± {perr[0]:.3g}")
    # print(f"a      = {a:.6f} ± {perr[1]:.3g}")
    # print(f"b = {b:.6f}   ± {perr[2]:.3g}")
    # print(f"alpha      = {alpha:.10f} ± {perr[3]:.3g}")
    # print(f"beta = {beta:.10f}   ± {perr[4]:.3g}")
    # yhat = sigma_model2(n, L_inf, a, b, alpha, beta)
    # res  = Lambda0_sigma[4:] - yhat[4:]
    # RSS  = float(np.dot(res, res))
    # TSS  = float(np.dot(Lambda0_sigma[4:] - Lambda0_sigma[4:].mean(),
    #                     Lambda0_sigma[4:] - Lambda0_sigma[4:].mean()))
    # R2   = 1.0 - RSS/TSS
    # N    = n.size
    # k    = 3
    # df   = N - k
    # RMSE = np.sqrt(RSS/N)
    # sigma_hat = np.sqrt(RSS/df)   # “standard error of regression”
    # AIC = N*np.log(RSS/N) + 2*k
    # BIC = N*np.log(RSS/N) + k*np.log(N)
    
    # print(R2, RMSE, sigma_hat, AIC, BIC)
    
    # fig, ax = plt.subplots(figsize=(10,7))
    # ax.loglog(n,Lambda0_sigma, label='STD', marker='d')
    # ax.loglog(n, sigma_model2(n, L_inf, a, b, alpha, beta), label='Fit', marker='s')
    # ax.legend()
    # fig.tight_layout()
    # plt.show()
    
    # popt, pcov = curve_fit(kurt_model, n[4:], Lambda0_kurt[4:], maxfev=10**6)
    
    # K, B = popt
    # perr = np.sqrt(np.diag(pcov))  # 1σ uncertainties from covariance
    
    # print(f"K  = {K:.10f} ± {perr[0]:.3g}")
    # print(f"B      = {B:.10f} ± {perr[1]:.3g}")
    # yhat = kurt_model(n, K, B)
    # res  = Lambda0_kurt[4:] - yhat[4:]
    # RSS  = float(np.dot(res, res))
    # TSS  = float(np.dot(Lambda0_kurt[4:] - Lambda0_kurt[4:].mean(),
    #                     Lambda0_kurt[4:] - Lambda0_kurt[4:].mean()))
    # R2   = 1.0 - RSS/TSS
    # N    = n.size
    # k    = 3
    # df   = N - k
    # RMSE = np.sqrt(RSS/N)
    # kurt_hat = np.sqrt(RSS/df)   # “standard error of regression”
    # AIC = N*np.log(RSS/N) + 2*k
    # BIC = N*np.log(RSS/N) + k*np.log(N)
    
    # print(R2, RMSE, kurt_hat, AIC, BIC)
    
    # fig, ax = plt.subplots(figsize=(10,7))
    # ax.plot(n, Lambda0_kurt, label='Kurtosis', marker='d')
    # ax.plot(n, kurt_model(n, K, B), label='Fit', marker='s')
    # ax.set_xscale('log')
    # ax.set_yscale('symlog')
    # ax.legend()
    # fig.tight_layout()
    # plt.show()
