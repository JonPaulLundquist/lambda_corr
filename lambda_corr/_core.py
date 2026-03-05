#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 18:08:51 2026

@author: jplundquist
"""

from numba import njit, objmode
import numpy as np

# --- in-place nan-ignoring median via quickselect using _partition and _select_kth_inplace---
@njit(cache=True, nogil=True, inline='always')
def _partition(a, left, right, pivot_index):
    pivot_value = a[pivot_index]
    a[pivot_index], a[right] = a[right], a[pivot_index]
    store = left
    for i in range(left, right):
        if a[i] < pivot_value:
            a[store], a[i] = a[i], a[store]
            store += 1
    a[right], a[store] = a[store], a[right]
    return store

@njit(cache=True, nogil=True, inline='always')
def _select_kth_inplace(a, k, n):
    """Quickselect on a[:length], places k-th smallest at index k."""
    left = 0
    right = n - 1
    while True:
        if left == right:
            return a[k]
        pivot_index = left + (right - left) // 2
        pivot_index = _partition(a, left, right, pivot_index)
        if k == pivot_index:
            return a[k]
        elif k < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1

@njit(cache=True, nogil=True, inline='always')
def _median_inplace(a, n):
    #Median of a[:length] ignoring NaNs
    # compact non-NaNs to front
    m = 0
    for i in range(n):
        v = a[i]
        if v == v:
            a[m] = v
            m += 1
    if m == 0:
        return np.nan

    mid = m // 2
    hi = _select_kth_inplace(a, mid, m)
    if m & 1:  #bitwise odd
        return hi

    # lower median is max of left partition
    lo = a[0]
    for i in range(1, mid):
        if a[i] > lo:
            lo = a[i]
    return 0.5 * (lo + hi)

# --- fast mean-of-medians of slopes ---
@njit(cache=True, nogil=True, inline='always')
def _mean_medians_slope(rx, ry, n):
    slopes_buf = np.empty(n - 1, dtype=np.float64)
    s = 0.0
    c = 0

    for i in range(n):
        xi = rx[i]
        yi = ry[i]
        k = 0

        # collect slopes to the left of i
        for j in range(i):
            dx = rx[j] - xi
            if dx != 0.0:
                slopes_buf[k] = (ry[j] - yi) / dx
                k += 1
        # collect slopes to the right of i
        for j in range(i + 1, n):
            dx = rx[j] - xi
            if dx != 0.0:
                slopes_buf[k] = (ry[j] - yi) / dx
                k += 1

        if k > 0:
            m = _median_inplace(slopes_buf, k)
            if m == m:
                s += m
                c += 1

    return s / c if c > 0 else np.nan

# Numpy sort is faster for about n > 1000
@njit(cache=True)
def _argsort(a, n, threshold=1000):
    if n > threshold:
        # NumPy's C argsort via objmode (usually faster for large n)
        with objmode(inds='intp[:]'):
            inds = np.argsort(a, kind='quicksort')
    else:
        # Numba-lowered argsort (often faster for small n)
        ac = np.ascontiguousarray(a)   # cheap if already contiguous
        inds = np.argsort(ac)          # returns intp dtype
    return inds

#ranks with averaged ties
@njit(cache=True, nogil=True, inline='always')
def _rankdata_avg_ties(x, n):
    #sorter = np.argsort(x, kind='quicksort')                 # stable not required here
    sorter = _argsort(x, n)
    rx = np.empty(n, np.float64)

    i = 0
    while i < n:
        j = i + 1
        xi = x[sorter[i]]
        while j < n and x[sorter[j]] == xi:
            j += 1
        r = 0.5 * (i + 1 + j)              # average rank for the tie block
        for k in range(i, j):
            rx[sorter[k]] = r

        i = j
    return rx

#standardize the ranks
@njit(cache=True, nogil=True, inline='always')
def _std_ranks(a, n):
    r = _rankdata_avg_ties(a, n) #scipy.stats.rankdata(a, method='average')

    # Doesn't affect Lambda_s. Affects Lambda_yx/Lambda_xy, the most when there are ties.
    # Tests compared to Somers' D better agrees on asymmetry when standardization is done
    # e.g. on binary data. Decreases the number of sign disagreements for 
    #Lambda_yx/Lambda_xy for various scenarios see /tests/test_opposites.py
    r = (r - np.mean(r)) / np.std(r)
    return r

@njit(cache=True, nogil=True, inline='always')
def _lambda_stats(rx, ry, n):
    
    # mean of median slopes of standardized ranks in both directions
    Lambda_yx = _mean_medians_slope(rx, ry, n) # Λ(Y|X): (Δry/Δrx, y given x)
    Lambda_xy = _mean_medians_slope(ry, rx, n) # Λ(X|Y): (Δrx/Δry, x given y)

    # NOTE (range guardrail):
    # Λ_yx and Λ_xy are robust slope-like functionals in rank space and are not
    # theoretically guaranteed to lie in [-1, 1] for all permutations. Extremely
    # rare, highly-structured near-(anti)monotone permutations can yield |Λ_asym|>1
    # (seen in permutation search starting from Λ=1; significantly rare and unseen 
    # calculating the Monte Carlo bivariate null used for p-value calibration). To enforce the 
    # conventional correlation range and restoring order vs τ/ρ in this rare regime, a 
    # reciprocal fold-back implemented as Λ_asym <- 1/Λ_asym when |Λ_asym|>1 (identity otherwise) is applied.
    # This is equivalent to a transform of f(Λ_asym) = sign(Λ_asym) * exp(-|ln(|Λ_asym|)|) for
    # *all* Λ_asym.

    # The below transform code is equivalent to Lambda[abs(Lambda)>1] = 1/Lambda[abs(Lambda)>1]
    # but is defined for all L (with f(0)=0) and serves the dual purpose of limiting to [-1,1] and 
    # increasing the linear correlation between Lambda and Kendall's tau / Spearman's rho.
    # if np.abs(Lambda_yx) > 0:
    #     Lambda_yx = math.sign(Lambda_yx)*math.exp(-math.abs(math.log(math.abs(Lambda_yx))))
    # if np.abs(Lambda_xy) > 0:
    #     Lambda_xy = math.sign(Lambda_xy)*math.exp(-math.abs(math.log(math.abs(Lambda_xy))))
    
    #Simpler for code
    if np.abs(Lambda_yx)>1:
        Lambda_yx = 1/Lambda_yx
    if np.abs(Lambda_xy)>1:
        Lambda_xy = 1/Lambda_xy
            
    # Symmetrized robust correlation
    prod = Lambda_yx * Lambda_xy
    if (not np.isfinite(prod)) or (prod <= 0.0):
        #What to do when asymmetrical measures disagree on sign of correlation.
        #A numerical oddity; fall back to 0 with disagreeing directions. 
        #This only happens for small Lambda_asym<0.05 for null distributions, is rare, 
        #and Kendall's tau is on average approximately zero.
        #Setting to zero is a choice and there are a few different options
        Lambda_s = 0.0
    else:
        Lambda_s = np.sign(Lambda_yx) * np.sqrt(prod)
        Lambda_s = float( min(max(Lambda_s, -1.0), 1.0)) #Shouldn't ever be necessary. Floating-point error maybe.
    
    #This is another option when the asymmetric measures signs disagree. 
    # if not np.isfinite(prod) or prod == 0.0:
    #     Lambda_s = 0.0
    # else:
    #     s = Lambda_xy + Lambda_yx
    #     if abs(s) < 1e-7:
    #         Lambda_s = 0.0
    #     else:
    #         Lambda_s = np.sign(s) * np.sqrt(abs(prod))
    #         Lambda_s = float( np.minimum(np.maximum(Lambda_s, -1.0), 1.0))

    return Lambda_s, Lambda_yx, Lambda_xy