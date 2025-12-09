#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Wed Oct  8 19:56:06 2025

    Repeated-Average-Rank Correlation Λ (Lambda) 
    
    Introduction
    ----------
    The Repeated-Average-Rank correlation Λ (Lambda) introduced here is a new family 
    of robust, symmetric and asymmetric measures of monotone association based on 
    pairwise slopes in rank space. Compared with traditional rank-based measures 
    (Spearman’s ρ and Kendall’s τ [1,2]), Lambda is more resistant to noise contamination 
    and exhibits much less bias relative to Pearson’s linear correlation [3]. For moderate 
    to strong signals, its estimation accuracy is comparable to, and can exceed, 
    that of ρ and τ. These gains come at the cost of a modest reduction in asymptotic 
    efficiency (≈81% vs. ≈91% for ρ and τ).
    
    Canonical definition of Λ_s
    ---------------------------
    Λ_s, is the symmetrized mean of the median of pairwise rank slopes (inspired by 
    Siegel's repeated median slope [4]) that combines the high noise breakdown robustness 
    of the repeated median with the unbiased symmetry of an outer mean, producing 
    a measure that is mroe resistant noise while retaining interpretability as a 
    standard measure of monotonic trend/association.
    
    Let (x_i, y_i), i = 1..n. Let rx = ranks(x) and ry = ranks(y) (average ranks for
    ties), then standardize to zero mean and unit variance:

        rxt = (rx - mean(rx)) / std(rx)
        ryt = (ry - mean(ry)) / std(ry)

    For each anchor i, compute the median slope in rank space:

        b_i = median_{j != i, rxt[j] != rxt[i]} ( (ryt[j] - ryt[i]) / (rxt[j] - rxt[i]) )

    The asymmetric Lambda is the outer mean:

        Lambda_xy = (1/n) * sum_i b_i

    Compute Λ_yx by swapping x and y. The symmetric correlation Λ_s is the
    sign-preserving geometric mean (just as Kendall's τ_a is the geometric mean 
                                    of the asymmetric Somers' D):

        Λ_s = sgn(Lambda_xy) * sqrt(|Lambda_xy * Lambda_yx|)
    
    Parameters
    ----------
    x, y : 1-D array_like 
        Two input samples of equal length (n ≥ 3).
    
    pvals : {True, False}, optional
        Flag for p-value calculation. Default: True.
        If False, all returned p-values are NaN and no permutation/asymptotic
        p-value calculations are performed.

    ptype : {"default", "asymp", "perm"}, optional
        Type of p-value calculation. Default: "default".
        - "default": If n < 25 do permutation calculation. 
                     If n ≥ 25 use asymptotic approximation.
        - "asymp": Use asymptotic approximation. Assumes no ties. 
                   The more ties the less accurate.
        - "perm": Calculates p-value using permutations. 
                  Valid with any fraction of ties.
    
    p_tol : float, optional
        If uncertainty on p-value is less than p_tol stop permutation calculation. 
        Default: 1e-5.
    
    n_perm : integer, optional
        Maximum number of Monte Carlo permutations for p-value estimation. Default: 1e4.
        Estimation will terminate earlier if the p-value uncertainty falls below p_tol. 
    
    alt : {"two-sided", "greater", "less"}, optional
        Alternative hypothesis relative to the null of no correlation. 
        Default: "two-sided".
        - "two-sided": Probability of getting larger magnitude Λ ([-1, 1]) with population 
        correlation of zero.
        - "greater": Probability of getting a greater Λ ([Λ, 1]) with population 
        correlation of zero.
        - "less": Probability of getting a smaller Λ ([-1, Λ]) with population correlation
        of zero.

    Returns
    -------
    Lambda_s (Λ_s) : symmetric repeated-average-rank correlation in [-1, 1]
    p_s      : p-value for Λ_s
    Lambda_xy (Λ(X|Y)): directional slope of x given y in rank space
    p_xy     : p-value of Λ_xy
    Lambda_yx (Λ(Y|X)): directional slope of y given x in rank space
    p_yx     : p-value of Λ_yx
    Lambda_a : normalized asymmetry = |Λ_xy - Λ_yx| / (|Λ_xy| + |Λ_yx|) in [0,1]

    Properties
    ----------
    - Robust to outliers and heavy-tailed noise; highest sign breakdown with 
      adversarial noise among rank methods.
    - Less biased than Spearman/Kendall relative to Pearson.
    - Similar or better accuracy than Spearman/Kendall for stronger associations.
    - Asymptotic efficiency: ~81% vs. ~91% for Spearman and Kendall.
    - Null distribution: centered, symmetric, slightly heavier tails than Spearman.
    - Symmetric: Λ_s(x,y) == Λ_s(y,x).
    - Invariant to strictly monotone transforms.

    Variants
    --------
    A continuum exists between mean-of-means (outside loop - inside loop) and 
    median-of-medians estimators:

        Spearman (ρ) ≈ Λ(mean-mean) <--> Λ(mean-median) <--> Λ(median-mean)  
                                                 <--> Λ(median-median) ≈ Siegel

    The canonical choice Λ_s(mean-median) achieves the best efficiency/robustness
    balance, especially at low statistics.
    
    Implementation Notes
    --------------------
    - If asymmetric Λ_xy/Λ_yx have opposite signs Λ_s is taken as zero.
    - Skip vertical pairs where rxt[j] == rxt[i].
    - If all slopes for an i are undefined (e.g., all rxt equal), set b_i = NaN and
      ignore in the outer mean.

    References
    ----------
    - [1] Spearman, C. The proof and measurement of association between two things. 
          American Journal of Psychology, 15(1), 72–101, 1904.
    - [2] Kendall, M.G., Rank Correlation Methods (4th Edition), Charles 
          Griffin & Co., 1970.
    - [3] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    - [4] Siegel, A.F., Robust Regression Using Repeated Medians, Biometrika, 
          Vol. 69, pp. 242-244, 1982.
    
@author: Jon Paul Lundquist
"""

import numpy as np
from math import erf, sqrt, exp, pi
from numba import njit, objmode, prange

##Mean-Mean (outer-inner) is just the same as mean of all pairwise slopes (ave_slope with mean)
##aka Theil-sen with mean instead of median. Median-median is Seigel in rank space.
#NOT OPTIMIZED CODE COMMENTED OUT FOR REFRENCE
# @njit(cache=True, nogil=True, fastmath=True, inline='always')
# def _mean_medians_slope(rx, ry, n):
#     """Mean over i of the median slopes (y_j - y_i) / (x_j - x_i), skipping verticals."""

#     bi = np.empty(n)
#     for i in range(n):
#         dx = rx - rx[i]
#         m = dx != 0.0
#         if not np.any(m):
#             bi[i] = np.nan
#             continue
#         bi[i] = np.nanmean((ry[m] - ry[i]) / dx[m])
#     return float(np.nanmedian(bi)) 

# @njit(cache=True, nogil=True, fastmath=True, inline='always')
# def _ave_slope(rx, ry, n, use_median=True):
#     """
#     median/mean of all pairwise slopes in rank space:
#       stat = mean_or_median_{i<j, rx[j]!=rx[i]} ( (ry[j]-ry[i]) / (rx[j]-rx[i]) )

#     Allocates C(n,2) then uses only the filled prefix.
#     Returns NaN if no valid slopes exist (all dx == 0).
#     """
 
#     mcap = n * (n - 1) // 2                # upper bound = total pairs
#     buf = np.empty(mcap, dtype=np.float64)  # over-allocate once
#     k = 0

#     for i in range(n - 1):
#         xi = rx[i]; yi = ry[i]
#         for j in range(i + 1, n):
#             dx = rx[j] - xi
#             if dx != 0.0:
#                 s = (ry[j] - yi) / dx
#                 # ranks should be finite, but keep a guard:
#                 if np.isfinite(s):
#                     buf[k] = s
#                     k += 1

#     if k == 0:
#         return np.nan

#     arr = buf[:k]  # use only the filled portion
#     return float(np.median(arr) if use_median else np.mean(arr))

# --- utilities: nanmean on first k items, in-place quickselect kth ---
@njit(cache=True, nogil=True, inline='always')
def _nanmean_k(a, k):
    s = 0.0
    c = 0
    for i in range(k):
        v = a[i]
        if v == v:  # not NaN
            s += v
            c += 1
    return s / c if c > 0 else np.nan

@njit(cache=True, nogil=True, inline='always')
def _partition(a, left, right, pivot_index):
    pivot_value = a[pivot_index]
    # move pivot to end
    a[pivot_index], a[right] = a[right], a[pivot_index]
    store = left
    for i in range(left, right):
        if a[i] < pivot_value or (a[i] != a[i] and pivot_value == pivot_value):  # NaNs to the end
            a[store], a[i] = a[i], a[store]
            store += 1
    # move pivot to its final place
    a[right], a[store] = a[store], a[right]
    return store

@njit(cache=True, nogil=True, inline='always')
def _select_kth_inplace(a, k, length):
    """Hoare-style quickselect on a[:length], places k-th smallest at index k."""
    left = 0
    right = length - 1
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
def _median_inplace(a, length):
    """Median of a[:length] ignoring NaNs (NaNs are pushed to the end by comparator)."""
    # Compact valid values (non-NaN) to front
    m = 0
    for i in range(length):
        v = a[i]
        if v == v:
            a[m] = v
            m += 1
    if m == 0:
        return np.nan
    mid = m // 2
    if m % 2 == 1:
        return _select_kth_inplace(a, mid, m)
    else:
        lo = _select_kth_inplace(a, mid - 1, m)
        hi = _select_kth_inplace(a, mid, m)
        return 0.5 * (lo + hi)

# --- fast mean-of-medians of slopes ---
@njit(cache=True, nogil=True, inline='always')
def _mean_medians_slope(rx, ry, n):
    bi = np.empty(n, dtype=np.float64)
    slopes_buf = np.empty(n - 1, dtype=np.float64)

    for i in range(n):
        xi = rx[i]
        yi = ry[i]
        k = 0
        # collect slopes to the right of i
        for j in range(i + 1, n):
            dx = rx[j] - xi
            if dx != 0.0:
                slopes_buf[k] = (ry[j] - yi) / dx
                k += 1
        # collect slopes to the left of i
        for j in range(0, i):
            dx = rx[j] - xi
            if dx != 0.0:
                slopes_buf[k] = (ry[j] - yi) / dx
                k += 1

        if k == 0:
            bi[i] = np.nan
        else:
            bi[i] = _median_inplace(slopes_buf, k)

    # nanmean of bi
    return _nanmean_k(bi, n)

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
    #Sxx = 0.0

    i = 0
    while i < n:
        j = i + 1
        xi = x[sorter[i]]
        while j < n and x[sorter[j]] == xi:
            j += 1
        r = 0.5 * (i + 1 + j)              # average rank for the tie block
        # write block and accumulate Sxx
        for k in range(i, j):
            rx[sorter[k]] = r
        #m = j - i
        #Sxx += m * r * r                    # faster than summing r*r in the loop
        i = j
    return rx #, Sxx

#standardize the ranks
@njit(cache=True, nogil=True, inline='always')
def _std_ranks(a, n):
    r = _rankdata_avg_ties(a, n) #scipy.stats.rankdata(a, method='average')

    r = (r - np.mean(r)) / np.std(r) # Numba-compatible, ddof=0
    return r

@njit(cache=True, nogil=True, inline='always')
def _lambda_stats(rx, ry, n):
    
    # mean of median slopes of standardized ranks in both directions
    Lambda_xy = _mean_medians_slope(rx, ry, n) # Λ_xy: inner median-of-slopes, outer mean
    Lambda_yx = _mean_medians_slope(ry, rx, n) # Λ_yx: inner median-of-slopes, outer mean
    
    # Symmetrized robust correlation
    prod = Lambda_yx * Lambda_xy
    if (not np.isfinite(prod)) or (prod <= 0.0):
        #What to do when asymmetrical measures disagree on sign of correlation.
        #A numerical oddity; fall back to 0 with disagreeing directions. 
        #This only happens for small Lambda_asym<0.05 for null distributions and is rare.
        #Setting to zero is a choice and there are a few different options
        Lambda_s = 0.0
    else:
        Lambda_s = np.sign(Lambda_yx) * np.sqrt(prod)
        Lambda_s = float( np.minimum(np.maximum(Lambda_s, -1.0), 1.0))
    
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

    return Lambda_s, Lambda_xy, Lambda_yx

#Check if p-value uncertainty is smaller than threshold
@njit(cache=True, nogil=True, inline='always')
def _check_stop(h, p_tol, N):
    if (N<50) or (h<5) or ((N-h)<5):
        check = False
    else:
        p_hat = (h + 1.0) / (N + 1.0)
        se = np.sqrt(p_hat * (1.0 - p_hat) / (N + 1.0))
        rel = se / max(p_hat, np.float64(2.220446049250313e-16))
        check = rel <= p_tol
    return check

#Numba parallelized p-value permutation test
@njit(cache=True, nogil=True, inline='always', parallel=True)
def _lambda_pvals(rx, ry, n, Lambda_s, Lambda_xy, Lambda_yx, p_tol = 1e-5, 
                  n_perm=1e4, alt="two-sided"):
    # ---- Permutation test ----

    c_s = 0
    c_xy = 0
    c_yx = 0
    N = 0
    for i in prange(n_perm):  #PARALLEL LOOP
        perm = np.random.permutation(n)
        l_b, Lambda_xy_b, Lambda_yx_b = _lambda_stats(rx, ry[perm], n)
        if alt == "two-sided":
            hit_s = (abs(l_b) >= abs(Lambda_s))
            hit_xy = (abs(Lambda_xy_b) >= abs(Lambda_xy))
            hit_yx = (abs(Lambda_yx_b) >= abs(Lambda_yx))
        elif alt == "greater":
            hit_s = (l_b >= Lambda_s)
            hit_xy = (Lambda_xy_b >= Lambda_xy)
            hit_yx = (Lambda_yx_b >= Lambda_yx)
        else:
            hit_s = (l_b <= Lambda_s)
            hit_xy = (Lambda_xy_b <= Lambda_xy)
            hit_yx = (Lambda_yx_b <= Lambda_yx)

        c_s += int(hit_s)
        c_xy += int(hit_xy)
        c_yx += int(hit_yx)
        
        N = N + 1
        #We will consider the p-value accuracy on the symmetric correlation only
        if _check_stop(c_s, p_tol, N):
            break

    p_s = (c_s + 1.0) / (N + 1.0)
    p_xy = (c_xy + 1.0) / (N + 1.0)
    p_yx = (c_yx + 1.0) / (N + 1.0)
    
    return p_s, p_xy, p_yx

@njit(cache=True, nogil=True, inline='always')
def _lambda_p_asymptotic(Lambda_s, n, alt="two-sided"):
    def Phi(t): return 0.5*(1.0 + erf(t / sqrt(2.0)))
    def phi(t): return exp(-0.5*t**2) / sqrt(2.0*pi)
    
    def sigma_model(n, L_inf, a, alpha):
        return L_inf + a * n**(-alpha)

    def kurt_model(n, A, B):
        return -A / n - B / n**2

    #test_limit.py confirmed there probably was an asymptotic distribution.
    #Fit using /tests/find_limit.py functions. Validated with test_asymp.py.
    sig0 = sigma_model(n, 1.1118112478, 0.5263109338, 0.699885)
    kurt0 = kurt_model(n, 11.2182780407, -63.0789971809)

    z = (Lambda_s - 0.0) * (n**0.5) / sig0

    # Edgeworth expansion for CDF (kurtosis correction) with first two kurtosis terms.

    P_z = Phi(z) - phi(z) * (kurt0/24.0) * (z**3 - 3.0*z) \
          + phi(z) * (kurt0**2/576.0) * (z**6 - 15.0*z**4 + 45.0*z**2 - 15.0)

    P_z = max(0.0, min(1.0, P_z))
    
    if alt == "greater": return 1.0 - P_z
    if alt == "less":    return P_z
    if z >= 0:
        return 2.0 * (1.0 - P_z)
    else:
        return 2.0 * P_z

@njit(cache=True, nogil=True)
def lambda_corr(x, y, pvals=True, ptype="default", p_tol=1e-5, n_perm=1e4, alt="two-sided"):
    
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size
    
    if n != y.size or n < 3:
        raise ValueError("x and y must be same length, n >= 3")
    
    # Remove pairs where either x or y is not finite
    indx = np.isfinite(x) & np.isfinite(y)
    m = np.sum(indx)
    if m < 3:
        # Not enough valid data
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    x = x[indx]
    y = y[indx]
    n = m
    if np.std(x) == 0 or np.std(y) == 0:
        #Constant input: the correlation coefficient is undefined.
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Standardized ranks with averaged ties
    rx = _std_ranks(x, n)
    ry = _std_ranks(y, n)
    # Get Lambda correlations - symmetric and asymmetric
    Lambda_s, Lambda_xy, Lambda_yx = _lambda_stats(rx, ry, n)
    
    if pvals:
        if (ptype=="default" and (n < 25)) or ptype=="perm":
                p_s, p_xy, p_yx = _lambda_pvals(rx, ry, n, Lambda_s, Lambda_xy, Lambda_yx, 
                                                p_tol=p_tol, n_perm=n_perm, alt=alt)
        elif (ptype=="default" and n >= 25) or ptype=="asymp":
            p_s = _lambda_p_asymptotic(Lambda_s, n, alt=alt) 
            #The null distribution for the asymmetric measures was not calculated seperately
            #but these two Gaussian-ish random variables are not independent; 
            #they are very strongly correlated and nearly identically distributed under the null.
            #Therefore, the geometric average should have approximately the same distribution.
            #MC testing confirms this.
            p_xy = _lambda_p_asymptotic(Lambda_xy, n, alt=alt)
            p_yx = _lambda_p_asymptotic(Lambda_yx, n, alt=alt)
        else:
            p_s = p_xy = p_yx = np.nan
    else:
        p_s = p_xy = p_yx = np.nan
    
    # Asymmetry index with safe denominator
    denom = abs(Lambda_yx) + abs(Lambda_xy)
    Lambda_a = 0.0 if denom == 0.0 else float(abs(Lambda_yx - Lambda_xy) / denom)
    
    return Lambda_s, p_s, Lambda_xy, p_xy, Lambda_yx, p_yx, Lambda_a