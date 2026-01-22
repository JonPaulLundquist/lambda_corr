#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 01:41:46 2025

@author: jplundquist
"""

import numpy as np
import numpy.random as nr
from math import erf, sqrt, exp, pi
from numba import njit, objmode #, prange
import warnings

def median_even_rule_inplace(a):
    """Median with your convention: average of middle two when even length."""
    a = a[np.isfinite(a)]
    m = a.size
    if m == 0:
        return np.nan
    a.sort()
    mid = m // 2
    if m % 2 == 1:
        return a[mid]
    return 0.5 * (a[mid - 1] + a[mid])

def mean_of_point_medians_identity_x(ry):
    """
    x ranks are [1..n] in this order; y ranks are ry (permutation of 1..n).
    For each i: median_{j!=i} (ry[j]-ry[i])/(j-i). Then mean over i.
    """
    n = ry.size
    total = 0.0
    slopes = np.empty(n - 1, dtype=float)
    for i in range(n):
        yi = ry[i]
        t = 0
        for j in range(n):
            if j == i:
                continue
            slopes[t] = (ry[j] - yi) / (j - i)
            t += 1
        total += median_even_rule_inplace(slopes)
    return total / n

def mean_of_point_medians_general(rx, ry):
    """
    General case: rx and ry are rank vectors (permutations).
    We convert to x-sorted order so denominators are (j-i) in that order.
    """
    n = rx.size
    # positions of each x-rank in the given order
    pos = np.empty(n, dtype=int)
    pos[rx - 1] = np.arange(n)
    ry_xorder = ry[pos]  # y ranks in x-sorted order
    return mean_of_point_medians_identity_x(ry_xorder)

def sym_geom(Mxy, Myx):
    if Mxy == 0.0 or Myx == 0.0:
        return 0.0
    mag = np.sqrt(abs(Mxy * Myx))
    return np.sign(Myx) * mag

#numba likes loops
@njit(cache=True, nogil=True, inline='always')
def _nanmean(a, n):
    s = 0.0
    c = 0
    for i in range(n):
        v = a[i]
        if v == v:  # not NaN
            s += v
            c += 1
    return s / c if c > 0 else np.nan

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
    return _nanmean(bi, n)

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
    # Tests compared to Somers' D better agree on asymmetry when standardization is done
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
    # if np.abs(Lambda_yx)>1:
    #     Lambda_yx = 1/Lambda_yx
    # if np.abs(Lambda_xy)>1:
    #     Lambda_xy = 1/Lambda_xy
            
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
        #Lambda_s = float( min(max(Lambda_s, -1.0), 1.0)) #Shouldn't ever be necessary. Floating-point error maybe.
    
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

#Numba p-value permutation test
@njit(cache=True, nogil=True, inline='always') #, parallel=True
def _lambda_pvals(rx, ry, n, Lambda_s, Lambda_yx, Lambda_xy, p_tol=1e-4, 
                  n_perm=10000, alt="two-sided"):
    # ---- Permutation test ----

    c_s = 0
    c_xy = 0
    c_yx = 0
    N = 0
    for i in range(n_perm): #prange(n_perm):  #PARALLEL LOOP not possible with early exit
        perm = np.random.permutation(n)
        l_b, Lambda_yx_b, Lambda_xy_b = _lambda_stats(rx, ry[perm], n)
        if alt == "two-sided":
            hit_s = (abs(l_b) >= abs(Lambda_s))
            hit_yx = (abs(Lambda_yx_b) >= abs(Lambda_yx))
            hit_xy = (abs(Lambda_xy_b) >= abs(Lambda_xy))
        elif (alt == "greater"):
            hit_s = (l_b >= Lambda_s)
            hit_yx = (Lambda_yx_b >= Lambda_yx)
            hit_xy = (Lambda_xy_b >= Lambda_xy)
        else: #alt == less
            hit_s = (l_b <= Lambda_s)
            hit_yx = (Lambda_yx_b <= Lambda_yx)
            hit_xy = (Lambda_xy_b <= Lambda_xy)

        c_s += int(hit_s)
        c_yx += int(hit_yx)
        c_xy += int(hit_xy)
        
        N = N + 1
        #We will consider the p-value accuracy on the symmetric correlation only
        if _check_stop(c_s, p_tol, N):
            break

    p_s = (c_s + 1.0) / (N + 1.0)
    p_yx = (c_yx + 1.0) / (N + 1.0)
    p_xy = (c_xy + 1.0) / (N + 1.0)
    
    return p_s, p_yx, p_xy

@njit(cache=True, nogil=True, inline='always')
def _lambda_p_asymptotic(Lambda_s, n, alt="two-sided"):
    def Phi(t): return 0.5*(1.0 + erf(t / sqrt(2.0)))
    def phi(t): return exp(-0.5*t**2) / sqrt(2.0*pi)
    
    def sigma_model(n, L_inf, a, alpha):
        return L_inf + a * n**(-alpha)

    def kurt_model(n, A, B):
        return -A / n - B / n**2

    #/tests/test_limit.py confirmed there is an asymptotic distribution.
    #Fit using /tests/find_limit.py functions. Validated with test_asymp.py.
    sig0 = sigma_model(n, 1.1118112478, 0.5263109338, 0.699885)
    kurt0 = kurt_model(n, 11.2182780407, -63.0789971809)

    z = (Lambda_s - 0.0) * (n**0.5) / sig0
        
    if alt == "two-sided":
        t = z if z >= 0.0 else -z  # |z|
        # first-order Edgeworth CDF at |z| #\ #second order kurtosis term caused instability
        #"+ phi(z) * (kurt0**2/576.0) * (z**6 - 15.0*z**4 + 45.0*z**2 - 15.0)"
        P_z = Phi(t) - phi(t) * (kurt0/24.0) * (t*t*t - 3.0*t)
        # clamp CDF once
        P_z = max(0.0, min(1.0, P_z))
        return 2.0 * (1.0 - P_z)   # already in [0,1] for symmetric case

    else:
        # one-sided uses signed z
        P_z = Phi(z) - phi(z) * (kurt0/24.0) * (z*z*z - 3.0*z)
        P_z = max(0.0, min(1.0, P_z))
        if alt == "greater":
            return 1.0 - P_z
        else:  # "less"
            return P_z

#@njit(cache=True, nogil=True) #njit not compatible with warnings
def lambda_corr(x, y, pvals=True, ptype="default", p_tol=1e-4, n_perm=10000, alt="two-sided"):
    
    if pvals not in [True, False]:
        raise ValueError(f"pvals must be True or False, got '{pvals}'")
    
    if pvals:
        if ptype not in ["default", "perm", "asymp"]:
            raise ValueError(f"ptype must be 'default', 'perm', or 'asymp', got '{ptype}'")
        
        if alt not in ["two-sided", "greater", "less"]:
            raise ValueError(f"alt must be 'two-sided', 'greater', or 'less', got '{alt}'")

    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size
    
    if n != y.size or n < 3:
        raise ValueError("x and y must be same length, n >= 3")
        
    # Remove pairs where either x or y is not finite
    indx = np.isfinite(x) & np.isfinite(y)
    n = np.sum(indx)
    if n < 3:
        # Not enough valid data
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    x = x[indx]
    y = y[indx]

    if np.std(x) == 0 or np.std(y) == 0:
        #Constant input: the correlation coefficient is undefined.
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Will we actually use permutation p-values?
    use_perm_pvals = (
        pvals and (
            ptype == "perm" or
            (ptype == "default" and n < 25)
        )
    )

    n_perm = int(n_perm) #even if isn't used it should be what numba expects
    if use_perm_pvals:
        # --- p_tol checks ---
        # Basic domain check
        if not (0 < p_tol < 1):
            raise ValueError("p_tol must be in the interval (0, 1).")
        # Practical lower bound
        if p_tol < 1e-15:
            raise ValueError("p_tol is too small to be practical; use p_tol ≥ 1e-15.")
        # Very small but allowed: warn
        if p_tol < 1e-10:
            warnings.warn(
                f"p_tol={p_tol:g} is extremely small; detecting >6σ effects requires a very "
                "large n_perm for stable permutation p-values.",
                UserWarning,
            )
        
        # --- n_perm checks ---
        if n_perm < 1:
            raise ValueError("n_perm must be an integer ≥ 1")
        
        if n_perm < n:
            warnings.warn(
                f"n_perm={n_perm} is smaller than sample size n={n}; "
                "for stable permutation p-values, using n_perm ≥ n is strongly recommended.",
                UserWarning
            )
   
        min_needed = int(1.0 / p_tol)
        
        if n_perm < min_needed:
            warnings.warn(
                f"n_perm={n_perm} possibly too small for p_tol={p_tol}; "
                f"use n_perm>{min_needed} permutations for stable p-values.",
                UserWarning
            )

    # Standardized ranks with averaged ties
    rx = _std_ranks(x, n)
    ry = _std_ranks(y, n)
    # Get Lambda correlations - symmetric and asymmetric
    Lambda_s, Lambda_yx, Lambda_xy  = _lambda_stats(rx, ry, n)
    
    if pvals:
        if (ptype=="default" and (n < 25)) or ptype=="perm":
                p_s, p_yx, p_xy = _lambda_pvals(rx, ry, n, Lambda_s, Lambda_yx, Lambda_xy, 
                                                p_tol=p_tol, n_perm=n_perm, alt=alt)
        elif (ptype=="default" and n >= 25) or ptype=="asymp":
            p_s = _lambda_p_asymptotic(Lambda_s, n, alt=alt) 
            #The null distribution for the asymmetric measures was not calculated seperately
            #but these two Gaussian-ish random variables are not independent; 
            #they are very strongly correlated and nearly identically distributed under the null.
            #Therefore, the geometric average should have approximately the same distribution.
            #MC testing confirms this.
            p_yx = _lambda_p_asymptotic(Lambda_yx, n, alt=alt)
            p_xy = _lambda_p_asymptotic(Lambda_xy, n, alt=alt)
        else:
            p_s = p_xy = p_yx = np.nan
    else:
        p_s = p_xy = p_yx = np.nan
    
    # Asymmetry index with safe denominator
    denom = abs(Lambda_yx) + abs(Lambda_xy)
    Lambda_a = 0.0 if denom == 0.0 else float(abs(Lambda_yx - Lambda_xy) / denom)
    
    return Lambda_s, p_s, Lambda_yx, p_yx, Lambda_xy, p_xy, Lambda_a

def G_for_perm(ry):
    rx = np.arange(1, ry.size + 1)
    #Mxy = mean_of_point_medians_identity_x(ry)
    # swap x<->y: treat ry as the x-ranks (in given order) and identity as y
    #Myx = mean_of_point_medians_general(ry, rx)
    #G = sym_geom(Mxy, Myx, sign_from=sign_from)
    G, _, M_yx, _, M_xy, _, _ = lambda_corr(rx, ry, pvals=False)
    return G, M_xy, M_yx

def improve_by_random_swaps_anneal(ry0, steps, T0=1e-3, Tend=1e-6, keep_best=True):
    """
    Stochastic local search with annealing.
    - ry0: permutation values 1..n
    - steps: proposals
    Returns: (best_ry, bestG, bestMxy, bestMyx)
    """
    ry = ry0.copy()
    G, Mxy, Myx = G_for_perm(ry)
    cur = abs(G)

    best_ry = ry.copy()
    bestG, bestMxy, bestMyx = G, Mxy, Myx
    best = cur

    n = ry.size
    for t in range(steps):
        # geometric cooling
        frac = t / max(steps - 1, 1)
        T = T0 * (Tend / T0) ** frac

        i = int(nr.randint(0, n))
        j = int(nr.randint(0, n-1))
        if j >= i:
            j += 1  # ensure j != i

        ry[i], ry[j] = ry[j], ry[i]
        G2, Mxy2, Myx2 = G_for_perm(ry)
        new = abs(G2)

        d = new - cur
        if d >= 0 or (T > 0 and nr.random() < np.exp(d / T)):
            # accept
            cur = new
            G, Mxy, Myx = G2, Mxy2, Myx2
            if keep_best and cur > best:
                best = cur
                best_ry = ry.copy()
                bestG, bestMxy, bestMyx = G, Mxy, Myx
        else:
            # reject: revert
            ry[i], ry[j] = ry[j], ry[i]

    return best_ry, bestG, bestMxy, bestMyx

def refine_best(best_ry, tries=50, steps=20000):
    #n = best_ry.size
    bestAbs = -1.0
    best = best_ry.copy()
    bestStats = None

    for _ in range(tries):
        ry, G, Mxy, Myx = improve_by_random_swaps_anneal(
            best, steps=steps, T0=1e-4, Tend=1e-7
        )
        if abs(G) > bestAbs:
            bestAbs = abs(G)
            best = ry
            bestStats = (G, Mxy, Myx)

    return bestAbs, best, bestStats

def estimate_Cn(n, ry0=False, restarts=100, steps_per_restart=50000):
    bestAbs = -1.0
    best = None
    bestStats = None
    
    if ry0 is not False:
        for i in range(restarts):
            print(i)
            ry2, G, Mxy, Myx = improve_by_random_swaps_anneal(
                ry0, steps=steps_per_restart, T0=5e-3, Tend=5e-6
            )
    
            # Intensify: start from the best found in this restart, cool more aggressively
            ry2, G2, Mxy2, Myx2 = improve_by_random_swaps_anneal(
                ry2, steps=max(2000, steps_per_restart//5), T0=5e-5, Tend=1e-7
            )
    
            if abs(G2) > bestAbs:
                bestAbs = abs(G2)
                best = ry2
                bestStats = (G2, Mxy2, Myx2)
                ry0 = ry2
                print(bestStats)
                print(best)
    else:
        
        for i in range(restarts):
            print(i)
            ry0 = nr.permutation(n) + 1
    
            ry, G, Mxy, Myx = improve_by_random_swaps_anneal(
                ry0, steps=steps_per_restart,  T0=5e-3, Tend=5e-6
            )
    
            # Intensify: start from the best found in this restart, cool more aggressively
            ry2, G2, Mxy2, Myx2 = improve_by_random_swaps_anneal(
                ry, steps=max(2000, steps_per_restart//5), T0=5e-5, Tend=1e-7
            )
    
            if abs(G2) > bestAbs:
                bestAbs = abs(G2)
                best = ry2
                bestStats = (G2, Mxy2, Myx2)
                print(bestStats)
                print(best)
    bestAbs, best, bestStats = refine_best(best)
    return bestAbs, best, bestStats

# Example:
# C25, ry_best, (G, Mxy, Myx) = estimate_Cn(25, restarts=80, steps_per_restart=12000, seed=1)
# print("C(25)~", C25, "G=", G, "Mxy=", Mxy, "Myx=", Myx)
# print("ry:", ry_best.tolist())