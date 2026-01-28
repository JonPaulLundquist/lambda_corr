#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Wed Oct  8 19:56:06 2025

    Repeated-Average Rank Correlation Λ (Lambda) 
    
    Introduction
    ------------
    The Repeated-Average Rank correlation Λ (Lambda) introduced here is a new family 
    of robust, symmetric, and asymmetric measures of monotone association based on 
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
    a measure that is more resistant to noise while retaining interpretability as a 
    standard measure of monotonic trend/association.
    
    Let (x_i, y_i), i = 1..n. Let rx = ranks(x) and ry = ranks(y) (average ranks for
    ties), then standardize to zero mean and unit variance:

        rxt = (rx - mean(rx)) / std(rx)
        ryt = (ry - mean(ry)) / std(ry)

    For each anchor sample i, compute the median slope in rank space:

        b_i = median_{j != i, rxt[j] != rxt[i]} ( (ryt[j] - ryt[i]) / (rxt[j] - rxt[i]) )

    The asymmetric Lambda is the outer mean over i slopes (y given x):

        Λ_yx = (1/n) * sum_i b_i
    
    Compute Λ_xy by swapping x and y. 
    
    A fold-back transform is applied to the asymmetric components to enforce the
    conventional range [-1, 1], and to restore the correct ordering relative to τ/ρ, 
    for extremely rare, highly structured near-(anti)monotone rank configurations
    (see Fold-Back Transform section below):
        
        Λ_asym ← sign(Λ_asym) · exp(−|log|Λ_asym||) 
    
    Equivalently: Λ_asym is unchanged for |Λ_asym|≤1 and mapped to 1/Λ_asym for |Λ_asym|>1.
    
    The symmetric correlation Λ_s is the sign-preserving geometric mean:

        Λ_s = sgn(Λ_yx) * sqrt(|Λ_yx * Λ_xy|)
    
    This construction mirrors classical correlation measures: Kendall’s τ_b can
    be written as the signed geometric mean of the asymmetric Somers’ D
    statistics D_{Y|X} and D_{X|Y}, and Pearson’s r can be written as the
    signed geometric mean of the two ordinary least-squares regression slopes
    m_{Y|X} = cov(X, Y) / var(X) and m_{X|Y} = cov(X, Y) / var(Y).
    Λ_s extends this same geometric-mean symmetrization to robust repeated average 
    rank-slope correlations.

    Functions
    ---------    
    - lambda_corr(x, y, ...): main user-facing wrapper with validation, finite filtering, 
      and warnings.
    - lambda_corr_nb(x, y, n, ...): Numba-compatible core for use inside @njit code.
      Assumes x and y are already prevalidated (same length n≥3, finite, non-constant).
  
    Parameters
    ----------
    x, y : 1-D array_like 
        Two input samples of equal length (n ≥ 3).
    n : integer (only for lambda_corr_nb)
        Size of x and y.
    pvals : {True, False}, optional
        Whether to compute p-values. Default: True.
        If False, all returned p-values are NaN and no permutation/asymptotic
        p-value calculations are performed.

    ptype : {"default", "asymp", "perm"}, optional
        Method for p-value calculation. Default: "default".
        
        - "default": If n < 25, use a Monte Carlo permutation test. 
                     If n ≥ 25, use asymptotic approximation.
                     
        - "asymp": Use asymptotic approximation  (Edgeworth expansion). Assumes 
                   no ties; accuracy decreases as tie frequency increases.
                   Note: The same null distribution is used for Λ_s, Λ_xy, 
                   and Λ_yx, as they have matching null moments (mean, 
                   variance, skewness, kurtosis) under independence.
           
        - "perm": Use Monte Carlo permutation test. Valid for any tie structure. 
                  Note: This is approximate unless all permutations are
                  enumerated, which is only feasible for very small n.
                  The RNG is re-seeded for every call so permutation p-values vary across 
                  runs by default.
                  
        Note:
            The permutation test samples from the *conditional* null distribution, generated 
            by permuting the observed y-values while keeping x fixed. This distribution 
            depends directly on the observed marginal distributions and tie structure. 
            Therefore, when the *underlying population is genuinely discrete*, the permutation 
            method can be more accurate because it automatically reflects the correct amount 
            and pattern of ties.
            
            In contrast, the asymptotic p-values approximate the *unconditional* null distribution 
            of Λ, calibrated from extremely large Monte Carlo simulations. As a result, they 
            tend to be more stable and often more accurate for moderate–large n, especially 
            when the *underlying population is continuous* (even if the sample exhibits ties 
            due to rounding, censoring, or finite precision) or when the data are skewed.

    p_tol : float, optional
        Stopping tolerance on p-value uncertainty in the permutation test. 
        Default: 1e-4.
        Sampling stops early if p-value uncertainty falls below p_tol
        (or once n_perm permutations are reached).
    
    n_perm : integer, optional
        Maximum number of MC permutations for p-value estimation. Default: 10000.
        The procedure will terminate earlier if p-value uncertainty falls 
        below p_tol. 
    
    alt : {"two-sided", "greater", "less"}, optional
        Alternative hypothesis relative to the null of zero monotonic association. 
        Default: "two-sided".
        
        - "two-sided": Probability of observing |Λ| as large or larger than |Λ_obs| 
                       ([-1, 1]) under the null (population Λ of zero).
                       
        - "greater": Probability of observing Λ ≥ Λ_obs (upper tail, [Λ_obs, 1])
                     under the null (population Λ of zero).
                     
        - "less": Probability of observing Λ ≤ Λ_obs ([-1, Λ_obs]) under the null 
                  (population Λ of zero).

    Returns
    -------
    Lambda_s (Λ_s) : symmetric repeated-average-rank correlation in [-1, 1]
    p_s      : p-value for Λ_s
    Lambda_yx (Λ(Y|X)): directional slope of y given x in rank space
    p_yx     : p-value of Λ_yx
    Lambda_xy (Λ(X|Y)): directional slope of x given y in rank space
    p_xy     : p-value of Λ_xy
    Lambda_a : normalized asymmetry index = |Λ_yx - Λ_xy| / (|Λ_yx| + |Λ_xy|) in [0,1]

    Properties
    ----------
    - Robust to outliers and heavy-tailed noise; highest sign breakdown with 
      adversarial noise among rank methods.
    - Less biased than Spearman/Kendall relative to Pearson.
    - Similar or better accuracy than Spearman/Kendall for stronger associations.
    - Asymptotic efficiency for bivariate normal: ~81% vs. ~91% for Spearman and Kendall.
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

    Fold-Back Transform
    -------------------
    The mean-of-medians construction can very rarely produce |Λ_yx| or |Λ_xy| slightly larger 
    than 1. These cases arise for extremely rare, highly structured near-(anti)monotone rank 
    configurations in which the set of pairwise rank slopes for one or more anchor points 
    becomes strongly discrete and imbalanced (often exhibiting a localized oscillatory 
    defect / weave-like structure). Such configurations are difficult to encounter by random 
    permutations, but can be found more efficiently by stochastic swap/annealing searches that 
    explicitly maximize |Λ|. Empirically, observed overshoots are small (|Λ_asym| ≲ 1.08 in 
    search-constructed examples; values depend on n and on the search procedure).
    
    Within this overshoot regime, larger |Λ| corresponds to *weaker* monotone association
    when compared to Kendall’s τ and Spearman’s ρ (i.e., among overshoot cases, |Λ_raw|
    tends to anti-correlate with |τ| and |ρ|). To enforce the conventional correlation
    range [-1,1] and restore the desired ordering in this regime, a reciprocal fold-back
    mapping is applied to the asymmetric components (prior to geometric-mean symmetrization):
    
        f(Λ_asym) = sign(Λ_asym) · exp(−|log|Λ_asym||),  with f(0)=0,
    
    which is the identity on [−1,1], preserves sign, and maps |Λ_asym|>1 back into (0,1]
    via reciprocal inversion.
    
    This transform is equivalent to:
        Λ_asym ← Λ_asym                  if |Λ_asym| ≤ 1
        Λ_asym ← 1 / Λ_asym              if |Λ_asym| > 1

    In the Monte Carlo calibration runs used for the asymptotic null and the bivariate-Gaussian 
    benchmarks, fold-back was never activated (zero occurrences in billions of draws). Therefore, 
    it had no effect on the calibrated null distribution or benchmark results.
    
    Alternative stabilizations (e.g., Harrell–Davis quantile estimator per anchor, or
    Monte Carlo/permutation-based bias correction) can only reduce overshoot frequency and
    magnitude, but they materially change Λ and its null behavior; fold-back
    is used as a simple, deterministic guardrail.

    Implementation Notes
    --------------------
    - If asymmetric Λ_yx/Λ_xy have opposite signs Λ_s is taken as zero.
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
    
    Example
    --------
    Compute the symmetric Lambda correlation Λ_s and its directional components
    for a simple monotonic relationship:

    import numpy as np
    import math
    from lambda_corr import lambda_corr

    rng = np.random.default_rng(seed=0)

    n = 50
    rho = 0.5   # correlation strength
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    c = math.sqrt((1 - rho) * (1 + rho))
    y = np.exp(rho * x + c * z)   # any monotonic transformation

    # Compute Lambda correlations
    Lambda_s, p_s, Lambda_yx, p_yx, Lambda_xy, p_xy, Lambda_a = lambda_corr(x, y)

    # Nicely formatted output
    print(f"Λ_s       = {Lambda_s: .4f}   (p = {p_s: .4g})")
    print(f"Λ(y|x)    = {Lambda_yx: .4f}   (p = {p_yx: .4g})")
    print(f"Λ(x|y)    = {Lambda_xy: .4f}   (p = {p_xy: .4g})")
    print(f"Asymmetry = {Lambda_a: .4f}")
    
    # Example output:
    # Λ_s       =  0.4130   (p =  0.0087)     #Result will be close to rho
    # Λ(y|x)    =  0.4145   (p =  0.008419)
    # Λ(x|y)    =  0.4114   (p =  0.008988)
    # Asymmetry =  0.0038
    
@author: Jon Paul Lundquist
"""

import numpy as np
from math import erf, sqrt, exp, pi
from numba import njit, objmode #, prange
import warnings
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lambda-corr")
except PackageNotFoundError:
    __version__ = "0.0.0"

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

#Numbda compatible entry
@njit(cache=True, nogil=True, fastmath=True)
def lambda_corr_nb(x, y, n, pvals=True, ptype="default", p_tol=1e-4, n_perm=10000, alt="two-sided"):

    # assume: x,y already arrays of same length, n>=3, finite, and non-constant
    # Standardized ranks with averaged ties
    rx = _std_ranks(x, n)
    ry = _std_ranks(y, n)
    # Get Lambda correlations - symmetric and asymmetric
    Lambda_s, Lambda_yx, Lambda_xy  = _lambda_stats(rx, ry, n)
    
    if pvals:
        if (ptype=="perm") or ((ptype=="default") and (n < 25)):
                p_s, p_yx, p_xy = _lambda_pvals(rx, ry, n, Lambda_s, Lambda_yx, Lambda_xy, 
                                                p_tol=p_tol, n_perm=n_perm, alt=alt)
        elif (ptype=="asymp") or ((ptype=="default") and (n >= 25)):
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

    Lambda_s, p_s, Lambda_yx, p_yx, Lambda_xy, p_xy, Lambda_a = \
        lambda_corr_nb(x, y, n, pvals=pvals, ptype=ptype, p_tol=p_tol, n_perm=n_perm, 
                       alt=alt)
    
    return Lambda_s, p_s, Lambda_yx, p_yx, Lambda_xy, p_xy, Lambda_a
