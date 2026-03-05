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
            If False, all returned p-values are NaN and no permutation/exact/approximate
            p-value calculations are performed.

    ptype : {"default", "approx", "perm"}, optional
            Method for p-value calculation. Default: "default".
        
            If ties=False and n ≤ 10, an exact lookup table is used for Λ_s regardless of ptype.
            p-values for Λ_xy and Λ_yx are returned only if permutation test is 
            done, otherwise they are NaN.
            
            - "default":
                        If n > 10 and ties=False, use the "approx" method 
                        (Beta-mixture null model). 
                        If ties=True, falls back to "perm."

            - "approx": Approximate p-value from an n-dependent Beta-mixture null 
                        model for |Λ_s| with point masses at 0 and ±1 and a Beta 
                        fit on (0,1). Model parameters (p0(n), p1(n), α(n), β(n)) 
                        are calibrated from extremely large Monte Carlo null simulations 
                        and then parametrically interpolated/extrapolated in n for 
                        intermediate/large sample sizes.
                        
                        Assumes no ties; accuracy degrades as tie frequency increases.
                        
                        Approximate p-values are provided only for the symmetric 
                        statistic Λ_s. Directional components Λ_xy and Λ_yx require 
                        permutation for valid p-values.

            - "perm":   Monte Carlo permutation test by permuting the observed y values 
                        while keeping x fixed (conditional null). Valid for any tie 
                        structure. Note: This is stochastic by intent unless all 
                        permutations are enumerated (feasible only for very small n).
                        Re-running the function can be useful for gauging the sampling 
                        uncertainty, especially when n_perm is modest or when early 
                        stopping (p_tol) triggers.
                        
                        Special case:
                            If ties=False and n <= 10, an exact lookup table is used 
                            only for Λ_s regardless of ptype.
            Note:
                 The permutation test samples the *conditional* null distribution, 
                 which depends on the observed marginal distributions and tie structure. 
                 When the underlying population is discrete, permutation can be more 
                 accurate because it reflects the correct amount/pattern of ties.

                 In contrast, the "approx" method targets an *unconditional* null 
                 distribution for Λ_s, calibrated from Monte Carlo simulations under 
                 continuous no-tie assumptions.
        
    p_tol :  float, optional
             Stopping tolerance on p-value uncertainty in the permutation test. 
             Default: 1e-4.
             Sampling stops early if p-value uncertainty falls below p_tol
             (or once n_perm permutations are reached).
    
    n_perm : integer, optional
             Maximum number of MC permutations for p-value estimation. Default: 10000.
             This will terminate earlier if p-value uncertainty falls below p_tol. 
    
    alt  :   {"two-sided", "greater", "less"}, optional
             Alternative hypothesis relative to the null of zero monotonic association. 
             Default: "two-sided".
        
             - "two-sided": Probability of observing |Λ| as large or larger than |Λ_obs| 
                            ([-1, 1]) under the null (population Λ=zero).
                       
             - "greater": Probability of observing Λ ≥ Λ_obs (upper tail, [Λ_obs, 1])
                          under the null (population Λ=zero).
                     
             - "less": Probability of observing Λ ≤ Λ_obs ([-1, Λ_obs]) under the null 
                       (population Λ=zero).
                  
    ties  :  {False, True}, optional. 
             Whether x or y may contain ties. Default: False.
             If ties = True permutation is used to calculate p-value; unless ptype='approx,' 
             then the accuracy decreases as tie frequency increases.
    
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
from numba import njit
import warnings
from importlib.metadata import version, PackageNotFoundError
from ._pvals import _lambda_p_perm, _lambda_p_beta, _lambda_p_exact
from ._core import _std_ranks, _lambda_stats

try:
    __version__ = version("lambda-corr")
except PackageNotFoundError:
    __version__ = "0.0.0"

#Numbda compatible entry
@njit(nogil=True)
def lambda_corr_nb(x, y, n, pvals=True, ptype="default", p_tol=1e-4, n_perm=10000, alt="two-sided", ties=False):

    # assume: x,y already arrays of same length, n>=3, finite, and non-constant
    # Standardized ranks with averaged ties
    rx = _std_ranks(x, n)
    ry = _std_ranks(y, n)
    # Get Lambda correlations - symmetric and asymmetric
    Lambda_s, Lambda_yx, Lambda_xy  = _lambda_stats(rx, ry, n)
    
    if pvals:
        if (ptype == "perm") or (ties and ptype == "default"):
                # --- permutation path ---
                p_s, p_yx, p_xy = _lambda_p_perm(rx, ry, n, Lambda_s, Lambda_yx, Lambda_xy, 
                                                p_tol=p_tol, n_perm=n_perm, alt=alt)
                
                #We can do better than permutation if ties=False
                if (not ties) and (n<=10):
                    p_s = _lambda_p_exact(Lambda_s, n, alt=alt)
                    
        else:
                # --- approx path (ptype=="approx" and (ties=True OR ties=False)) OR (ptype="default" and ties==False) ---
                if n <= 10:
                    p_s = _lambda_p_exact(Lambda_s, n, alt=alt) # exact only if no ties
                else: 
                    p_s = _lambda_p_beta(Lambda_s, n, alt=alt) 
                #The null distribution for the asymmetric measures was not calculated seperately
                p_yx = np.nan
                p_xy = np.nan

    else:
        p_s = p_xy = p_yx = np.nan
        
    # Asymmetry index with safe denominator
    denom = abs(Lambda_yx) + abs(Lambda_xy)
    Lambda_a = 0.0 if denom == 0.0 else float(abs(Lambda_yx - Lambda_xy) / denom)
    
    return Lambda_s, p_s, Lambda_yx, p_yx, Lambda_xy, p_xy, Lambda_a

#@njit(cache=True, nogil=True) #njit not compatible with warnings
def lambda_corr(x, y, pvals=True, ptype="default", p_tol=1e-4, n_perm=10000, alt="two-sided", ties=False):
    
    if pvals not in [True, False]:
        raise ValueError(f"pvals must be True or False, got '{pvals}'")
    
    if pvals:
        if ptype not in ["default", "perm", "approx"]:
            raise ValueError(f"ptype must be 'default', 'perm', or 'approx', got '{ptype}'")
        
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
                "for stable permutation p-values, using n_perm ≥≥ n is strongly recommended.",
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
                       alt=alt, ties=ties)
    
    return Lambda_s, p_s, Lambda_yx, p_yx, Lambda_xy, p_xy, Lambda_a
