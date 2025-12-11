#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Thu Oct  9 18:29:41 2025

Λ_S (Lambda_S) correlation evaluation suite
===========================================

This script compares Lambda_S (RA-rank symmetric correlation) against Spearman and Kendall on:

1) Efficiency (variance in clean Gaussian data)
2) Robustness to contamination (vertical outliers)
3) Null calibration / Type I error (permutation p-values)
4) Power under small effects
5) Monotone invariance (Y = f(X) + noise, with monotone f)

Author: Jon Paul Lundquist
"""

import math
import time
import numpy as np
from scipy.stats import t as student_t # pearsonr, spearmanr, kendalltau, 
from hyper_corr import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
from lambda_corr import lambda_corr
from scipy.special import erfcinv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
    
# -----------------------------
# Permutation p-values (studentized, sequential relative-uncertainty stopping)
# -----------------------------

def sigma_RA_permtest_studentized(x, y, *, stat, alternative="two-sided", warmup_B=1500,
    min_B=3000, max_B=200000, batch=1000, rel_tol=0.10):
    """
    Generic studentized permutation test for a rank-based statistic 'stat(x,y)'.
    Returns dict with p, T_obs, mu0, sd0, B_used, hits.
    """
    rng = np.random.default_rng()
    x = np.asarray(x); y = np.asarray(y)
    n = x.size
    if n != y.size or n < 3:
        raise ValueError("x,y same length with n>=3")

    # observed
    s_obs = stat(x, y)

    # warm-up: estimate null mean/sd under independence by permuting y
    S_w = np.empty(warmup_B, dtype=float)
    for b in range(warmup_B):
        perm = rng.permutation(n)
        S_w[b] = stat(x, y[perm])
    mu0 = float(np.mean(S_w))
    sd0 = float(np.std(S_w, ddof=1))
    if sd0 == 0.0 or not np.isfinite(sd0):
        return dict(p=1.0, T_obs=0.0, mu0=mu0, sd0=sd0, B_used=0, hits=0)

    def to_T(s: float) -> float:
        return (s - mu0) / sd0

    T_obs = to_T(s_obs)
    thr = abs(T_obs)

    hits = 0
    B = 0
    eps = 1e-12

    def stop(h, B):
        if B < min_B: return False
        p_hat = (h + 1.0) / (B + 1.0)
        se = math.sqrt(p_hat * (1.0 - p_hat) / (B + 1.0))
        rel = se / max(p_hat, eps)
        return (rel <= rel_tol)

    while B < max_B:
        k = min(batch, max_B - B)
        Tb = np.empty(k, dtype=float)
        for t in range(k):
            perm = rng.permutation(n)
            sb = stat(x, y[perm])
            Tb[t] = (sb - mu0) / sd0

        if alternative == "greater":
            hits += int(np.sum(Tb >= T_obs))
        elif alternative == "less":
            hits += int(np.sum(Tb <= T_obs))
        else:
            hits += int(np.sum(np.abs(Tb) >= thr))

        B += k
        if stop(hits, B):
            break

    p = (hits + 1.0) / (B + 1.0)
    return dict(p=p, T_obs=T_obs, mu0=mu0, sd0=sd0, B_used=B, hits=hits)


# -----------------------------
# Data generators & contamination
# -----------------------------

def draw_bivariate_normal(n, rho, rng):
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = rho * x + math.sqrt(max(1e-12, 1 - rho**2)) * z
    return x, y

def contaminate_vertical(y, frac = 0.05, strength = 10.0, rng=None):
    """
    ----------
    y : array_like
        Response values to be contaminated.
    frac : float
        Fraction of points to contaminate (0 < frac <= 1).
    strength : float
        Multiplier of y's population-scale (std) for the contamination size.
    rng : np.random.Generator or None
        RNG for sign/noise. If None, a fresh Generator is created.

    Returns
    -------
    y2 : np.ndarray
        Contaminated copy of y.
    """
    if rng is None: rng = np.random.default_rng()
    n = y.size
    k = max(1, int(frac * n))
    idx = rng.choice(n, size=k, replace=False)
    y2 = y.copy()
    y_std = y.std(ddof=0) or 1.0
    y2[idx] = y2[idx] + strength * y_std * np.sign(rng.standard_normal(k))
    return y2

def contaminate_leverage(x, y, frac = 0.05,
                         strength = 10.0, rng=None):
    """
    'Leverage' contamination: pick the k observations with the largest |x - median(x)|
    (i.e., highest leverage in the design space) and shove their y-values far away.

    Parameters
    ----------
    x : array_like
        Predictor values (used to find high-leverage points).
    y : array_like
        Response values to be contaminated at those leverage points.
    frac : float
        Fraction of points to contaminate (0 < frac <= 1).
    strength : float
        Multiplier of y's population-scale (std) for the contamination size.
    rng : np.random.Generator or None
        RNG for sign/noise. If None, a fresh Generator is created.

    Returns
    -------
    y2 : np.ndarray
        Contaminated copy of y.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = y.size
    k = max(1, int(frac * n))

    # leverage score proxy: distance from median in x-space
    x_center = np.median(x)
    lev = np.abs(x - x_center)

    # indices of the k largest leverage points
    idx = np.argpartition(lev, n - k)[-k:]

    # contaminate their y's strongly (random up/down like your vertical case)
    y2 = y.copy()
    y_std = y.std(ddof=0) or 1.0
    y2[idx] = y2[idx] + strength * y_std * np.sign(rng.standard_normal(k))
    return y2

def contaminate_uniform(x, y, frac=0.05, strength=10.0, rng=None):
    """
    Random contamination: replace observations with uniform random noise.
    This simulates measurement errors or data entry mistakes.
    
    Parameters
    ----------
    x, y : array_like
        Original data
    frac : float
        Fraction of points to contaminate
    strength : float
        Multiplier for contamination scale (relative to data scale)
    rng : np.random.Generator or None
    
    Returns
    -------
    x2, y2 : np.ndarray
        Contaminated data
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(x)
    k = max(1, int(frac * n))
    idx = rng.choice(n, size=k, replace=False)
    
    x2 = x.copy()
    y2 = y.copy()
    
    # Replace with random noise scaled to data range
    x_scale = x.std(ddof=0) or 1.0
    y_scale = y.std(ddof=0) or 1.0
    
    #Uniform random in expanded range
    x2[idx] = x.mean() + strength * x_scale * rng.uniform(-1, 1, size=k)
    y2[idx] = y.mean() + strength * y_scale * rng.uniform(-1, 1, size=k)
    
    # Option 2: Independent Gaussian noise
    # x2[idx] = rng.normal(x.mean(), strength * x_scale, size=k)
    # y2[idx] = rng.normal(y.mean(), strength * y_scale, size=k)
    
    return x2, y2

def discretize(a, bins = 6):
    edges = np.quantile(a, np.linspace(0, 1, bins+1))
    edges[0] -= 1e-12; edges[-1] += 1e-12
    idx = np.digitize(a, edges) - 1
    mids = 0.5 * (edges[:-1] + edges[1:])
    return mids[idx]

def heavy_tail_pair(n, rho, rng):
    x = rng.standard_normal(n)
    z = student_t.rvs(df=3, size=n, random_state=rng)
    z = (z - np.mean(z)) / np.std(z, ddof=0)
    y = rho * x + math.sqrt(max(1e-12, 1 - rho**2)) * z
    return x, y

def monotone_signal(n, rho, rng, f):
    # Build x ~ N(0,1); y = f(x) + noise scaled to target Pearson ρ approximately.
    x = rng.standard_normal(n)
    fx = f(x)
    fx = (fx - fx.mean()) / fx.std(ddof=0)
    # choose noise to achieve approximate Pearson ρ with linear proxy
    # y = a*fx + b*z; set a=ρ and b=sqrt(1-ρ^2)
    z = rng.standard_normal(n)
    y = rho * fx + math.sqrt(max(1e-12, 1 - rho**2)) * z
    return x, y

def toy_linear(n=100, slope=1.0, intercept=0.0, noise=0.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.linspace(0, 1, n)
    y = slope * x + intercept + rng.normal(scale=noise, size=n)
    return x, y

def toy_nonlinear_power(n=100, exponent=3.0, noise=0.1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.linspace(0, 1, n)
    y = (x ** exponent) + rng.normal(scale=noise, size=n)
    return x, y

def toy_u_shape(n=100, noise=0.2, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.linspace(-1, 1, n)
    y = - (x ** 2) + rng.normal(scale=noise, size=n)
    return x, y

def toy_floor_ceiling(n=100, split=30, high=10, low=0, noise=0.1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.linspace(0,1,n)
    y = np.empty(n)
    # first part: floor (low noise), then line, then ceiling
    y[:split] = low + rng.normal(scale=noise, size=split)
    y[split:] = high + rng.normal(scale=noise, size=n-split)
    return x, y

def toy_contaminated_linear(n=100, slope=1.0, noise=0.0, n_outliers=3, outlier_mag=10.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x, y = toy_linear(n, slope=slope, noise=noise, rng=rng)
    # place outliers
    inds = rng.choice(n, size=n_outliers, replace=False)
    y[inds] += rng.normal(scale=outlier_mag, size=n_outliers)
    return x, y

def toy_ties_discrete(n=100, n_levels=5, noise=0.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.linspace(0, 1, n)
    y_true = x  # baseline
    # map y_true to one of n_levels discrete levels + small noise
    levels = np.linspace(0, 1, n_levels)
    y = levels[np.argmin(np.abs(y_true[:,None] - levels[None,:]), axis=1)]
    y = y + rng.normal(scale=noise, size=n)
    return x, y

# -----------------------------
# Metrics / experiments
# -----------------------------

# def pearson(x, y):
#     xx = (x - x.mean()) / x.std(ddof=0)
#     yy = (y - y.mean()) / y.std(ddof=0)
#     pr = float(np.mean(xx * yy))
#     return pr

def compute_stats(x, y):
    #lamS, lam_yx, lam_xy, lamA = lambda_symmetric(x, y)
    lamS, _, lam_xy, _, lam_yx, _, lamA = lambda_corr(x, y, pvals=False)
    sp, _ = spearmanr(x, y, pvals=False)
    kt, _ = kendalltau(x, y, pvals=False)
    # Pearson for reference
    pr, _ = pearsonr(x, y, pvals=False)
    return dict(Lambda_S=lamS, Lambda_YX=lam_yx, Lambda_XY=lam_xy, Lambda_A=lamA,
                Spearman=sp, Kendall=kt, Pearson=pr)

# Theoretical expectations under BVN (large-n)
def E_spearman_given_rho(rho):
    # E[ρ_S] ≈ (6/π) * arcsin(ρ/2)
    return (6.0/np.pi) * np.arcsin(np.clip(rho, -1, 1) / 2.0)

def E_kendall_given_rho(rho):
    # E[τ] = (2/π) * arcsin(ρ)
    return (2.0/np.pi) * np.arcsin(np.clip(rho, -1, 1))

def bias_study(rhos=(0.1, 0.3, 0.5, 0.7, 0.9), n=300, trials=1000):
    rng = np.random.default_rng()
    out = []
    for rho in rhos:
        vals = {k: [] for k in ["Lambda_S", "Spearman", "Kendall", "Pearson"]}
        for _ in range(trials):
            x, y = draw_bivariate_normal(n, rho, rng)
            s = compute_stats(x, y)  # expects keys: "Lambda_S","Spearman","Kendall","Pearson"
            for k in vals:
                vals[k].append(s[k])

        mean = {k: float(np.mean(v)) for k, v in vals.items()}

        # Bias vs true Pearson-ρ target
        bias_vs_rho = {k + "_bias": mean[k] - rho for k in vals}

        # Also report bias vs *theoretical expectation* (Spearman/Kendall only)
        sp_theory = float(E_spearman_given_rho(rho))
        kt_theory = float(E_kendall_given_rho(rho))
        bias_vs_theory = {
            "Spearman_bias_vs_E": mean["Spearman"] - sp_theory,
            "Kendall_bias_vs_E":  mean["Kendall"]  - kt_theory,
        }

        out.append(dict(
            rho=rho, n=n,
            **{k + "_mean": m for k, m in mean.items()},
            **bias_vs_rho,
            **bias_vs_theory
        ))
    return out

def efficiency_study(rhos=(0.1,0.3,0.5,0.7,0.9), n=300, trials=50000):
    rng = np.random.default_rng()
    out = []
    for rho in rhos:
        vals = {k: [] for k in ["Lambda_S","Spearman","Kendall","Pearson"]}
        for _ in range(trials):
            x, y = draw_bivariate_normal(n, rho, rng)
            s = compute_stats(x, y)
            for k in vals: vals[k].append(s[k])
        # variances
        var = {k: float(np.var(vals[k], ddof=1)) for k in vals}
        # relative efficiency vs Pearson (smaller var is better → higher eff)
        eff = {k+"_eff_vs_P": var["Pearson"] / var[k] for k in vals if k!="Pearson"}
        out.append(dict(rho=rho, n=n, **{k+"_var": v for k,v in var.items()}, **eff))
    return out

# -----------------------------
# Accuracy (MSE / RMSE) study
# -----------------------------

def accuracy_study(rhos=(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9), n=300, trials=50000):
    """
    For each rho, simulate 'trials' datasets, collect estimators,
    and compute MSE and RMSE vs the *true* Pearson rho.
    Returns a list of dict rows mirroring your other studies.
    """
    rng = np.random.default_rng()
    out = []
    methods = ["Lambda_S","Spearman","Kendall","Pearson"]

    for rho in rhos:
        vals = {k: [] for k in methods}
        for _ in range(trials):
            x, y = draw_bivariate_normal(n, rho, rng)
            s = compute_stats(x, y)  # expects keys: Lambda_S, Spearman, Kendall, Pearson
            for k in methods:
                vals[k].append(s[k])

        # Mean & Var for convenience
        mean = {k: float(np.mean(vals[k])) for k in methods}
        var  = {k: float(np.var(vals[k], ddof=1)) for k in methods}

        # Accuracy vs true rho: MSE = Var + Bias^2; RMSE = sqrt(MSE)
        mse = {k+"_MSE": var[k] + (mean[k] - rho)**2 for k in methods}
        rmse = {k+"_RMSE": float(np.sqrt(mse[k+"_MSE"])) for k in methods}

        out.append(dict(rho=rho, n=n,
                        **{k+"_mean": mean[k] for k in methods},
                        **{k+"_var": var[k] for k in methods},
                        **mse, **rmse))
    return out

def robustness_study(n=300, rho=0.7, frac_list=(0,0.02,0.05,0.1,0.2), trials=50000, 
                     ver="Vertical"):
    rng = np.random.default_rng()
    results = []
    for frac in frac_list:
        vals = {k: [] for k in ["Lambda_S","Spearman","Kendall","Pearson"]}
        
        for _ in range(trials):
            x, y = draw_bivariate_normal(n, rho, rng)
            if ver=="Vertical":
                x2 = x
                y2 = contaminate_vertical(y, frac=frac, strength=10.0, rng=rng)
                
            elif ver=="Uniform":
                x2, y2 = contaminate_uniform(x, y, frac=frac, strength=10.0, rng=rng)
                
            else:
                x2 = x
                y2 = contaminate_leverage(x, y, frac=frac, strength=10.0, rng=rng)
            
            s = compute_stats(x2, y2)
            for k in vals: vals[k].append(s[k])
            
        # median & IQR: robustness summaries
        med = {k+"_med": float(np.median(vals[k])) for k in vals}
        iqr = {k+"_IQR": float(np.percentile(vals[k], 75)-np.percentile(vals[k], 25)) for k in vals}
        results.append(dict(frac=frac, n=n, rho=rho, **med, **iqr))
    return results

def null_calibration(n=200, trials=50000):
    rng = np.random.default_rng()
    stats = {k: [] for k in ["Lambda_S","Spearman","Kendall","Pearson"]}
    for _ in range(trials):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)  # independent
        s = compute_stats(x, y)
        for k in stats: stats[k].append(s[k])
    out = {}
    for k, arr in stats.items():
        a = np.array(arr, float)
        out[k] = dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                      q95=float(np.percentile(a, 95)), q99=float(np.percentile(a, 99)))
    return out

def power_study(n=200, rhos=(0.1,0.2,0.3,0.4), alpha=0.05, trials=50000):
    """
    Reports rejection rate (power).
    """
    rng = np.random.default_rng()
    out = []
    for rho in rhos:
        rej_LS = rej_SP = rej_KT = rej_PR = 0
        for _ in range(trials):
            x, y = draw_bivariate_normal(n, rho, rng)
            # Lambda_S permutation p
            # p_LS = sigma_RA_permtest_studentized(x, y, stat=lambda u,v: lambda_symmetric(u,v)[0],
            #                                      alternative="two-sided", warmup_B=600,
            #                                      min_B=1500, max_B=30000, batch=1000,
            #                                      rel_tol=0.12)["p"]
            p_LS = lambda_corr(x, y, p_tol=0.005)[1]
            # Spearman/Kendall p from SciPy (large-sample approx)
            sp, p_sp = spearmanr(x, y)
            kt, p_kt = kendalltau(x, y)
            pr, p_pr = pearsonr(x, y)
            rej_LS += int(p_LS < alpha)
            rej_SP += int(p_sp < alpha)
            rej_KT += int(p_kt < alpha)
            rej_PR += int(p_pr < alpha)
        out.append(dict(n=n, rho=rho, alpha=alpha,
                        power_LambdaS=rej_LS/trials,
                        power_Spearman=rej_SP/trials,
                        power_Kendall=rej_KT/trials,power_Pearson=rej_PR/trials))
    return out

def monotone_invariance_study(n=400, rho=0.7, trials=50000):
    rng = np.random.default_rng()
    # choose monotone transforms
    Fs = [np.tanh, np.exp, lambda z: z**3, lambda z: 1/(1+np.exp(-z))]
    names = ["tanh","exp","cube","logistic"]
    # measure correlation between Lambda_S on (x,y) and on (x,f(y))
    vals = {name: [] for name in names}
    for _ in range(trials):
        x, y = draw_bivariate_normal(n, rho, rng)
        #base = lambda_symmetric(x, y)[0]
        base = lambda_corr(x, y, pvals=False)[0]
        for f,name in zip(Fs, names):
            yf = f(y)
            yf = (yf - yf.mean()) / (yf.std(ddof=0) or 1.0)
            vals[name].append( (base, lambda_corr(x, yf, pvals=False)[0]) )
    # compute linear correlation between base and transformed Lambda_S
    res = {}
    for name in names:
        a = np.array(vals[name], float)
        b0 = a[:,0]; b1 = a[:,1]
        # Pearson on paired Lambda_S values to summarize invariance
        b0 = (b0 - b0.mean()) / b0.std(ddof=0)
        b1 = (b1 - b1.mean()) / b1.std(ddof=0)
        res[name] = float(np.mean(b0*b1))
    return res

def contaminate_corner(x, y, m, rng):
    """Return (x', y') after replacing m points adversarially.
       Puts contaminated points at opposing extremes (worst for positive signal).
    """
    rng = np.random.default_rng() if rng is None else rng
    n = x.size
    #idx = np.arange(n)
    # choose m indices to replace
    idx = rng.choice(n, size=m, replace=False)
    x_adv , y_adv  = x.copy(), y.copy()

    # Create extreme values in data space
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    
    # Push to opposing corners (flip sign)
    x_adv[idx] = x.max() + 10 * x_range  # Push X high
    y_adv[idx] = y.min() - 10 * y_range  # Push Y low

    return x_adv, y_adv

def contaminate_neg_subset(x, y, m, T):
    """
    Bounded permutation adversarial contamination on a random subset of size m.

    - Choose m indices at random.
    - Within that subset, reorder x and y to inject an association 
      opposite in sign to T.
    - Outside the subset, points are unchanged.

    This preserves the marginal distributions of x and y globally, but
    injects a maximally opposite monotone block in the subset.
    """
    rng = np.random.default_rng()

    n = x.size
    if m <= 0 or m > n:
        raise ValueError("m must be in 1..n")

    # Random subset
    idx = rng.choice(n, size=m, replace=False)

    x_adv = x.copy()
    y_adv = y.copy()

    # Subset values
    x_sub = x[idx]
    y_sub = y[idx]

    # Order subset indices by x (ascending)
    order_by_x = np.argsort(x_sub)       # positions 0..m-1 sorted by x
    idx_ordered = idx[order_by_x]
    x_sorted    = x_sub[order_by_x]      # x increasing

    if T > 0:
        # Original association positive → inject negative association:
        # increasing x, decreasing y
        y_sorted = np.sort(y_sub)[::-1]  # descending
    else:
        # Original association negative → inject positive association:
        # increasing x, increasing y
        y_sorted = np.sort(y_sub)        # ascending

    # Assign reordered block
    x_adv[idx_ordered] = x_sorted
    y_adv[idx_ordered] = y_sorted
    
    return x_adv, y_adv

def eps_breakdown(stat_fn, x, y, grid=None, trials=50000, rng=None):
    """
    stat_fn: callable(x,y)-> scalar correlation (e.g., Lambda_S, Spearman, Kendall, Pearson)
    mode: 'sign' (flip sign) or 'mag' (reach |T| >= 1-delta)
    returns: {'eps': estimated breakdown fraction, 'details': list of per-f results}
    """
    rng = np.random.default_rng() if rng is None else rng
    if grid is None:
        grid = np.linspace(0.0, 1, 101)  # scan 0..70%

    n = x.size

    T0 = float(stat_fn(x, y)) # Observed correlation on clean data

    worst_reached = []
    for f in grid:
        m = int(np.floor(f * n))
        if m == 0:
            worst_reached.append(False)
            continue
        
        hit = False
        for _ in range(trials):
            # try to flip signs;
            x_adv, y_adv = contaminate_neg_subset(x, y, m, T0)

            # Compute contaminated correlation
            T = float(stat_fn(x_adv, y_adv))
            if np.sign(T) != np.sign(T0) and T != 0.0:
                hit = True
                break
                
        worst_reached.append(hit)
        if hit:  # first f where adversary wins
            return {"eps": f, "details": list(zip(grid, worst_reached))}

    return {"eps": None, "details": list(zip(grid, worst_reached))}

def wilson_ci(k, n, conf=0.95):
    z = 1.959963984540054 if conf == 0.95 else np.sqrt(2) * erfcinv(2*(1-conf))
    ph = k / n
    denom = 1.0 + z*z/n
    center = (ph + z*z/(2*n)) / denom
    half = z * np.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / denom
    return (center - half, center + half)

def size_study(n=300, alpha=0.05, trials=50000):
    """
    Empirical Type-I error under independence (ρ=0).
    Same machinery as power_study, just with rho=0.
    """
    rng = np.random.default_rng()
    rej_LS = rej_SP = rej_KT = rej_PR = 0
    rho = 0.0
    for _ in range(trials):
        x, y = draw_bivariate_normal(n, rho, rng)  # independence
        p_LS = lambda_corr(x, y, p_tol=0.005)[1]                # your permutation p for Λ_S
        sp, p_sp = spearmanr(x, y)
        kt, p_kt = kendalltau(x, y)
        pr, p_pr = pearsonr(x, y)
        rej_LS += int(p_LS < alpha)
        rej_SP += int(p_sp < alpha)
        rej_KT += int(p_kt < alpha)
        rej_PR += int(p_pr < alpha)
        
    out = []
    for name, k in [("LambdaS", rej_LS), ("Spearman", rej_SP), ("Kendall", rej_KT), ("Pearson", rej_PR)]:
        lo, hi = wilson_ci(k, trials, conf=0.95)
        out.append(dict(
            n=n, alpha=alpha, method=name,
            hat_alpha=k/trials, ci95=(lo, hi), trials=trials
        ))
    return out

# Plot helpers

def plot_bias_means(rows):
    rhos = [r["rho"] for r in rows]
    mL  = [r["Lambda_S_mean"] for r in rows]
    mS  = [r["Spearman_mean"] for r in rows]
    mK  = [r["Kendall_mean"]  for r in rows]
    mP  = [r["Pearson_mean"]  for r in rows]

    plt.figure(figsize=(7,4))
    plt.plot(rhos, rhos, "--", alpha=0.6, label="y = ρ (ideal)", color='blue')
    plt.plot(rhos, mP, "-o", label="Pearson", color='C0')
    plt.plot(rhos, mS, "-o", label="Spearman", color='C1')
    plt.plot(rhos, mK, "-o", label="Kendall", color='C2')
    plt.plot(rhos, mL, "-o", label="Λ_S", color='C3')
    plt.xlabel("True ρ (Gaussian)"); plt.ylabel("Mean estimate")
    plt.title("Calibration: mean(estimator) vs true ρ")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_bias_curves(rows):
    rhos = [r["rho"] for r in rows]
    bL = [r["Lambda_S_bias"] for r in rows]
    bS = [r["Spearman_bias"] for r in rows]
    bK = [r["Kendall_bias"]  for r in rows]
    bP = [r["Pearson_bias"]  for r in rows]

    plt.figure(figsize=(7,4))
    plt.axhline(0.0, ls="--", alpha=0.6, label="zero bias", color='blue')
    plt.plot(rhos, bP, "-o", label="Pearson bias", color='C0')
    plt.plot(rhos, bS, "-o", label="Spearman bias (vs ρ)", color='C1')
    plt.plot(rhos, bK, "-o", label="Kendall bias (vs ρ)", color='C2')
    plt.plot(rhos, bL, "-o", label="Λ_S bias (vs ρ)", color='C3')
    plt.xlabel("True ρ (Gaussian)"); plt.ylabel("Bias (mean − ρ)")
    plt.title("Bias vs true ρ")
    plt.legend(); plt.tight_layout(); plt.show()
    
def plot_efficiency_table(eff_rows):
    rhos = [row["rho"] for row in eff_rows]
    eff_LS = [row["Lambda_S_eff_vs_P"] for row in eff_rows]
    eff_SP = [row["Spearman_eff_vs_P"] for row in eff_rows]
    eff_KT = [row["Kendall_eff_vs_P"] for row in eff_rows]

    plt.figure(figsize=(7,4))
    plt.axhline(1.0, linestyle='--', alpha=0.6, label="Pearson (reference = 1)", color='blue')
    plt.plot(rhos, eff_SP, marker='o', label="Spearman vs Pearson", color='C1')
    plt.plot(rhos, eff_KT, marker='o', label="Kendall vs Pearson", color='C2')
    plt.plot(rhos, eff_LS, marker='o', label="Λ_S vs Pearson", color='C3')
    # Pearson reference at 1.0 with legend entry

    plt.xlabel("True ρ (Gaussian)")
    plt.ylabel("Relative efficiency (↑ better)")
    plt.title("Efficiency (variance ratio) vs ρ")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_accuracy_table(acc_rows, metric="RMSE"):
    """
    Plot RMSE (default) or MSE vs rho for each method.
    Lower is better.
    """
    rhos = [row["rho"] for row in acc_rows]
    key = "_RMSE" if metric.upper()=="RMSE" else "_MSE"

    yL = [row["Lambda_S"+key] for row in acc_rows]
    yS = [row["Spearman"+key] for row in acc_rows]
    yK = [row["Kendall"+key]  for row in acc_rows]
    yP = [row["Pearson"+key]  for row in acc_rows]

    plt.figure(figsize=(7,4))
    plt.plot(rhos, yP, "-o", label="Pearson", color='C0')
    plt.plot(rhos, yS, "-o", label="Spearman", color='C1')
    plt.plot(rhos, yK, "-o", label="Kendall", color='C2')
    plt.plot(rhos, yL, "-o", label="Λ_S", color='C3')
    plt.xlabel("True ρ (Gaussian)")
    plt.ylabel(metric.upper() + " (↓ better)")
    plt.title(f"Accuracy ({metric.upper()}) vs ρ")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_robustness_table(rob_rows, ver):
    fracs = np.array([row["frac"] for row in rob_rows])

    med_LS = np.array([row["Lambda_S_med"] for row in rob_rows])
    med_SP = np.array([row["Spearman_med"] for row in rob_rows])
    med_KT = np.array([row["Kendall_med"] for row in rob_rows])
    med_PR = np.array([row["Pearson_med"] for row in rob_rows])

    iqr_LS = np.array([row["Lambda_S_IQR"] for row in rob_rows])
    iqr_SP = np.array([row["Spearman_IQR"] for row in rob_rows])
    iqr_KT = np.array([row["Kendall_IQR"] for row in rob_rows])
    iqr_PR = np.array([row["Pearson_IQR"] for row in rob_rows])

    # Symmetric error bars: half-IQR above and below the median
    err_LS = iqr_LS / 2.0
    err_SP = iqr_SP / 2.0
    err_KT = iqr_KT / 2.0
    err_PR = iqr_PR / 2.0

    plt.figure(figsize=(7,4))
    plt.errorbar(fracs, med_PR, yerr=err_PR, fmt='-o', capsize=3, label="Pearson median ± IQR/2", color='C0')
    plt.errorbar(fracs, med_SP, yerr=err_SP, fmt='-o', capsize=3, label="Spearman median ± IQR/2", color='C1')
    plt.errorbar(fracs, med_KT, yerr=err_KT, fmt='-o', capsize=3, label="Kendall median ± IQR/2", color='C2')
    plt.errorbar(fracs, med_LS, yerr=err_LS, fmt='-o', capsize=3, label="Λ_S median ± IQR/2", color='C3')
    plt.xlabel("Outlier fraction")
    plt.ylabel("Median estimate (error bars = IQR/2)")
    plt.title("Robustness: Median vs " + ver + "Contamination (IQR error bars)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_power_table(power_rows):
    """
    Plot:
      1) Power vs rho for each method (with Pearson theory curve).
      2) Power ratio vs Pearson-theory power for each method.
    """
    from scipy.stats import norm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    def pearson_power_approx(rho, n, alpha=0.05):
        """Approximate power for Pearson correlation test using Fisher z."""
        # Fisher z transformation (works fine for rho=0 too)
        if abs(rho) >= 1:
            # perfect correlation => essentially power ~1 for any reasonable alpha
            return 1.0
        z_rho = 0.5 * np.log((1 + rho) / (1 - rho))
        se = 1.0 / np.sqrt(n - 3)
        z_crit = norm.ppf(1 - alpha/2)
        # two-sided power
        power = 1 - norm.cdf(z_crit - abs(z_rho) / se) + norm.cdf(-z_crit - abs(z_rho) / se)
        return power

    rhos = [row["rho"] for row in power_rows]

    yL = np.array([row["power_LambdaS"] for row in power_rows])
    yS = np.array([row["power_Spearman"] for row in power_rows])
    yK = np.array([row["power_Kendall"]  for row in power_rows])
    yP = np.array([row["power_Pearson"]  for row in power_rows])

    n = power_rows[0]['n']
    alpha = power_rows[0]['alpha']

    # --- 1) Absolute power vs rho ---

    fig, ax = plt.subplots(figsize=(7,4))

    # Theoretical Pearson power on a fine grid
    rhos_theory = np.linspace(0, 1, 200)
    power_theory = [pearson_power_approx(r, n, alpha) for r in rhos_theory]
    ax.plot(rhos_theory, power_theory, '--', color='C0', alpha=0.5,
            linewidth=2, label='Pearson (theoretical)')

    # Empirical power curves
    ax.plot(rhos, yP, "-o", label="Pearson (empirical)", color='C0')
    ax.plot(rhos, yS, "-o", label="Spearman",            color='C1')
    ax.plot(rhos, yK, "-o", label="Kendall",             color='C2')
    ax.plot(rhos, yL, "-o", label="Λ_S",                 color='C3')

    ax.set_xlabel("True ρ (Gaussian)")
    ax.set_ylabel("Power (↑ better)")
    ax.set_title(f"Power vs ρ (n={n}, α={alpha})")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    axins = inset_axes(ax, width="30%", height="30%", loc='upper left',
                       bbox_to_anchor=(0.08, 0.05, 0.9, 0.9),
                       bbox_transform=ax.transAxes)

    zoom_idx = [i for i, r in enumerate(rhos) if r <= 0.25]
    rhos_zoom = [rhos[i] for i in zoom_idx]

    rhos_theory_zoom = np.linspace(0, 0.25, 100)
    power_theory_zoom = [pearson_power_approx(r, n, alpha) for r in rhos_theory_zoom]
    axins.plot(rhos_theory_zoom, power_theory_zoom, '--', color='C0', alpha=0.5, linewidth=2)

    axins.plot(rhos_zoom, [yP[i] for i in zoom_idx], "-o", color='C0', markersize=4)
    axins.plot(rhos_zoom, [yS[i] for i in zoom_idx], "-o", color='C1', markersize=4)
    axins.plot(rhos_zoom, [yK[i] for i in zoom_idx], "-o", color='C2', markersize=4)
    axins.plot(rhos_zoom, [yL[i] for i in zoom_idx], "-o", color='C3', markersize=4)

    axins.set_xlim(0, 0.21)
    axins.set_ylim(alpha/2, 0.1)
    axins.grid(True, alpha=0.3)
    axins.set_xlabel('ρ', fontsize=9)
    axins.set_ylabel('Power', fontsize=9)
    axins.tick_params(labelsize=8)
    axins.set_yscale('log')
    plt.tight_layout()
    plt.show()

    # --- 2) Power ratio vs Pearson-theory ---

    # Pearson theoretical power at your discrete rhos
    base_P = np.array([pearson_power_approx(r, n, alpha) for r in rhos])

    # To avoid division noise at rho=0 (power ~ alpha for all),
    # you can either keep it (ratio ~1) or skip it:
    # mask = base_P > 1e-6
    mask = np.ones_like(base_P, dtype=bool)  # keep all for now

    ratio_L = yL[mask] / base_P[mask]
    ratio_S = yS[mask] / base_P[mask]
    ratio_K = yK[mask] / base_P[mask]
    ratio_P = yP[mask] / base_P[mask]   # sanity check: should be ~1

    rhos_eff = np.array(rhos)[mask]

    fig2, ax2 = plt.subplots(figsize=(7,4))

    ax2.axhline(1.0, color='k', linestyle='--', linewidth=1, label="Pearson (theory baseline)")

    ax2.plot(rhos_eff, ratio_P, "-o", color="C0", label="Pearson (empirical / theory)")
    ax2.plot(rhos_eff, ratio_S, "-o", color="C1", label="Spearman / Pearson-theory")
    ax2.plot(rhos_eff, ratio_K, "-o", color="C2", label="Kendall / Pearson-theory")
    ax2.plot(rhos_eff, ratio_L, "-o", color="C3", label="Λ_S / Pearson-theory")

    ax2.set_xlabel("True ρ (Gaussian)")
    ax2.set_ylabel("Power / Pearson-theory (↑ better)")
    ax2.set_title(f"Power Ratio vs Pearson (n={n}, α={alpha})")
    ax2.grid(True, alpha=0.3)

    # Tight y-limits around 1 to show subtle differences
    ymin = min(ratio_L.min(), ratio_S.min(), ratio_K.min(), ratio_P.min())
    ymax = max(ratio_L.max(), ratio_S.max(), ratio_K.max(), ratio_P.max())
    margin = 0.2 * (ymax - ymin if ymax > ymin else 0.1)
    ax2.set_ylim(max(0.0, ymin - margin), ymax + margin)

    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()
 
def plot_breakdown_study(rhos, n=25, trials=2000, n_datasets=50):
    """
    Plot sign-breakdown epsilon vs rho for each method with error bars (SEM).
    Lower epsilon = more vulnerable to sign reversal.

    Parameters
    ----------
    rhos : list
        True correlation values to test
    n : int
        Sample size for each dataset
    trials : int
        Number of adversarial trials per breakdown test
    n_datasets : int
        Number of independent (x,y) datasets to average over
    """
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    results = {
        'rhos': [],
        'Lambda_S_mean': [],
        'Lambda_S_sem': [],
        'Spearman_mean': [],
        'Spearman_sem': [],
        'Kendall_mean': [],
        'Kendall_sem': [],
        'Pearson_mean': [],
        'Pearson_sem': []
    }

    def stat_lambdaS(a, b):
        return lambda_corr(a, b, pvals=False)[0]

    for rho in rhos:
        print(f"Computing breakdown for rho={rho}...")

        # Collect breakdown values over multiple datasets
        eps_lambda = []
        eps_spear = []
        eps_kend = []
        eps_pear = []

        for dataset_idx in range(n_datasets):
            x, y = draw_bivariate_normal(n=n, rho=rho, rng=rng)

            res_lambda = eps_breakdown(stat_lambdaS, x, y, trials=trials, rng=rng)
            res_spear = eps_breakdown(lambda a,b: spearmanr(a,b, pvals=False).statistic, 
                                      x, y, trials=trials, rng=rng)
            res_kend = eps_breakdown(lambda a,b: kendalltau(a,b, pvals=False).statistic, 
                                    x, y, trials=trials, rng=rng)
            res_pear = eps_breakdown(lambda a,b: pearsonr(a,b, pvals=False).statistic, 
                                    x, y, trials=trials, rng=rng)

            # Handle None (breakdown > 50%)
            eps_lambda.append(res_lambda['eps'] )
            eps_spear.append(res_spear['eps'])
            eps_kend.append(res_kend['eps'])
            eps_pear.append(res_pear['eps'])

        # Calculate mean and SEM = std / sqrt(n_datasets)
        results['rhos'].append(rho)
        results['Lambda_S_mean'].append(np.mean(eps_lambda))
        results['Lambda_S_sem'].append(np.std(eps_lambda, ddof=1) / np.sqrt(n_datasets))
        results['Spearman_mean'].append(np.mean(eps_spear))
        results['Spearman_sem'].append(np.std(eps_spear, ddof=1) / np.sqrt(n_datasets))
        results['Kendall_mean'].append(np.mean(eps_kend))
        results['Kendall_sem'].append(np.std(eps_kend, ddof=1) / np.sqrt(n_datasets))
        results['Pearson_mean'].append(np.mean(eps_pear))
        results['Pearson_sem'].append(np.std(eps_pear, ddof=1) / np.sqrt(n_datasets))

        print(f"  Pearson: {results['Pearson_mean'][-1]:.3f}±{results['Pearson_sem'][-1]:.3f}, "
              f"Spearman: {results['Spearman_mean'][-1]:.3f}±{results['Spearman_sem'][-1]:.3f}, "
              f"Kendall: {results['Kendall_mean'][-1]:.3f}±{results['Kendall_sem'][-1]:.3f}, "
              f"Λ_S: {results['Lambda_S_mean'][-1]:.3f}±{results['Lambda_S_sem'][-1]:.3f}")

    # Plot with error bars (SEM)
    fig, ax = plt.subplots(figsize=(7,4))

    ax.errorbar(results['rhos'], results['Pearson_mean'], yerr=results['Pearson_sem'],
                fmt="-o", label="Pearson", color='C0', capsize=3)
    ax.errorbar(results['rhos'], results['Spearman_mean'], yerr=results['Spearman_sem'],
                fmt="-o", label="Spearman", color='C1', capsize=3)
    ax.errorbar(results['rhos'], results['Kendall_mean'], yerr=results['Kendall_sem'],
                fmt="-o", label="Kendall", color='C2', capsize=3)
    ax.errorbar(results['rhos'], results['Lambda_S_mean'], yerr=results['Lambda_S_sem'],
                fmt="-o", label="Λ_S", color='C3', capsize=3)

    ax.set_xlabel("True ρ (Gaussian)")
    ax.set_ylabel("Sign-breakdown ε (↑ better)")
    ax.set_title(f"Adversarial Sign-Breakdown Point vs ρ (n={n})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results

def stat_lambdaS(a, b):
    return lambda_corr(a, b, pvals=False)[0]

def _breakdown_for_single_rho(rho, n, trials, n_datasets, grid, base_seed):
    """
    Run breakdown study for a single rho in a separate process.
    Returns a dict with mean & SEM for each method at this rho.
    """
    # Independent RNG per process
    rng = np.random.default_rng(base_seed)

    eps_lambda = []
    eps_spear  = []
    eps_kend   = []
    eps_pear   = []

    for dataset_idx in range(n_datasets):
        # New RNG per dataset (optional, but cleaner)
        ds_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

        x, y = draw_bivariate_normal(n=n, rho=rho, rng=ds_rng)

        res_lambda = eps_breakdown(
            stat_lambdaS, x, y, grid=grid, trials=trials, rng=ds_rng
        )
        res_spear = eps_breakdown(
            lambda a, b: spearmanr(a, b, pvals=False).statistic,
            x, y, grid=grid, trials=trials, rng=ds_rng
        )
        res_kend = eps_breakdown(
            lambda a, b: kendalltau(a, b, pvals=False).statistic,
            x, y, grid=grid, trials=trials, rng=ds_rng
        )
        res_pear = eps_breakdown(
            lambda a, b: pearsonr(a, b, pvals=False).statistic,
            x, y, grid=grid, trials=trials, rng=ds_rng
        )

        eps_lambda.append(res_lambda['eps'])
        eps_spear.append(res_spear['eps'])
        eps_kend.append(res_kend['eps'])
        eps_pear.append(res_pear['eps'])

    # Convert to arrays
    eps_lambda = np.asarray(eps_lambda)
    eps_spear  = np.asarray(eps_spear)
    eps_kend   = np.asarray(eps_kend)
    eps_pear   = np.asarray(eps_pear)

    def mean_sem(arr):
        m = float(np.mean(arr))
        # Need n_datasets >= 2 for ddof=1; you are using 25 so fine
        s = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        return m, s

    Lambda_mean, Lambda_sem = mean_sem(eps_lambda)
    Spear_mean,  Spear_sem  = mean_sem(eps_spear)
    Kend_mean,   Kend_sem   = mean_sem(eps_kend)
    Pear_mean,   Pear_sem   = mean_sem(eps_pear)

    return {
        "rho": rho,
        "Lambda_S_mean": Lambda_mean,
        "Lambda_S_sem":  Lambda_sem,
        "Spearman_mean": Spear_mean,
        "Spearman_sem":  Spear_sem,
        "Kendall_mean":  Kend_mean,
        "Kendall_sem":   Kend_sem,
        "Pearson_mean":  Pear_mean,
        "Pearson_sem":   Pear_sem,
    }

def plot_breakdown_study_parallel(
    rhos, n=25, trials=2000, n_datasets=25, max_workers=None
):
    """
    Parallel version: each rho is handled in a separate process.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    # Common grid used by all workers
    grid = np.linspace(0.0, 0.95, 96)

    results = {
        'rhos': [],
        'Lambda_S_mean': [],
        'Lambda_S_sem': [],
        'Spearman_mean': [],
        'Spearman_sem': [],
        'Kendall_mean': [],
        'Kendall_sem': [],
        'Pearson_mean': [],
        'Pearson_sem': []
    }

    # Launch workers
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i, rho in enumerate(rhos):
            # different base seed per rho
            base_seed = 42 + 10_000 * i
            fut = ex.submit(
                _breakdown_for_single_rho,
                rho, n, trials, n_datasets, grid, base_seed
            )
            futures.append(fut)

        # Gather results as they complete
        for fut in as_completed(futures):
            res = fut.result()
            rho = res["rho"]
            print(
                f"rho={rho}: "
                f"Pearson={res['Pearson_mean']:.3f}±{res['Pearson_sem']:.3f}, "
                f"Spearman={res['Spearman_mean']:.3f}±{res['Spearman_sem']:.3f}, "
                f"Kendall={res['Kendall_mean']:.3f}±{res['Kendall_sem']:.3f}, "
                f"Λ_S={res['Lambda_S_mean']:.3f}±{res['Lambda_S_sem']:.3f}"
            )
            results['rhos'].append(rho)
            results['Lambda_S_mean'].append(res['Lambda_S_mean'])
            results['Lambda_S_sem'].append(res['Lambda_S_sem'])
            results['Spearman_mean'].append(res['Spearman_mean'])
            results['Spearman_sem'].append(res['Spearman_sem'])
            results['Kendall_mean'].append(res['Kendall_mean'])
            results['Kendall_sem'].append(res['Kendall_sem'])
            results['Pearson_mean'].append(res['Pearson_mean'])
            results['Pearson_sem'].append(res['Pearson_sem'])

    # Sort by rho (as as_completed returns in completion order)
    order = np.argsort(results['rhos'])
    for key in results:
        results[key] = [results[key][i] for i in order]

    # Plot like before
    fig, ax = plt.subplots(figsize=(7, 4))

    rhos_sorted = results['rhos']

    ax.errorbar(rhos_sorted, results['Pearson_mean'], yerr=results['Pearson_sem'],
                fmt="-o", label="Pearson", capsize=3)
    ax.errorbar(rhos_sorted, results['Spearman_mean'], yerr=results['Spearman_sem'],
                fmt="-o", label="Spearman", capsize=3)
    ax.errorbar(rhos_sorted, results['Kendall_mean'], yerr=results['Kendall_sem'],
                fmt="-o", label="Kendall", capsize=3)
    ax.errorbar(rhos_sorted, results['Lambda_S_mean'], yerr=results['Lambda_S_sem'],
                fmt="-o", label="Λ_S", capsize=3)

    ax.set_xlabel("True ρ (Gaussian)")
    ax.set_ylabel("Sign-breakdown ε (↑ better)")
    ax.set_title(f"Adversarial Sign-Breakdown Point vs ρ (n={n})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results
# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    t0 = time.time()

    n = 25
    
    A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=int)  # Natural
    B = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=int)  # Inverse
    _ = lambda_corr(A, B)
    # 0) Bias (clean Gaussian)
    bias = bias_study(rhos=(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95), n=n, trials=50000)
    print("\n=== Bias (means & bias vs true ρ; plus bias vs theoretical Expectation) ===")
    for row in bias:
        # round floats for compact display
        pretty = {k: (round(v,4) if isinstance(v, float) else v) for k,v in row.items()}
        print(pretty)

    # Bias plots
    plot_bias_means(bias)   # mean(estimator) vs true ρ
    plot_bias_curves(bias)  # bias (mean − ρ) vs ρ

    # 1) Efficiency (clean Gaussian)
    eff = efficiency_study(rhos=(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95), n=n, trials=50000)
    print("\n=== Efficiency (variance; relative efficiency vs Pearson) ===")
    for row in eff:
        print({k: (round(v,4) if isinstance(v, float) else v) for k,v in row.items()})
    plot_efficiency_table(eff)

    # 2) Accuracy (MSE/RMSE vs true ρ)
    acc = accuracy_study(rhos=(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95), n=n, trials=50000)
    print("\n=== Accuracy (MSE and RMSE vs true ρ) ===")
    for row in acc:
        pretty = {k: (round(v,4) if isinstance(v, float) else v) for k,v in row.items()}
        print(pretty)

    # Plots (choose one)
    plot_accuracy_table(acc, metric="RMSE")
    #plot_accuracy_table(acc, metric="MSE")

    # 2) Robustness (vertical outliers)
    ver = "Vertical"
    rob = robustness_study(n=n, rho=0.5, frac_list=(0,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6, 0.7), 
                           trials=50000, ver=ver)
    print("\n=== Robustness (median & IQR across trials): Increasing Vertical Contamination ===")
    for row in rob:
        print({k: (round(v,4) if isinstance(v, float) else v) for k,v in row.items()})
    plot_robustness_table(rob, ver)

    ver = "Leverage"
    rob = robustness_study(n=n, rho=0.5, frac_list=(0,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7), 
                           trials=50000, ver=ver)
    print("\n=== Robustness (median & IQR across trials): Increasing Leverage Contamination ===")
    for row in rob:
        print({k: (round(v,4) if isinstance(v, float) else v) for k,v in row.items()})
    plot_robustness_table(rob, ver)
    
    ver = "Uniform"
    rob = robustness_study(n=n, rho=0.5, frac_list=(0,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7), 
                           trials=50000, ver=ver)
    print("\n=== Robustness (median & IQR across trials): Increasing Uniform Contamination ===")
    for row in rob:
        print({k: (round(v,4) if isinstance(v, float) else v) for k,v in row.items()})
    plot_robustness_table(rob, ver)
    
    # 3) Null calibration (independence)
    null = null_calibration(n=n, trials=50000)
    print("\n=== Null calibration (independence): mean, sd, q95, q99 ===")
    for k, v in null.items():
        print(k, {kk: round(vv,4) for kk,vv in v.items()})

    # 4) Pow
    alpha = 0.05
    power = power_study(n=n, rhos=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0), alpha=alpha, trials=1000000)
    print("\n=== Power (rejection rate at alpha=0.05) ===")
    for row in power:
        print({k: (round(v,4) if isinstance(v, float) else v) for k,v in row.items()})

    plot_power_table(power)
    
    # Type I Error
    # Example usage:
    print(' ')
    print('=== Type I Error ===')
    res = size_study(n=n, alpha=0.05, trials=50000)
    for row in res:
        lo, hi = row["ci95"]
        print(f'{row["method"]}: hat α={row["hat_alpha"]:.3f}  CI95=({lo:.3f},{hi:.3f})  n={row["n"]}  trials={row["trials"]}')

    # Usage:
    rhos_breakdown = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    breakdown_results = plot_breakdown_study_parallel(rhos_breakdown, n=n, trials=50000)
    
    # Print results table
    print("\n=== Sign-Breakdown Results ===")
    for i, rho in enumerate(breakdown_results['rhos']):
        print(f"ρ={rho:.1f}: Pearson={breakdown_results['Pearson'][i]}, "
              f"Spearman={breakdown_results['Spearman'][i]}, "
              f"Kendall={breakdown_results['Kendall'][i]}, "
              f"Λ_S={breakdown_results['Lambda_S'][i]}")

    print(f"\nDone in {time.time()-t0:.1f}s")
