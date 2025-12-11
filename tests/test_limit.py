#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Sat Oct 11 20:11:14 2025

@author: Jon Paul Lundquist
"""

import numpy as np
import math
from scipy.stats import jarque_bera, normaltest, anderson, kstest, norm, skew, kurtosis
from lambda_corr import lambda_corr


# ---- helper: draw BVN ----
def draw_bivariate_normal(n, rho, rng):
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = rho * x + math.sqrt(max(1e-12, 1 - rho**2)) * z
    return x, y

# ---- main study ----
def gaussianity_study(stat_fn, rhos=(0.0, 0.3, 0.7), n_list=(30, 60, 120, 240), trials=3000, seed=0):
    """
    For each (rho, n):
      - simulate 'trials' datasets
      - compute T_i = stat_fn(x,y)
      - standardize Z_i = (T_i - mean(T))/std(T)
      - report shape tests (JB, D’Agostino, AD, KS), skew, kurt, sd, sd*sqrt(n)
    Returns: list of dict rows.
    """
    rng = np.random.default_rng(seed)
    out = []
    for rho in rhos:
        for n in n_list:
            Ts = np.empty(trials, dtype=np.float64)
            for t in range(trials):
                x, y = draw_bivariate_normal(n, rho, rng)
                Ts[t] = float(stat_fn(x, y))
            m = float(np.mean(Ts))
            s = float(np.std(Ts, ddof=1)) or 1.0
            Z = (Ts - m) / s  # standardize (empirical)

            # Moments
            skew = float(np.mean(((Z - 0.0))**3))
            kurt_excess = float(np.mean(Z**4) - 3.0)

            # Tests (use with care: with huge trials they reject tiny deviations)
            jb_stat, jb_p = jarque_bera(Z)
            dp_stat, dp_p = normaltest(Z)  # D’Agostino–Pearson
            ad = anderson(Z, dist='norm')  # critical values depend on n; returns statistic & critvals
            ks_stat, ks_p = kstest(Z, 'norm')  # vs N(0,1) since already standardized

            row = dict(
                rho=rho, n=n, trials=trials,
                mean=float(m), sd=float(s),
                sd_sqrt_n=float(s * math.sqrt(n)),  # should stabilize if Var ~ c/n
                skew=skew, kurt_excess=kurt_excess,
                JB_stat=float(jb_stat), JB_p=float(jb_p),
                DP_stat=float(dp_stat), DP_p=float(dp_p),
                AD_stat=float(ad.statistic),
                AD_crit=list(map(float, ad.critical_values)), AD_sig=list(map(float, ad.significance_level)),
                KS_stat=float(ks_stat), KS_p=float(ks_p)
            )
            out.append(row)
    return out

# ---- Q–Q plot for a chosen (rho, n) ----
import matplotlib.pyplot as plt

def qq_plot_Z(stat_fn, rho=0.0, n=120, trials=5000, seed=0, title=None):
    rng = np.random.default_rng(seed)
    Ts = np.empty(trials, dtype=np.float64)
    for t in range(trials):
        x, y = draw_bivariate_normal(n, rho, rng)
        Ts[t] = float(stat_fn(x, y))
    m = float(np.mean(Ts))
    s = float(np.std(Ts, ddof=1) or 1.0)
    Z = (Ts - m) / s

    Zs = np.sort(Z)
    qs = np.linspace(0.5/trials, 1-0.5/trials, trials)
    qN = norm.ppf(qs)

    plt.figure(figsize=(5.5,5.5))
    plt.plot(qN, Zs, ".", markersize=2, label="empirical")
    lo, hi = np.percentile(Z, [1,99])
    span = max(abs(lo), abs(hi))
    xs = np.linspace(-span, span, 200)
    plt.plot(xs, xs, "k--", alpha=0.6, label="y=x (N(0,1))")
    plt.xlabel("Theoretical quantiles N(0,1)")
    plt.ylabel("Empirical quantiles of Z")
    plt.title(title or f"Q–Q of standardized statistic (rho={rho}, n={n})")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---- Example bindings for your Lambda measures ----
# Symmetric:
def stat_lambda_S(x, y):
    # returns Λ_S only (no pvals)
    return lambda_corr(x, y, pvals=False)[0]

# Directional components:
def stat_lambda_YX(x, y):
    # returns Λ(Y|X)
    _, _, lam_xy, _, lam_yx, _, _ = lambda_corr(x, y, pvals=False)
    return lam_yx

def stat_lambda_XY(x, y):
    # returns Λ(X|Y)
    _, _, lam_xy, _, lam_yx, _, _ = lambda_corr(x, y, pvals=False)
    return lam_xy

# choose settings
rhos   = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
n_list = (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 250, 300)
trials = 10000

rows = gaussianity_study(stat_lambda_S, rhos=rhos, n_list=n_list, trials=trials, seed=0)

# quick, readable summary per (rho,n)
#What to look for

    #sd*sqrt(n) stabilizing as n grows → variance ~ c/n.

    #skew → 0, kurt_excess → 0.

    #Normality p-values (JB/DP/KS) not systematically tiny (with 3k trials they’re strict; trends matter).
for r in rows:
    print(
        f"rho={r['rho']:.1f} n={r['n']:>3}  "
        f"mean={r['mean']:+.4f} sd={r['sd']:.4f}  sd*sqrt(n)={r['sd_sqrt_n']:.4f}  "
        f"skew={r['skew']:+.3f} kurt_ex={r['kurt_excess']:+.3f}  "
        f"JB p={r['JB_p']:.3f}  DP p={r['DP_p']:.3f}  KS p={r['KS_p']:.3f}"
    )
    
# symmetric Λ_S at rho=0.3, n=120
qq_plot_Z(stat_lambda_S, rho=0.3, n=120, trials=5000, seed=1,
          title="Λ_S: Q–Q vs N(0,1), rho=0.3, n=120")

# try a directional component too
qq_plot_Z(stat_lambda_YX, rho=0.3, n=120, trials=5000, seed=2,
          title="Λ(Y|X): Q–Q vs N(0,1), rho=0.3, n=120")

rows_rho07 = gaussianity_study(stat_lambda_S, rhos=(0.7,), n_list=(20, 40, 80, 160, 320, 640, 1280), trials=3000, seed=3)
print("\n√n scaling at ρ=0.7:")
for r in rows_rho07:
    print(f"n={r['n']:>3}  sd={r['sd']:.5f}  sd*sqrt(n)={r['sd_sqrt_n']:.5f}")
    
# empirical rate-of-convergence check
skews = [r['skew'] for r in rows if r['rho']==0.4]
ns = np.array([r['n'] for r in rows if r['rho']==0.4])
plt.loglog(ns, np.abs(skews), '-o'); plt.xlabel('n'); plt.ylabel('|skew|');
