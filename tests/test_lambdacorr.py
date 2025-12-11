#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Wed Oct  8 20:25:46 2025

@author: Jon Paul Lundquist
"""

# Monte Carlo comparing correlations: Pearson r, Spearman rho_s, Kendall tau, 
# Theil–Sen rank-based slope correlation (TS-rank), and Repeated-Average Λ (Lambda).
# One figure per scenario; print numeric summaries too.

import numpy as np
import math
from scipy.stats import spearmanr, kendalltau, t as student_t
import matplotlib.pyplot as plt
from lambda_corr import lambda_corr

rng = np.random.default_rng(0)

def standardized_ranks(a):
    a = np.asarray(a, float)
    # average ranks for ties: do it by correcting equal blocks
    # Faster: use rankdata, but let's avoid pandas; scipy.rankdata is acceptable but we'll implement simple tie average.
    # We'll use scipy.stats.rankdata for correctness and brevity.
    from scipy.stats import rankdata
    r = rankdata(a, method='average')
    r = (r - r.mean()) / r.std(ddof=0)
    return r

def theil_sen_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size
    slopes = []
    for i in range(n-1):
        dx = x[i+1:] - x[i]
        dy = y[i+1:] - y[i]
        m = dx != 0.0
        if m.any():
            slopes.extend((dy[m] / dx[m]).tolist())
    if not slopes:
        return np.nan
    return float(np.median(np.array(slopes)))

def ts_rank_corr(x, y):
    rx = standardized_ranks(x)
    ry = standardized_ranks(y)
    b_yx = theil_sen_slope(rx, ry)
    b_xy = theil_sen_slope(ry, rx)
    prod = b_yx * b_xy
    if not np.isfinite(prod) or prod < 0:
        rts = 0.0
    else:
        rts = math.sqrt(prod)
    # sign from Kendall concordance on ranks
    dx = rx[:, None] - rx[None, :]
    dy = ry[:, None] - ry[None, :]
    iu = np.triu_indices(rx.size, 1)
    S = np.sign(dx[iu] * dy[iu]).sum()
    sgn = 1.0 if S >= 0 else -1.0
    return float(np.clip(sgn * rts, -1.0, 1.0))

def pearson_r(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = (x - x.mean()) / x.std(ddof=0)
    y = (y - y.mean()) / y.std(ddof=0)
    return float(np.mean(x*y))

def draw_gaussian(n, rho, rng):
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = rho * x + math.sqrt(1 - rho**2) * z
    return x, y

def contaminate_vertical(y, frac=0.05, strength=10.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = y.size
    k = max(1, int(frac * n))
    idx = rng.choice(n, size=k, replace=False)
    y2 = y.copy()
    y_std = y.std(ddof=0)
    y2[idx] = y2[idx] + strength * y_std * np.sign(rng.standard_normal(k))
    return y2

def discretize(a, bins=6):
    # quantize to given number of bins to induce ties
    edges = np.quantile(a, np.linspace(0, 1, bins+1))
    # ensure unique edges
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    idx = np.digitize(a, edges) - 1
    # map bins to midpoints
    mids = 0.5*(edges[:-1] + edges[1:])
    return mids[idx]

def run_scenario(name, gen_func, *, n=300, rho=0, trials=300, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    stats = {
        "pearson": [],
        "spearman": [],
        "kendall": [],
        "ts_rank": [],
        "l_rank": [],
    }
    for _ in range(trials):
        x, y = gen_func(n, rho, rng)
        stats["pearson"].append(pearson_r(x, y))
        stats["spearman"].append(spearmanr(x, y)[0])
        stats["kendall"].append(kendalltau(x, y)[0])
        stats["ts_rank"].append(ts_rank_corr(x, y))
        stats["l_rank"].append(lambda_corr(x, y)[0])
    # summarize
    summary = {}
    for k, arr in stats.items():
        a = np.array(arr, float)
        med = float(np.median(a))
        iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
        summary[k] = (med, iqr)
    return summary, stats

# Scenario generators
def gen_gaussian(n, rho, rng):
    return draw_gaussian(n, rho, rng)

def gen_gaussian_with_vertical_outliers(n, rho, rng):
    x, y = draw_gaussian(n, rho, rng)
    y2 = contaminate_vertical(y, frac=0.05, strength=10.0, rng=rng)
    return x, y2

def gen_gaussian_with_ties(n, rho, rng):
    x, y = draw_gaussian(n, rho, rng)
    # discretize y to induce ties
    y2 = discretize(y, bins=6)
    return x, y2

def gen_heavy_tail(n, rho, rng):
    # linear model with t3 noise replacing normal z
    x = rng.standard_normal(n)
    z = student_t.rvs(df=3, size=n, random_state=rng)
    z = (z - np.mean(z)) / np.std(z, ddof=0)
    y = rho * x + math.sqrt(1 - rho**2) * z
    return x, y

# Run scenarios
scenarios = [
    ("Gaussian ρ=0", gen_gaussian),
    ("Gaussian + 5% vertical outliers", gen_gaussian_with_vertical_outliers),
    ("Gaussian + ties (6 bins on y)", gen_gaussian_with_ties),
    ("Heavy-tail (t3 noise)", gen_heavy_tail),
]

all_results = []
for name, gf in scenarios:
    summary, stats = run_scenario(name, gf, n=100, rho=0.5, trials=300, rng=rng)
    all_results.append((name, summary, stats))

# Print summaries
print("Monte Carlo medians (IQR) over 300 trials, n=100, target ρ=0.5\n")
for name, summary, _ in all_results:
    print(f"== {name} ==")
    for k in ["pearson","spearman","kendall","ts_rank","l_rank"]:
        med, iqr = summary[k]
        print(f"  {k:9s}: median={med: .3f}   IQR={iqr: .3f}")
    print()

# Plot for each scenario: boxplot of the five statistics
labels_order = ["pearson","spearman","kendall","ts_rank","l_rank"]
for name, summary, stats in all_results:
    data = [np.array(stats[k], float) for k in labels_order]
    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=labels_order, showmeans=True)
    plt.axhline(0, linestyle='--')  # target Pearson rho
    plt.title(f"Correlation estimates — {name}")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()
