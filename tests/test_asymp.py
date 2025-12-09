#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Tue Dec  2 18:56:16 2025

@author: Jon Paul Lundquist
"""

import numpy as np
import math
import lambda_corr


def draw_bivariate_normal(n, rho, rng):
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = rho * x + math.sqrt(max(1e-12, 1 - rho**2)) * z
    return x, y

def compare_lambda_pvals(n=30, rho=0.0, M=2000, 
                         B_perm=10000, seed=12345,
                         alpha=0.05):
    """
    Monte Carlo comparison of permutation vs asymptotic p-values for Λ_s.

    n      : sample size
    rho    : true Pearson correlation for the generator
    M      : number of Monte Carlo replications
    B_perm : number of permutations in lambda_corr (ptype='perm')
    alpha  : significance level for rejection-rate comparison
    """

    rng = np.random.default_rng(seed)

    p_perm  = np.empty(M)
    p_asymp = np.empty(M)
    lambda_s_vals = np.empty(M)

    for m in range(M):
        x, y = draw_bivariate_normal(n, rho, rng)

        # Permutation p-values
        Lambda_s_perm, p_s_perm, *_ = lambda_corr(
            x, y,
            pvals=True,
            ptype="perm",
            B=B_perm,
            alt="two-sided"
        )

        # Asymptotic p-values
        Lambda_s_asymp, p_s_asymp, *_ = lambda_corr(
            x, y,
            pvals=True,
            ptype="asymp",
            alt="two-sided"
        )

        # Sanity check: statistic should match
        lambda_s_vals[m] = Lambda_s_perm
        p_perm[m]  = p_s_perm
        p_asymp[m] = p_s_asymp

    # Some simple diagnostics
    diff = p_asymp - p_perm

    # Rejection rates (type I error under rho=0, power otherwise)
    rej_perm  = np.mean(p_perm  < alpha)
    rej_asymp = np.mean(p_asymp < alpha)

    # Correlation between the two p-value sets
    num = np.sum((p_perm - p_perm.mean()) * (p_asymp - p_asymp.mean()))
    den = (np.sqrt(np.sum((p_perm - p_perm.mean())**2)) *
           np.sqrt(np.sum((p_asymp - p_asymp.mean())**2)))
    p_corr = num / den if den != 0 else np.nan

    summary = {
        "n": n,
        "rho": rho,
        "M": M,
        "B_perm": B_perm,
        "alpha": alpha,
        "mean_p_perm": float(p_perm.mean()),
        "mean_p_asymp": float(p_asymp.mean()),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "p_corr": float(p_corr),
        "rej_rate_perm": float(rej_perm),
        "rej_rate_asymp": float(rej_asymp),
    }

    return summary, p_perm, p_asymp, lambda_s_vals

summary, p_perm, p_asymp, lambda_s_vals = compare_lambda_pvals(
    n=30, rho=0.0, M=5000, B_perm=10000
)

print(summary)