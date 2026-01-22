#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 22:15:39 2025

@author: jplundquist
"""

import numpy as np
from numba import njit

@njit(cache=True, nogil=True, inline='always')
def _shuffle_int_inplace(a, n):
    # Fisher–Yates shuffle
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp

@njit(cache=True, nogil=True, inline='always')
def _apply_perm_float(src, perm, out, n):
    for i in range(n):
        out[i] = src[perm[i]]

@njit(cache=True, nogil=True)
def calibrate_perm_median_offsets(rx, ry, n, B=2000, seed=12345):
    """
    Permutation-median offsets for Lambda_yx and Lambda_xy.
    rx, ry should already be standardized ranks (your _std_ranks outputs).
    """
    np.random.seed(seed)

    perm = np.empty(n, np.int64)
    for i in range(n):
        perm[i] = i

    ry_perm = np.empty(n, np.float64)
    x_perm  = np.empty(n, np.float64)  # for Lambda_xy we need permuted "x" i.e. rx under same perm

    vals_yx = np.empty(B, np.float64)
    vals_xy = np.empty(B, np.float64)

    for b in range(B):
        _shuffle_int_inplace(perm, n)
        _apply_perm_float(ry, perm, ry_perm, n)
        _apply_perm_float(rx, perm, x_perm,  n)

        # Under null: random relabeling of y across fixed x (and vice versa for the reverse direction)
        vals_yx[b] = _mean_medians_slope(rx, ry_perm, n)  # Λ(Y|X) under perm
        vals_xy[b] = _mean_medians_slope(ry, x_perm,  n)  # Λ(X|Y) under perm

    # permutation medians
    m_yx = _median_inplace(vals_yx, B)
    m_xy = _median_inplace(vals_xy, B)
    return m_yx, m_xy

@njit(cache=True, nogil=True)
def lambda_stats_perm_median_centered(rx, ry, n, B=2000, seed=12345):
    """
    Returns centered Lambda_s, Lambda_yx, Lambda_xy where centering makes
    the permutation-median (null) approximately 0 for each asymmetric component.
    """
    # observed
    Lambda_yx = _mean_medians_slope(rx, ry, n)
    Lambda_xy = _mean_medians_slope(ry, rx, n)

    # null offsets
    m_yx, m_xy = calibrate_perm_median_offsets(rx, ry, n, B, seed)

    # centered asymmetrics
    Lyx = Lambda_yx - m_yx
    Lxy = Lambda_xy - m_xy

    # symmetrize (your logic)
    prod = Lyx * Lxy
    if (not np.isfinite(prod)) or (prod <= 0.0):
        Lambda_s = 0.0
    else:
        Lambda_s = np.sign(Lyx) * np.sqrt(prod)
        if Lambda_s > 1.0:
            Lambda_s = 1.0
        elif Lambda_s < -1.0:
            Lambda_s = -1.0

    return Lambda_s, Lyx, Lxy, m_yx, m_xy