#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Tue Dec  9 14:54:51 2025

@author: Jon Paul Lundquist
"""
from numba import njit
import numpy as np

#Mean-Mean (outer-inner) is just the same as mean of all pairwise slopes (ave_slope with mean)
#aka Theil-sen with mean instead of median. Median-median is Seigel in rank space.
#NOT OPTIMIZED SIMPLIFIED CODE
@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _mean_medians_slope(rx, ry, n):
    """Mean over i of the median slopes (y_j - y_i) / (x_j - x_i), skipping verticals."""

    bi = np.empty(n)
    for i in range(n):
        dx = rx - rx[i]
        m = dx != 0.0
        if not np.any(m):
            bi[i] = np.nan
            continue
        bi[i] = np.nanmean((ry[m] - ry[i]) / dx[m])
    return float(np.nanmedian(bi)) 

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _ave_slope(rx, ry, n, use_median=True):
    """
    median/mean of all pairwise slopes in rank space:
      stat = mean_or_median_{i<j, rx[j]!=rx[i]} ( (ry[j]-ry[i]) / (rx[j]-rx[i]) )

    Allocates C(n,2) then uses only the filled prefix.
    Returns NaN if no valid slopes exist (all dx == 0).
    """
 
    mcap = n * (n - 1) // 2                # upper bound = total pairs
    buf = np.empty(mcap, dtype=np.float64)  # over-allocate once
    k = 0

    for i in range(n - 1):
        xi = rx[i]; yi = ry[i]
        for j in range(i + 1, n):
            dx = rx[j] - xi
            if dx != 0.0:
                s = (ry[j] - yi) / dx
                # ranks should be finite, but keep a guard:
                if np.isfinite(s):
                    buf[k] = s
                    k += 1

    if k == 0:
        return np.nan

    arr = buf[:k]  # use only the filled portion
    return float(np.median(arr) if use_median else np.mean(arr))
