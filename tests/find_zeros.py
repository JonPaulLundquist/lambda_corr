#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 02:50:06 2026

@author: jplundquist
"""

import math
import numpy as np
from numba import njit, prange
from tqdm import tqdm
from lambda_corr import lambda_corr_nb

@njit(cache=True, nogil=True, inline='always')
def draw_bivariate_normal_numba(n, rho):
    x = np.random.standard_normal(n)
    z = np.random.standard_normal(n)
    c = math.sqrt(max(1e-12, 1.0 - rho * rho))
    y = rho * x + c * z
    return x, y

@njit(cache=True, nogil=True, parallel=True)
def compute_lambda0_counts_block(N, batch, rho, seed):
    """
    Return counts for this block instead of samples.
    Counts are exact comparisons: == 0.0 and abs()==1.0
    """
    np.random.seed(seed)

    c0_s = 0
    c1_s = 0

    c0_xy = 0
    c1_xy = 0

    c0_yx = 0
    c1_yx = 0

    for i in prange(batch):
        x, y = draw_bivariate_normal_numba(N, rho)
        Lam_s, _, Lam_xy, _, Lam_yx, _, _ = lambda_corr_nb(x, y, N, pvals=False)

        # Lambda_s
        if Lam_s == 0.0:
            c0_s += 1
        if abs(Lam_s) == 1.0:
            c1_s += 1

        # Lambda_xy
        if Lam_xy == 0.0:
            c0_xy += 1
        if abs(Lam_xy) == 1.0:
            c1_xy += 1

        # Lambda_yx
        if Lam_yx == 0.0:
            c0_yx += 1
        if abs(Lam_yx) == 1.0:
            c1_yx += 1

    return c0_s, c1_s, c0_xy, c1_xy, c0_yx, c1_yx

def compute_lambda0_counts(N, total, rho=0.0, batch=200_000, seed=12345, heartbeat=True):
    """
    Stream blocks and accumulate counts only.
    Prints running estimates after each block.
    """
    n_batches = total // batch
    rem = total % batch

    # totals
    T = 0

    C0_s = C1_s = 0
    C0_xy = C1_xy = 0
    C0_yx = C1_yx = 0

    cur_seed = seed

    with tqdm(total=total, unit="samples", desc=f"Λ₀ counts (N={N})") as bar:
        for _ in range(n_batches):
            c0s, c1s, c0xy, c1xy, c0yx, c1yx = compute_lambda0_counts_block(N, batch, rho, cur_seed)

            T += batch
            C0_s  += c0s;  C1_s  += c1s
            C0_xy += c0xy; C1_xy += c1xy
            C0_yx += c0yx; C1_yx += c1yx

            cur_seed += 1
            bar.update(batch)

            if heartbeat:
                # running estimates
                p0s  = C0_s / T
                p1s  = C1_s / T
                p0xy = C0_xy / T
                p1xy = C1_xy / T
                p0yx = C0_yx / T
                p1yx = C1_yx / T

                print(
                    f"T={T:,}  "
                    f"p0_s={p0s:.6g} p1_s={p1s:.6g}  "
                    f"p0_xy={p0xy:.6g} p1_xy={p1xy:.6g}  "
                    f"p0_yx={p0yx:.6g} p1_yx={p1yx:.6g}",
                    flush=True
                )

        if rem > 0:
            c0s, c1s, c0xy, c1xy, c0yx, c1yx = compute_lambda0_counts_block(N, rem, rho, cur_seed)

            T += rem
            C0_s  += c0s;  C1_s  += c1s
            C0_xy += c0xy; C1_xy += c1xy
            C0_yx += c0yx; C1_yx += c1yx

            bar.update(rem)

            if heartbeat:
                p0s  = C0_s / T
                p1s  = C1_s / T
                p0xy = C0_xy / T
                p1xy = C1_xy / T
                p0yx = C0_yx / T
                p1yx = C1_yx / T
                print(
                    f"T={T:,}  "
                    f"p0_s={p0s:.6g} p1_s={p1s:.6g}  "
                    f"p0_xy={p0xy:.6g} p1_xy={p1xy:.6g}  "
                    f"p0_yx={p0yx:.6g} p1_yx={p1yx:.6g}",
                    flush=True
                )

    print("\a", end="", flush=True)
    return {
        "total": T,
        "C0_s": C0_s, "C1_s": C1_s,
        "C0_xy": C0_xy, "C1_xy": C1_xy,
        "C0_yx": C0_yx, "C1_yx": C1_yx,
    }

res = compute_lambda0_counts(N=5, total=5_000_000_000, rho=0.0, batch=200_000, seed=12345)
print(res)