#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 10:21:37 2025

@author: jplundquist
"""

import numpy as np
from scipy.special import betainc  # regularized incomplete beta I_x(a,b)
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

def apply_k_disjoint_swaps(y, K, rng):
    """
    Apply K disjoint random swaps (transpositions) to permutation y in-place.
    Chooses 2K distinct indices, pairs them randomly, swaps within each pair.
    """
    n = y.size
    if K <= 0:
        return y
    K = min(K, n // 2)
    idx = rng.choice(n, size=2*K, replace=False)
    rng.shuffle(idx)
    for a, b in idx.reshape(K, 2):
        y[a], y[b] = y[b], y[a]
    return y

def apply_m_out_of_place(y, m, rng):
    """
    Randomly permute exactly m chosen positions of y.
    m can be odd or even.
    """
    n = y.size
    if m <= 1:
        return y
    m = min(m, n)
    idx = rng.choice(n, size=m, replace=False)
    vals = y[idx].copy()
    rng.shuffle(vals)
    y[idx] = vals
    return y

def apply_m_out_of_place_no_fixed(y, m, rng, max_tries=50):
    """
    Choose m indices and permute their values so that none stays in its original
    selected position (derangement on the selected positions).
    Falls back to a single rotation if rejection fails.
    """
    n = y.size
    if m <= 1:
        return y
    m = min(m, n)

    idx = rng.choice(n, size=m, replace=False)
    vals = y[idx].copy()

    # Try to find a shuffle with no fixed points relative to original vals
    for _ in range(max_tries):
        perm = rng.permutation(m)
        if np.all(perm != np.arange(m)):   # no fixed points
            y[idx] = vals[perm]
            return y

    # Fallback: deterministic derangement by rotation (always works if m>1)
    y[idx] = np.roll(vals, 1)
    return y

# ---------- helpers ----------
def _median(a):
    """Median: for even length, average middle two."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    m = a.size
    if m == 0:
        return np.nan
    a.sort()
    mid = m // 2
    if m % 2 == 1:
        return a[mid]
    return 0.5 * (a[mid - 1] + a[mid])

def harrell_davis_quantile(a, q=0.5):
    """
    Harrell–Davis quantile estimator.
    Returns a weighted sum of order statistics with Beta CDF difference weights.
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    m = a.size
    if m == 0:
        return np.nan
    a.sort()
    if m == 1:
        return a[0]

    # Beta parameters
    alpha = (m + 1) * q
    beta  = (m + 1) * (1.0 - q)

    # weight for order statistic i (1..m):
    # w_i = P((i-1)/m < U <= i/m), U~Beta(alpha,beta)
    i = np.arange(1, m+1, dtype=float)
    x0 = (i - 1.0) / m
    x1 = i / m
    w = betainc(alpha, beta, x1) - betainc(alpha, beta, x0)

    # numerical cleanup
    w = np.maximum(w, 0.0)
    w /= w.sum()

    return float(np.dot(w, a))

def harrell_davis_median(a):
    return harrell_davis_quantile(a, q=0.5)

def kendall_tau_from_ranks(ry):
    """
    Kendall tau-a between x=[1..n] and y ranks ry (permutation, no ties).
    Equivalent to 1 - 4*inv/(n(n-1)), where inv = inversion count of ry.
    O(n^2) inversion count for simplicity.
    """
    n = ry.size
    inv = 0
    for i in range(n-1):
        inv += np.sum(ry[i+1:] < ry[i])
    return 1.0 - 4.0*inv/(n*(n-1))

def spearman_rho_from_ranks(ry):
    """Spearman rho between x=[1..n] and y ranks ry (no ties)."""
    n = ry.size
    x = np.arange(1, n+1, dtype=float)
    y = ry.astype(float)
    d = x - y
    return 1.0 - (6.0*np.sum(d*d))/(n*(n*n - 1.0))

# ---------- your Lambda-like statistic ----------
def mean_HDmedian_slopes(rx, ry):
    """
    Given rank vectors rx, ry (1..n permutations), compute:
      M_xy = mean_i median_{j!=i} (ry[j]-ry[i])/(rx[j]-rx[i])
    using your even-median rule.
    Pairs with zero denom (ties in rx) are skipped (not expected for perms).
    """
    n = rx.size
    total = 0.0
    slopes = np.empty(n - 1, dtype=float)
    for i in range(n):
        t = 0
        for j in range(n):
            if j == i:
                continue
            denom = rx[j] - rx[i]
            if denom == 0:
                continue
            slopes[t] = (ry[j] - ry[i]) / denom
            t += 1
        total += harrell_davis_median(slopes[:t])
    return total / n

def mean_adaptive_median_slopes(rx, ry):
    n = rx.size
    slopes = np.empty(n - 1, dtype=float)
    med = np.empty(n, dtype=float)
    med_hd = np.empty(n, dtype=float)

    for i in range(n):
        t = 0
        for j in range(n):
            if j == i:
                continue
            denom = rx[j] - rx[i]
            if denom == 0:
                continue
            slopes[t] = (ry[j] - ry[i]) / denom
            t += 1
        med[i] = _median(slopes[:t])
        med_hd[i] = harrell_davis_median(slopes[:t])

    # start with classical medians
    used_hd = np.zeros(n, dtype=bool)
    L = med.mean()

    # only intervene if needed
    if abs(L) <= 1.0:
        return float(L)

    # iterate at most n times: each step can flip at most one i to HD
    for _ in range(n):
        if abs(L) <= 1.0:
            break

        # choose candidate index based on sign
        if L > 1.0:
            b = int(np.argmax(med))
        else:  # L < -1
            b = int(np.argmin(med))

        if used_hd[b]:
            # we're stuck: can't improve further with this rule
            break

        # only accept switch if it reduces |L|
        old = med[b]
        new = med_hd[b]
        L_new = L + (new - old) / n

        if abs(L_new) >= abs(L):
            # switching doesn't help; try next best by temporarily masking
            # (simple fallback: mark as used and continue)
            used_hd[b] = True
            continue

        med[b] = new
        used_hd[b] = True
        L = L_new

    return float(L)

def mean_median_slopes(rx, ry):
    """
    Given rank vectors rx, ry (1..n permutations), compute:
      M_xy = mean_i median_{j!=i} (ry[j]-ry[i])/(rx[j]-rx[i])
    using your even-median rule.
    Pairs with zero denom (ties in rx) are skipped (not expected for perms).
    """
    n = rx.size
    total = 0.0
    slopes = np.empty(n - 1, dtype=float)
    for i in range(n):
        t = 0
        for j in range(n):
            if j == i:
                continue
            denom = rx[j] - rx[i]
            if denom == 0:
                continue
            slopes[t] = (ry[j] - ry[i]) / denom
            t += 1
        total += _median(slopes[:t])
    return total / n

def lambda_raw(rx, ry):
    """
    Compute your symmetric raw statistic:
      M_xy = mean_of_point_median_slopes(ranks(x), ranks(y))
      M_yx = mean_of_point_median_slopes(ranks(y), ranks(x))
      Lambda_raw = sign(M_yx)*sqrt(|M_xy*M_yx|)  (if sign_from="yx")
    """
    Mxy = mean_median_slopes(rx, ry)
    Myx = mean_median_slopes(ry, rx)
    if Mxy == 0.0 or Myx == 0.0:
        Lam = 0.0
    elif np.sign(Mxy) != np.sign(Myx):
        Lam =  0.0
    else:
        mag = np.sqrt(abs(Mxy * Myx))
        Lam = np.sign(Myx) * mag
    return Lam, Mxy, Myx

def lambda_raw_HD(rx, ry):
    """
    Compute your symmetric raw statistic:
      M_xy = mean_of_point_median_slopes(ranks(x), ranks(y))
      M_yx = mean_of_point_median_slopes(ranks(y), ranks(x))
      Lambda_raw = sign(M_yx)*sqrt(|M_xy*M_yx|)  (if sign_from="yx")
    """
    Mxy = mean_adaptive_median_slopes(rx, ry)
    Myx = mean_adaptive_median_slopes(ry, rx)
    if Mxy == 0.0 or Myx == 0.0:
        Lam = 0.0
    elif np.sign(Mxy) != np.sign(Myx):
        Lam =  0.0
    else:
        mag = np.sqrt(abs(Mxy * Myx))
        Lam = np.sign(Myx) * mag
    return Lam, Mxy, Myx

def foldback_reciprocal(L_yx,L_xy):
    """sign(L) * min(|L|, 1/|L|) with L=0 handled."""
    if np.sign(L_yx) != np.sign(L_xy):
        return 0
    if (L_yx > 1.0) or (L_xy > 1.0):
        if L_yx > 1.0:
            L_yx = 1/L_yx
        if L_xy > 1.0:
            L_xy = 1/L_xy
        
    return np.sign(L_yx)*np.sqrt(np.abs(L_yx*L_xy))

# ---------- experiment ----------
def run_test_swaps(n=25, trials=20000, Kmax=1, seed=0):
    """
    Start with perfect monotone ranks:
      x = [1..n], y = [1..n]
    Then apply exactly K random swaps to y.
    Compute Lambda_raw, Kendall_tau, Spearman_rho.
    Among cases with |Lambda_raw|>1, test whether tau/rho decrease as |Lambda_raw| increases.
    """
    rng = np.random.default_rng(seed)

    Lam = np.empty(trials, dtype=float)
    Lam_yx = np.empty(trials, dtype=float)
    Lam_xy = np.empty(trials, dtype=float)
    Lam_fold = np.empty(trials, dtype=float)
    Lam_HD = np.empty(trials, dtype=float)
    absLam = np.empty(trials, dtype=float)
    tau = np.empty(trials, dtype=float)
    rho = np.empty(trials, dtype=float)
    absLam_fold = np.empty(trials, dtype=float)
    Ks = np.empty(trials,dtype=int)
    x = np.arange(1, n+1, dtype=float)-1
    y = np.nan
    Lam_examp = 1
    for t in range(trials):
        print(f"Trial: {t}")
        y = np.arange(1, n+1, dtype=int)-1
        #K = np.random.randint(3,Kmax+1)
        K = int(rng.integers(3, Kmax + 1))   # K can be 3..Kmax
        Ks[t] = K
        apply_m_out_of_place_no_fixed(y, K, rng)

        # Use your existing function; it ranks internally, but ranks are already 1..n.
        # If you want to avoid reranking, I can show a lambda_raw_from_ranks(rx, ry).
        Lam[t], Lam_yx[t], Lam_xy[t] = lambda_raw(x, y.astype(float))
        Lam_HD[t], _, _ = lambda_raw_HD(x, y.astype(float))
        absLam[t] = abs(Lam_HD[t])
        Lam_fold[t] = foldback_reciprocal(Lam_yx[t], Lam_xy[t])
        absLam_fold[t] = abs(Lam_fold[t])
        tau[t] = kendall_tau_from_ranks(y)
        rho[t] = spearman_rho_from_ranks(y)
        if absLam[t]>Lam_examp:
            Lam_examp = absLam[t]
            y_examp = y

    mask = absLam > 1.0
    k = int(np.sum(mask))
    d = np.abs(Lam - Lam_HD)
    print(f"min HD diff {d.min():.6g}, max HD diff {d.max():.6g}, mean HD diff {d.mean():.6g}")
    print(f"count(d>1e-12) = {np.sum(d > 1e-12)}")
    print(f"n={n}, trials={trials}, K(adj swaps)={Kmax}")
    print(f"HD overshoot count (|Lambda_raw|>1): {k} ({k/trials:.4%})")
    if k>1:
        print(f"Overshoot Min. {min(Ks[mask])} Overshoot Max. {max(Ks[mask])}")
    
    if k < 50:
        print("Too few overshoot cases; increase trials or adjust K/trials.")
        return Lam, Lam_yx, Lam_xy, Lam_fold, Lam_HD, tau, rho, y_examp

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
        return float(np.dot(a, b) / denom) if denom > 0 else np.nan
    
    c_tau = corr(absLam[mask], abs(tau[mask]))
    c_rho = corr(absLam[mask], abs(rho[mask]))
    print(f"corr(|Lambda_raw|, Kendall_tau) within overshoot: {c_tau:+.4f}")
    print(f"corr(|Lambda_raw|, Spearman_rho) within overshoot: {c_rho:+.4f}")

    q = np.quantile(absLam[mask], [0.0, 0.25, 0.5, 0.75, 1.0])
    print("\nBinned by |Lambda_raw| (overshoot only):")
    for a0, a1 in zip(q[:-1], q[1:]):
        if a1 != q[-1]:
            sel = mask & (absLam >= a0) & (absLam < a1)
        else:
            sel = mask & (absLam >= a0) & (absLam <= a1)
        if np.sum(sel) == 0:
            continue
        print(f"  bin [{a0:.6f}, {a1:.6f}): "
              f"count={np.sum(sel):5d}, "
              f"mean tau={tau[sel].mean():+.4f}, mean rho={rho[sel].mean():+.4f}")

    c_rho_fold = corr(absLam_fold[mask], rho[mask])
    c_tau_fold = corr(absLam_fold[mask], tau[mask])
    print(f"\nOvershoot corr(|Lambda_fold|, Spearman_rho): {c_rho_fold:+.4f}")
    print(f"Overshoot corr(|Lambda_fold|, Kendall_tau): {c_tau_fold:+.4f}")

    print(f"Min abs overshoot: {np.min(absLam[mask]):.6f}   Max abs overshoot: {np.max(absLam[mask]):.6f}")
    print(f"Min Lambda(overshoot): {np.min(Lam_HD[mask]):+.6f} Max Lambda(overshoot): {np.max(Lam_HD[mask]):+.6f}")

    plt.figure()
    ind1 = Lam>1
    Lmin = min(Lam_fold[ind1])
    #ind2 = Lam_fold>=0.8
    ind2 = Lam_fold>=Lmin
    ind3 = (Lam>=Lmin)
    bins = np.arange(np.round(Lmin*100)/100,1.015,0.0025)
    
    mean1, edges1, _ = binned_statistic(Lam[ind1], tau[ind1], statistic="mean", bins=7)
    
    # count in bins
    count1, _, _ = binned_statistic(Lam[ind1], tau[ind1], statistic="count", bins=7)
    
    # std in bins
    std1, _, _ = binned_statistic(Lam[ind1], tau[ind1], statistic="std", bins=7)
    
    centers1 = 0.5 * (edges1[:-1] + edges1[1:])
    sem1 = std1 / np.sqrt(count1)
    
    mean2, edges2, _ = binned_statistic(Lam_fold[ind1], tau[ind1], statistic="mean", bins=7)
    
    # count in bins
    count2, _, _ = binned_statistic(Lam_fold[ind1], tau[ind1], statistic="count", bins=7)
    
    # std in bins
    std2, _, _ = binned_statistic(Lam_fold[ind1], tau[ind1], statistic="std", bins=7)
    
    centers2 = 0.5 * (edges2[:-1] + edges2[1:])
    sem2 = std2 / np.sqrt(count2)
    
    mean3, edges3, _ = binned_statistic(Lam_fold[ind2], tau[ind2], statistic="mean", bins=7)
    
    # count in bins
    count3, _, _ = binned_statistic(Lam_fold[ind2], tau[ind2], statistic="count", bins=7)
    
    # std in bins
    std3, _, _ = binned_statistic(Lam_fold[ind2], tau[ind2], statistic="std", bins=7)
    
    centers3 = 0.5 * (edges3[:-1] + edges3[1:])
    
    plt.errorbar(centers1, mean1, yerr=std1, fmt='s', capsize=3, label='$\Lambda_{raw}$',color='black')
    #plt.errorbar(centers3, mean3, yerr=std3, fmt='s', capsize=3, label='$\Lambda$',color='green')
    plt.errorbar(centers2, mean2, yerr=std2, fmt='s', capsize=3, label='$\Lambda_{folded}$',color='red')
    
    plt.xlabel(r"$\mathbf{\Lambda}$", fontweight='semibold')
    plt.ylabel(r"Kendall's $\mathbf{\langle \tau \rangle}$ with STD.", fontweight='semibold')
    plt.legend(fontsize=16, frameon=True, prop={'weight': 'semibold'})
    plt.show()
    
    bins = 50
    plt.figure()

    mean4, edges4, _ = binned_statistic(Lam, tau, statistic="mean", bins=bins)
    # count in bins
    count4, _, _ = binned_statistic(Lam, tau, statistic="count", bins=bins)

    # std in bins
    std4, _, _ = binned_statistic(Lam, tau, statistic="std", bins=bins)
    centers4 = 0.5 * (edges4[:-1] + edges4[1:])
        
    mean5, edges5, _ = binned_statistic(Lam_fold, tau, statistic="mean", bins=edges4)
    # count in bins
    count5, _, _ = binned_statistic(Lam_fold, tau, statistic="count", bins=edges4)

    # std in bins
    std5, _, _ = binned_statistic(Lam_fold, tau, statistic="std", bins=edges4)
    centers5 = 0.5 * (edges5[:-1] + edges5[1:])
    
    plt.errorbar(centers4, mean4, yerr=std4, fmt='s', capsize=3, label='$\Lambda_{raw}$',color='black')
    plt.errorbar(centers5, mean5, yerr=std5, fmt='s', capsize=3, label='$\Lambda_{folded}$',color='red')
    plt.xlabel(r"$\mathbf{\Lambda}$", fontweight='semibold')
    plt.ylabel(r"Kendall's $\mathbf{\langle \tau \rangle}$ with STD.", fontweight='semibold')
    plt.legend(fontsize=16, frameon=True, prop={'weight': 'semibold'})
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    plt.show()
    
    plt.figure()
    ind1 = Lam>1
    Lmin = min(Lam_fold[ind1])
    #ind2 = Lam_fold>=0.8
    ind2 = Lam_fold>=Lmin
    ind3 = (Lam>=Lmin)
    bins = np.arange(np.round(Lmin*100)/100,1.015,0.0025)
    
    mean1, edges1, _ = binned_statistic(Lam[ind1], rho[ind1], statistic="mean", bins=7)
    
    # count in bins
    count1, _, _ = binned_statistic(Lam[ind1], rho[ind3], statistic="count", bins=7)
    
    # std in bins
    std1, _, _ = binned_statistic(Lam[ind1], rho[ind1], statistic="std", bins=7)
    
    centers1 = 0.5 * (edges1[:-1] + edges1[1:])
    sem1 = std1 / np.sqrt(count1)
    
    mean2, edges2, _ = binned_statistic(Lam_fold[ind1], rho[ind1], statistic="mean", bins=7)
    
    # count in bins
    count2, _, _ = binned_statistic(Lam_fold[ind1], rho[ind1], statistic="count", bins=7)
    
    # std in bins
    std2, _, _ = binned_statistic(Lam_fold[ind1], rho[ind1], statistic="std", bins=7)
    
    centers2 = 0.5 * (edges2[:-1] + edges2[1:])
    sem2 = std2 / np.sqrt(count2)
    
    mean3, edges3, _ = binned_statistic(Lam_fold[ind2], rho[ind2], statistic="mean", bins=7)
    
    # count in bins
    count3, _, _ = binned_statistic(Lam_fold[ind2], rho[ind2], statistic="count", bins=7)
    
    # std in bins
    std3, _, _ = binned_statistic(Lam_fold[ind2], rho[ind2], statistic="std", bins=7)
    
    centers3 = 0.5 * (edges3[:-1] + edges3[1:])
    
    plt.errorbar(centers1, mean1, yerr=std1, fmt='s', capsize=3, label='$\Lambda_{raw}$',color='black')
    #plt.errorbar(centers3, mean3, yerr=std3, fmt='s', capsize=3, label='$\Lambda$',color='green')
    plt.errorbar(centers2, mean2, yerr=std2, fmt='s', capsize=3, label='$\Lambda_{folded}$',color='red')
    
    plt.xlabel(r"$\mathbf{\Lambda}$", fontweight='semibold')
    plt.ylabel(r"Spearman's $\mathbf{\langle \rho \rangle}$ with STD.", fontweight='semibold')
    plt.legend(fontsize=16, frameon=True, prop={'weight': 'semibold'})
    plt.show()
    
    bins = 50
    plt.figure()

    mean4, edges4, _ = binned_statistic(Lam, tau, statistic="mean", bins=bins)
    # count in bins
    count4, _, _ = binned_statistic(Lam, tau, statistic="count", bins=bins)

    # std in bins
    std4, _, _ = binned_statistic(Lam, tau, statistic="std", bins=bins)
    centers4 = 0.5 * (edges4[:-1] + edges4[1:])
        
    mean5, edges5, _ = binned_statistic(Lam_fold, tau, statistic="mean", bins=edges4)
    # count in bins
    count5, _, _ = binned_statistic(Lam_fold, tau, statistic="count", bins=edges4)

    # std in bins
    std5, _, _ = binned_statistic(Lam_fold, tau, statistic="std", bins=edges4)
    centers5 = 0.5 * (edges5[:-1] + edges5[1:])
    
    plt.errorbar(centers4, mean4, yerr=std4, fmt='s', capsize=3, label='$\Lambda_{raw}$',color='black')
    plt.errorbar(centers5, mean5, yerr=std5, fmt='s', capsize=3, label='$\Lambda_{folded}$',color='red')
    plt.xlabel(r"$\mathbf{\Lambda}$", fontweight='semibold')
    plt.ylabel(r"Spearman's $\mathbf{\langle \rho \rangle}$ with STD.", fontweight='semibold')
    plt.legend(fontsize=16, frameon=True, prop={'weight': 'semibold'})
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    plt.show()
    return Lam, Lam_yx, Lam_xy, Lam_fold, Lam_HD, tau, rho, y_examp