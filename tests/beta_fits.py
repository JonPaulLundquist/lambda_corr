#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:24:51 2026

@author: Jon Paul Lundquist
"""

import numpy as np
import matplotlib.pyplot as plt
import gc
import math
from scipy.stats import beta

# del L0
# for _ in range(3):
#     gc.collect()
# N = 18
# L0, _, _ = compute_lambda0_numba(N, 1_000_000_000, rho=0.0)
# L0_mean, L0_var, L0_skew, L0_kurt = online_moments(L0) #uses far less memory for large arrays
# L0_sigma = math.sqrt(L0_var * N)   # ≈ constant ≈ sd*sqrt(n)
# for _ in range(3):
#     gc.collect()
# print(np.sum(L0==0)/L0.size)
# print(np.sum(np.abs(L0)==1)/L0.size)
# L0 = L0[:500_000_000]
# beta.fit(np.abs(L0[(L0!=0) & (np.abs(L0)!=1)]), floc=0.0, fscale=1.0)

    # #N = 950
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.029838691
    # p1 = 0.0
    # alpha = 1.5359283689755698
    # beta = 50.24456877336037

    # #N = 975
    #Beta distribution with 0 and 1 probabilities for N_MC = 1,000,000,000
    # p0 = 0.029816765
    # p1 = 0.0
    # alpha = 1.5362005077273875
    # beta = 50.932650989089765
    
N = np.array([10, 11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   
              22,   23,   24,   25,   26,   27,   28,   29,   30,   35,   40,   
              45,   50,   55,   60,   65,   70,   75,   80,   85,   90,   95,  
              100,  125,  150,  175,  200,  225,  250,  275,  300,  325,  350,  
              375,  400,  425,  450,  475,  500,  525,  550,  575,  600,  625,  
              650,  675,  700,  725,  750,  775,  800,  825,  850,  875,  900, 
              925, 950, 975, 1000], dtype=np.float64)

P0 = np.array([0.06776014109347443, 0.0708556798140131, 0.06714529554807333, 
               0.06600454715038048, 0.06496984457983962, 0.06238369283333333, 0.062640989, 
               0.059537863333333337, 0.0604504015, 0.057216893, 0.05846732416666667, 
               0.05526443416666667, 0.056689878, 0.05361363, 0.055135552, 0.052150197, 
               0.053692919, 0.050890785, 0.052419307, 0.049762158, 0.051266391, 
               0.047058179, 0.046875418, 0.043886314, 0.043944742, 0.041731068, 
               0.041834139, 0.040133117, 0.040235615, 0.038882284, 0.039020723, 
               0.037892605, 0.038032011, 0.03708275, 0.037203714, 0.035357447, 
               0.034638402, 0.033672204, 0.033266614, 0.032681559, 0.032427826, 
               0.032034477, 0.031861522, 0.031575871, 0.031441454, 0.031220126, 
               0.031163257, 0.030977032, 0.030888612, 0.030751585, 0.030699806, 
               0.030585179, 0.030540247, 0.030446717, 0.030377763, 0.030312046, 
               0.030270575, 0.030216366, 0.030182576, 0.030133224, 0.03011434, 
               0.030066833, 0.030032078, 0.029965395, 0.029968917, 0.029930739, 
               0.029911014, 0.029861657, 0.029838691, 0.029816765, 0.029800397], dtype=np.float64)

P1 = np.array([3.031305114638448e-05, 2.0041686708353374e-06, 6.513548180214847e-07, 
               3.789934345489901e-08, 1.0553085949911346e-08, 4.1753513975736197e-10, 
               1.4663436455764497e-10, 5.263047980134815e-12, 1.4366546569705612e-12, 
               4.1251147667560884e-14, 1.069422439233359e-14, 2.675229584544318e-16, 
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

ALPHA = np.array([1.4031180011417934, 1.3964874007690553, 1.4378036180809, 
                  1.4274144119465353, 1.4552545703790116, 1.4447437136748829, 
                  1.4671825739472821, 1.45608952243845, 1.4748300243343895, 
                  1.4644181779078853, 1.4805092620739457, 1.4702557385709127, 
                  1.4845023191449047, 1.4751097546757268, 1.4876743353571005, 
                  1.47888092293881, 1.4902126712917316, 1.481845624216896, 
                  1.4922291390943292, 1.4845170545731274, 1.4939833744022095, 
                  1.4898825870424968, 1.499302409292932, 1.4955990847250638, 
                  1.502465594762055, 1.4992361161278511, 1.5044662234906618, 
                  1.5022242332542404, 1.5062253835425243, 1.5042723097595319, 
                  1.5079053750994416, 1.50612090780228, 1.5089013366844017, 
                  1.5075254654395471, 1.5102234802411307, 1.5113284065657007, 
                  1.514566218290616, 1.51565912355074, 1.5181813633855017, 
                  1.518913532842462, 1.5205505305248728, 1.5214446822928052, 
                  1.5229574230485505, 1.5236559811756816, 1.5247232196563312, 
                  1.5254575297765316, 1.526318019911779, 1.526845897280529, 
                  1.5277077843012459, 1.528195807582924, 1.5288456636813765, 
                  1.5292162131170872, 1.5300201255867647, 1.530457233851254, 
                  1.5310201246651984, 1.5317665585255542, 1.5317090284693289, 
                  1.5322985276327077, 1.5326493081879098, 1.533454878326082, 
                  1.533373190827337, 1.5337987163954845, 1.5341584699361657, 
                  1.5344860867391108, 1.5349810276918976, 1.5354628204406753, 
                  1.5353554431348562, 1.5356060363460318, 1.5359283689755698,
                  1.5362005077273875, 1.5365810229894095], dtype=np.float64)

BETA = np.array([2.6579089754819853, 2.9759841690111215, 3.2336691391510604, 
                 3.518822685429833, 3.7485744560979195, 4.00762594005941, 
                 4.223033938955358, 4.45823438758898, 4.661919097481938, 
                 4.879683679160764, 5.0736720165789455, 5.275428822561992, 
                 5.461907884453, 5.651686059204632, 5.830734762026783, 
                 6.0102024965515985, 6.183505864613792, 6.352736078846976, 
                 6.5203056989847, 6.683121593579365, 6.845129043097732, 
                 7.602912186121273, 8.315653246239936, 8.969373209567118, 
                 9.602286385340967, 10.185169129999263, 10.761719585499176, 
                 11.29663472972513, 11.824783962631374, 12.319867078793214, 
                 12.81532905149589, 13.278011578898136, 13.7419513511831, 
                 14.17886825457339, 14.620807506130546, 16.631275961748585, 
                 18.46135081454609, 20.13302497097813, 21.701969321294357, 
                 23.163118691724332, 24.552094234696717, 25.870278062401255, 
                 27.136947451508497, 28.342616246520397, 29.507296214303995, 
                 30.631514126811364, 31.7192585034286, 32.768332687149446, 
                 33.79554550614419, 34.78506303697264, 35.754911133449475, 
                 36.698065581167064, 37.623570262759436, 38.52795788132418, 
                 39.411441943365325, 40.28258751199046, 41.11695548772955, 
                 41.95506071921157, 42.768932424746424, 43.58453326711583, 
                 44.35884279549922, 45.137388992535854, 45.898482432935275, 
                 46.65003909704631, 47.39322880772411, 48.12850686411679, 
                 48.8360834115559, 49.542915424547346, 50.24456877336037,
                 50.932650989089765, 51.61939556620012], dtype=np.float64)

# N_A = np.arange(11,31)
# P0_A = np.array([0.0708556798140131, 0.06714529554807333, 0.06600454715038048, 0.06496984457983962, 0.06238369283333333, 0.062640989,
#       0.059537863333333337, 0.0604504015, 0.057216893, 0.05846732416666667, 0.05526443416666667, 0.056689878, 0.05361363,
#       0.055135552, 0.052150197, 0.053692919, 0.050890785, 0.052419307, 0.049762158, 0.051266391])

# P1_A = np.array([2.0041686708353374e-06, 6.513548180214847e-07, 3.789934345489901e-08, 1.0553085949911346e-08, 4.1753513975736197e-10, 1.4663436455764497e-10,
#                5.263047980134815e-12, 1.4366546569705612e-12, 4.1251147667560884e-14, 1.069422439233359e-14, 1.762739072169017e-16,
#                0, 0, 0, 0, 0, 0, 0, 0, 0])

# ALPHA_A = np.array([1.3964874007690553, 1.4378036180809, 1.4274144119465353, 1.4552545703790116, 1.4447437136748829, 1.4671825739472821,
#                   1.45608952243845, 1.4748300243343895, 1.4644181779078853, 1.4805092620739457, 1.4702557385709127, 1.4845023191449047, 
#                   1.4751097546757268, 1.4876743353571005, 1.47888092293881, 1.4902126712917316, 1.481845624216896, 1.4922291390943292, 
#                   1.4845170545731274, 1.4939833744022095])
# BETA_A = np.array([2.9759841690111215, 3.2336691391510604, 3.518822685429833, 3.7485744560979195, 4.00762594005941, 4.223033938955358,
#                  4.45823438758898, 4.661919097481938, 4.879683679160764, 5.0736720165789455, 5.275428822561992, 5.461907884453,
#                  5.651686059204632, 5.830734762026783, 6.0102024965515985, 6.183505864613792, 6.352736078846976, 6.5203056989847,
#                  6.683121593579365, 6.845129043097732])
    
# -----------------------------
# Core helpers: linear-basis LOOCV
# -----------------------------
def _design_matrix(N, basis_funcs):
    N = np.asarray(N, dtype=float)
    return np.column_stack([f(N) for f in basis_funcs])

def _fit_predict_linear_basis(N_train, y_train, N_test, basis_funcs):
    X = _design_matrix(N_train, basis_funcs)
    scale = np.linalg.norm(X, axis=0)
    scale[scale == 0] = 1.0
    Xs = X / scale

    coef_s, *_ = np.linalg.lstsq(Xs, y_train, rcond=None)
    coef = coef_s / scale

    Nt = np.array([N_test], dtype=float)
    Xt = _design_matrix(Nt, basis_funcs)
    return (Xt @ coef).item()

def loocv_linear_basis(N, y, basis_funcs):
    N = np.asarray(N, dtype=float)
    y = np.asarray(y, dtype=float)

    yhat = np.empty_like(y, dtype=float)
    for k in range(len(N)):
        mask = np.ones(len(N), dtype=bool)
        mask[k] = False
        yhat[k] = _fit_predict_linear_basis(N[mask], y[mask], N[k], basis_funcs)

    r = y - yhat
    rmse = np.sqrt(np.mean(r * r))
    mae  = np.mean(np.abs(r))
    maxae = np.max(np.abs(r))
    return yhat, rmse, mae, maxae

def fit_full(N, y, basis_funcs):
    N = np.asarray(N, dtype=float)
    y = np.asarray(y, dtype=float)

    X = _design_matrix(N, basis_funcs)
    scale = np.linalg.norm(X, axis=0)
    scale[scale == 0.0] = 1.0

    Xs = X / scale
    coef_s, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    coef = coef_s / scale
    return coef


# -----------------------------
# joint mean + parity-decay fit for BETA
# -----------------------------
def fit_mean_plus_parity_decay(N, y, mean_bases, delta_bases, fit_min_N=15):
    N = np.asarray(N, np.int64)
    y = np.asarray(y, float)

    m = (N >= fit_min_N)
    Nf = N[m].astype(float)
    yf = y[m]

    sgn = np.where((Nf.astype(int) % 2) == 0, 1.0, -1.0)  # even:+1, odd:-1

    Xm = _design_matrix(Nf, mean_bases)
    Xd = _design_matrix(Nf, delta_bases)
    X  = np.column_stack([Xm, Xd * sgn[:, None]])

    scale = np.linalg.norm(X, axis=0)
    scale[scale == 0.0] = 1.0
    Xs = X / scale

    coef_s, *_ = np.linalg.lstsq(Xs, yf, rcond=None)
    coef = coef_s / scale

    cm = coef[:Xm.shape[1]]
    cd = coef[Xm.shape[1]:]
    return cm, cd

def predict_mean_plus_parity_decay(Nvals, cm, cd, mean_bases, delta_bases):
    Nvals = np.asarray(Nvals, np.int64)
    Nv = Nvals.astype(float)
    sgn = np.where((Nvals % 2) == 0, 1.0, -1.0)

    Xm = _design_matrix(Nv, mean_bases)
    Xd = _design_matrix(Nv, delta_bases)
    return (Xm @ cm) + sgn * (Xd @ cd)

def loocv_mean_plus_parity_decay(N, y, mean_bases, delta_bases, fit_min_N=25):
    N = np.asarray(N, np.int64)
    y = np.asarray(y, float)

    m = (N >= fit_min_N)
    Nf = N[m]
    yf = y[m]

    yhat = np.empty_like(yf, dtype=float)

    for k in range(len(Nf)):
        mask = np.ones(len(Nf), dtype=bool)
        mask[k] = False
        cm, cd = fit_mean_plus_parity_decay(
            Nf[mask], yf[mask], mean_bases, delta_bases,
            fit_min_N=-1  # don't re-filter; we already filtered to N>=fit_min_N
        )
        yhat[k] = predict_mean_plus_parity_decay(
            np.array([Nf[k]], dtype=np.int64), cm, cd, mean_bases, delta_bases
        ).item()

    r = yf - yhat
    odd  = (Nf % 2 == 1)
    even = ~odd

    def stats(rr):
        rmse = np.sqrt(np.mean(rr * rr))
        mae  = np.mean(np.abs(rr))
        maxae = np.max(np.abs(rr))
        return rmse, mae, maxae

    rmse_o, mae_o, maxae_o = stats(r[odd])
    rmse_e, mae_e, maxae_e = stats(r[even])

    return yhat, rmse_o, mae_o, maxae_o, odd.sum(), rmse_e, mae_e, maxae_e, even.sum()

def fit_parity_shared_constant(N, y, basis_funcs):
    """
    Fit odd/even with a SINGLE shared intercept (basis_funcs[0] must be constant-like),
    and separate coefficients for the remaining basis terms.

    Returns: (coef_odd_full, coef_even_full) where both share coef[0].
    """
    N = np.asarray(N, np.int64)
    y = np.asarray(y, float)

    if len(basis_funcs) < 2:
        raise ValueError("Need at least a constant basis and one other basis term.")

    bases_rest = basis_funcs[1:]
    p = len(bases_rest)

    odd = (N % 2 == 1)
    even = ~odd

    # Design matrices for the "rest" terms
    Xo = _design_matrix(N[odd].astype(float), bases_rest)   # (n_odd, p)
    Xe = _design_matrix(N[even].astype(float), bases_rest)  # (n_even, p)

    # Big design: [ intercept | odd-rest | even-rest ]
    n = N.size
    X = np.zeros((n, 1 + 2*p), dtype=float)
    X[:, 0] = 1.0

    X[odd,  1:1+p]     = Xo
    X[even, 1+p:1+2*p] = Xe

    # Column scaling for numerical stability
    scale = np.linalg.norm(X, axis=0)
    scale[scale == 0.0] = 1.0
    Xs = X / scale

    coef_s, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    coef = coef_s / scale

    c0 = coef[0]
    c_odd_rest  = coef[1:1+p]
    c_even_rest = coef[1+p:1+2*p]

    coef_odd_full  = np.concatenate(([c0], c_odd_rest))
    coef_even_full = np.concatenate(([c0], c_even_rest))
    return coef_odd_full, coef_even_full


def predict_parity_from_coefs(Nvals, basis_funcs, coef_odd, coef_even):
    """
    Predict for arbitrary Nvals using parity-specific full coefficient vectors.
    """
    Nvals = np.asarray(Nvals, np.int64)
    X = _design_matrix(Nvals.astype(float), basis_funcs)
    out = np.empty(Nvals.shape, dtype=float)
    odd = (Nvals % 2 == 1)
    out[odd]  = X[odd]  @ coef_odd
    out[~odd] = X[~odd] @ coef_even
    return out


def loocv_shared_constant_parity(N, y, basis_funcs):
    """
    LOOCV where the intercept is shared between odd and even (refit each leave-out).
    Returns: (yhat, rmse_odd, mae_odd, maxae_odd, n_odd, rmse_even, mae_even, maxae_even, n_even)
    """
    N = np.asarray(N, np.int64)
    y = np.asarray(y, float)

    yhat = np.empty_like(y, dtype=float)
    for k in range(N.size):
        mask = np.ones(N.size, dtype=bool)
        mask[k] = False
        co, ce = fit_parity_shared_constant(N[mask], y[mask], basis_funcs)
        yhat[k] = predict_parity_from_coefs(np.array([N[k]], dtype=np.int64), basis_funcs, co, ce).item()

    r = y - yhat
    odd = (N % 2 == 1)
    even = ~odd

    def stats(rr):
        rmse = np.sqrt(np.mean(rr * rr))
        mae  = np.mean(np.abs(rr))
        maxae = np.max(np.abs(rr))
        return rmse, mae, maxae

    rmse_o, mae_o, maxae_o = stats(r[odd])
    rmse_e, mae_e, maxae_e = stats(r[even])
    return yhat, rmse_o, mae_o, maxae_o, odd.sum(), rmse_e, mae_e, maxae_e, even.sum()

# -----------------------------
# Bases (yours)
# -----------------------------
BASES_P0 = [
    lambda n: np.ones_like(n, dtype=float),
    lambda n: n**(-4/3),
    lambda n: 1.0/(n*np.sqrt(n)),   # n^(-3/2)
    lambda n: n**(-5/3),
    lambda n: n**(-5/2),
]

BASES_ALPHA = [
    lambda n: np.ones_like(n, dtype=float),
    lambda n: 1.0/n,
    lambda n: n**(-4/3),
    lambda n: 1.0/(n*n),            # n^(-2)
    lambda n: 1.0/(np.log(n)**3),   # 1/log(n)^3
]

MEAN_BETA = [
    lambda n: np.ones_like(n, float),
    lambda n: np.sqrt(n),
    lambda n: 1.0/np.sqrt(n),
    lambda n: 1/n**(2/3)
]

DELTA_BETA = [
    lambda n: n**(-1/3),
    lambda n: 1.0/np.sqrt(n),
    lambda n: 1.0/n,
    lambda n: 1.0/(np.log(n)**2),
]


# -----------------------------
# Plot helpers
# -----------------------------
def _plot_parity_fit(N_all, y_all, basis_funcs, coef_odd, coef_even, fit_min_N,
                     title, ylabel=None, show_training=True, yscale=None):
    N_all = np.asarray(N_all, dtype=np.int64)
    y_all = np.asarray(y_all, dtype=float)

    Ngrid = np.arange(int(N_all.min()), int(N_all.max()) + 1, dtype=np.int64)
    Ng = Ngrid.astype(float)

    oddg  = (Ngrid % 2 == 1)
    eveng = ~oddg

    yhat = np.empty_like(Ng, dtype=float)
    yhat[oddg]  = (_design_matrix(Ng[oddg],  basis_funcs) @ coef_odd)
    yhat[eveng] = (_design_matrix(Ng[eveng], basis_funcs) @ coef_even)

    odd_all  = (N_all % 2 == 1)
    even_all = ~odd_all
    mfit = (N_all >= fit_min_N)

    plt.figure(figsize=(9, 4.8))
    ax = plt.gca()
    ax.set_title(f"{title} vs N (parity-aware fits trained on N ≥ {fit_min_N})")
    ax.set_xlabel("N")
    ax.set_ylabel(ylabel if ylabel is not None else title)

    line_odd,  = ax.plot(Ngrid[oddg],  yhat[oddg],  linewidth=2.0, label="odd fit")
    line_even, = ax.plot(Ngrid[eveng], yhat[eveng], linewidth=2.0, label="even fit")

    ax.plot(N_all[odd_all],  y_all[odd_all],  marker='o', linestyle='None',
            color=line_odd.get_color(),  label="odd data")
    ax.plot(N_all[even_all], y_all[even_all], marker='o', linestyle='None',
            color=line_even.get_color(), label="even data")

    if show_training:
        ax.plot(N_all[mfit & odd_all],  y_all[mfit & odd_all],  marker='s', linestyle='None',
                color=line_odd.get_color(),  label="odd training")
        ax.plot(N_all[mfit & even_all], y_all[mfit & even_all], marker='s', linestyle='None',
                color=line_even.get_color(), label="even training")

    if yscale is not None:
        ax.set_yscale(yscale)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncols=2)
    plt.tight_layout()
    plt.show()

def _plot_parity_fit_joint(N_all, y_all, mean_bases, delta_bases, cm, cd, fit_min_N,
                           title, ylabel=None, show_training=True, yscale=None):
    N_all = np.asarray(N_all, dtype=np.int64)
    y_all = np.asarray(y_all, dtype=float)

    Ngrid = np.arange(int(N_all.min()), int(N_all.max()) + 1, dtype=np.int64)
    yhat = predict_mean_plus_parity_decay(Ngrid, cm, cd, mean_bases, delta_bases)

    oddg  = (Ngrid % 2 == 1)
    eveng = ~oddg

    odd_all  = (N_all % 2 == 1)
    even_all = ~odd_all
    mfit = (N_all >= fit_min_N)

    plt.figure(figsize=(9, 4.8))
    ax = plt.gca()
    ax.set_title(f"{title} vs N (joint mean + parity-decay fit trained on N ≥ {fit_min_N})")
    ax.set_xlabel("N")
    ax.set_ylabel(ylabel if ylabel is not None else title)

    line_odd,  = ax.plot(Ngrid[oddg],  yhat[oddg],  linewidth=2.0, label="odd fit")
    line_even, = ax.plot(Ngrid[eveng], yhat[eveng], linewidth=2.0, label="even fit")

    ax.plot(N_all[odd_all],  y_all[odd_all],  marker='o', linestyle='None',
            color=line_odd.get_color(),  label="odd data")
    ax.plot(N_all[even_all], y_all[even_all], marker='o', linestyle='None',
            color=line_even.get_color(), label="even data")

    if show_training:
        ax.plot(N_all[mfit & odd_all],  y_all[mfit & odd_all],  marker='s', linestyle='None',
                color=line_odd.get_color(),  label="odd training")
        ax.plot(N_all[mfit & even_all], y_all[mfit & even_all], marker='s', linestyle='None',
                color=line_even.get_color(), label="even training")

    if yscale is not None:
        ax.set_yscale(yscale)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncols=2)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Reporting
# -----------------------------
def _print_block(name, rmse_o, mae_o, maxae_o, n_o, rmse_e, mae_e, maxae_e, n_e):
    print(f"{name} LOOCV odd : RMSE={rmse_o:.6g}  MAE={mae_o:.6g}  MaxAE={maxae_o:.6g}  (n={n_o})")
    print(f"{name} LOOCV even: RMSE={rmse_e:.6g}  MAE={mae_e:.6g}  MaxAE={maxae_e:.6g}  (n={n_e})")
    print()

def overall_report(N, P0, ALPHA, BETA, fit_min_N=25, fit_min_BETA=10, plot=True, show_training=True):
    N = np.asarray(N, dtype=np.int64)
    P0 = np.asarray(P0, dtype=float)
    ALPHA = np.asarray(ALPHA, dtype=float)
    BETA = np.asarray(BETA, dtype=float)

    mfit = (N >= fit_min_N)
    
    # Training arrays (filtered)
    Nf = N[mfit].astype(float)
    P0f = P0[mfit]
    ALf = ALPHA[mfit]

    out = {}

    # --- P0 (shared constant between odd/even) ---
    _, rmse_o, mae_o, maxae_o, n_o, rmse_e, mae_e, maxae_e, n_e = loocv_shared_constant_parity(
        Nf.astype(np.int64), P0f, BASES_P0
    )
    _print_block(f"P0 shared-c0 (N>={fit_min_N})", rmse_o, mae_o, maxae_o, n_o,
                 rmse_e, mae_e, maxae_e, n_e)

    coef_p0_odd, coef_p0_even = fit_parity_shared_constant(Nf.astype(np.int64), P0f, BASES_P0)
    out["P0_odd_coef"]  = coef_p0_odd
    out["P0_even_coef"] = coef_p0_even

    if plot:
        _plot_parity_fit(N, P0, BASES_P0, coef_p0_odd, coef_p0_even, fit_min_N,
                         title="P0", ylabel="P0", show_training=show_training)

    # --- ALPHA (shared constant between odd/even) ---
    _, rmse_o, mae_o, maxae_o, n_o, rmse_e, mae_e, maxae_e, n_e = loocv_shared_constant_parity(
        Nf.astype(np.int64), ALf, BASES_ALPHA
    )
    _print_block(f"ALPHA shared-c0 (N>={fit_min_N})", rmse_o, mae_o, maxae_o, n_o,
                 rmse_e, mae_e, maxae_e, n_e)

    coef_a_odd, coef_a_even = fit_parity_shared_constant(Nf.astype(np.int64), ALf, BASES_ALPHA)
    out["ALPHA_odd_coef"]  = coef_a_odd
    out["ALPHA_even_coef"] = coef_a_even

    if plot:
        _plot_parity_fit(N, ALPHA, BASES_ALPHA, coef_a_odd, coef_a_even, fit_min_N,
                         title="ALPHA", ylabel="alpha", show_training=show_training)

    # --- BETA (JOINT mean + parity-decay) ---
    _, rmse_o, mae_o, maxae_o, n_o, rmse_e, mae_e, maxae_e, n_e = loocv_mean_plus_parity_decay(
        N, BETA, MEAN_BETA, DELTA_BETA, fit_min_N=fit_min_BETA
    )
    _print_block(f"BETA (joint, N>={fit_min_BETA})", rmse_o, mae_o, maxae_o, n_o,
                 rmse_e, mae_e, maxae_e, n_e)

    cm, cd = fit_mean_plus_parity_decay(N, BETA, MEAN_BETA, DELTA_BETA, fit_min_N=fit_min_BETA)
    out["BETA_cm"] = cm
    out["BETA_cd"] = cd

    if plot:
        _plot_parity_fit_joint(N, BETA, MEAN_BETA, DELTA_BETA, cm, cd, fit_min_BETA,
                               title="BETA", ylabel="beta", show_training=show_training)

    return out

# coeffs = overall_report(N, P0, ALPHA, BETA, fit_min_N=25)
# print(coeffs.keys())
    