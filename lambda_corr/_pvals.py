#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 17:52:55 2026

@author: jplundquist
"""

from numba import njit
import numpy as np
from importlib.resources import files
from ._core import _lambda_stats
from math import exp, lgamma, log, log1p, sqrt

#Exact p-value calculation globals
#predefine lookup tables to stop IDE warnings
LUT_VALS_N3 = LUT_VALS_N4 = LUT_VALS_N5 = LUT_VALS_N6 = LUT_VALS_N7 = LUT_VALS_N8 = LUT_VALS_N9 = LUT_VALS_N10 = None
LUT_CC_N3 = LUT_CC_N4 = LUT_CC_N5 = LUT_CC_N6 = LUT_CC_N7 = LUT_CC_N8 = LUT_CC_N9 = LUT_CC_N10 = None
LUT_ABS_VALS_N3 = LUT_ABS_VALS_N4 = LUT_ABS_VALS_N5 = LUT_ABS_VALS_N6 = LUT_ABS_VALS_N7 = LUT_ABS_VALS_N8 = LUT_ABS_VALS_N9 = LUT_ABS_VALS_N10 = None
LUT_ABS_TAIL_N3 = LUT_ABS_TAIL_N4 = LUT_ABS_TAIL_N5 = LUT_ABS_TAIL_N6 = LUT_ABS_TAIL_N7 = LUT_ABS_TAIL_N8 = LUT_ABS_TAIL_N9 = LUT_ABS_TAIL_N10 = None

#Load lookup tables
_LUT_PATH = files("lambda_corr").joinpath("data", "lambda_lut_n3_n10.npz")
with np.load(_LUT_PATH, allow_pickle=False) as z:
    globals().update({k: z[k] for k in z.files})

#p-value Beta distribution globals
N_A = np.arange(11,31)
P0_A = np.array([0.0708556798140131, 0.06714529554807333, 0.06600454715038048, 0.06496984457983962, 0.06238369283333333, 0.062640989,
                 0.059537863333333337, 0.0604504015, 0.057216893, 0.05846732416666667, 0.05526443416666667, 0.056689878, 0.05361363,
                 0.055135552, 0.052150197, 0.053692919, 0.050890785, 0.052419307, 0.049762158, 0.051266391], dtype=np.float64)
P1_A = np.array([2.0041686708353374e-06, 6.513548180214847e-07, 3.789934345489901e-08, 1.0553085949911346e-08, 4.1753513975736197e-10, 
                 1.4663436455764497e-10, 5.263047980134815e-12, 1.4366546569705612e-12, 4.1251147667560884e-14, 1.069422439233359e-14, 
                 1.762739072169017e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

ALPHA_A = np.array([1.3964874007690553, 1.4378036180809, 1.4274144119465353, 1.4552545703790116, 1.4447437136748829, 1.4671825739472821,
                    1.45608952243845, 1.4748300243343895, 1.4644181779078853, 1.4805092620739457, 1.4702557385709127, 1.4845023191449047, 
                    1.4751097546757268, 1.4876743353571005, 1.47888092293881, 1.4902126712917316, 1.481845624216896, 1.4922291390943292, 
                    1.4845170545731274, 1.4939833744022095], dtype=np.float64)
BETA_A = np.array([2.9759841690111215, 3.2336691391510604, 3.518822685429833, 3.7485744560979195, 4.00762594005941, 4.223033938955358,
                    4.45823438758898, 4.661919097481938, 4.879683679160764, 5.0736720165789455, 5.275428822561992, 5.461907884453,
                    5.651686059204632, 5.830734762026783, 6.0102024965515985, 6.183505864613792, 6.352736078846976, 6.5203056989847,
                    6.683121593579365, 6.845129043097732], dtype=np.float64)
    
P0_odd_coef = np.array([0.029002086096860582, 25.904072374554012, -75.33127655371946, 59.04726289172854, -15.030544411676352], dtype=np.float64)
P0_even_coef = np.array([0.029002086096860582, 25.237305319544447, -72.36060891036985, 56.922475180883616, -22.777536208741292], dtype=np.float64)
ALPHA_odd_coef = np.array([1.5736326140535608, 11.84181244804074, -13.066381141090687, 50.47418489312429, -15.689595928752894], dtype=np.float64)
ALPHA_even_coef = np.array([1.5736326140535608, 11.831773837778785, -11.576831224054585, 45.710298261783606, -15.764548077471224], dtype=np.float64)
BETA_cm = np.array([-2.1472504419053378, 1.7181047852900624, -0.08124648038375559, -4.307157321356942], dtype=np.float64)
BETA_cd = np.array([0.9980956667928251, 0.6688934142825781, -2.5049256378879714, 2.106273285559509], dtype=np.float64)

#p-value permutation test

#Check if p-value uncertainty is smaller than threshold
@njit(cache=True, nogil=True, inline='always')
def _check_stop(h, p_tol, N):
    if (N<50) or (h<5) or ((N-h)<5):
        check = False
    else:
        p_hat = (h + 1.0) / (N + 1.0)
        se = np.sqrt(p_hat * (1.0 - p_hat) / (N + 1.0))
        rel = se / max(p_hat, np.float64(2.220446049250313e-16))
        check = rel <= p_tol
    return check

@njit(cache=True, nogil=True, inline='always') #, parallel=True
def _lambda_p_perm(rx, ry, n, Lambda_s, Lambda_yx, Lambda_xy, p_tol=1e-4, 
                  n_perm=10000, alt="two-sided"):
    # ---- Permutation test ----

    c_s = 0
    c_xy = 0
    c_yx = 0
    N = 0
    for i in range(n_perm): #prange(n_perm):  #PARALLEL LOOP not possible with early exit
        perm = np.random.permutation(n)
        l_b, l_yx_b, l_xy_b = _lambda_stats(rx, ry[perm], n)
        if alt == "two-sided":
            hit_s = (abs(l_b) >= abs(Lambda_s))
            hit_yx = (abs(l_yx_b) >= abs(Lambda_yx))
            hit_xy = (abs(l_xy_b) >= abs(Lambda_xy))
        elif (alt == "greater"):
            hit_s = (l_b >= Lambda_s)
            hit_yx = (l_yx_b >= Lambda_yx)
            hit_xy = (l_xy_b >= Lambda_xy)
        else: #alt == less
            hit_s = (l_b <= Lambda_s)
            hit_yx = (l_yx_b <= Lambda_yx)
            hit_xy = (l_xy_b <= Lambda_xy)

        c_s += int(hit_s)
        c_yx += int(hit_yx)
        c_xy += int(hit_xy)
        
        N = N + 1
        #We will consider the p-value accuracy on the symmetric correlation only
        if _check_stop(c_s, p_tol, N):
            break

    p_s = (c_s + 1.0) / (N + 1.0)
    p_yx = (c_yx + 1.0) / (N + 1.0)
    p_xy = (c_xy + 1.0) / (N + 1.0)
    
    return p_s, p_yx, p_xy

#p-value Beta distribution
@njit(cache=True, nogil=True, inline='always')
def _betacf(a, b, x, maxiter=1000, eps=1e-15):
    # Continued fraction for incomplete beta function
    FPMIN = 1e-300
    m2 = 0
    aa = 0.0
    c = 1.0
    
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d

    for m in range(1, maxiter + 1):
        m2 = 2.0 * m
        # even term
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: 
            d = FPMIN
        
        c = 1.0 + aa / c
        if abs(c) < FPMIN: 
            c = FPMIN
        
        d = 1.0 / d
        h *= d * c
        # odd term
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: 
            d = FPMIN
        
        c = 1.0 + aa / c
        if abs(c) < FPMIN: 
            c = FPMIN
        
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    return h

@njit(cache=True, nogil=True, inline='always')
def _incbet_series(a, b, x, maxiter=1000, eps=1e-15):
    # Computes I_x(a,b) via series expansion (good for small x)
    # term recurrence: term_{n+1} = term_n * ( (a+n)*(1-b+n) / ((a+1+n)*(n+1)) ) * x
    # This is one common form; there are several equivalent series.
    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(a*log(x) + b*log1p(-x) - lbeta) / a

    # Sum the series for the continued fraction alternative near x~0
    s = 1.0
    term = 1.0
    for n in range(1, maxiter + 1):
        term *= ( (a + n - 1.0) * (1.0 - b + n - 1.0) / ((a + n) * n) ) * x
        s_old = s
        s += term
        if abs(s - s_old) <= eps * abs(s):
            break

    return front * s

@njit(cache=True, nogil=True, inline='always')
def _incbet(a, b, x):
    # Regularized incomplete beta I_x(a,b)
    if x <= 0.0: 
        return 0.0
    
    if x >= 1.0: 
        return 1.0
    
    # exact symmetry for a==b at x=0.5
    if a == b and abs(x - 0.5) < 1e-15:
        return 0.5

    # choose a conservative threshold
    X_SER = 1e-7

    # lower tail: use series directly
    if x < X_SER:
        return _incbet_series(a, b, x)

    # upper tail: use symmetry + series on (1-x)
    if x > 1.0 - X_SER:
        return 1.0 - _incbet_series(b, a, 1.0 - x)

    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    lnum  = a*log(x) + b*log1p(-x) - lbeta

    if x < (a+1.0)/(a+b+2.0):
        # forward: I_x(a,b)
        return exp(lnum) * _betacf(a, b, x) / a
    else:
        # backward: 1 - I_{1-x}(b,a)
        return 1.0 - (exp(lnum) * _betacf(b, a, 1.0 - x) / b)

# P0 bases:
# [1, n^(-4/3), n^(-3/2), n^(-5/3), n^(-5/2)]
@njit(cache=True, nogil=True)
def predict_p0(N, c):
    n = float(N)

    inv = 1.0 / n
    inv_sqrt = 1.0 / sqrt(n)

    n_m43 = inv * (inv ** (1.0/3.0))          # n^(-4/3) = n^-1 * n^(-1/3)
    n_m32 = inv * inv_sqrt                    # n^(-3/2)
    n_m53 = inv * (inv ** (2.0/3.0))          # n^(-5/3) = n^-1 * n^(-2/3)
    n_m52 = (inv * inv_sqrt) * inv            # n^(-5/2) = n^(-3/2)*n^-1

    return (c[0]
            + c[1] * n_m43
            + c[2] * n_m32
            + c[3] * n_m53
            + c[4] * n_m52)


# ALPHA bases:
# [1, 1/n, n^(-4/3), 1/n^2, 1/log(n)^3]
@njit(cache=True, nogil=True)
def predict_alpha(N, c):
    n = float(N)

    inv = 1.0 / n
    inv2 = inv * inv
    n_m43 = inv * (inv ** (1.0/3.0))

    ln = log(n)
    invlog3 = 1.0 / (ln * ln * ln)

    return (c[0]
            + c[1] * inv
            + c[2] * n_m43
            + c[3] * inv2
            + c[4] * invlog3)


# BETA joint model:
# beta(N) = mean(N) + sgn(N)*delta(N), sgn even=+1, odd=-1
# mean bases: [1, sqrt(n), log(n), 1/n]
# delta bases: [n^(-1/3), 1/log(n)^(1/2), 1/log(n), 1/log(n)^3]
@njit(cache=True, nogil=True)
def predict_beta(N, cm, cd, s_beta):
    n = float(N)

    inv = 1.0 / n
    sq = sqrt(n)
    ln = log(n)

    # mean
    mean = (cm[0]
            + cm[1] * sq
            + cm[2] * ln
            + cm[3] * inv)

    # delta
    n_m13 = inv ** (1.0/3.0)             # n^(-1/3)
    invlog = 1.0 / ln
    invlog_half = 1.0 / sqrt(ln)    # 1/log(n)^(1/2)
    invlog3 = invlog * invlog * invlog

    delta = (cd[0] * n_m13
             + cd[1] * invlog_half
             + cd[2] * invlog
             + cd[3] * invlog3)

    return mean + s_beta * delta

@njit(cache=True, nogil=True, inline='always')
def get_tail(x, alpha, beta):
    if x <= 0.0:
        return 1.0
    if x >= 1.0:
        return 0.0
    if x > 0.5:
        tail = _incbet(beta, alpha, 1.0 - x) # stable near 1
    else:
        tail = 1.0 - _incbet(alpha, beta, x)
        
    return tail
   
@njit(cache=True, nogil=True, inline='always')
def _lambda_p_beta(Lambda, n, alt="two-sided"):   
    
    t = abs(Lambda)
    if 10 < n < 31:

        i = n-11
        p0 = P0_A[i]
        p1 = P1_A[i]
        alpha = ALPHA_A[i]
        beta = BETA_A[i]
    
    else:
        odd = (n & 1) == 1
        s_beta = -1.0 if odd else 1.0
        if odd: 
            c_p0 = P0_odd_coef
            c_alpha = ALPHA_odd_coef
        else:
            c_p0 = P0_even_coef
            c_alpha = ALPHA_even_coef
            
        p0 = predict_p0(n, c_p0)
        alpha = predict_alpha(n, c_alpha)
        beta = predict_beta(n, BETA_cm, BETA_cd, s_beta)
            
    if alt == "two-sided":
        if t == 0:
            p_val = 1.0
            
        elif t >= 1.0:
            p_val = p1
            
        else:
            pc = (1.0 - p0) - p1
            if pc < 0.0 and pc > -1e-15:
                pc = 0.0
            tail = get_tail(t, alpha, beta)
            p_val = p1 + pc * tail
    
    elif alt == "greater":
        if Lambda <= -1.0:
            p_val = 1.0
            
        elif Lambda >= 1.0:
            p_val = 0.5 * p1
            
        elif Lambda == 0.0:
            p_val = 0.5 * (1.0 + p0)        # P(Lambda >= 0) includes the mass at 0
    
        else:
            tail = get_tail(t, alpha, beta)
            pc = (1.0 - p0) - p1
            if pc < 0.0 and pc > -1e-15:
                pc = 0.0
            
            pos_tail = 0.5 * (p1 + pc * tail) # P(L >= |x|) for x>0              
            if Lambda > 0.0:     
                p_val = pos_tail # P(L >= x)
            
            else:
                p_val = 1.0 - pos_tail # P(L >= -|x|)
    
    else:  # "less"
        if Lambda <= -1.0:
            p_val = 0.5 * p1

        elif Lambda >= 1.0:
            p_val = 1.0

        elif Lambda == 0.0:
            p_val = 0.5 * (1.0 + p0)              # inclusive at 0
            
        else:
            tail = get_tail(t, alpha, beta)
            pc = (1.0 - p0) - p1
            if pc < 0.0 and pc > -1e-15:
                pc = 0.0
            
            neg_tail = 0.5 * (p1 + pc * tail)    # P(L <= -|x|) for x<0
            
            if Lambda < 0.0:
                p_val = neg_tail                  # P(L <= x)
            else:
                p_val = 1.0 - neg_tail            # P(L <= +|x|)
                
    return max(0.0, min(1.0, p_val))

@njit(nogil=True, inline='always') #cache=True,  Can't cache with globals
def _lambda_p_exact(Lambda, n, alt="two-sided"):
    
    if alt == "two-sided":
        # ---- select LUT arrays for this n (3..9) ----
        if n == 3:
            abs_vals = LUT_ABS_VALS_N3; abs_tail = LUT_ABS_TAIL_N3
        elif n == 4:
            abs_vals = LUT_ABS_VALS_N4; abs_tail = LUT_ABS_TAIL_N4
        elif n == 5:
            abs_vals = LUT_ABS_VALS_N5; abs_tail = LUT_ABS_TAIL_N5
        elif n == 6:
            abs_vals = LUT_ABS_VALS_N6; abs_tail = LUT_ABS_TAIL_N6
        elif n == 7:
            abs_vals = LUT_ABS_VALS_N7; abs_tail = LUT_ABS_TAIL_N7
        elif n == 8:
            abs_vals = LUT_ABS_VALS_N8; abs_tail = LUT_ABS_TAIL_N8
        elif n == 9:
            abs_vals = LUT_ABS_VALS_N9; abs_tail = LUT_ABS_TAIL_N9
        else:  # n == 10
            abs_vals = LUT_ABS_VALS_N10; abs_tail = LUT_ABS_TAIL_N10
            
        k = np.searchsorted(abs_vals, abs(Lambda), side="left")
        if k >= abs_vals.size:
            p_val = 0.0
        else:
            p_val = float(abs_tail[k]) / float(abs_tail[0])

    else:
        # ---- select LUT arrays for this n (3..9) ----
        if n == 3:
            vals = LUT_VALS_N3; cc = LUT_CC_N3
        elif n == 4:
            vals = LUT_VALS_N4; cc = LUT_CC_N4
        elif n == 5:
            vals = LUT_VALS_N5; cc = LUT_CC_N5
        elif n == 6:
            vals = LUT_VALS_N6; cc = LUT_CC_N6
        elif n == 7:
            vals = LUT_VALS_N7; cc = LUT_CC_N7
        elif n == 8:
            vals = LUT_VALS_N8; cc = LUT_CC_N8
        elif n == 9:
            vals = LUT_VALS_N9; cc = LUT_CC_N9
        else:  # n == 10
            vals = LUT_VALS_N10; cc = LUT_CC_N10
                
        N = float(cc[-1])

        if alt == "less":
            # P(L <= Lambda): include any exact ties at Lambda
            j = np.searchsorted(vals, Lambda, side="right") - 1
            if j < 0:
                p_val = 0.0
            else:
                p_val = float(cc[j]) / N

        else:  # alt == "greater"
            # P(L >= Lambda): include any exact ties at Lambda
            j = np.searchsorted(vals, Lambda, side="left")
            if j >= vals.size:
                p_val = 0.0
            else:
                below = 0 if j == 0 else float(cc[j - 1])  # count(L < vals[j])
                p_val = (N - below) / N

    return max(0.0, min(1.0, p_val))