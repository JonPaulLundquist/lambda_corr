#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 18:16:18 2025

@author: Jon Paul Lundquist
"""

import numpy as np
from scipy.optimize import curve_fit, minimize, minimize_scalar
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.pipeline import make_pipeline
from scipy.special import gammaln

def student_t_loglike(resid, nu=10.0):
    resid = np.asarray(resid, float)
    #n = resid.size
    const = gammaln((nu + 1)/2) - gammaln(nu/2) - 0.5*np.log(nu*np.pi)

    # optimize over log(s) for stability
    def neg_ll(logs):
        s = np.exp(logs) + 1e-300
        r2 = (resid / s)**2
        return -(np.sum(const - np.log(s) - 0.5*(nu + 1)*np.log1p(r2/nu)))

    # bracket around a robust scale guess
    s0 = 1.4826 * np.median(np.abs(resid)) + 1e-12
    res = minimize_scalar(neg_ll, bracket=(np.log(s0)-8, np.log(s0), np.log(s0)+8))
    return -res.fun  # maximized log-likelihood

def loglike_gaussian(resid):
    """
    Gaussian log‐likelihood (exact, up to constants) for residuals r_i.
    """
    n = len(resid)
    σ2 = np.mean(resid**2)                  # MLE σ²
    # exact ℓ = −n/2 [ln(2πσ²) + 1]
    return -0.5*n*(np.log(2*np.pi*σ2) + 1)

def loglike_laplace(resid):
    """
    Laplace log‐likelihood for residuals r_i:
      p(r)=1/(2b) exp(−|r|/b),  b = mean(|r|).
    """
    n = len(resid)
    b = np.mean(np.abs(resid))              # MLE scale
    # ℓ = −n·ln(2b) − (1/b)∑|r|
    return -n*np.log(2*b) - np.sum(np.abs(resid))/b

def rho_tukey(u, c=4.685):
    """
    Tukey’s biweight ρ(u):
      ρ(u) = (c²/6)[1 − (1 − u²)³]    for |u| ≤ 1
           = c²/6                     for |u| > 1
    """
    u = np.asarray(u)
    ρ = np.empty_like(u)
    mask = np.abs(u) <= 1
    ρ[mask]    = (c**2/6)*(1 - (1 - u[mask]**2)**3)
    ρ[~mask]   = (c**2/6)
    return ρ

def loglike_bisquare(resid, h, c=4.685):
    """
    Pseudo–log‐likelihood for Tukey’s biweight:
      ℓ ≈ − ∑ ρ( u_i ),  u_i = r_i / (s·c),
    where we estimate the scale s via MAD:
      s = 1.4826 · median( |r_i| ).
    This drops the normalizing constant but is
    perfectly fine for ranking.
    """
    #n = len(resid)
    # robust scale estimate
    s = 1.4826 * np.median(np.abs(resid))
    if not np.isfinite(s) or s <= 0:
        s = 1e-12
    u = resid / (s*c*np.sqrt(np.maximum(1.0 - h, 1e-12)))
    Q = np.sum(rho_tukey(u, c))
    return -Q

def weighted_adj_r2(y, y_pred, w, p):
    """
    Return (R2, R2_adj) for data y vs. y_pred,
    with weights w and p fitted parameters.
    """
    # 1) only keep points with w>0
    mask = w > 0
    y = y[mask]
    y_pred = y_pred[mask]
    w = w[mask]
    
    # 2) weighted mean of y
    W = w.sum()
    y_bar = np.dot(w, y) / W
    
    # 3) sums of squares
    ss_tot = np.dot(w, (y - y_bar)**2)
    ss_res = np.dot(w, (y - y_pred)**2)
    
    # 4) weighted R^2
    R2 = 1 - ss_res/ss_tot
    
    # 5) adjusted R^2
    n = len(y)
    R2_adj = 1 - (1 - R2)*(n - 1)/(n - p - 1)
    
    return R2_adj

def tukey_weights(residuals, scale, h, c=4.685):
    denom = c * scale * np.sqrt(np.maximum(1.0 - h, 1e-12))
    u = residuals / denom
    w = np.zeros_like(u)
    mask = np.abs(u) < 1
    w[mask] = (1 - u[mask]**2)**2
    return w

def _jacobian(func, x, p, eps=None):
    """
    Numerically approximate J_ij = ∂f(x_i; p)/∂p_j by central differences.
    """
    p = np.asarray(p, float)
    n = p.size
    m = x.shape[0]
    J = np.zeros((m, n))
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps) * (np.abs(p) + 1)
    for j in range(n):
        dp = np.zeros_like(p)
        h = eps[j]
        dp[j] = h
        f_plus  = func(x, *(p + dp))
        f_minus = func(x, *(p - dp))
        J[:, j] = (f_plus - f_minus) / (2*h)
    return J

def bisquare_fit(func, x, y, sigma=None, p0=None, max_iter=1000, tol=1e-9, c=4.685):
    if sigma is None:
        prior_w = np.ones_like(y)
    else:
        prior_w = 1/np.asarray(sigma)**2
    
    # 1) Initial OLS fit
    #popt, pcov = curve_fit(func, x, y, sigma=sigma, p0=p0, maxfev=40000)
    sigma0 = 1/np.sqrt(prior_w)
    popt, _ = curve_fit(func, x, y, sigma=sigma0, p0=p0, absolute_sigma=True,
                        maxfev=40000)
    J0 = _jacobian(func, x, popt)    
    JTJ_inv = np.linalg.inv(J0.T @ J0)
    h = np.einsum('ij,jk,ik->i', J0, JTJ_inv, J0)
    for _ in range(max_iter):
        # 2) Compute residuals and Tukey weights
        resid = y - func(x, *popt)
        scale = np.median(np.abs(resid)) / 0.6745
        if not np.isfinite(scale) or scale <= 0:
            scale = 1e-12
        w_bisq = tukey_weights(resid, scale, h, c=c)
        w = prior_w * w_bisq
        # 3) Convert weights -> effective sigma for curve_fit
        sigma_eff = np.full(w.shape, np.inf, dtype=float)
        np.divide(1.0, np.sqrt(w), out=sigma_eff, where=(w > 0))
        
        # 4) Weighted fit
        popt_new, pcov_new = curve_fit(func, x, y, sigma=sigma_eff, absolute_sigma=True,
                                       p0=popt, maxfev=40000)
        
        # 5) Check convergence
        if np.allclose(popt, popt_new, atol=tol, rtol=tol):
            popt = popt_new
            break
        
        popt = popt_new

    y_fit = func(x, *popt)    
    resid = y - y_fit
    scale = np.median(np.abs(resid)) / 0.6745
    w_final = prior_w * tukey_weights(resid, scale, h, c=c)
    R2_adj = weighted_adj_r2(y, y_fit, w_final, p=len(popt))
    
    return popt, R2_adj, h

def least_absolute_fit(func, x, y, p0, method='Powell', tol=1e-7):
    """
    Fit func(x, *p) by minimizing sum |y - func(x,p)|, then
    compute adjusted R^2 on squared errors.
    
    Parameters
    ----------
    func : callable
        Model function f(x, *p).
    x : array_like
        Independent-variable data.
    y : array_like
        Dependent-variable data.
    p0 : array_like
        Initial guess for parameters.
    method : str, optional
        Optimization method for `scipy.optimize.minimize`.
    tol : float, optional
        Tolerance for termination.
    
    Returns
    -------
    popt : ndarray
        Best‐fit parameters.
    R2_adj : float
        Adjusted R² of the fit.
    """
    # 1) objective: sum of absolute residuals
    obj = lambda p: np.sum(np.abs(y - func(x, *p)))
    
    # 2) minimize
    res = minimize(obj, p0, method=method, tol=tol)
    popt = res.x
    
    # 3) compute predictions & uniform weights
    y_fit = func(x, *popt)
    w = np.ones_like(y)
    
    # 4) compute R2_adj
    R2_adj = weighted_adj_r2(y, y_fit, w, p=len(popt))
    
    return popt, R2_adj

def robust_fit(func, x, y, sigma=None, p0=None):
    """
    Try three fits—OLS, bisquare, L1—and return the best fit.

    Returns
    -------
    best_popt : ndarray
    best_R2_adj : float
    best_w : ndarray
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # default sigma = 1
    if sigma is None:
        sigma = np.ones_like(y)
    else:
        sigma = np.asarray(sigma)
    
    # 1) OLS fit
    popt_ols, _ = curve_fit(func, x, y, sigma=sigma, p0=p0, maxfev=40000)

    # 2) Bisquare, init from OLS
    popt_bisq, R2_adj_bisq, h = bisquare_fit(func, x, y, sigma=sigma, p0=popt_ols)

    # 3) L1, init from bisquare
    popt_l1, R2_adj_l1 = least_absolute_fit(func, x, y, p0=popt_bisq)
    
    popts   = [popt_ols, popt_bisq, popt_l1]
    names   = ["OLS", "Bisquare", "L1"]
    
    ll_ols  = loglike_gaussian(y - func(x, *popt_ols))
    ll_bisq = loglike_bisquare(y - func(x, *popt_bisq), h)
    ll_l1   = loglike_laplace(y - func(x, *popt_l1))

    scores = np.array([ll_ols, ll_bisq, ll_l1])


    i_best  = int(np.argmax(scores))   # gives 0, 1, or 2
    
    # r_ols  = y - func(x, *popt_ols)
    # r_bisq = y - func(x, *popt_bisq)
    # r_l1   = y - func(x, *popt_l1)
    
    # scores = np.array([
    #     student_t_loglike(r_ols,  nu=4.0),
    #     student_t_loglike(r_bisq, nu=4.0),
    #     student_t_loglike(r_l1,   nu=4.0),
    # ])
    # i_best = int(np.argmax(scores))

    best_popt, best_score, best_name = popts[i_best], scores[i_best], names[i_best]
    
    print(f"Selected {best_name} fit (max score={best_score:.4f})")
    
    return best_popt, best_score

def theil_parabola(x, y, sigma=None, random_state=0):
    """
    Fit y = a*(x - x0)^2 + y0 robustly via Theil–Sen on a quadratic basis.

    Parameters
    ----------
    x, y : 1D arrays, shape (n,)
        Data points.
    sigma : 1D array of shape (n,), optional
        Measurement errors on y.  If given, used as weights ~ 1/sigma^2.
    random_state : int, optional
        Seed for Theil–Sen’s internal sampling.

    Returns
    -------
    a, x0, y0 : floats
        Fitted parabola parameters so that y ≈ a*(x - x0)**2 + y0.
    """
    X = np.vstack([x]).T  # shape (n,1)

    # Build quadratic features [x, x^2]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X2 = poly.fit_transform(X)
    #   X2 columns are [x, x^2]

    # Setup Theil–Sen with optional weighting
    ts = TheilSenRegressor(
        fit_intercept=True,
        n_subsamples=None,        # smallest subset size → max # of combos
        max_subpopulation=10**50,   # enumerate them all
        max_iter=10**50,      # lots of median iterations
        tol=1e-15,                # high‐precision convergence
        n_jobs=-1,                # use every CPU core
        verbose=True,
        random_state=0
    )
    if sigma is None:
        ts.fit(X2, y)
    else:
        # weights = 1/sigma^2
        sw = 1.0/(sigma**2)
        m = np.round(sw / sw.min()).astype(int)  # normalize so min→1

        X2_rep = np.repeat(X2, m, axis=0)
        y_rep  = np.repeat(y, m, axis=0)
        ts.fit(X2_rep, y_rep)

    # Extract the quadratic model:
    #   y = w0 + w1*x + w2*x^2
    w1, w2 = ts.coef_      # [x , x^2] coefficients
    w0     = ts.intercept_

    # Convert back to vertex form
    a  = w2
    # x0 = -w1/(2*a)
    # y0 = w0 - w1^2/(4*a)
    x0 = -w1/(2.0*a)
    y0 = w0 - (w1**2)/(4.0*a)

    return a, x0, y0

def RANSAC_parabola(x, y, sigma=None, random_state=0,
                   max_iter=1000, cond_thresh=1e3):
    """
    Robust fit of y = a*(x-x0)^2 + y0 via Theil–Sen + RANSAC
    on quadratic basis, with standard scaling of x.
    """
    # 1. scale x
    X = x.reshape(-1, 1)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)           # zero-mean, unit-std

    # 2. build quadratic features on scaled x
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X2_sc = poly.fit_transform(X_sc)         # [x_sc, x_sc^2]

    # 3. helper to reject ill-conditioned or poor-R2 subsets
    def is_model_valid(estimator, X_sub, y_sub):
        Rtest = estimator.score(X_sub, y_sub) > 0
        Ctest = np.linalg.cond(X_sub) < cond_thresh
        return bool(Rtest and Ctest)

    # prepare RANSAC
    ts = RANSACRegressor(min_samples=0.5,
                         max_trials=max_iter,
                         is_model_valid=is_model_valid,
                         random_state=random_state)

    # 4. run many trials, collect true (a, x0, y0)
    A = np.empty(max_iter)
    X0 = np.empty(max_iter)
    Y0 = np.empty(max_iter)

    for i in range(max_iter):
        if sigma is None:
            ts.fit(X2_sc, y)
        else:
            w = 1.0/(sigma**2)
            mult = np.round(w / w.min()).astype(int)
            X_rep = np.repeat(X2_sc, mult, axis=0)
            y_rep = np.repeat(y,   mult, axis=0)
            ts.fit(X_rep, y_rep)

        b0 = ts.estimator_.intercept_
        b1, b2 = ts.estimator_.coef_           # on [x_sc, x_sc^2]

        # vertex in scaled space
        t0     = -b1 / (2.0*b2)
        # back-transform to original parabola
        A[i]   = b2 / (scaler.scale_[0]**2)
        X0[i]  = t0*scaler.scale_[0] + scaler.mean_[0]
        Y0[i]  = b0 + b1*t0 + b2*(t0**2)

    # aggregate by medians
    a  = np.median(A)
    x0 = np.median(X0)
    y0 = np.median(Y0)

    return a, x0, y0

def huber_parabola(x, y, sigma=None, epsilon=1.35, max_iter=1000):
    """
    Robust quadratic fit via HuberRegressor, with proper preprocessing.
    """
    # 1) build pipeline:
    #    - expand to [x, x^2]
    #    - center+scale each column to unit variance
    #    - fit the Huber loss
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        HuberRegressor(epsilon=epsilon, max_iter=max_iter)
    )

    # 2) if you have uncertainties, convert to sample_weight
    if sigma is None:
        model.fit(x[:,None], y)
    else:
        sw = 1.0/(sigma**2)
        model.fit(x[:,None], y, huberregressor__sample_weight=sw)

    # 3) extract the final linear model on the standardized space
    #    pipeline.named_steps['huberregressor'].coef_ corresponds
    #    to the scaled [x, x^2] columns
    hr = model.named_steps['huberregressor']
    ss = model.named_steps['standardscaler']

    # un-scale the coefficients back to original basis
    # if X2_std = (X2 - mean_)/scale_, then coef_orig = coef_std/scale_
    coef_std = hr.coef_            # [c1, c2] on [x, x^2] *after* scaling
    coef_unscaled = coef_std / ss.scale_
    w1, w2 = coef_unscaled        # note: scale_ is array([scale_x, scale_x2])
    w0     = hr.intercept_ - np.dot(ss.mean_/ss.scale_, coef_std)

    # 4) convert back to vertex‐form
    a  = w2
    x0 = -w1/(2.0*a)
    y0 = w0 - (w1**2)/(4.0*a)
    return a, x0, y0