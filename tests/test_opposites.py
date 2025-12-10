#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:56:45 2025

@author: jplundquist
"""

import numpy as np
from lambda_corr import lambda_corr

def search_for_opposite_signs(n_trials=10000, seed_start=0):
    """
    Search for random samples where Lambda_yx and Lambda_xy have opposite signs.

    Strategy: Use small sample sizes and try various data generation methods
    that might produce weak or ambiguous associations.
    """

    found_cases = []

    print("Searching for cases where Λ_yx and Λ_xy have opposite signs...")
    print("=" * 70)

    for trial in range(n_trials):
        rng = np.random.default_rng(seed=seed_start + trial)

        # Try different scenarios that might produce sign disagreement
        scenario = trial % 5

        if scenario == 0:
            # Very small sample with pure noise
            n = 5
            x = rng.standard_normal(n)
            y = rng.standard_normal(n)

        elif scenario == 1:
            # Small sample with outlier
            n = 8
            x = rng.standard_normal(n)
            y = rng.standard_normal(n)
            # Add one strong outlier
            x[-1] = 3 * np.std(x)
            y[-1] = -3 * np.std(y)

        elif scenario == 2:
            # Small sample with non-monotonic pattern
            n = 10
            x = np.linspace(-2, 2, n)
            y = x**2 + 0.5 * rng.standard_normal(n)

        elif scenario == 3:
            # Small sample with heavy ties
            n = 12
            x = rng.choice([1, 2, 3], size=n)
            y = rng.choice([1, 2, 3], size=n)

        else:
            # Small sample with mixed correlation structure
            n = 15
            x = rng.standard_normal(n)
            # First half positive correlation, second half negative
            y = np.zeros(n)
            mid = n // 2
            y[:mid] = x[:mid] + 0.3 * rng.standard_normal(mid)
            y[mid:] = -x[mid:] + 0.3 * rng.standard_normal(n - mid)

        try:
            Lambda_s, p_s, Lambda_yx, p_yx, Lambda_xy, p_xy, Lambda_a = lambda_corr(
                x, y, pvals=False
            )

            # Check for opposite signs
            if np.isfinite(Lambda_yx) and np.isfinite(Lambda_xy):
                if Lambda_yx * Lambda_xy < 0:  # Opposite signs
                    found_cases.append({
                        'trial': trial,
                        'seed': seed_start + trial,
                        'scenario': scenario,
                        'n': n,
                        'x': x.copy(),
                        'y': y.copy(),
                        'Lambda_s': Lambda_s,
                        'Lambda_yx': Lambda_yx,
                        'Lambda_xy': Lambda_xy,
                        'Lambda_a': Lambda_a
                    })

                    print(f"Found case #{len(found_cases)} at trial {trial} (seed={seed_start + trial}):")
                    print(f"  n = {n}, scenario = {scenario}")
                    print(f"  Λ_s       = {Lambda_s: .6f}")
                    print(f"  Λ(y|x)    = {Lambda_yx: .6f}")
                    print(f"  Λ(x|y)    = {Lambda_xy: .6f}")
                    print(f"  Asymmetry = {Lambda_a: .6f}")
                    print(f"  x = {x}")
                    print(f"  y = {y}")
                    print("-" * 70)

        except Exception as e:
            pass  # Skip any errors

    print(f"\n{'=' * 70}")
    print(f"Search complete: Found {len(found_cases)} cases out of {n_trials} trials")
    print(f"Rate: {len(found_cases)/n_trials*100:.2f}%")

    if found_cases:
        print("\nAnalyzing characteristics of found cases:")
        sample_sizes = [c['n'] for c in found_cases]
        lambda_s_values = [c['Lambda_s'] for c in found_cases]
        lambda_yx_abs = [abs(c['Lambda_yx']) for c in found_cases]
        lambda_xy_abs = [abs(c['Lambda_xy']) for c in found_cases]

        print(f"  Sample size range: {min(sample_sizes)} - {max(sample_sizes)}")
        print(f"  |Λ_yx| range: {min(lambda_yx_abs):.4f} - {max(lambda_yx_abs):.4f}")
        print(f"  |Λ_xy| range: {min(lambda_xy_abs):.4f} - {max(lambda_xy_abs):.4f}")
        print(f"  All Λ_s values: {set(lambda_s_values)}")

    return found_cases

def test_specific_case(x, y):
    """Test a specific case in detail."""
    print("\nDetailed analysis of specific case:")
    print("=" * 70)

    Lambda_s, p_s, Lambda_yx, p_yx, Lambda_xy, p_xy, Lambda_a = lambda_corr(
        x, y, pvals=True, ptype='perm', n_perm=5000
    )

    print(f"x = {x}")
    print(f"y = {y}")
    print(f"\nΛ_s       = {Lambda_s: .6f}   (p = {p_s: .4g})")
    print(f"Λ(y|x)    = {Lambda_yx: .6f}   (p = {p_yx: .4g})")
    print(f"Λ(x|y)    = {Lambda_xy: .6f}   (p = {p_xy: .4g})")
    print(f"Asymmetry = {Lambda_a: .6f}")

    # Also compute standard rank correlations for comparison
    from scipy.stats import spearmanr, kendalltau
    rho, p_rho = spearmanr(x, y)
    tau, p_tau = kendalltau(x, y)

    print(f"\nFor comparison:")
    print(f"Spearman ρ = {rho: .6f}   (p = {p_rho: .4g})")
    print(f"Kendall τ  = {tau: .6f}   (p = {p_tau: .4g})")

    # Key question: Is Kendall's tau always zero when signs disagree?
    if abs(tau) < 1e-10:
        print(f"\n*** Kendall τ ≈ 0 (within numerical precision) ***")
    else:
        print(f"\n*** Kendall τ ≠ 0 (τ = {tau}) ***")

if __name__ == "__main__":
    from scipy.stats import kendalltau, spearmanr

    # Search for cases
    cases = search_for_opposite_signs(n_trials=10000, seed_start=0)

    # Analyze Kendall's tau for all found cases
    if cases:
        print("\n" + "=" * 70)
        print("KENDALL'S TAU AND SPEARMAN'S RHO ANALYSIS FOR ALL CASES:")
        print("=" * 70)

        tau_values = []
        rho_values = []
        lambda_values = []
        for i, case in enumerate(cases):
            tau, _ = kendalltau(case['x'], case['y'])
            rho, _ = spearmanr(case['x'], case['y'])
            _, _,l_yx,_,l_xy,_,_ = lambda_corr(case['x'], case['y'])
            tau_values.append(tau)
            rho_values.append(rho)
            lambda_values.append(np.abs(l_yx))
            lambda_values.append(np.abs(l_xy))
            print(f"Case {i+1}: Λ(y|x)={case['Lambda_yx']:+.4f}, "
                  f"Λ(x|y)={case['Lambda_xy']:+.4f}, "
                  f"Kendall τ={tau:+.6f}, "
                  f"Spearman ρ={rho:+.6f}")

        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS:")
        print("=" * 70)
        print(f"  Total cases found: {len(cases)}")

        print(f"\n  Lambda_xy, Lambda_yx:")
        print(f"    Mean:  {np.mean(lambda_values):+.6f}")
        print(f"    Std:   {np.std(lambda_values):.6f}")
        print(f"    Min:   {np.min(lambda_values):+.6f}")
        print(f"    Max:   {np.max(lambda_values):+.6f}")

        print(f"\n  Kendall's τ:")
        print(f"    Range: [{min(tau_values):+.6f}, {max(tau_values):+.6f}]")
        print(f"    Mean:  {np.mean(tau_values):+.6f}")
        print(f"    Std:   {np.std(tau_values):.6f}")
        print(f"    Cases with |τ| < 1e-10: {sum(abs(t) < 1e-10 for t in tau_values)}")
        print(f"    Cases with |τ| < 1e-6:  {sum(abs(t) < 1e-6 for t in tau_values)}")
        print(f"    Cases with |τ| < 0.01:  {sum(abs(t) < 0.01 for t in tau_values)}")

        print(f"\n  Spearman's ρ:")
        print(f"    Range: [{min(rho_values):+.6f}, {max(rho_values):+.6f}]")
        print(f"    Mean:  {np.mean(rho_values):+.6f}")
        print(f"    Std:   {np.std(rho_values):.6f}")
        print(f"    Cases with |ρ| < 1e-10: {sum(abs(r) < 1e-10 for r in rho_values)}")
        print(f"    Cases with |ρ| < 1e-6:  {sum(abs(r) < 1e-6 for r in rho_values)}")
        print(f"    Cases with |ρ| < 0.01:  {sum(abs(r) < 0.01 for r in rho_values)}")

        print("\n" + "=" * 70)
        print("CONCLUSIONS:")
        if all(abs(t) < 1e-10 for t in tau_values):
            print("*** ALL cases have Kendall τ ≈ 0 (within machine precision) ***")
        elif all(abs(t) < 0.01 for t in tau_values):
            print("*** ALL cases have Kendall τ very close to 0 (|τ| < 0.01) ***")
        else:
            print("*** Some cases have non-negligible Kendall τ ***")

        if all(abs(r) < 1e-10 for r in rho_values):
            print("*** ALL cases have Spearman ρ ≈ 0 (within machine precision) ***")
        elif all(abs(r) < 0.01 for r in rho_values):
            print("*** ALL cases have Spearman ρ very close to 0 (|ρ| < 0.01) ***")
        else:
            print("*** Some cases have non-negligible Spearman ρ ***")

        print("\n" + "=" * 70)
        print("DETAILED ANALYSIS OF FIRST CASE:")
        test_specific_case(cases[0]['x'], cases[0]['y'])