# lambda_corr -- Repeated-Average-Rank Correlation Λ (Lambda)

The Repeated-Average-Rank correlation Λ (Lambda) defines a family of robust, 
symmetric and asymmetric measures of monotone association based on pairwise rank 
slopes. Compared with traditional rank-based measures (Spearman’s ρ and Kendall’s τ), 
Lambda is substantially more resistant to noise contamination and exhibits markedly 
less bias relative to Pearson’s linear correlation. For moderate to strong signals, 
its estimation accuracy is comparable to, and often exceeds, that of ρ and τ. These 
advantages come at the cost of a modest reduction in asymptotic efficiency: 
approximately 81% for Lambda, compared to ~91% for Spearman’s ρ and Kendall’s τ.

The canonical version, Λₛ, is the mean of median pairwise rank slopes -- that 
is for each observation i, the median slope of y ranks versus x ranks is 
calculated, then the average is taken to obtain a single robust slope estimate 
(inspired by Siegal slope [1]). The resulting coefficient is symmetrized as 
the geometric mean of Λ for x vs y and Λ for y vs x -- in the same manner as 
Kendall's symmetric τ_a can be found from the asymmetric Somers' D.

The Λₛ (mean-median) correlation combines the high noise breakdown robustness 
of the repeated median with the unbiased symmetry of an outer mean, producing 
a measure that is resistant to outliers and heavy-tailed noise while retaining 
interpretability as a standard measure of monotonic trend/association.
    
Let (x_i, y_i), i=1..n. Let rx = ranks(x) and ry = ranks(y) using average ranks
for ties, then z-score the ranks to zero mean / unit variance:
  rxt = (rx - mean(rx)) / std(rx)
  ryt = (ry - mean(ry)) / std(ry)

For each anchor i, compute the median slope in rank space:
  b_i = median_{j != i, rxt[j] != rxt[i]} ( (rxt[j] - rxt[i]) / (ryt[j] - ryt[i]) )

Aggregate by the outer mean for the asymmetric Λ(X|Y) and (Y|X):
  Lambda_xy = (1/n) * sum_i b_i
Do the same with x and y swapped to get Lambda_yx.

Symmetric Λₛ correlation is the geometric mean (just as Kendall's tau_a as 
the geometric mean of the asymmetric Somers' D):
  Λₛ = sgn(Lambda_yx) * sqrt( Lambda_yx * Lambda_xy )

Parameters:
    x, y : 1-D array_like 
        Two input samples of equal length (n ≥ 3).
    
    pvals : {True, False}, optional
        Flag for p-value calculation. Default: True.
        If False, all returned p-values are NaN and no permutation/asymptotic
        p-value calculations are performed.
    
    ptype : {"default", "asymp", "perm"}, optional
        Type of p-value calculation. Default: "default".
        - "default": If n < 25 do permutation calculation. 
                     If n ≥ 25 use asymptotic approximation.
        - "asymp": Use asymptotic approximation. Assumes no ties. 
                   The more ties the less accurate.
        - "perm": Calculates p-value using permutations. 
                  Valid with any fraction of ties.
    
    p_tol : float, optional
        If uncertainty on p-value is less than p_tol stop permutation calculation. 
        Default: 1e-5.
    
    n_perm : integer, optional
        Maximum number of Monte Carlo permutations for p-value estimation. Default: 1e4.
        Estimation will terminate earlier if the p-value uncertainty falls below p_tol. 
    
    alt : {"two-sided", "greater", "less"}, optional
        Alternative hypothesis relative to the null of no correlation. 
        Default: "two-sided".
        - "two-sided": Probability of getting larger magnitude Λ ([-1, 1]) with population 
        correlation of zero.
        - "greater": Probability of getting a greater Λ ([Λ, 1]) with population 
        correlation of zero.
        - "less": Probability of getting a smaller Λ ([-1, Λ]) with population correlation
        of zero.
        
Returns:
    Lambda_s  = Λₛ  (symmetric repeated-average-rank-correlation [-1,1])
    p_s p-value of Λₛ
    Lambda_xy = Λ(x|y)  directional slope of x on y in rank space
    p_xy p-value of Λ_xy
    Lambda_yx = Λ(y|x)  directional slope of y on x in rank space
    p_yx p-value of Λ_yx
    Lambda_a  = Λ_a  normalized asymmetry = |Λ(Y|X)-Λ(X|Y)| / (|Λ(Y|X)|+|Λ(X|Y)|) in [0,1]
    
## Properties
- As determined by MC Λₛ is slightly less efficient than Spearman/Kendall [2,3] 
  in clean Gaussian data, particularly at small ρ. For ρ>~0.6 efficiency is 
  the same as Spearman. Λₛ efficiency ~0.8 at moderate ρ → it needs about 
  1 / 0.8 ≈ 1.25× as many samples as Pearson [4] to reach the same variance.
- Asymptotic efficiency = var_opt/var_Λₛ = (1/N)/(1.112^2/N) = 81% versus
  Spearman's ρ and Kendall's τ ~91% (Siegal median of medians is ~37%).
- The most robust overall compared to Spearman, Kendall, and Pearson: with 
  increasing noise fraction it retains the strongest central signal
  with Spearman-like variability and much better stability than Pearson. As 
  outlier fraction grows, Λₛ’s median stays highest of the rank methods and 
  far above Pearson (which collapses).
- Null behavior: Λₛ is well centered, approximately symmetric, and has a
  slightly heavier null tail than Spearman.
- Symmetric: rho_RA(x, y) == rho_RA(y, x).
- Invariant to strictly monotone transforms of x or y (rank-based).
    
## Installation
The library targets Python 3.8+ and requires NumPy and Numba (for speed).

```bash
pip install numba numpy

#Some statistical tests that are not the main function make use of SciPy
pip install scipy

# optional for fast math optimizations on Intel CPUs
pip install icc_rt

#Install hyper-corr from pypi with pip
pip install lambda-corr

# or local install from source
pip install -e .

```

## Quick Example
```python
import numpy as np
import math
from lambda_corr import lambda_corr

rng = np.random.default_rng(seed=0)

n = 50
rho = 0.5 #correlation strength
x = rng.standard_normal(n)
z = rng.standard_normal(n)
c = math.sqrt(max(1e-12, 1.0 - rho * rho))
y = np.exp(rho * x + c * z) #any monotonic function

#Lambda_s will be close to rho
Lambda_s, p_S, Lambda_xy, p_xy, Lambda_yx, p_yx, Lambda_A = lambda_corr(x,y)

```

## Notes
- A fully repeated-median Λ has maximal robustness but reduced asymptotic 
  efficiency, while a mean-of-medians variant recovers much of the efficiency 
  at minimal loss of breakdown.
- A mean-of-means Λ is Theil-Sen in rank-space and is essentially Spearman in 
  both efficiency and null spread but gives up most of the robustness advantage 
  compared to mean of medians.
- Continuum of Lambda variants behavior (outside loop - inside loop):
  Spearman (ρ) ≈ Λₛ^(mean-mean)  <->  Λₛ^(mean-median)  <-> Λₛ^(median-mean)  <->  
                                                          Λₛ^(median-median) ≈ Siegel
  Canonical choice: Λₛ^(mean-median) — best efficiency/robustness balance 
                                              especially at low statistics)

## Implementation Notes
- Skip vertical pairs where rxt[j] == rxt[i].
- If all slopes for an i are undefined (e.g., all rxt equal), set b_i = NaN and
  ignore in the outer mean; if all b_i are NaN, return NaN.
- If asymmetric Λ_xy/Λ_yx have opposite signs Λ_s is taken as zero.
    
## References
[1] Spearman, C. The proof and measurement of association between two things. 
      American Journal of Psychology, 15(1), 72–101, 1904.
[2] Kendall, M.G., Rank Correlation Methods (4th Edition), Charles 
      Griffin & Co., 1970.
[3] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
[4]Siegel, A.F., Robust Regression Using Repeated Medians, Biometrika, 
      Vol. 69, pp. 242-244, 1982.
      
## Dependencies
- Python ≥ 3.8  
- NumPy ≥ 1.23
- Numba ≥ 0.61
- SciPy ≥ 1.9 #If needed for verification of statistical testing

## License
This project is licensed under the MIT license.  
See the full text in [LICENSE](./LICENSE).
