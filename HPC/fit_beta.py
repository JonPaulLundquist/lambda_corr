from numba import njit, prange, get_num_threads
import numpy as np
import math
from lambda_corr import lambda_corr_nb
from tqdm import tqdm

@njit(cache=True, nogil=True, inline='always')
def draw_bivariate_normal_numba(n, rho):
    x = np.random.standard_normal(n)
    z = np.random.standard_normal(n)
    c = math.sqrt(max(1e-12, 1.0 - rho * rho))
    y = rho * x + c * z
    return x, y

@njit(cache=True, nogil=True, parallel=True)
def compute_lambda0_block(N, batch, rho, seed):
    """
    Compute 'batch' samples of Lambda_s (L0) with per-thread array reuse.
    Much less allocation than drawing fresh vectors each sample.
    """
    out = np.empty(batch, dtype=np.float64)

    nthreads = get_num_threads()
    # More chunks than threads helps load balance if lambda_corr cost varies slightly
    nchunks = nthreads * 2
    if nchunks > batch:
        nchunks = batch
    if nchunks < 1:
        nchunks = 1

    for c in prange(nchunks):
        # Each chunk gets its own RNG seed (Numba seed is thread-local)
        np.random.seed(seed + 1000003 * c)

        start = (c * batch) // nchunks
        end   = ((c + 1) * batch) // nchunks

        # Allocate once per chunk (reused for many samples)
        x = np.empty(N, dtype=np.float64)
        z = np.empty(N, dtype=np.float64)
        y = np.empty(N, dtype=np.float64)

        cfac = math.sqrt(max(1e-12, 1.0 - rho * rho))

        for i in range(start, end):
            # Fill arrays in-place with scalar RNG
            for j in range(N):
                xj = np.random.standard_normal()
                zj = np.random.standard_normal()
                x[j] = xj
                z[j] = zj
                y[j] = rho * xj + cfac * zj

            out[i] = lambda_corr_nb(x, y, N, pvals=False)[0]

    return out

def compute_lambda0(N, total, rho=0.0, batch=1_000_000, seed=12345):
    out = np.empty(total, dtype=np.float64)

    n_batches = total // batch
    rem = total % batch
    pos = 0
    cur_seed = seed

    # If you keep tqdm, consider disable=True in SLURM to avoid huge logs
    with tqdm(total=total, unit="samples", desc=f"Λ₀ (N={N})") as bar:
        for _ in range(n_batches):
            bs = compute_lambda0_block(N, batch, rho, cur_seed)
            out[pos:pos+batch] = bs
            pos += batch
            cur_seed += 1
            bar.update(batch)

        if rem:
            bs = compute_lambda0_block(N, rem, rho, cur_seed)
            out[pos:pos+rem] = bs
            bar.update(batch)

    return out

if __name__ == "__main__":
    print("SLURM_CPUS_PER_TASK =", __import__("os").environ.get("SLURM_CPUS_PER_TASK"))
    print("NUMBA default threads =", get_num_threads())
    # warm-up compile (important for ETA)
    _ = compute_lambda0_block(10, 20, 0.0, 123)

    N = 225
    L0 = compute_lambda0(N, 1_000_000_000, rho=0.0)

    mask = (L0 != 0.0) & (np.abs(L0) != 1.0)
    alpha, beta_param, loc, scale = beta.fit(np.abs(L0[mask]), floc=0.0, fscale=1.0)

    print(repr(np.mean(L0 == 0.0)))
    print(repr(np.mean(np.abs(L0) == 1.0)))
    print(repr(alpha))
    print(repr(beta_param))
