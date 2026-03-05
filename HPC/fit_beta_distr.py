from lambda_corr import lambda_corr_nb
import os
import numpy as np
import math
from mpi4py import MPI
import numba as nb
from numba import njit, prange, get_num_threads
from scipy.stats import beta
import gc
import socket
import time

comm = MPI.COMM_WORLD
if comm.rank == 0:
    print("MPI size =", comm.size, flush=True)
    print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))
    print("NUMBA_NUM_THREADS   =", os.environ.get("NUMBA_NUM_THREADS"))
    print("numba.get_num_threads() =", nb.get_num_threads(), flush=True)

print(f"rank {comm.rank}/{comm.size} on {socket.gethostname()}", flush=True)

#want = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) * 2
#nb.set_num_threads(want)

nb.set_num_threads(int(os.environ.get("NUMBA_NUM_THREADS", "28")))

@njit(cache=True, nogil=True, parallel=True)
def compute_lambda0_block(N, batch, seed):
    out = np.empty(batch, dtype=np.float64)

    nthreads = get_num_threads()
    nchunks = nthreads * 2
    if nchunks > batch:
        nchunks = batch
    if nchunks < 1:
        nchunks = 1

    #cfac = math.sqrt(max(1e-12, 1.0 - rho * rho))

    for c in prange(nchunks):
        np.random.seed(seed + 1000003 * c)

        start = (c * batch) // nchunks
        end   = ((c + 1) * batch) // nchunks

        # reuse these within chunk
        x = np.empty(N, dtype=np.float64)
        y = np.empty(N, dtype=np.float64)

        for i in range(start, end):
            for j in range(N):
                #xj = np.random.standard_normal()
                yj = np.random.standard_normal()
                #x[j] = xj
                x[j] = j + 1.0
                y[j] = yj #rho * xj + cfac * zj

            out[i] = lambda_corr_nb(x, y, N, pvals=False)[0]

    return out

def compute_L0_local(N, local_total, batch, seed_base, comm, rank, TOTAL, size):
    out = np.empty(local_total, dtype=np.float64)
    pos = 0
    b = 0

    t0 = time.time()
    next_report = t0 + 60.0

    while pos < local_total:
        this_batch = min(batch, local_total - pos)
        bs = compute_lambda0_block(N, this_batch, seed_base + b)
        out[pos:pos+this_batch] = bs
        pos += this_batch
        b += 1

        # SAFE progress: rank 0 only, no MPI calls
        if rank == 0:
            now = time.time()
            if now >= next_report:
                frac = pos / local_total
                rate0 = pos / (now - t0)          # rank0 samples/s
                rate_est = rate0 * size           # global samples/s estimate
                eta = (TOTAL * (1 - frac)) / rate_est if rate_est > 0 else float("inf")

                print(f"[est] {pos*size:,}/{TOTAL:,} ({frac:.2%})  "
                      f"{rate_est:,.0f} samples/s  ETA {eta/3600:.2f} h",
                      flush=True)

                while next_report <= now:
                    next_report += 60.0

    print(f"[rank {rank}] Finished compute_L0_local (pos={pos:,})", flush=True)
    return out

def peak_rss_kb():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0

def main():
    N     = int(os.environ.get("L0_N", "425"))
    TOTAL = int(os.environ.get("L0_TOTAL", "1000000000"))
    TOTAL = int(os.environ.get("L0_TOTAL", "1000000000")) 
    BATCH = int(os.environ.get("L0_BATCH", "200000")) 
    #RHO = float(os.environ.get("L0_RHO", "0.0")) 
    SEED0 = int(os.environ.get("L0_SEED", "21645"))

    comm = MPI.COMM_WORLD 
    rank = comm.Get_rank() 
    size = comm.Get_size()
    cpus = int(os.environ.get("NUMBA_NUM_THREADS", "28")) 
    nb.set_num_threads(cpus) 
    # split TOTAL across ranks 
    base = TOTAL // size 
    rem = TOTAL % size 
    local_total = base + (1 if rank < rem else 0)

    seed_base = SEED0 + rank * 10_000_000
    L0_local = compute_L0_local(N, local_total, BATCH, seed_base, comm, rank, TOTAL, size)

    comm.Barrier()
    if rank == 0:
        print("All ranks finished compute; starting gather phase", flush=True)
    comm.Barrier()

    counts = comm.gather(local_total, root=0)

    if rank == 0:
        counts = np.array(counts, dtype=np.int64)
        displs = np.zeros(size, dtype=np.int64)
        displs[1:] = np.cumsum(counts[:-1])
        L0 = np.empty(TOTAL, dtype=np.float64)
    else:
        displs = None
        L0 = None
        counts = None

    comm.Barrier()
    if rank == 0:
        print("Starting gather...", flush=True)
    comm.Barrier()
    t = time.time()
    comm.Gatherv(L0_local, [L0, counts, displs, MPI.DOUBLE], root=0)

    comm.Barrier()
    if rank == 0:
        print(f"Finished Gatherv in {(time.time()-t)/60:.1f} min", flush=True)
        t = time.time()
        
        p0 = np.mean(L0 == 0.0)

        np.abs(L0, out=L0)   # L0 becomes |L0| in-place (no extra 8GB abs array)
        p1 = np.mean(L0 == 1.0)

        mask = (L0 != 0.0) & (L0 != 1.0)
        x = L0[mask]         # x in (0,1)
        
        del mask, L0
        gc.collect()
        gc.collect()

        print(f"Prepared x (len={x.size:,}) in {(time.time()-t)/60:.1f} min", flush=True)
        t = time.time()

        a, b, loc, scale = beta.fit(x, floc=0.0, fscale=1.0)
        print(f"beta.fit done in {(time.time()-t):.1f} s", flush=True)

        print(N)
        print(repr(p0))
        print(repr(p1))
        print(repr(a))
        print(repr(b))

    info = (rank, socket.gethostname(), peak_rss_kb())
    allinfo = comm.gather(info, root=0)
    if rank == 0:
        worst = max(allinfo, key=lambda t: t[2])
        print(f"Peak RSS (worst rank): {worst[2]/1024/1024:.2f} GiB "
              f"(rank {worst[0]} on {worst[1]})",
              flush=True)

if __name__ == "__main__":
    main()
