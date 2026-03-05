#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 06:37:39 2026

@author: jplundquist
"""

import os
import numpy as np
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from lambda_corr import lambda_corr_nb


# ---------------------------
# Core Lambda helpers
# ---------------------------
def G_for_perm(ry):
    rx = np.arange(1, ry.size + 1, dtype=np.float64)
    ry = ry.astype(np.float64, copy=False)
    G, _, M_yx, _, M_xy, _, _ = lambda_corr_nb(rx, ry, ry.size, pvals=False)
    return G, M_xy, M_yx

def lambda_from_perm(perm):
    """Return Lambda_s for permutation perm (values 1..n)."""
    n = len(perm)
    rx = np.arange(1, n + 1, dtype=np.float64)
    ry = np.asarray(perm, dtype=np.float64)
    return lambda_corr_nb(rx, ry, n, pvals=False)[0]


# ---------------------------
# RNG-clean versions of your stochastic search
# ---------------------------
def improve_by_random_swaps_anneal(ry0, steps, rng, T0=1e-3, Tend=1e-6, keep_best=True):
    """
    Stochastic local search with annealing.
    - ry0: permutation values 1..n (ndarray)
    - steps: proposals
    Returns: (best_ry, bestG, bestMxy, bestMyx)
    """
    ry = ry0.copy()
    G, Mxy, Myx = G_for_perm(ry)
    cur = abs(G)

    best_ry = ry.copy()
    bestG, bestMxy, bestMyx = G, Mxy, Myx
    best = cur

    n = ry.size
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        T = T0 * (Tend / T0) ** frac

        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1

        ry[i], ry[j] = ry[j], ry[i]
        G2, Mxy2, Myx2 = G_for_perm(ry)
        new = abs(G2)

        d = new - cur
        if d >= 0 or (T > 0 and rng.random() < np.exp(d / T)):
            cur = new
            G, Mxy, Myx = G2, Mxy2, Myx2
            if keep_best and cur > best:
                best = cur
                best_ry = ry.copy()
                bestG, bestMxy, bestMyx = G, Mxy, Myx
        else:
            ry[i], ry[j] = ry[j], ry[i]

    return best_ry, bestG, bestMxy, bestMyx


def kick_perm(perm, rng, n_kicks=10):
    """Apply a few random (non-adjacent) swaps to jump basins/components."""
    ry = np.array(perm, dtype=np.int64).copy()
    n = ry.size
    for _ in range(n_kicks):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        ry[i], ry[j] = ry[j], ry[i]
    return ry


# ---------------------------
# BFS component expansion (single-process)
# ---------------------------
def component_bfs_adjacent(seed_perm, target=1.0):
    """
    Return the connected component (under adjacent swaps) inside {Lambda==target}
    starting from seed_perm.
    """
    if lambda_from_perm(seed_perm) != target:
        return set()

    n = len(seed_perm)
    seen = set([seed_perm])
    q = deque([seed_perm])

    p = list(seed_perm)
    while q:
        cur = q.popleft()
        p[:] = cur
        for i in range(n - 1):
            p[i], p[i + 1] = p[i + 1], p[i]
            nb_perm = tuple(p)
            p[i], p[i + 1] = p[i + 1], p[i]

            if nb_perm in seen:
                continue
            if lambda_from_perm(nb_perm) == target:
                seen.add(nb_perm)
                q.append(nb_perm)

    return seen


# ---------------------------
# Parallel seed hunting
# ---------------------------
def _seed_worker(args):
    """
    Worker: try to discover a seed permutation with bestG == target that is not in known_snapshot.
    Returns the seed tuple or None.
    """
    (n, target, known_list, known_snapshot,
     tries, steps, kick_swaps, p_kick, seed0) = args

    rng = np.random.default_rng(seed0)

    for _ in range(tries):
        if known_list is not None and rng.random() < p_kick:
            base = known_list[int(rng.integers(0, len(known_list)))]
            ry0 = kick_perm(base, rng, n_kicks=kick_swaps)
        else:
            ry0 = rng.permutation(n) + 1

        best_ry, bestG, *_ = improve_by_random_swaps_anneal(
            ry0, steps=steps, rng=rng, T0=5e-3, Tend=5e-6, keep_best=True
        )

        if bestG == target:
            seed = tuple(int(v) for v in best_ry)
            if seed not in known_snapshot:
                return seed

    return None


def find_new_component_seed_parallel(
    n, target, known_set,
    tries=400, steps=40000,
    kick_swaps=20, p_kick=0.7,
    workers=None,
    tries_per_task=25,
    max_tasks_in_flight=None,
    heartbeat_every_tasks=0
):
    """
    Run the seed-hunt in parallel. Returns a new seed tuple or None.
    """
    if workers is None:
        workers = max(1, (os.cpu_count() or 1))

    if max_tasks_in_flight is None:
        max_tasks_in_flight = workers * 2

    known_list = list(known_set) if len(known_set) > 0 else None
    known_snapshot = set(known_set)

    n_tasks = (tries + tries_per_task - 1) // tries_per_task
    base_seed = int(np.random.randint(0, 2**31 - 1))

    tasks = []
    for k in range(n_tasks):
        seed0 = (base_seed + 1000003 * k) % (2**31 - 1)
        tasks.append((n, target, known_list, known_snapshot,
                      tries_per_task, steps, kick_swaps, p_kick, int(seed0)))

    submitted = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        it = iter(tasks)

        # Prime the pump
        for _ in range(min(max_tasks_in_flight, n_tasks)):
            futures.append(ex.submit(_seed_worker, next(it)))
            submitted += 1

        while futures:
            for fut in as_completed(futures):
                futures.remove(fut)
                completed += 1

                seed = fut.result()
                if heartbeat_every_tasks and (completed % heartbeat_every_tasks == 0):
                    print(f"  seed tasks done: {completed}/{n_tasks} (known={len(known_set)})")

                if seed is not None and seed not in known_set:
                    return seed

                # Submit another if remaining
                try:
                    futures.append(ex.submit(_seed_worker, next(it)))
                    submitted += 1
                except StopIteration:
                    pass

    return None


# ---------------------------
# Main harvesting logic
# ---------------------------
def harvest_all_lambda_components(
    n, target=1.0,
    seed_tries=400, seed_steps=40000,
    max_rounds=5000, stall_rounds=2000,
    kick_swaps=20, verbose=True,
    target_count=None,
    # parallel knobs:
    workers=None, tries_per_task=25
):
    all_perms = set()
    stall = 0

    # canonical seeds
    if target == 1.0:
        canonical = tuple(range(1, n + 1))
    elif target == -1.0:
        canonical = tuple(range(n, 0, -1))
    else:
        canonical = None

    if canonical is not None and lambda_from_perm(canonical) == target:
        comp = component_bfs_adjacent(canonical, target=target)
        all_perms |= comp
        if verbose:
            print(f"Init from canonical: +{len(comp)} (total {len(all_perms)})")

    if target_count is not None and len(all_perms) >= target_count:
        if verbose:
            print(f"Reached target_count={target_count}. Stopping.")
        return all_perms

    for r in range(max_rounds):
        # seed = find_new_component_seed_parallel(
        #     n, target, all_perms,
        #     tries=seed_tries, steps=seed_steps,
        #     kick_swaps=kick_swaps, p_kick=0.7,
        #     workers=workers, tries_per_task=tries_per_task,
        #     max_tasks_in_flight=None,
        #     heartbeat_every_tasks=0
        # )

        seed = find_new_component_seed_threaded(
            n, target, all_perms,
            tries=seed_tries, steps=seed_steps,
            kick_swaps=kick_swaps, p_kick=0.7,
            workers=workers, tries_per_worker=25
            )
        
        if seed is None:
            stall += 1
            if verbose and (r % 25 == 0):
                print(f"round {r}: no new seed (stall {stall}/{stall_rounds}) total={len(all_perms)}")
            if stall >= stall_rounds:
                break
            continue

        comp = component_bfs_adjacent(seed, target=target)

        # Optional: if this component is already entirely known, don’t pretend we gained anything
        if comp.issubset(all_perms):
            stall += 1
            if verbose:
                print(f"round {r}: seed hit known component (stall {stall}/{stall_rounds}) total={len(all_perms)}")
            if stall >= stall_rounds:
                break
            continue

        before = len(all_perms)
        all_perms |= comp
        gained = len(all_perms) - before

        if gained == 0:
            stall += 1
        else:
            stall = 0

        if verbose:
            print(f"round {r}: comp={len(comp)}, gained={gained}, total={len(all_perms)}, stall={stall}")

        if target_count is not None and len(all_perms) >= target_count:
            if verbose:
                print(f"Reached target_count={target_count}. Stopping.")
            break

        if stall >= stall_rounds:
            break

    return all_perms

def _thread_seed_worker(n, target, known_list, known_set,
                        tries, steps, kick_swaps, p_kick, seed0):
    rng = np.random.default_rng(seed0)

    for _ in range(tries):
        if known_list is not None and rng.random() < p_kick:
            base = known_list[int(rng.integers(0, len(known_list)))]
            ry0 = kick_perm(base, rng, n_kicks=kick_swaps)
        else:
            ry0 = rng.permutation(n) + 1

        best_ry, bestG, *_ = improve_by_random_swaps_anneal(
            ry0, steps=steps, rng=rng, T0=5e-3, Tend=5e-6, keep_best=True
        )

        if bestG == target:
            seed = tuple(int(v) for v in best_ry)
            if seed not in known_set:   # shared set read is fine
                return seed

    return None


def find_new_component_seed_threaded(
    n, target, known_set,
    tries=400, steps=40000,
    kick_swaps=20, p_kick=0.7,
    workers=None,
    tries_per_worker=25
):
    if workers is None:
        # threads beyond cores can still help if GIL is released inside Numba
        workers = max(1, (os.cpu_count() or 1))

    known_list = list(known_set) if len(known_set) > 0 else None

    # Split total tries across threads
    n_workers = workers
    blocks = (tries + tries_per_worker - 1) // tries_per_worker

    base_seed = int(np.random.randint(0, 2**31 - 1))

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = []
        for k in range(blocks):
            seed0 = (base_seed + 1000003 * k) % (2**31 - 1)
            futures.append(ex.submit(
                _thread_seed_worker,
                n, target, known_list, known_set,
                tries_per_worker, steps, kick_swaps, p_kick, int(seed0)
            ))

        for fut in as_completed(futures):
            seed = fut.result()
            if seed is not None and seed not in known_set:
                return seed

    return None

# ---------------------------
# Example usage (MUST be under __main__ on Windows)
# ---------------------------
if __name__ == "__main__":
    n = 13

    # If lambda_corr_nb uses numba parallelism internally, you may want fewer workers
    # workers = max(1, (os.cpu_count() or 1) // 2)
    workers = None  # default: all cores

    pos = harvest_all_lambda_components(
        n, target=1.0, verbose=True,
        workers=workers, tries_per_task=25
    )
    neg = harvest_all_lambda_components(
        n, target=-1.0, verbose=True,
        workers=workers, tries_per_task=25
    )

    print("Lambda==+1:", len(pos))
    print("Lambda==-1:", len(neg))
    print("abs==1 total:", len(pos) + len(neg))

    id_perm  = tuple(range(1, n + 1))
    rev_perm = tuple(range(n, 0, -1))
    print("Lambda(identity) =", lambda_from_perm(id_perm), "in pos?", id_perm in pos)
    print("Lambda(reverse)  =", lambda_from_perm(rev_perm), "in neg?", rev_perm in neg)