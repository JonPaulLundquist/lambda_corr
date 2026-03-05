#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 16:57:23 2026

@author: jplundquist
"""

import numpy.random as nr
import numpy as np
from lambda_corr import lambda_corr_nb
from collections import deque

def G_for_perm(ry):
    rx = np.arange(1, ry.size + 1)
    #Mxy = mean_of_point_medians_identity_x(ry)
    # swap x<->y: treat ry as the x-ranks (in given order) and identity as y
    #Myx = mean_of_point_medians_general(ry, rx)
    #G = sym_geom(Mxy, Myx, sign_from=sign_from)
    G, _, M_yx, _, M_xy, _, _ = lambda_corr_nb(rx, ry, ry.size, pvals=False)
    return G, M_xy, M_yx

def improve_by_random_swaps_anneal(ry0, steps, T0=1e-3, Tend=1e-6, keep_best=True):
    """
    Stochastic local search with annealing.
    - ry0: permutation values 1..n
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
        # geometric cooling
        frac = t / max(steps - 1, 1)
        T = T0 * (Tend / T0) ** frac

        i = int(nr.randint(0, n))
        j = int(nr.randint(0, n-1))
        if j >= i:
            j += 1  # ensure j != i

        ry[i], ry[j] = ry[j], ry[i]
        G2, Mxy2, Myx2 = G_for_perm(ry)
        new = abs(G2)

        d = new - cur
        if d >= 0 or (T > 0 and nr.random() < np.exp(d / T)):
            # accept
            cur = new
            G, Mxy, Myx = G2, Mxy2, Myx2
            if keep_best and cur > best:
                best = cur
                best_ry = ry.copy()
                bestG, bestMxy, bestMyx = G, Mxy, Myx
        else:
            # reject: revert
            ry[i], ry[j] = ry[j], ry[i]

    return best_ry, bestG, bestMxy, bestMyx

def lambda_from_perm(perm):
    """Return Lambda_s for permutation perm (values 1..n)."""
    n = len(perm)
    rx = np.arange(1, n + 1, dtype=np.float64)
    ry = np.asarray(perm, dtype=np.float64)
    return lambda_corr_nb(rx, ry, n, pvals=False)[0]  # no starred unpack


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


def kick_perm(perm, n_kicks=10):
    """Apply a few random (non-adjacent) swaps to a permutation to jump basins/components."""
    ry = np.array(perm, dtype=np.int64).copy()
    n = ry.size
    for _ in range(n_kicks):
        i = int(nr.randint(0, n))
        j = int(nr.randint(0, n - 1))
        if j >= i:
            j += 1
        ry[i], ry[j] = ry[j], ry[i]
    return ry


def find_new_component_seed(n, target, known_set,
                            tries=400, steps=40000,
                            kick_from_known=True, kick_swaps=20,
                            p_kick=0.7, progress_every=50):
    known_list = list(known_set) if (kick_from_known and len(known_set) > 0) else None

    for t in range(tries):
        if known_list is not None and nr.random() < p_kick:
            base = known_list[int(nr.randint(0, len(known_list)))]
            ry0 = kick_perm(base, n_kicks=kick_swaps)
        else:
            ry0 = nr.permutation(n) + 1

        best_ry, bestG, *_ = improve_by_random_swaps_anneal(
            ry0, steps=steps, T0=5e-3, Tend=5e-6, keep_best=True
        )

        if (t + 1) % progress_every == 0:
            # small heartbeat so it doesn't look frozen
            print(f"  seed attempts: {t+1}/{tries} (known={len(known_set)})")

        if bestG == target:
            seed = tuple(int(v) for v in best_ry)
            if seed not in known_set:
                return seed

    return None


def harvest_all_lambda_components(n, target=1.0,
                                  seed_tries=400, seed_steps=40000,
                                  max_rounds=5000, stall_rounds=2000,
                                  kick_swaps=20, verbose=True,
                                  target_count=None):
    all_perms = set()
    stall = 0

    # canonical seeds
    if target == 1.0:
        canonical = tuple(range(1, n + 1))
    elif target == -1.0:
        canonical = tuple(range(n, 0, -1))
    else:
        canonical = None

    # include canonical component if it matches target
    if canonical is not None and lambda_from_perm(canonical) == target:
        comp = component_bfs_adjacent(canonical, target=target)
        all_perms |= comp
        if verbose:
            print(f"Init from canonical: +{len(comp)} (total {len(all_perms)})")

    # If we already hit target_count from canonical, stop immediately
    if target_count is not None and len(all_perms) >= target_count:
        if verbose:
            print(f"Reached target_count={target_count}. Stopping.")
        return all_perms

    for r in range(max_rounds):
        seed = find_new_component_seed(
            n, target, all_perms,
            tries=seed_tries, steps=seed_steps,
            kick_from_known=True, kick_swaps=kick_swaps
        )

        if seed is None:
            stall += 1
            if verbose and (r % 25 == 0):
                print(f"round {r}: no new seed (stall {stall}/{stall_rounds}) total={len(all_perms)}")
            if stall >= stall_rounds:
                break
            continue

        comp = component_bfs_adjacent(seed, target=target)

        before = len(all_perms)
        all_perms |= comp
        gained = len(all_perms) - before

        if gained == 0:
            stall += 1
        else:
            stall = 0

        if verbose:
            print(f"round {r}: comp={len(comp)}, gained={gained}, total={len(all_perms)}, stall={stall}")

        # STOP as soon as we have enough
        if target_count is not None and len(all_perms) >= target_count:
            if verbose:
                print(f"Reached target_count={target_count}. Stopping.")
            break

        if stall >= stall_rounds:
            break

    return all_perms


# ---------------------------
# Example usage:
# ---------------------------
n = 13
pos = harvest_all_lambda_components(n, target=1.0, verbose=True)  #This keeps going until you force stop.
neg = harvest_all_lambda_components(n, target=-1.0, verbose=True)
print("Lambda==+1:", len(pos))
print("Lambda==-1:", len(neg))
print("abs==1 total:", len(pos) + len(neg))

# sanity check that starting points are included:
id_perm  = tuple(range(1, n+1))
rev_perm = tuple(range(n, 0, -1))
print("Lambda(identity) =", lambda_from_perm(id_perm), "in pos?", id_perm in pos)
print("Lambda(reverse)  =", lambda_from_perm(rev_perm), "in neg?", rev_perm in neg)



