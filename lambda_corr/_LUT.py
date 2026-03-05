#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 18:33:53 2026

Generate look-up table

@author: Jon Paul Lundquist
"""

import numpy as np
import itertools
import math
from lambda_corr import lambda_corr_nb

#Generation of lookup table values
def enumerate_lambda(n: int) -> np.ndarray:
    """Full enumeration of Lambda_s under permutation null (x fixed as 1..n, y permuted)."""
    x = np.arange(1, n + 1, dtype=np.float64)
    out = np.empty(math.factorial(n), dtype=np.float64)

    k = 0
    for y_perm in itertools.permutations(range(1, n + 1)):
        y = np.array(y_perm, dtype=np.float64)
        L, *_ = lambda_corr_nb(x, y, n, pvals=False)
        out[k] = L
        k += 1

    return out

def unique_counts(x: np.ndarray, atol=0.0, rtol=0.0):
    """
    Get sorted unique values and counts.
    If you want tolerance merging, set atol>0 (e.g. 1e-14). If you believe all values
    are meaningful, leave atol=0.
    """
    xs = np.sort(np.asarray(x, dtype=np.float64))

    if xs.size == 0:
        return np.empty(0, np.float64), np.empty(0,np.uint32)

    if atol == 0.0 and rtol == 0.0:
        vals, cnt = np.unique(xs, return_counts=True)
        return vals.astype(np.float64), cnt.astype(np.uint32)

    vals = []
    cnts = []

    cur_rep = xs[0]
    cur_sum = xs[0]
    cur_cnt = 1

    for v in xs[1:]:
        if np.isclose(v, cur_rep, atol=atol, rtol=rtol):
            cur_sum += v
            cur_cnt += 1
            cur_rep = cur_sum / cur_cnt
        else:
            vals.append(cur_rep)
            cnts.append(cur_cnt)
            cur_rep = v
            cur_sum = v
            cur_cnt = 1

    vals.append(cur_rep)
    cnts.append(cur_cnt)

    return np.array(vals, np.float64), np.array(cnts, np.uint32)


def build_from_counts(vals: np.ndarray, cnt: np.ndarray):
    """
    Build:
      - signed vals
      - signed CC: C(L <= v)
      - abs vals
      - abs tail: C(|L| >= t)
    """
    vals = np.asarray(vals, np.float64)
    cnt  = np.asarray(cnt,  np.uint32)
    #N = int(cnt.sum())

    # signed cumulative counts
    cc = np.cumsum(cnt).astype(np.uint32) #/ float(N)

    # abs aggregation
    abs_v = np.abs(vals)
    order = np.argsort(abs_v)
    abs_v = abs_v[order]
    abs_c = cnt[order]

    # merge identical abs-values (exact match; abs derived from vals)
    abs_vals = []
    abs_cnts = []
    cur = abs_v[0]
    curc = int(abs_c[0])
    for i in range(1, abs_v.size):
        if abs_v[i] == cur:
            curc += int(abs_c[i])
        else:
            abs_vals.append(cur)
            abs_cnts.append(curc)
            cur = abs_v[i]
            curc = int(abs_c[i])
    abs_vals.append(cur)
    abs_cnts.append(curc)

    abs_vals = np.array(abs_vals, np.float64)
    abs_cnts = np.array(abs_cnts, np.int64)

    # abs tail: P(|L| >= t). (abs_vals is ascending)
    abs_tail = np.cumsum(abs_cnts[::-1])[::-1].astype(np.uint32)
    #abs_tail = abs_tail #/ float(N)

    return vals, cc, abs_vals, abs_tail


def _print_array(name, arr, dtype, max_line=110):
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        items = [f"{v:.17g}" for v in arr.tolist()]
    else:
        items = [str(int(v)) for v in arr.tolist()]

    lines, cur = [], ""
    for s in items:
        if not cur:
            cur = s
        elif len(cur) + 2 + len(s) <= max_line:
            cur += ", " + s
        else:
            lines.append(cur)
            cur = s
    if cur:
        lines.append(cur)

    body = ",\n        ".join(lines)
    print(f"{name} = np.array([\n        {body}\n    ], dtype={dtype})")


def print_lut(n: int, atol=0.0, rtol=0.0):
    """
    Generates and prints the 4 arrays for a given n:
      LUT_VALS_Nn, LUT_CC_Nn, LUT_ABS_VALS_Nn, LUT_ABS_TAIL_Nn
    """
    full = enumerate_lambda(n)
    vals, cnt = unique_counts(full, atol=atol, rtol=rtol)

    # sanity
    N_expected = math.factorial(n)
    if int(cnt.sum()) != N_expected:
        raise RuntimeError(f"Count mismatch for n={n}: got {cnt.sum()}, expected {N_expected}")

    vals, cc, abs_vals, abs_tail = build_from_counts(vals, cnt)

    print(f"\n# ---------- n = {n} (total perms = {N_expected}) ----------")
    _print_array(f"LUT_VALS_N{n}", vals, "np.float64")
    _print_array(f"LUT_CC_N{n}", cc, "np.uint32")
    _print_array(f"LUT_ABS_VALS_N{n}", abs_vals, "np.float64")
    _print_array(f"LUT_ABS_TAIL_N{n}", abs_tail, "np.uint32")


def generate_luts(n_min=3, n_max=9, atol=0.0, rtol=0.0):
    """
    Print LUT blocks for n in [n_min, n_max].
    n=9 is usually fine; n=10 starts getting large.
    """
    for n in range(n_min, n_max + 1):
        print_lut(n, atol=atol, rtol=rtol)
        
def save_lut_npz(filename: str, n_min=3, n_max=9, atol=0.0, rtol=0.0):
    """
    Generate LUT arrays for n in [n_min, n_max] and save to a compressed NPZ.

    Saved keys per n:
        LUT_VALS_N{n}      (float64)  signed unique values
        LUT_CC_N{n}        (uint32)   cumulative counts: count(L <= vals[i])
        LUT_ABS_VALS_N{n}  (float64)  unique |L| support (ascending)
        LUT_ABS_TAIL_N{n}  (uint32)   tail counts: count(|L| >= abs_vals[i])
    """
    data = {}
    for n in range(n_min, n_max + 1):
        full = enumerate_lambda(n)
        vals, cnt = unique_counts(full, atol=atol, rtol=rtol)

        # sanity
        N_expected = math.factorial(n)
        if int(cnt.sum()) != N_expected:
            raise RuntimeError(
                f"Count mismatch for n={n}: got {int(cnt.sum())}, expected {N_expected}"
            )

        vals, cc, abs_vals, abs_tail = build_from_counts(vals, cnt)

        data[f"LUT_VALS_N{n}"] = np.ascontiguousarray(vals, dtype=np.float64)
        data[f"LUT_CC_N{n}"] = np.ascontiguousarray(cc, dtype=np.uint32)
        data[f"LUT_ABS_VALS_N{n}"] = np.ascontiguousarray(abs_vals, dtype=np.float64)
        data[f"LUT_ABS_TAIL_N{n}"] = np.ascontiguousarray(abs_tail, dtype=np.uint32)

        print(f"n={n}: vals={vals.size}, abs_vals={abs_vals.size}")

    np.savez_compressed(filename, **data)
    print(f"\nSaved LUT NPZ: {filename}")


# convenience wrapper matching your old name
def generate_luts(n_min=3, n_max=10, atol=0.0, rtol=0.0, filename="lambda_lut_n3_n10.npz"):
    save_lut_npz(filename, n_min=n_min, n_max=n_max, atol=atol, rtol=rtol)
