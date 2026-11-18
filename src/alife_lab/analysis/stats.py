"""Small, dependency-free statistics for seed-level comparisons (NumPy only).

The experimental unit is one independently seeded run (one value per run). Time points of a
single trajectory are never treated as independent samples.

* ``describe``: n, mean, sd (ddof=1), median, min, max.
* ``permutation_test``: two-sided test of the difference in means. Exact (all relabellings)
  when the number of distinct splits is ≤ ``exact_limit``; otherwise Monte Carlo with a fixed
  seed, using the (k+1)/(m+1) estimator so that p is never 0.
* ``cliffs_delta``: P(X > Y) − P(X < Y), robust ordinal effect size in [−1, 1].
* ``bootstrap_mean_diff_ci``: percentile bootstrap CI of mean(x) − mean(y), fixed seed.
* ``holm``: Holm–Bonferroni step-down adjustment.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

EXACT_LIMIT = 200_000
MC_PERMUTATIONS = 20_000


def describe(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None, "min": None, "max": None}
    return {
        "n": n,
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if n > 1 else None,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def permutation_test(
    x: list[float], y: list[float], exact_limit: int = EXACT_LIMIT, seed: int = 0
) -> dict[str, Any]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    nx, ny = a.size, b.size
    if nx == 0 or ny == 0:
        return {"p_value": None, "method": "undefined", "permutations": 0}
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    tol = 1e-12 * max(1.0, abs(observed))
    total = math.comb(nx + ny, nx)
    if total <= exact_limit:
        count = 0
        idx = np.arange(nx + ny)
        pooled_sum = pooled.sum()
        for combo in itertools.combinations(idx, nx):
            sx = pooled[list(combo)].sum()
            diff = abs(sx / nx - (pooled_sum - sx) / ny)
            if diff >= observed - tol:
                count += 1
        return {"p_value": count / total, "method": "exact", "permutations": total}
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(MC_PERMUTATIONS):
        perm = rng.permutation(pooled)
        if abs(perm[:nx].mean() - perm[nx:].mean()) >= observed - tol:
            count += 1
    return {
        "p_value": (count + 1) / (MC_PERMUTATIONS + 1),
        "method": "monte_carlo",
        "permutations": MC_PERMUTATIONS,
        "seed": seed,
    }


def cliffs_delta(x: list[float], y: list[float]) -> float | None:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return None
    greater = (a[:, None] > b[None, :]).sum()
    less = (a[:, None] < b[None, :]).sum()
    return float((greater - less) / (a.size * b.size))


def bootstrap_mean_diff_ci(
    x: list[float], y: list[float], level: float = 0.95, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float] | None:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return None
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(diffs, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def holm(p_values: list[float | None]) -> list[float | None]:
    indexed = [(p, i) for i, p in enumerate(p_values) if p is not None]
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(indexed)):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)  # enforce monotonicity
        adjusted[i] = running
    return adjusted


def compare(treatment: list[float], reference: list[float], seed: int = 0) -> dict[str, Any]:
    """Treatment-minus-reference comparison with effect sizes and a permutation test."""
    perm = permutation_test(treatment, reference, seed=seed)
    ci = bootstrap_mean_diff_ci(treatment, reference, seed=seed)
    t = describe(treatment)
    r = describe(reference)
    diff = None if t["mean"] is None or r["mean"] is None else t["mean"] - r["mean"]
    return {
        "mean_difference": diff,
        "mean_difference_ci95": list(ci) if ci else None,
        "ci_method": "percentile bootstrap, 10000 resamples, fixed seed",
        "cliffs_delta": cliffs_delta(treatment, reference),
        "permutation": perm,
        "n_treatment": t["n"],
        "n_reference": r["n"],
    }
