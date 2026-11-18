"""Statistics helpers against brute-force / hand-computed values."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from alife_lab.analysis.stats import (
    bootstrap_mean_diff_ci,
    cliffs_delta,
    compare,
    describe,
    holm,
    permutation_test,
)


def brute_force_p(x: list[float], y: list[float]) -> float:
    pooled = x + y
    obs = abs(np.mean(x) - np.mean(y))
    hits = total = 0
    for combo in itertools.combinations(range(len(pooled)), len(x)):
        a = [pooled[i] for i in combo]
        b = [pooled[i] for i in range(len(pooled)) if i not in combo]
        total += 1
        hits += abs(np.mean(a) - np.mean(b)) >= obs - 1e-12
    return hits / total


def test_describe() -> None:
    d = describe([1.0, 2.0, 3.0, 4.0])
    assert d["n"] == 4 and d["mean"] == 2.5 and d["median"] == 2.5
    assert d["sd"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert describe([])["mean"] is None and describe([5.0])["sd"] is None


def test_exact_permutation_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    for _ in range(5):
        x = rng.normal(0, 1, 5).round(3).tolist()
        y = rng.normal(0.8, 1, 6).round(3).tolist()
        res = permutation_test(x, y)
        assert res["method"] == "exact"
        assert res["p_value"] == pytest.approx(brute_force_p(x, y))


def test_complete_separation_6v6() -> None:
    res = permutation_test([10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5])
    # Two-sided: only the observed split and its mirror are as extreme -> 2 / C(12, 6).
    assert res["p_value"] == pytest.approx(2 / 924)
    assert cliffs_delta([10, 11, 12], [0, 1, 2]) == 1.0
    assert cliffs_delta([0, 1, 2], [10, 11, 12]) == -1.0


def test_identical_groups_are_not_significant() -> None:
    res = permutation_test([1, 2, 3, 4], [1, 2, 3, 4])
    assert res["p_value"] == 1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_monte_carlo_is_seeded_and_never_zero() -> None:
    x = list(np.arange(20.0))
    y = list(np.arange(20.0) + 100)
    a = permutation_test(x, y, exact_limit=10)
    b = permutation_test(x, y, exact_limit=10)
    assert a["method"] == "monte_carlo" and a == b
    assert 0 < a["p_value"] < 0.001


def test_bootstrap_ci_contains_true_difference() -> None:
    rng = np.random.default_rng(1)
    x = (rng.normal(5, 1, 40)).tolist()
    y = (rng.normal(3, 1, 40)).tolist()
    lo, hi = bootstrap_mean_diff_ci(x, y)  # type: ignore[misc]
    assert lo < 2.0 < hi and hi - lo < 1.5
    assert bootstrap_mean_diff_ci([1.0], [2.0]) is None


def test_holm() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, None]) == [0.5, None]


def test_compare_shape() -> None:
    out = compare([3.0, 4.0, 5.0], [1.0, 2.0, 3.0])
    assert out["mean_difference"] == 2.0
    assert out["cliffs_delta"] == pytest.approx(8 / 9)  # 8 of 9 pairs greater, one tie (3, 3)
    assert out["permutation"]["method"] == "exact"
