"""CMA-ES oracle test: our NumPy update equals Hansen's purecma (pycma 4.x) when both are told
the same candidates and f-values; plus convergence, invariance and serialisation checks."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from alife_lab.evolution.cmaes import CMAParams, CMAState, ask, eigensystem, tell

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    purecma = pytest.importorskip("cma.more_algorithms.purecma")


def elli(x: np.ndarray) -> float:
    n = x.size
    return float(sum(1e6 ** (i / (n - 1)) * x[i] ** 2 for i in range(n)))


def rosen(x: np.ndarray) -> float:
    return float(sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


@pytest.mark.parametrize(("n", "popsize", "f"), [(2, None, rosen), (3, 8, elli), (4, None, elli)])
def test_update_matches_purecma(n: int, popsize: int | None, f) -> None:  # type: ignore[no-untyped-def]
    x0 = [0.3] * n
    es = purecma.CMAES(x0, 0.5, popsize=popsize or purecma.CMAESParameters.default_popsize)
    par = CMAParams.default(n, popsize)
    assert par.lam == es.params.lam and par.mu == es.params.mu
    # purecma refreshes C^-1/2 lazily; make it refresh at every ask, as ours does.
    es.params.lazy_gap_evals = 0
    for attr in ("mueff", "cc", "cs", "c1", "cmu", "damps"):
        assert getattr(par, attr) == pytest.approx(getattr(es.params, attr), rel=1e-15)
    st = CMAState.start(np.array(x0), 0.5)
    for _ in range(40):
        arx = es.ask()  # purecma's own sampling; both see identical candidates
        fit = [f(np.asarray(x)) for x in arx]
        eigensystem(st)
        tell(st, par, np.asarray(arx, dtype=np.float64), np.asarray(fit))
        es.tell(arx, fit)
        np.testing.assert_allclose(st.mean, es.xmean, rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(st.C, np.asarray([list(r) for r in es.C]), rtol=1e-9, atol=1e-14)
        assert st.sigma == pytest.approx(es.sigma, rel=1e-10)


def test_converges_on_sphere_and_is_deterministic() -> None:
    def run(seed: int) -> CMAState:
        rng = np.random.Generator(np.random.PCG64(seed))
        par = CMAParams.default(5)
        st = CMAState.start(np.full(5, 3.0), 1.0)
        for _ in range(200):
            x = ask(st, par, rng)
            tell(st, par, x, (x**2).sum(axis=1))
        return st

    a, b = run(1), run(1)
    assert np.array_equal(a.mean, b.mean) and a.sigma == b.sigma
    assert float((a.mean**2).sum()) < 1e-12


def test_rank_based_invariance_to_monotone_transform() -> None:
    """CMA-ES uses only the ranking of f, so f and log1p(f) give identical trajectories."""
    states = []
    for g in (lambda v: v, np.log1p):
        rng = np.random.Generator(np.random.PCG64(3))
        par = CMAParams.default(3)
        st = CMAState.start(np.ones(3), 0.4)
        for _ in range(30):
            x = ask(st, par, rng)
            tell(st, par, x, g(np.array([elli(r) for r in x]) * 1e-3))
        states.append(st)
    assert np.array_equal(states[0].mean, states[1].mean)
    assert states[0].sigma == states[1].sigma


def test_json_roundtrip_continues_identically() -> None:
    rng = np.random.Generator(np.random.PCG64(9))
    par = CMAParams.default(3)
    st = CMAState.start(np.zeros(3), 0.3)
    for _ in range(5):
        x = ask(st, par, rng)
        tell(st, par, x, np.array([rosen(r) for r in x]))
    copy = CMAState.from_json(st.to_json())
    rng2 = np.random.Generator(np.random.PCG64())
    rng2.bit_generator.state = rng.bit_generator.state
    for s, r in ((st, rng), (copy, rng2)):
        x = ask(s, par, r)
        tell(s, par, x, np.array([rosen(v) for v in x]))
    assert np.array_equal(st.mean, copy.mean) and np.array_equal(st.C, copy.C)
