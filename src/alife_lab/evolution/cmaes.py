"""CMA-ES with the update equations of Hansen's ``purecma`` (public domain, shipped with pycma
as ``cma.more_algorithms.purecma``), vectorised with NumPy.

Same defaults as purecma: popsize ``4 + int(3 ln N)`` unless given, ``mu = lambda // 2``,
positive log-rank weights, rank-one + rank-mu covariance update with the ``hsig`` stall, and the
squared-norm CSA step-size rule ``sigma *= exp(min(1, cs/damps * (|ps|^2/N - 1) / 2))``.
One difference: the eigendecomposition is refreshed at every ``ask`` (purecma refreshes it
lazily to save O(N^3) work); with purecma's lazy gap set to 0 both agree to rounding error
(tests/reference/test_cmaes.py).

All state is plain arrays/floats so it can be checkpointed without pickle. Minimisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CMAParams:
    n: int
    lam: int
    mu: int
    weights: F64  # (mu,) positive, sum 1
    mueff: float
    cc: float
    cs: float
    c1: float
    cmu: float
    damps: float

    @staticmethod
    def default(n: int, popsize: int | None = None) -> CMAParams:
        lam = popsize if popsize else 4 + int(3 * math.log(n))
        mu = lam // 2
        if mu < 1:
            raise ValueError("CMA-ES needs a population of at least 2")
        raw = np.array([math.log(lam / 2 + 0.5) - math.log(i + 1) for i in range(mu)])
        weights = raw / raw.sum()
        mueff = float(weights.sum() ** 2 / (weights**2).sum())
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 2 * mueff / lam + 0.3 + cs
        return CMAParams(n, lam, mu, weights, mueff, cc, cs, c1, cmu, damps)


@dataclass
class CMAState:
    mean: F64
    sigma: float
    C: F64
    pc: F64
    ps: F64
    counteval: int = 0
    # Eigensystem of C, refreshed by ``ask``.
    B: F64 | None = None
    D: F64 | None = None  # sqrt of eigenvalues

    @staticmethod
    def start(x0: F64, sigma0: float) -> CMAState:
        n = x0.size
        return CMAState(
            mean=np.array(x0, dtype=np.float64),
            sigma=float(sigma0),
            C=np.eye(n),
            pc=np.zeros(n),
            ps=np.zeros(n),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "sigma": self.sigma,
            "C": self.C.tolist(),
            "pc": self.pc.tolist(),
            "ps": self.ps.tolist(),
            "counteval": self.counteval,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> CMAState:
        return CMAState(
            mean=np.asarray(d["mean"], dtype=np.float64),
            sigma=float(d["sigma"]),
            C=np.asarray(d["C"], dtype=np.float64),
            pc=np.asarray(d["pc"], dtype=np.float64),
            ps=np.asarray(d["ps"], dtype=np.float64),
            counteval=int(d["counteval"]),
        )


def eigensystem(st: CMAState) -> None:
    c = (st.C + st.C.T) / 2  # enforce symmetry, as purecma does
    vals, vecs = np.linalg.eigh(c)
    if vals.min() <= 0:
        raise RuntimeError("CMA-ES covariance matrix is not positive definite")
    st.C = c
    st.B = vecs
    st.D = np.sqrt(vals)


def ask(st: CMAState, par: CMAParams, rng: np.random.Generator) -> F64:
    """Sample ``lam`` candidates, shape (lam, n): m + sigma * B diag(D) z, z ~ N(0, I)."""
    eigensystem(st)
    assert st.B is not None and st.D is not None
    z = rng.standard_normal((par.lam, par.n))
    out: F64 = st.mean + st.sigma * (z * st.D) @ st.B.T
    return out


def tell(st: CMAState, par: CMAParams, arx: F64, fitvals: F64) -> None:
    """Update mean, paths, C and sigma from candidates ``arx`` (lam, n) and their f-values."""
    if st.B is None or st.D is None:
        eigensystem(st)
    assert st.B is not None and st.D is not None
    n = par.n
    st.counteval += len(fitvals)
    order = np.argsort(fitvals, kind="stable")
    xs = arx[order]
    xold = st.mean
    st.mean = par.weights @ xs[: par.mu]

    y = st.mean - xold
    invsqrt = st.B @ np.diag(1.0 / st.D) @ st.B.T
    z = invsqrt @ y
    csn = math.sqrt(par.cs * (2 - par.cs) * par.mueff) / st.sigma
    st.ps = (1 - par.cs) * st.ps + csn * z
    ccn = math.sqrt(par.cc * (2 - par.cc) * par.mueff) / st.sigma
    ps_norm2 = float((st.ps**2).sum()) / n / (1 - (1 - par.cs) ** (2 * st.counteval / par.lam))
    hsig = float(ps_norm2 < 2 + 4.0 / (n + 1))
    st.pc = (1 - par.cc) * st.pc + ccn * hsig * y

    c1a = par.c1 * (1 - (1 - hsig**2) * par.cc * (2 - par.cc))
    st.C = st.C * (1 - c1a - par.cmu * par.weights.sum())
    st.C = st.C + par.c1 * np.outer(st.pc, st.pc)
    d = xs[: par.mu] - xold
    st.C = st.C + (par.cmu / st.sigma**2) * (d.T * par.weights) @ d

    cn = par.cs / par.damps
    st.sigma *= math.exp(min(1.0, cn * ((st.ps**2).sum() / n - 1) / 2))
