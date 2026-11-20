"""``search``: an external optimisation run over the parameters of a target world.

One tick = one generation: the algorithm proposes ``population`` candidates in the normalised
box, each candidate's target spec is run headless with the target's own adapter on the common
search seeds and scored by the pre-declared objective (``alife_lab.evolution.objectives``).
Whenever the best candidate changes it is re-scored on disjoint holdout seeds, which never
drive the search. The archive of every evaluation is part of the checkpointed state.

With a ``qd`` section every algorithm also maintains a MAP-Elites grid (passively, for
coverage / QD-score); ``map_elites`` selects parents uniformly from its filled cells and
mutates them with N(0, mutation_sigma) in the normalised box (clipped). The first generation
and any generation with an empty grid sample uniformly. ``parent`` records the elite a child
was mutated from (-1 otherwise: uniform samples and CMA-ES draws have no single parent).

The target substrate is not modified: the score is computed outside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import ExperimentSpec, SearchExperiment
from alife_lab.evolution import OBJECTIVE_VERSIONS, cmaes, qd
from alife_lab.evolution.space import bound_violation, candidate_spec, decode, seeds_for
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest

F64 = npt.NDArray[np.float64]
SERIALIZE_VERSION = 2  # 2: + mass, parent (version 1 checkpoints load with NaN / -1)
BOUND_PENALTY = 10.0  # CMA-ES f = -score + BOUND_PENALTY * |u - clip(u)|^2
_ARCHIVE = ("u", "x", "score", "viable", "alive_fraction", "speed", "mass", "parent",
            "generation")  # fmt: skip


def _se(spec: ExperimentSpec) -> SearchExperiment:
    exp = spec.experiment
    if not isinstance(exp, SearchExperiment):
        raise TypeError("search adapter received a different spec")
    return exp


@dataclass
class SearchState:
    tick: int  # completed generations
    rng: np.random.Generator = field(repr=False)
    cma: cmaes.CMAState | None
    u: F64  # (n, d) proposed normalised coordinates (unclipped)
    x: F64  # (n, d) decoded parameter values
    score: F64  # (n,) mean score over the search seeds
    viable: F64  # (n,) fraction of search seeds on which the candidate was viable
    alive_fraction: F64  # (n,) mean final alive fraction
    speed: F64  # (n,) mean speed
    mass: F64  # (n,) mean final mass
    parent: F64  # (n,) archive index of the parent (-1: none)
    generation: F64  # (n,) generation index (1-based)
    elites: npt.NDArray[np.int64] | None = field(default=None, repr=False)  # derived (QD grid)
    grid: tuple[int, int] | None = None  # QD bins (descriptor 0, descriptor 1), from the spec
    holdout: dict[int, float] = field(default_factory=dict)  # archive index -> holdout score
    best_field: F64 | None = field(default=None, repr=False)
    last: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> int | None:
        return int(np.argmax(self.score)) if self.score.size else None


_M = MetricDefinition
_METRICS = (
    _M("best_score", "metric.best_score", "score", "generation",
       "Highest mean search-seed score of any candidate so far.", "max", "every generation",
       1, "never null after generation 1"),
    _M("generation_best", "metric.generation_best", "score", "generation",
       "Highest score in this generation.", "max", "every generation", 1, "never null"),
    _M("generation_mean", "metric.generation_mean", "score", "generation",
       "Mean score of this generation.", "mean", "every generation", 1, "never null"),
    _M("viable_fraction", "metric.viable_fraction", "fraction", "generation",
       "Fraction of (candidate, seed) evaluations of this generation that were viable.",
       "mean", "every generation", 1, "never null"),
    _M("evaluations", "metric.evaluations", "evaluations", "generation",
       "Cumulative (candidate, search seed) evaluations; holdout re-scoring is not counted.",
       "sum", "every generation", 1, "never null"),
    _M("holdout_best", "metric.holdout_best", "score", "generation",
       "Mean holdout-seed score of the current best candidate (null without holdout seeds).",
       "none", "every generation", 1, "null without holdout seeds"),
    _M("sigma", "metric.sigma", "normalised", "generation",
       "CMA-ES step size in normalised units (null for random search).", "none",
       "every generation", 1, "null for random search"),
    _M("coverage", "metric.coverage", "fraction", "generation",
       "Fraction of QD grid cells holding an elite (needs a qd section).", "none",
       "every generation", 1, "absent without qd"),
    _M("qd_score", "metric.qd_score", "score", "generation",
       "Sum of the elites' scores over the QD grid (needs a qd section).", "sum",
       "every generation", 1, "absent without qd"),
    _M("archive_size", "metric.archive_size", "cells", "generation",
       "Number of filled QD grid cells (needs a qd section).", "none", "every generation", 1,
       "absent without qd"),
)  # fmt: skip

SEARCH_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.search",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="search",
    display_name_key="substrate.search",
    scientific_modes=("exploratory",),
    capabilities=Capabilities(
        state_kind="network",
        clock_unit="generation",
        supports_pause=True,
        supports_interventions=False,
        supports_inspection=True,
        supports_lineage=False,
        supports_differentiation=False,
        supports_checkpoint=True,
        supports_deterministic_cpu=True,
        intervention_types=(),
    ),
    backends=("numpy",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "External search over target-world parameters. random = uniform "
        "baseline; cma_es = CMA-ES with the update of Hansen's purecma (oracle-tested). "
        f"Objective versions: {OBJECTIVE_VERSIONS}.",
        "paper_urls": ["https://arxiv.org/abs/1604.00772"],
        "implementation": "this repository (NumPy)",
    },
)


def _empty(d: int) -> dict[str, F64]:
    return {
        "u": np.zeros((0, d)),
        "x": np.zeros((0, d)),
        "score": np.zeros(0),
        "viable": np.zeros(0),
        "alive_fraction": np.zeros(0),
        "speed": np.zeros(0),
        "mass": np.zeros(0),
        "parent": np.zeros(0),
        "generation": np.zeros(0),
    }


def _grid(exp: SearchExperiment) -> tuple[int, int] | None:
    if exp.qd is None:
        return None
    return exp.qd.descriptors[0].bins, exp.qd.descriptors[1].bins


class SearchAdapter:
    backend: str = "numpy"

    def set_backend(self, backend: str) -> None:
        if backend not in SEARCH_MANIFEST.backends:
            raise ValueError(f"unsupported search backend {backend!r}")
        self.backend = backend

    @property
    def manifest(self) -> PluginManifest:
        return SEARCH_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> SearchState:
        exp = _se(spec)
        d = len(exp.space)
        rng = np.random.Generator(np.random.PCG64(exp.seed))
        cma = (
            cmaes.CMAState.start(np.full(d, 0.5), exp.dynamics.sigma0)
            if exp.dynamics.algorithm == "cma_es"
            else None
        )
        e = _empty(d)
        return SearchState(
            tick=0, rng=rng, cma=cma, u=e["u"], x=e["x"], score=e["score"], viable=e["viable"],
            alive_fraction=e["alive_fraction"], speed=e["speed"], mass=e["mass"],
            parent=e["parent"], generation=e["generation"],
            elites=qd.empty(exp.qd) if exp.qd is not None else None,
            grid=_grid(exp),
        )  # fmt: skip

    # ------------------------------------------------------------------ evaluation

    def _evaluate(
        self, exp: SearchExperiment, params: dict[str, float | int], seeds: list[int]
    ) -> tuple[float, float, float, float, float, F64 | None]:
        """Mean score, viable fraction, alive fraction, speed, mass over seeds; first field."""
        from alife_lab.evolution.objectives import evaluate_lenia  # target adapters import us

        scores, viable, alive, speed, mass = [], [], [], [], []
        first: F64 | None = None
        for s in seeds:
            cand = candidate_spec(exp, params, s)
            if cand is None:  # invalid combination (e.g. kernel larger than the world)
                scores.append(0.0)
                viable.append(0.0)
                alive.append(0.0)
                speed.append(0.0)
                mass.append(0.0)
                continue
            ev = evaluate_lenia(cand, exp.dynamics.eval_steps, exp.dynamics.max_alive_fraction)
            scores.append(ev.score)
            viable.append(float(ev.viable))
            alive.append(ev.alive_fraction)
            speed.append(ev.speed)
            mass.append(ev.mass)
            if first is None:
                first = np.asarray(ev.field)
        return (
            float(np.mean(scores)),
            float(np.mean(viable)),
            float(np.mean(alive)),
            float(np.mean(speed)),
            float(np.mean(mass)),
            first,
        )

    def step(self, state: SearchState, spec: ExperimentSpec) -> SearchState:
        exp = _se(spec)
        d = exp.dynamics
        dim = len(exp.space)
        parents = np.full(d.population, -1.0)
        if d.algorithm == "cma_es":
            assert state.cma is not None
            par = cmaes.CMAParams.default(dim, d.population)
            u = cmaes.ask(state.cma, par, state.rng)
        elif d.algorithm == "map_elites" and state.elites is not None and (state.elites >= 0).any():
            assert exp.qd is not None
            filled = state.elites[state.elites >= 0]
            chosen = filled[state.rng.integers(0, filled.size, size=d.population)]
            noise = state.rng.standard_normal((d.population, dim)) * exp.qd.mutation_sigma
            u = np.clip(state.u[chosen] + noise, 0.0, 1.0)
            parents = chosen.astype(np.float64)
        else:
            u = state.rng.random((d.population, dim))
        search_seeds = seeds_for(exp, 1, d.eval_seeds)
        xs, stats = [], []
        fields: list[F64 | None] = []
        for ui in u:
            params = decode(exp, ui)
            score, viable, alive, speed, mass, fld = self._evaluate(exp, params, search_seeds)
            xs.append([float(v) for v in params.values()])
            stats.append((score, viable, alive, speed, mass))
            fields.append(fld)
        st = np.array(stats)  # (pop, 5)
        new_score = st[:, 0]
        gen = state.tick + 1
        prev_best = state.best
        state.u = np.vstack([state.u, u])
        state.x = np.vstack([state.x, np.array(xs)])
        state.score = np.concatenate([state.score, new_score])
        state.viable = np.concatenate([state.viable, st[:, 1]])
        state.alive_fraction = np.concatenate([state.alive_fraction, st[:, 2]])
        state.speed = np.concatenate([state.speed, st[:, 3]])
        state.mass = np.concatenate([state.mass, st[:, 4]])
        state.parent = np.concatenate([state.parent, parents])
        state.generation = np.concatenate([state.generation, np.full(len(u), gen, dtype=float)])
        if d.algorithm == "cma_es":
            assert state.cma is not None
            f = -new_score + BOUND_PENALTY * np.array([bound_violation(ui) for ui in u])
            cmaes.tell(state.cma, par, u, f)
        if exp.qd is not None:
            if state.elites is None:
                state.elites = qd.build(state, exp.qd)
            else:
                qd.insert(state.elites, state, exp.qd, state.score.size - len(u))
        best = state.best
        assert best is not None
        offset = state.score.size - len(u)
        if best != prev_best:
            if best >= offset:
                state.best_field = fields[best - offset]
            if d.holdout_seeds:
                params = decode(exp, state.u[best])
                holdout_seeds = seeds_for(exp, 2, d.holdout_seeds)
                state.holdout[best] = self._evaluate(exp, params, holdout_seeds)[0]
        state.tick = gen
        state.last = {
            "best_score": float(state.score[best]),
            "generation_best": float(new_score.max()),
            "generation_mean": float(new_score.mean()),
            "viable_fraction": float(st[:, 1].mean()),
            "evaluations": float(state.score.size * d.eval_seeds),
            "holdout_best": state.holdout.get(best, math.nan) if d.holdout_seeds else math.nan,
            "sigma": state.cma.sigma if state.cma is not None else math.nan,
        }
        if exp.qd is not None and state.elites is not None:
            state.last.update(qd.stats(state.elites, state))
        return state

    # ------------------------------------------------------------------ observation

    def metrics(self, state: SearchState, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_se(spec).observations.metrics)
        return {k: v for k, v in state.last.items() if k in wanted}

    def render(self, state: SearchState) -> tuple[dict[str, Any], bytes]:
        from alife_lab.substrates.search.render import render

        return render(state)

    def apply_intervention(
        self, state: SearchState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[SearchState, dict[str, Any]]:
        raise ValueError("search runs accept no interventions (optimisation run)")

    def inspect(
        self, state: SearchState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        from alife_lab.substrates.search.render import inspect

        return inspect(state, _se(spec), query)

    def state_distance(self, a: SearchState, b: SearchState) -> dict[str, Any]:
        same_shape = a.score.shape == b.score.shape
        diff = float(np.abs(a.score - b.score).max()) if same_shape and a.score.size else 0.0
        exact = (
            same_shape
            and np.array_equal(a.u, b.u)
            and np.array_equal(a.score, b.score)
            and np.array_equal(a.parent, b.parent)
            and a.holdout == b.holdout
        )
        if a.cma is not None and b.cma is not None:
            exact = exact and np.array_equal(a.cma.mean, b.cma.mean) and a.cma.sigma == b.cma.sigma
        return {
            "metric": "archive_score_max_abs_difference" if same_shape else "archive_size",
            "value": diff if same_shape else abs(a.score.size - b.score.size),
            "exact_equal": bool(exact),
        }

    def serialize_state(self, state: SearchState) -> dict[str, Any]:
        arrays: dict[str, Any] = {k: getattr(state, k) for k in _ARCHIVE}
        if state.best_field is not None:
            arrays["best_field"] = state.best_field
        return {
            "arrays": arrays,
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "rng_state": state.rng.bit_generator.state,
                "cma": state.cma.to_json() if state.cma is not None else None,
                "holdout": {str(k): v for k, v in state.holdout.items()},
                "last": state.last,
                "groups": {"archive": ", ".join(_ARCHIVE), "optimizer_state": "cma",
                           "rng": "rng_state", "display": "best_field"},
            },
        }  # fmt: skip

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> SearchState:
        exp = _se(spec)
        meta = payload["meta"]
        if meta["serialize_version"] not in (1, SERIALIZE_VERSION):
            raise ValueError("incompatible search checkpoint serialize_version")
        arr = dict(payload["arrays"])
        n = len(arr["score"])
        arr.setdefault("mass", np.full(n, np.nan))  # version 1 had no mass / parent
        arr.setdefault("parent", np.full(n, -1.0))
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = meta["rng_state"]
        archive = {k: np.asarray(arr[k], dtype=np.float64).copy() for k in _ARCHIVE}
        if archive["u"].ndim != 2 or archive["u"].shape[1] != len(exp.space):
            raise ValueError("search checkpoint does not match the search space")
        bf = arr.get("best_field")
        state = SearchState(
            tick=int(meta["tick"]),
            rng=rng,
            cma=cmaes.CMAState.from_json(meta["cma"]) if meta.get("cma") else None,
            holdout={int(k): float(v) for k, v in (meta.get("holdout") or {}).items()},
            best_field=np.asarray(bf, dtype=np.float64).copy() if bf is not None else None,
            last=dict(meta.get("last") or {}),
            u=archive["u"], x=archive["x"], score=archive["score"], viable=archive["viable"],
            alive_fraction=archive["alive_fraction"], speed=archive["speed"],
            mass=archive["mass"], parent=archive["parent"], generation=archive["generation"],
        )  # fmt: skip
        if exp.qd is not None:
            state.elites = qd.build(state, exp.qd)
            state.grid = _grid(exp)
        return state
