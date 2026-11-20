"""Search layer: objective positive/negative controls, determinism, checkpoint continuation,
holdout isolation and the guarantee that candidates run the unchanged Lenia substrate."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from alife_lab.core.spec import ExperimentSpec, SearchExperiment
from alife_lab.evolution.objectives import CHECK_EVERY, evaluate_lenia
from alife_lab.evolution.space import candidate_spec, decode, encode, seeds_for
from alife_lab.substrates.lenia.lenia import LeniaAdapter
from alife_lab.substrates.search.adapter import SearchAdapter


def search_spec(**dyn: Any) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "t_search",
                "substrate": "search",
                "seed": 7,
                "target": {
                    "name": "t",
                    "substrate": "lenia",
                    "seed": 0,
                    "world": {"width": 64, "height": 64},
                    "initialization": {"preset": "random", "size": 32},
                },
                "space": [
                    {"path": "dynamics.growth.m", "low": 0.1, "high": 0.3},
                    {"path": "dynamics.growth.s", "low": 0.005, "high": 0.05, "scale": "log"},
                ],
                "dynamics": {"population": 6, "eval_steps": 60, "eval_seeds": 2, **dyn},
                "runner": {"max_steps": 6, "checkpoint_interval": 3},
            }
        }
    )


def lenia(m: float, s: float, **init: Any) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "o",
                "substrate": "lenia",
                "seed": 0,
                "world": {"width": 64, "height": 64},
                "dynamics": {"growth": {"m": m, "s": s}},
                "initialization": init or {"preset": "pattern"},
            }
        }
    )


def test_objective_positive_and_negative_controls() -> None:
    orbium = evaluate_lenia(lenia(0.15, 0.015), 300, 0.3)
    assert orbium.viable and orbium.concentration > 0.99 and orbium.score > 0.5
    off_niche = evaluate_lenia(lenia(0.17, 0.015), 300, 0.3)  # dissolves
    assert not off_niche.viable and off_niche.score < 0.1
    # A world-filling lattice of rings (found by the first objective draft) is rejected.
    lattice = evaluate_lenia(lenia(0.1098, 0.0365, preset="random", size=32), 300, 0.3)
    assert not lattice.viable and lattice.concentration < 0.9


def test_partial_credit_grows_with_survival() -> None:
    short = evaluate_lenia(lenia(0.17, 0.015), 300, 0.3)
    longer = evaluate_lenia(lenia(0.16, 0.015), 300, 0.3)
    assert short.survived < longer.survived or longer.viable
    assert short.score <= longer.score


def test_candidates_run_the_unchanged_substrate() -> None:
    exp = search_spec().experiment
    assert isinstance(exp, SearchExperiment)
    params = {"dynamics.growth.m": 0.15, "dynamics.growth.s": 0.015}
    cand = candidate_spec(exp, params, seed=11)
    assert cand is not None
    ev = evaluate_lenia(cand, 60, 0.3)
    direct = lenia(0.15, 0.015, preset="random", size=32)
    raw = direct.model_dump(mode="json")
    raw["experiment"]["seed"] = 11
    spec = ExperimentSpec.model_validate(raw)
    a = LeniaAdapter()
    st = a.initialize(spec)
    # The evaluation stops at the first failed check (every CHECK_EVERY ticks).
    for _ in range(60 if ev.viable else min(60, ev.survived + CHECK_EVERY)):
        st = a.step(st, spec)
    assert np.array_equal(st.cells, ev.field)


def test_space_roundtrip_and_seeds() -> None:
    exp = search_spec().experiment
    assert isinstance(exp, SearchExperiment)
    u = np.array([0.25, 0.75])
    params = decode(exp, u)
    assert params["dynamics.growth.m"] == pytest.approx(0.15)
    np.testing.assert_allclose(encode(exp, params), u)  # type: ignore[arg-type]
    assert decode(exp, np.array([-1.0, 2.0])) == decode(exp, np.array([0.0, 1.0]))  # clipped
    s, h = seeds_for(exp, 1, 8), seeds_for(exp, 2, 8)
    assert not set(s) & set(h) and all(0 <= v < 2**63 for v in s + h)


def run(spec: ExperimentSpec, gens: int) -> Any:
    a = SearchAdapter()
    st = a.initialize(spec)
    for _ in range(gens):
        st = a.step(st, spec)
    return a, st


@pytest.mark.parametrize("algorithm", ["cma_es", "random"])
def test_deterministic_and_checkpoint_continuation(algorithm: str) -> None:
    spec = search_spec(algorithm=algorithm)
    a, full = run(spec, 4)
    _, again = run(spec, 4)
    assert a.state_distance(full, again)["exact_equal"]
    _, half = run(spec, 2)
    payload = a.serialize_state(half)
    payload = json.loads(json.dumps({**payload, "arrays": {}}))  # meta must be JSON
    payload["arrays"] = {k: np.asarray(v) for k, v in a.serialize_state(half)["arrays"].items()}
    resumed = a.deserialize_state(payload, spec)
    for _ in range(2):
        resumed = a.step(resumed, spec)
    assert a.state_distance(full, resumed)["exact_equal"]
    assert np.all((full.u >= 0) & (full.u <= 1)) if algorithm == "random" else True
    m = a.metrics(full, spec)
    assert m["evaluations"] == 4 * 6 * 2 and 0 <= m["viable_fraction"] <= 1


def test_holdout_never_drives_the_search() -> None:
    _, with_holdout = run(search_spec(holdout_seeds=3), 3)
    _, without = run(search_spec(holdout_seeds=0), 3)
    assert np.array_equal(with_holdout.u, without.u)
    assert np.array_equal(with_holdout.score, without.score)
    assert with_holdout.holdout and not without.holdout


def test_render_and_inspect() -> None:
    spec = search_spec()
    a, st = run(spec, 2)
    header, data = a.render(st)
    assert header["channels"] == 3 and len(data) == header["width"] * header["height"] * 3
    info = a.inspect(st, spec, {"x": 200, "y": 10})
    best = info["best"]
    assert best["score"] == st.score.max()
    rerun = ExperimentSpec.model_validate(best["spec"])  # materialisable as a normal Lenia run
    assert rerun.experiment.substrate == "lenia"


def test_invalid_search_specs() -> None:
    base = search_spec().model_dump(mode="json")
    for patch in (
        {"space": [{"path": "dynamics.kernel.b", "low": 0, "high": 1}]},  # not searchable
        {"space": [{"path": "dynamics.growth.s", "low": 0.0, "high": 0.1, "scale": "log"}]},
        {"space": [{"path": "dynamics.growth.m", "low": 0.3, "high": 0.1}]},
    ):
        raw = json.loads(json.dumps(base))
        raw["experiment"].update(patch)
        with pytest.raises(ValueError):
            ExperimentSpec.model_validate(raw)
