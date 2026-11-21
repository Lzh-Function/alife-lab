"""Novelty search and IMGEP: parents, archive, determinism, checkpoint continuation, spec
rules; the fixed-size illumination subset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from alife_lab.core.spec import ExperimentSpec
from alife_lab.evolution import asal, explore
from alife_lab.substrates.search.adapter import SearchAdapter

ROOT = Path(__file__).parents[2]


def qd_spec(algorithm: str, **dyn: Any) -> ExperimentSpec:
    raw = yaml.safe_load((ROOT / "configs/scenarios/search_lenia_qd.yaml").read_text())
    e = raw["experiment"]
    e["dynamics"].update(algorithm=algorithm, population=8, eval_steps=120, **dyn)
    e["explore"] = {"k": 5, "archive_add": 2, "mutation_sigma": 0.1}
    e["runner"] = {"max_steps": 6, "checkpoint_interval": 3}
    return ExperimentSpec.model_validate(raw)


def run(spec: ExperimentSpec, gens: int) -> tuple[SearchAdapter, Any]:
    a = SearchAdapter()
    st = a.initialize(spec)
    for _ in range(gens):
        st = a.step(st, spec)
    return a, st


@pytest.mark.parametrize("algorithm", ["novelty", "imgep"])
def test_parents_determinism_and_continuation(algorithm: str) -> None:
    spec = qd_spec(algorithm)
    a, full = run(spec, 5)
    _, again = run(spec, 5)
    assert a.state_distance(full, again)["exact_equal"]
    _, half = run(spec, 2)
    payload = a.serialize_state(half)
    json.dumps(payload["meta"])  # the archive and RNG state are plain JSON
    resumed = a.deserialize_state(payload, spec)
    for _ in range(3):
        resumed = a.step(resumed, spec)
    assert a.state_distance(full, resumed)["exact_equal"]
    children = np.flatnonzero(full.parent >= 0)
    assert children.size > 0 and np.all(full.generation[full.generation == 1] == 1)
    par = full.parent[children].astype(int)
    assert np.all(full.generation[par] < full.generation[children])
    assert np.all(full.viable[par] >= 1.0)  # parents always had a behaviour
    assert np.all((full.u[children] >= 0) & (full.u[children] <= 1))


def test_novelty_archive_and_metric() -> None:
    raw = qd_spec("novelty").model_dump(mode="json")
    raw["experiment"]["observations"]["metrics"].append("mean_novelty")
    spec = ExperimentSpec.model_validate(raw)
    a, st = run(spec, 4)
    assert 0 < len(st.novelty_archive) <= 4 * 2
    assert all(st.viable[i] >= 1.0 for i in st.novelty_archive)
    assert a.metrics(st, spec)["mean_novelty"] >= 0


def test_knn_novelty_and_farthest_subset() -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [np.nan, np.nan]])
    nov = explore.knn_novelty(pts, pts, k=1, self_in_reference=True)
    np.testing.assert_allclose(nov, [1.0, 1.0, 1.0, 0.0])
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 16))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    sub = asal.farthest_subset(z, 32)
    assert sub.shape == (32, 16)
    # A spread-out subset is more diverse than the first 32 rows; the set size is fixed.
    assert asal.illumination(sub) > asal.illumination(z[:32])


def test_explore_spec_rules() -> None:
    raw = qd_spec("novelty").model_dump(mode="json")
    for patch in ({"explore": None}, {"qd": None}):
        bad = json.loads(json.dumps(raw))
        bad["experiment"].update(patch)
        if patch.get("qd", 1) is None:
            bad["experiment"]["dynamics"]["algorithm"] = "imgep"
            bad["experiment"]["observations"] = {"metrics": ["best_score"]}
        bad["experiment"] = {k: v for k, v in bad["experiment"].items() if v is not None}
        with pytest.raises(ValueError):
            ExperimentSpec.model_validate(bad)
    plain = yaml.safe_load((ROOT / "configs/scenarios/search_lenia_qd.yaml").read_text())
    assert (
        "explore" not in ExperimentSpec.model_validate(plain).model_dump(mode="json")["experiment"]
    )  # digest-stable when absent
