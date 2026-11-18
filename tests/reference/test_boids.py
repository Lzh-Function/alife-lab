"""Boids: neighbour-search oracle, invariants, determinism, checkpoint round trip and the
alignment negative control (flocking order must vanish without the alignment rule)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.boids.boids import (
    BoidsAdapter,
    neighbor_pairs,
    neighbor_pairs_bruteforce,
    steering,
)

METRICS = ["population", "polarization", "mean_speed", "mean_neighbor_count"]


def make_spec(seed: int = 1, n: int = 120, **dyn: Any) -> ExperimentSpec:
    boundary = dyn.pop("boundary", "periodic")
    size = dyn.pop("size", 120.0)
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "t",
                "substrate": "boids",
                "seed": seed,
                "world": {"width": size, "height": size * 0.8, "boundary": boundary},
                "initialization": {"count": n},
                "dynamics": dyn,
                "observations": {"metrics": METRICS},
            }
        }
    )


def as_set(p: Any) -> set[tuple[int, int]]:
    return set(zip(p.i.tolist(), p.j.tolist(), strict=True))


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(0, 120),
    radius=st.floats(2.0, 15.0),
    width=st.floats(46.0, 150.0),
    height=st.floats(46.0, 150.0),
    periodic=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_cell_list_equals_bruteforce(
    n: int, radius: float, width: float, height: float, periodic: bool, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2)) * [width, height]
    fast = neighbor_pairs(pos, radius, width, height, periodic)
    slow = neighbor_pairs_bruteforce(pos, radius, width, height, periodic)
    assert as_set(fast) == as_set(slow)
    assert len(fast.i) == len(as_set(fast))  # no duplicated pairs


def test_steering_matches_bruteforce_neighbours() -> None:
    spec = make_spec(n=200)
    adapter = BoidsAdapter()
    state = adapter.initialize(spec)
    exp = spec.experiment
    args = (exp.dynamics.perception_radius, exp.world.width, exp.world.height, True)
    fast, c1 = steering(state.pos, state.vel, exp, neighbor_pairs(state.pos, *args))
    slow, c2 = steering(state.pos, state.vel, exp, neighbor_pairs_bruteforce(state.pos, *args))
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("boundary", ["periodic", "reflect"])
def test_invariants_speed_bounds_and_world(boundary: str) -> None:
    spec = make_spec(n=150, boundary=boundary, noise=0.05)
    exp = spec.experiment
    adapter = BoidsAdapter()
    state = adapter.initialize(spec)
    for _ in range(150):
        state = adapter.step(state, spec)
        speed = np.hypot(state.vel[:, 0], state.vel[:, 1])
        assert np.all(speed >= exp.dynamics.min_speed - 1e-9)
        assert np.all(speed <= exp.dynamics.max_speed + 1e-9)
        assert np.all((state.pos >= 0) & (state.pos <= [exp.world.width, exp.world.height]))
    assert state.pos.shape == (150, 2) and np.isfinite(state.pos).all()


def test_deterministic_and_checkpoint_roundtrip_with_noise() -> None:
    spec = make_spec(n=100, noise=0.1)
    adapter = BoidsAdapter()
    a = adapter.initialize(spec)
    for _ in range(60):
        a = adapter.step(a, spec)
    # Serialize mid-run (including the RNG state), continue both copies.
    ser = adapter.serialize_state(a)
    payload = decode_checkpoint(encode_checkpoint(ser["arrays"], ser["meta"]))
    b = adapter.deserialize_state(payload, spec)
    for _ in range(60):
        a = adapter.step(a, spec)
        b = adapter.step(b, spec)
    assert adapter.state_distance(a, b)["exact_equal"]
    c = adapter.initialize(spec)
    for _ in range(120):
        c = adapter.step(c, spec)
    assert adapter.state_distance(a, c)["exact_equal"]


def test_spawn_remove_and_distance() -> None:
    spec = make_spec(n=50)
    adapter = BoidsAdapter()
    base = adapter.initialize(spec)
    state = adapter.initialize(spec)
    state, res = adapter.apply_intervention(
        state, spec, "spawn", {"x": 60, "y": 40, "radius": 5, "count": 20}
    )
    assert res == {"population_before": 50, "population_after": 70}
    assert len(np.unique(state.ids)) == 70 and state.next_id == 70
    d = adapter.state_distance(base, state)
    assert not d["exact_equal"] and d["only_in_second"] == 20 and d["value"] == 0.0
    state, res = adapter.apply_intervention(state, spec, "remove", {"x": 60, "y": 40, "radius": 5})
    assert res["population_after"] <= 50
    for bad in ({"x": -1, "y": 0}, {"x": 1, "y": 1, "count": 0}, {"y": 3}):
        with pytest.raises(ValueError):
            adapter.apply_intervention(state, spec, "spawn", bad)
    with pytest.raises(ValueError):
        adapter.apply_intervention(state, spec, "poison", {"x": 1, "y": 1})


def test_polarization_null_policy_and_random_baseline() -> None:
    spec = make_spec(n=0)
    adapter = BoidsAdapter()
    assert math.isnan(adapter.metrics(adapter.initialize(spec), spec)["polarization"])
    n = 400
    values = [
        adapter.metrics(adapter.initialize(make_spec(seed=s, n=n)), make_spec(n=n))["polarization"]
        for s in range(20)
    ]
    # Random headings: E|mean unit vector| ≈ sqrt(pi / (4N)).
    assert np.mean(values) == pytest.approx(math.sqrt(math.pi / (4 * n)), rel=0.35)


def test_alignment_rule_is_required_for_polarized_flocks() -> None:
    """Negative control: identical runs without the alignment rule must not polarize."""

    def final_polarization(seed: int, align: float) -> float:
        spec = ExperimentSpec.model_validate(
            {
                "experiment": {
                    "name": "nc",
                    "substrate": "boids",
                    "seed": seed,
                    "world": {"width": 200, "height": 200},
                    "initialization": {"count": 300},
                    "dynamics": {"alignment_weight": align},
                    "observations": {"metrics": ["polarization"]},
                }
            }
        )
        adapter = BoidsAdapter()
        state = adapter.initialize(spec)
        for _ in range(600):
            state = adapter.step(state, spec)
        return adapter.metrics(state, spec)["polarization"]

    with_alignment = [final_polarization(s, 0.05) for s in range(6)]
    without = [final_polarization(s, 0.0) for s in range(6)]
    # Complete separation of 6 vs 6 seeds (one-sided Mann-Whitney exact p = 1/924).
    assert min(with_alignment) > max(without)
    assert np.mean(without) < 0.12
    assert np.mean(with_alignment) > 0.4
