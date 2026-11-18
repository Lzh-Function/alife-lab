"""Particle Life: force oracle, invariants, determinism, interventions and the
zero-matrix negative control for type segregation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.particle_life.particle_life import (
    ParticleLifeAdapter,
    accelerations,
    accelerations_oracle,
    force_profile,
)
from alife_lab.substrates.particles.neighbors import neighbor_pairs

SELF = [[1.0 if i == j else -0.2 for j in range(4)] for i in range(4)]
ZERO = [[0.0] * 4 for _ in range(4)]


def make(seed: int = 1, n: int = 120, matrix: Any = None, **dyn: Any) -> ExperimentSpec:
    boundary = dyn.pop("boundary", "periodic")
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "pl",
                "substrate": "particle_life",
                "seed": seed,
                "world": {"width": 120, "height": 90, "boundary": boundary},
                "initialization": {"count": n, "types": 4},
                "dynamics": {"matrix": matrix, **dyn},
            }
        }
    )


def test_force_profile_shape() -> None:
    r = np.array([0.0, 0.15, 0.3, 0.65, 1.0 - 1e-12])
    f = force_profile(r, np.full(5, 0.5), 0.3)
    np.testing.assert_allclose(f, [-1.0, -0.5, 0.0, 0.5, 0.0], atol=1e-9)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), boundary=st.sampled_from(["periodic", "reflect"]))
def test_accelerations_match_scalar_oracle(seed: int, boundary: str) -> None:
    sp = make(seed=seed, n=60, boundary=boundary)
    adapter = ParticleLifeAdapter()
    state = adapter.initialize(sp)
    for _ in range(5):
        state = adapter.step(state, sp)
    exp = sp.experiment
    pairs = neighbor_pairs(
        state.pos, exp.dynamics.r_max, exp.world.width, exp.world.height, boundary == "periodic"
    )
    fast = accelerations(state.pos, state.types, state.matrix, exp, pairs)
    slow = accelerations_oracle(state.pos, state.types, state.matrix, exp)
    np.testing.assert_allclose(fast, slow, rtol=1e-9, atol=1e-9)


def test_invariants_and_checkpoint_roundtrip() -> None:
    sp = make(n=150)
    adapter = ParticleLifeAdapter()
    a = adapter.initialize(sp)
    for _ in range(80):
        a = adapter.step(a, sp)
        assert np.isfinite(a.pos).all() and np.isfinite(a.vel).all()
        assert ((a.pos >= 0) & (a.pos <= [120, 90])).all()
    a, _ = adapter.apply_intervention(a, sp, "spawn", {"x": 30, "y": 30, "radius": 4, "count": 12})
    ser = adapter.serialize_state(a)
    b = adapter.deserialize_state(
        decode_checkpoint(encode_checkpoint(ser["arrays"], ser["meta"])), sp
    )
    for _ in range(50):
        a = adapter.step(a, sp)
        b = adapter.step(b, sp)
    assert adapter.state_distance(a, b)["exact_equal"]


def test_matrix_is_seeded_and_spec_validation() -> None:
    adapter = ParticleLifeAdapter()
    m1 = adapter.initialize(make(seed=3)).matrix
    m2 = adapter.initialize(make(seed=3)).matrix
    np.testing.assert_array_equal(m1, m2)
    assert m1.shape == (4, 4) and np.abs(m1).max() <= 1
    with pytest.raises(ValidationError):
        make(matrix=[[0.0, 1.0], [1.0, 0.0]])  # wrong shape for 4 types
    with pytest.raises(ValidationError):
        make(matrix=[[2.0] * 4] * 4)  # out of range


def test_interventions_and_inspect() -> None:
    sp = make(n=50, matrix=SELF)
    adapter = ParticleLifeAdapter()
    s = adapter.initialize(sp)
    s, res = adapter.apply_intervention(
        s, sp, "spawn", {"x": 60, "y": 45, "radius": 3, "count": 10, "type": 2}
    )
    assert res["population_after"] == 60 and (s.types[-10:] == 2).all()
    info = adapter.inspect(s, sp, {"x": 60, "y": 45, "radius": 3})
    assert info["type_counts_in_radius"][2] >= 10 and info["nearest_particle"] is not None
    assert len(info["matrix"]) == 4
    with pytest.raises(ValueError):
        adapter.apply_intervention(s, sp, "spawn", {"x": 1, "y": 1, "type": 9})
    s, res = adapter.apply_intervention(s, sp, "remove", {"x": 60, "y": 45, "radius": 3})
    assert res["population_after"] <= 50


def test_zero_matrix_negative_control_for_segregation() -> None:
    """Type sorting requires type-dependent attraction: zero matrix stays at sum p_k^2."""

    def final(seed: int, matrix: Any) -> tuple[float, float]:
        sp = ExperimentSpec.model_validate(
            {
                "experiment": {
                    "name": "nc",
                    "substrate": "particle_life",
                    "seed": seed,
                    "world": {"width": 200, "height": 200},
                    "initialization": {"count": 600, "types": 4},
                    "dynamics": {"matrix": matrix},
                }
            }
        )
        adapter = ParticleLifeAdapter()
        s = adapter.initialize(sp)
        for _ in range(800):
            s = adapter.step(s, sp)
        p = np.bincount(s.types, minlength=4) / s.types.size
        return adapter.metrics(s, sp)["segregation_index"], float((p**2).sum())

    sorted_runs = [final(s, SELF) for s in range(4)]
    control = [final(s, ZERO) for s in range(4)]
    assert min(v for v, _ in sorted_runs) > max(v for v, _ in control)
    for value, baseline in control:
        assert abs(value - baseline) < 0.02
    assert np.mean([v for v, _ in sorted_runs]) > 0.5
