"""Physarum (Jones 2010): motor-stage oracle vs JIT, exclusion invariant, sensory rule table,
diffusion/decay conservation, network formation vs field-removal controls, checkpoints."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.physarum import model
from alife_lab.substrates.physarum.physarum import PhysarumAdapter

A = PhysarumAdapter()


def make(n: int = 64, mode: str = "reference", **dyn: Any) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "p",
                "substrate": "physarum",
                "seed": 1,
                "mode": mode,
                "world": {"width": n, "height": n},
                "dynamics": dyn,
            }
        }
    )


def test_jit_motor_stage_equals_python_oracle() -> None:
    sp = make()
    s1, s2 = A.initialize(sp), A.initialize(sp)
    for _ in range(30):
        r = model.draw_randoms(s1.rng, s1.x.size)
        model.draw_randoms(s2.rng, s2.x.size)
        b1 = model.motor_stage(s1.x, s1.y, s1.h, s1.occ, s1.trail, r, step_size=1.0,
                               deposit=5.0, exclusion=True, jit=True)  # fmt: skip
        b2 = model.motor_stage(s2.x, s2.y, s2.h, s2.occ, s2.trail, r, step_size=1.0,
                               deposit=5.0, exclusion=True, jit=False)  # fmt: skip
        assert b1 == b2
        for arr in ("x", "y", "h", "trail", "occ"):
            assert np.array_equal(getattr(s1, arr), getattr(s2, arr)), arr
        model.sensory_stage(s1.x, s1.y, s1.h, s1.trail, r, sensor_angle=math.pi / 4,
                            sensor_offset=9, rotation_angle=math.pi / 4, p_random_turn=0,
                            sensing=True)  # fmt: skip
        s2.h[:] = s1.h


def test_exclusion_keeps_one_agent_per_cell() -> None:
    sp = make(n=48)
    s = A.initialize(sp)
    for _ in range(200):
        s = A.step(s, sp)
        cells = np.floor(s.y).astype(int) % 48 * 48 + np.floor(s.x).astype(int) % 48
        assert np.unique(cells).size == s.x.size
    assert int(s.occ.sum()) == s.x.size


def test_sensory_rule_table() -> None:
    trail = np.zeros((40, 40))
    x = np.array([20.5] * 4)
    y = np.array([20.5] * 4)
    h = np.zeros(4)  # facing +x; front sensor at (29.5, 20.5)
    # front strongest: keep; left: +RA; right: -RA; front weakest: random side
    cases = [(5, 1, 1), (1, 5, 1), (1, 1, 5), (0, 5, 5)]
    out = []
    for f, left, right in cases:
        tr = trail.copy()
        a = math.pi / 4
        for val, ang in ((f, 0.0), (left, a), (right, -a)):
            tr[int(20.5 + 9 * math.sin(ang)), int(20.5 + 9 * math.cos(ang))] = val
        hh = h[:1].copy()
        r = model.TickRandoms(np.zeros(1, np.int64), np.zeros(1), np.array([True]),
                              np.ones(1), np.zeros(1))  # fmt: skip
        model.sensory_stage(x[:1], y[:1], hh, tr, r, sensor_angle=a, sensor_offset=9,
                            rotation_angle=a, p_random_turn=0, sensing=True)  # fmt: skip
        out.append(round(math.degrees(hh[0])) % 360)
    assert out == [0, 45, 315, 45]  # turn_left draw -> +RA in the tie-both case


def test_diffusion_preserves_mass_before_decay() -> None:
    trail = np.random.default_rng(0).random((30, 30))
    assert model.diffuse_decay(trail, 0.0).sum() == pytest.approx(trail.sum(), rel=1e-12)
    assert model.diffuse_decay(trail, 0.1).sum() == pytest.approx(0.9 * trail.sum(), rel=1e-12)


def test_network_forms_but_not_without_sensing_or_deposit() -> None:
    def final(**dyn: Any) -> dict[str, float]:
        sp = make(n=100, mode="exploratory", **dyn)
        s = A.initialize(sp)
        for _ in range(600):
            s = A.step(s, sp)
        return A.metrics(s, sp)

    net, blind, empty = final(), final(sensing=False), final(deposit=0.0)
    assert net["trail_contrast"] > 1.0 > 0.4 > blind["trail_contrast"]
    assert empty["trail_mean"] == 0.0 and math.isnan(empty["trail_contrast"])


def test_spec_guards() -> None:
    with pytest.raises(ValidationError):
        make(collisions="none")  # reference mode requires exclusion
    with pytest.raises(ValidationError):
        make(food=[{"x": 500, "y": 1}])
    make(mode="exploratory", collisions="none")


def test_checkpoint_continues_bitwise_and_tools() -> None:
    sp = make(food=[{"x": 10, "y": 10}])
    s = A.initialize(sp)
    for _ in range(25):
        s = A.step(s, sp)
    s, info = A.apply_intervention(s, sp, "attract", {"x": 30, "y": 30, "radius": 2})
    assert info["cells"] == 13 and info["trail_after"] > info["trail_before"]
    q = A.inspect(s, sp, {"x": 30, "y": 30, "radius": 5})
    assert q["nearest_agent"]["decision"] in {"keep", "turn_left", "turn_right",
                                              "random_left_or_right"}  # fmt: skip
    ser = A.serialize_state(s)
    r = A.deserialize_state(decode_checkpoint(encode_checkpoint(ser["arrays"], ser["meta"])), sp)
    for _ in range(40):
        s, r = A.step(s, sp), A.step(r, sp)
    assert A.state_distance(s, r)["exact_equal"]
    assert A.metrics(s, sp) == A.metrics(r, sp)
    assert len(A.render(s)[1]) == 64 * 64
