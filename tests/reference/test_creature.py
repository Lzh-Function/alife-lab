"""Embodied creature: free fall against the analytic solution, model structure, control,
negative control, interventions, determinism and bitwise checkpoint continuation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from alife_lab.core.spec import CreatureExperiment, ExperimentSpec  # noqa: E402
from alife_lab.service.catalog import load_presets  # noqa: E402
from alife_lab.substrates.creature import model  # noqa: E402
from alife_lab.substrates.creature.adapter import CreatureAdapter  # noqa: E402

PRESETS = load_presets(Path(__file__).parents[2] / "configs")


def spec(name: str = "creature_quadruped_wave") -> ExperimentSpec:
    return PRESETS[name].spec


def _exp(s: ExperimentSpec) -> CreatureExperiment:
    assert isinstance(s.experiment, CreatureExperiment)
    return s.experiment


@pytest.mark.parametrize(("preset", "rel"), [("creature_quadruped_off", 1e-9),
                                             ("creature_quadruped_wave", 1e-2)])  # fmt: skip
def test_free_fall_matches_gravity(preset: str, rel: float) -> None:
    """Before the feet touch the floor the centre-of-mass velocity is -g t. Without motor
    torque it is exact. With moving legs it agrees to 1 %: the joints' armature (reflected
    rotor inertia, not part of any body) does not conserve the bodies' total momentum."""
    raw = spec(preset).model_dump(mode="json")
    raw["experiment"]["initialization"]["height"] = 2.0
    s = ExperimentSpec.model_validate(raw)
    w = model.make_world(_exp(s))
    m, d = w.model, w.data
    assert not m.dof_damping[:6].any() and not m.dof_armature[:6].any()  # free root joint
    for _ in range(4):  # 4 ticks = 20 steps = 0.2 s; the feet start ~1.7 m above the floor
        model.step(w, _exp(s))
    assert d.ncon == 0
    mujoco.mj_forward(m, d)  # refresh derived quantities for the current state
    mujoco.mj_subtreeVel(m, d)
    assert float(d.subtree_linvel[1][2]) == pytest.approx(-9.81 * d.time, rel=rel)


def test_model_structure_and_control() -> None:
    e = _exp(spec("creature_hexapod_tripod"))
    w = model.make_world(e)
    assert w.model.njnt == 1 + 12 and w.model.nu == 12 and w.model.ngeom == 2 + 12
    u = model.control(e, 0.25)
    c = e.dynamics.controller
    expected = np.asarray(c.amplitudes) * np.sin(
        2 * math.pi * (c.frequency_hz * 0.25 + np.asarray(c.phases))
    )
    np.testing.assert_allclose(u, expected)
    bad = spec().model_dump(mode="json")
    bad["experiment"]["dynamics"]["controller"]["amplitudes"] = [0.5] * 6
    with pytest.raises(ValueError):
        ExperimentSpec.model_validate(bad)


def run(s: ExperimentSpec, ticks: int) -> tuple[CreatureAdapter, Any]:
    a = CreatureAdapter()
    st = a.initialize(s)
    for _ in range(ticks):
        st = a.step(st, s)
    return a, st


def test_negative_control_and_gait() -> None:
    _, off = run(spec("creature_quadruped_off"), 400)
    a, wave = run(spec("creature_quadruped_wave"), 400)
    m_off = a.metrics(off, spec("creature_quadruped_off"))
    m_wave = a.metrics(wave, spec())
    assert m_off["displacement"] < 0.1 and m_off["energy"] == 0.0
    assert m_wave["displacement"] > 0.5 and m_wave["upright"] > 0.85


def test_push_determinism_and_checkpoint() -> None:
    s = spec()
    a, st = run(s, 30)
    st, info = a.apply_intervention(st, s, "push", {"vx": 1.5})
    assert info["velocity_after"][0] == pytest.approx(info["velocity_before"][0] + 1.5)
    with pytest.raises(ValueError):
        a.apply_intervention(st, s, "push", {"vx": 50})
    payload = a.serialize_state(st)
    json.dumps(payload["meta"])
    resumed = a.deserialize_state(payload, s)
    for _ in range(40):
        st, resumed = a.step(st, s), a.step(resumed, s)
    assert a.state_distance(st, resumed)["exact_equal"]
    _, one = run(s, 70)
    _, two = run(s, 70)
    assert a.state_distance(one, two)["exact_equal"]
    header, data = a.render(st)
    assert header["codec"] == "scene3d_f32" and len(data) == header["count"] * 12 * 4
    insp = a.inspect(st, s, {})
    assert len(insp["legs"]) == 4 and insp["mass_kg"] > 0
