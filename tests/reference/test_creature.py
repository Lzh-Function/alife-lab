"""Embodied creature: free fall against the analytic solution, model structure, control,
negative control, interventions, determinism and bitwise checkpoint continuation."""

from __future__ import annotations

import base64
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


def search_spec(preset: str, gens: int = 3) -> ExperimentSpec:
    raw = PRESETS[preset].spec.model_dump(mode="json")
    raw["experiment"]["dynamics"].update(population=6, eval_steps=60)
    raw["experiment"]["runner"] = {"max_steps": gens, "checkpoint_interval": gens}
    return ExperimentSpec.model_validate(raw)


def test_creature_objective_and_search_paths() -> None:
    from alife_lab.evolution.objectives import evaluate_creature
    from alife_lab.evolution.space import candidate_spec, decode

    off = evaluate_creature(spec("creature_quadruped_off"), 200)
    assert off.score < 0.1 and off.viable and off.mass == pytest.approx(26.8, abs=0.1)
    s = search_spec("search_creature_coevolve_cma")
    exp = s.experiment
    params = decode(exp, np.full(len(exp.space), 0.5))  # type: ignore[arg-type]
    assert params["dynamics.controller.amplitudes.3"] == pytest.approx(0.5)
    cand = candidate_spec(exp, params, 0)  # type: ignore[arg-type]
    assert cand is not None
    assert cand.experiment.dynamics.body.legs[2].upper == pytest.approx(0.375)  # type: ignore[union-attr]
    bad = s.model_dump(mode="json")
    bad["experiment"]["space"][0]["path"] = "dynamics.controller.amplitudes.8"  # only 8 joints
    with pytest.raises(ValueError):
        ExperimentSpec.model_validate(bad)


def test_creature_search_is_deterministic_and_resumable() -> None:
    from alife_lab.substrates.search.adapter import SearchAdapter

    s = search_spec("search_creature_qd_map_elites", gens=4)
    a = SearchAdapter()

    def go(n: int) -> Any:
        st = a.initialize(s)
        for _ in range(n):
            st = a.step(st, s)
        return st

    full, again = go(4), go(4)
    assert a.state_distance(full, again)["exact_equal"]
    resumed = a.deserialize_state(a.serialize_state(go(2)), s)
    for _ in range(2):
        resumed = a.step(resumed, s)
    assert a.state_distance(full, resumed)["exact_equal"]
    assert np.all((full.mass > 5) & (full.mass < 90))  # body mass is a descriptor here
    header, _ = a.render(full)  # parameter map and grid; no field panel for bodies
    assert header["channels"] == 3


def _with(preset: str, **experiment: Any) -> ExperimentSpec:
    raw = spec(preset).model_dump(mode="json")
    for key, value in experiment.items():
        raw["experiment"][key] = value
    return ExperimentSpec.model_validate(raw)


def test_terrain_is_seeded_flat_at_start_and_matches_the_hfield() -> None:
    s = spec("creature_rough_wave")
    e = _exp(s)
    assert e.terrain is not None
    h = model.terrain_heights(e)
    assert h is not None
    np.testing.assert_array_equal(h, model.terrain_heights(e))  # deterministic
    other = _exp(_with("creature_rough_wave", seed=e.seed + 1))
    assert not np.array_equal(h, model.terrain_heights(other))  # the seed changes the ground
    assert h.min() >= 0 and h.max() <= e.terrain.amplitude
    assert model.ground_height(e, 0.0, 0.0) == 0.0  # flat start area
    assert h.max() > 0.5 * e.terrain.amplitude
    w = model.make_world(e)
    hf = np.asarray(w.model.hfield_data).reshape(h.shape) * e.terrain.amplitude
    np.testing.assert_allclose(hf, h, atol=1e-6)  # MuJoCo stores float32
    # Spec digests: a spec without terrain/reflex serialises exactly as before.
    flat = spec().model_dump(mode="json")["experiment"]
    assert "terrain" not in flat and "reflex" not in flat["dynamics"]


def test_reflex_sign_and_negative_control() -> None:
    """The reflex pushes down the knee of a foot in the air (contact term) and of the legs on
    the side the torso leans to (tilt term); with gains 0 it changes nothing."""
    s = spec("creature_rough_wave_reflex")
    e = _exp(s)
    assert e.dynamics.reflex is not None
    w = model.make_world(e)
    w.contacts = [True] * len(w.contacts)
    tilt = math.radians(10)  # lean towards +x: the top of the torso moves to +x
    rot = np.array([[math.cos(tilt), 0, math.sin(tilt)], [0, 1, 0],
                    [-math.sin(tilt), 0, math.cos(tilt)]])  # fmt: skip
    w.sensed_rot = rot.ravel()
    out, tilt_deg = model.reflex_commands(w, e)
    assert tilt_deg == pytest.approx(10.0)
    angles = [math.radians(leg.angle_deg) for leg in e.dynamics.body.legs]
    for i, a in enumerate(angles):
        # up = (sin t, 0, cos t), leg direction = (cos t cos a, sin a, -sin t cos a); the
        # reflex uses their horizontal dot product.
        lean = math.sin(tilt) * math.cos(tilt) * math.cos(a)
        assert out[i] == pytest.approx(e.dynamics.reflex.tilt_gain * lean)
        assert (out[i] > 0) == (math.cos(a) > 1e-9)  # legs on the +x side push down
    w.contacts[0] = False
    out2, _ = model.reflex_commands(w, e)
    assert out2[0] - out[0] == pytest.approx(e.dynamics.reflex.contact_gain)
    zero = spec("creature_rough_wave_reflex").model_dump(mode="json")
    zero["experiment"]["dynamics"]["reflex"] = {"tilt_gain": 0.0, "contact_gain": 0.0}
    _, with_zero = run(ExperimentSpec.model_validate(zero), 60)
    _, without = run(spec("creature_rough_wave"), 60)
    np.testing.assert_array_equal(with_zero.data.qpos, without.data.qpos)


def test_reflex_on_terrain_checkpoint_continues_bitwise() -> None:
    s = spec("creature_rough_wave_reflex")
    a, st = run(s, 45)
    assert any(st.contacts) and st.path_length > 0
    payload = a.serialize_state(st)
    json.dumps(payload["meta"])
    resumed = a.deserialize_state(payload, s)
    assert resumed.path_length == st.path_length and resumed.contacts == st.contacts
    for _ in range(60):
        st, resumed = a.step(st, s), a.step(resumed, s)
    assert a.state_distance(st, resumed)["exact_equal"]
    assert resumed.path_length == pytest.approx(st.path_length, rel=1e-12)
    header, _ = a.render(st)
    assert len(header["contacts"]) == 4 and len(header["reflex"]) == 4
    assert "hfield" in {g["type"] for g in header["geoms"]} and header["trail"]
    scene = a.inspect(st, s, {"scene": True})["terrain"]
    n = scene["n"]
    assert n <= 241 and len(base64.b64decode(scene["heights_u16_b64"])) == 2 * n * n
    flat_a, flat = run(spec(), 1)
    assert flat_a.inspect(flat, spec(), {"scene": True})["terrain"] is None
