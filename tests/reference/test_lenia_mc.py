"""Multichannel Lenia: agreement with SakanaAI/asal lenia_impl.py (fixture written by
scripts/compare_lenia_mc_upstream.py from the upstream JAX code in float64), invariants,
negative controls, interventions and checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alife_lab.core.spec import ExperimentSpec, LeniaMCExperiment
from alife_lab.substrates.lenia_mc import model
from alife_lab.substrates.lenia_mc.adapter import LeniaMCAdapter

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures/lenia_mc_asal_reference.json").read_text()
)
# Rounding differences (rFFT here vs complex FFT upstream, no re-centring roll) grow
# chaotically; the bounds are ~30x the measured differences.
TOL = {50: 1e-10, 100: 1e-9, 200: 1e-6}


def spec(pattern: str = "5N7KKM", **exp: Any) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {"experiment": {"name": "t", "substrate": "lenia_mc", "seed": 0,
                        "initialization": {"preset": "pattern", "pattern": pattern}, **exp}}
    )  # fmt: skip


def _exp(s: ExperimentSpec) -> LeniaMCExperiment:
    assert isinstance(s.experiment, LeniaMCExperiment)
    return s.experiment


@pytest.mark.parametrize("pattern", ["5N7KKM", "VT049W"])
def test_agrees_with_asal_lenia_impl(pattern: str) -> None:
    ref = FIXTURE["patterns"][pattern]
    exp = _exp(spec(pattern))
    kf, world = model.kernels_fft(exp), model.initial_cells(exp)
    idx = np.asarray(ref["indices"])
    for t in range(1, 201):
        world = model.step(world, kf, exp)
        if str(t) in ref["mass"]:
            assert world.sum() == pytest.approx(ref["mass"][str(t)], rel=1e-9)
        snap = ref["snapshots"].get(str(t))
        if snap:
            aligned = np.roll(world, tuple(-np.asarray(snap["shift"])), axis=(1, 2))
            ours = aligned[idx[:, 0], idx[:, 1], idx[:, 2]]
            assert np.abs(ours - np.asarray(snap["values"])).max() < TOL[t]


def test_spec_fills_pattern_values_and_validates() -> None:
    s = spec()
    d = s.model_dump(mode="json")["experiment"]["dynamics"]
    assert d["channels"] == 3 and d["R"] == 12.0 and d["T"] == 2.0 and len(d["kernels"]) == 15
    assert ExperimentSpec.model_validate(s.model_dump(mode="json")).digest() == s.digest()
    bad = s.model_dump(mode="json")
    bad["experiment"]["dynamics"]["channels"] = 2  # kernels use channel 2
    with pytest.raises(ValueError):
        ExperimentSpec.model_validate(bad)


def test_kernels_normalised_and_translation_equivariant() -> None:
    exp = _exp(spec())
    kf = model.kernels_fft(exp)
    kern = np.fft.irfft2(kf, s=(128, 128))
    np.testing.assert_allclose(kern.sum(axis=(1, 2)), 1.0, atol=1e-12)
    w = model.initial_cells(exp)
    shifted = np.roll(w, (17, -5), axis=(1, 2))
    a, b = w, shifted
    for _ in range(20):
        a, b = model.step(a, kf, exp), model.step(b, kf, exp)
    assert np.abs(np.roll(a, (17, -5), axis=(1, 2)) - b).max() < 1e-12
    assert a.min() >= 0 and a.max() <= 1


def run_metrics(s: ExperimentSpec, ticks: int) -> dict[str, float]:
    a = LeniaMCAdapter()
    st = a.initialize(s)
    for _ in range(ticks):
        st = a.step(st, s)
    return a.metrics(st, s)


def test_negative_controls_from_presets() -> None:
    from alife_lab.service.catalog import load_presets

    presets = load_presets(Path(__file__).parents[2] / "configs")
    swarm = run_metrics(presets["lenia_mc_swarm"].spec, 300)
    assert 330 < swarm["mass"] < 360 and swarm["alive_fraction"] < 0.05
    off = run_metrics(presets["lenia_mc_off_niche"].spec, 150)
    assert off["mass"] < 1.0  # dissolves
    no_cross = run_metrics(presets["lenia_mc_no_cross_channel"].spec, 300)
    assert no_cross["alive_fraction"] > 0.4  # loses its shape and spreads


def test_interventions_inspection_and_checkpoint() -> None:
    s = spec()
    a = LeniaMCAdapter()
    st = a.initialize(s)
    for _ in range(5):
        st = a.step(st, s)
    st, info = a.apply_intervention(st, s, "paint", {"x": 10, "y": 10, "radius": 2, "channel": 1})
    assert info["cells"] > 0 and st.cells[1, 10, 10] == 1.0 and st.cells[0, 10, 10] == 0.0
    with pytest.raises(ValueError):
        a.apply_intervention(st, s, "paint", {"x": 10, "y": 10, "channel": 3})
    q = a.inspect(st, s, {"x": 64, "y": 64})
    assert len(q["values"]) == 3 and len(q["kernels"]) == 15
    payload = a.serialize_state(st)
    resumed = a.deserialize_state(payload, s)
    for _ in range(5):
        st, resumed = a.step(st, s), a.step(resumed, s)
    assert a.state_distance(st, resumed)["exact_equal"]
    header, data = a.render(st)
    assert header["channels"] == 3 and len(data) == 128 * 128 * 3


def test_torch_batch_matches_numpy_reference() -> None:
    pytest.importorskip("torch")
    from alife_lab.evolution.objectives import rollout_lenia_frames
    from alife_lab.substrates.lenia_mc import torch_batch

    s = spec()
    raw = s.model_dump(mode="json")
    raw["experiment"]["dynamics"]["kernels"][0]["m"] = 0.25  # a second, different candidate
    other = ExperimentSpec.model_validate(raw)
    ro = torch_batch.rollout([_exp(s), _exp(other)], 120, 2, 64, "cpu")
    for i, sp in enumerate((s, other)):
        ref = rollout_lenia_frames(sp, 120, 2, 1.0, 64)
        assert np.abs(ro.frames[i] - ref.frames).max() < 1e-9
        assert ro.mass[i] == pytest.approx(ref.mass, rel=1e-10)
        assert ro.speed[i] == pytest.approx(ref.speed, abs=1e-9)


def test_search_over_kernel_parameters_spec() -> None:
    import math

    from alife_lab.evolution.space import candidate_spec, decode

    target = spec().model_dump(mode="json")["experiment"]
    base = target["dynamics"]["kernels"][3]["m"]
    sig = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
    logit = math.log(base) - math.log1p(-base)
    raw = {"experiment": {
        "name": "t", "substrate": "search", "seed": 0, "target": target,
        "space": [{"path": "dynamics.kernels.3.m", "low": sig(logit - 1), "high": sig(logit + 1),
                   "scale": "logit"}],
        "dynamics": {"objective": "lenia_clip_target", "numeric_policy": "float32_clip_torch",
                     "max_alive_fraction": 1.0, "eval_steps": 10},
        "asal": {"prompts": ["a biological cell"], "frames": 1, "crop": 64},
        "observations": {"metrics": ["best_score"]}}}  # fmt: skip
    exp = ExperimentSpec.model_validate(raw).experiment
    params = decode(exp, np.array([0.5]))  # type: ignore[arg-type]
    assert params["dynamics.kernels.3.m"] == pytest.approx(base)  # box centre = the creature
    cand = candidate_spec(exp, {"dynamics.kernels.3.m": 0.3}, 7)  # type: ignore[arg-type]
    assert cand is not None and cand.experiment.dynamics.kernels[3].m == 0.3  # type: ignore[union-attr]
    for path in ("dynamics.kernels.15.m", "dynamics.kernels.3.c0", "dynamics.growth.m"):
        bad = json.loads(json.dumps(raw))
        bad["experiment"]["space"][0]["path"] = path
        with pytest.raises(ValueError):
            ExperimentSpec.model_validate(bad)
    bad = json.loads(json.dumps(raw))
    bad["experiment"]["dynamics"].update(objective="lenia_moving_localized",
                                         numeric_policy="float64_cpu")  # fmt: skip
    bad["experiment"].pop("asal")
    with pytest.raises(ValueError):  # that objective is defined for single-channel Lenia only
        ExperimentSpec.model_validate(bad)
