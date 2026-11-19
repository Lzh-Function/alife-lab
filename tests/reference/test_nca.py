"""Growing NCA (JAX, CPU): perception and alive mask against NumPy, untrained control, training
determinism and loss decrease, checkpoint continuation, export to an nca run, spec guards.
Skipped without JAX (uv sync --extra nca)."""

from __future__ import annotations

import base64
import importlib.util
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec, NCATarget
from alife_lab.substrates.nca.export import nca_spec_from_training
from alife_lab.substrates.nca.targets import padded_target, preset_rgba, target_rgba

HAS_JAX = importlib.util.find_spec("jax") is not None
needs_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed (uv sync --extra nca)")

TINY = {
    "target": {"preset": "disc", "size": 12},
    "dynamics": {"experiment": "persistent", "pad": 2, "batch_size": 4, "pool_size": 8,
                 "steps_min": 16, "steps_max": 24, "architecture": {"hidden": 32}},
}  # fmt: skip


def train_spec(**over: Any) -> ExperimentSpec:
    exp = {"name": "t", "substrate": "nca_train", "seed": 0, **TINY, **over}
    return ExperimentSpec.model_validate({"experiment": exp})


def test_targets_are_premultiplied_rgba_in_range() -> None:
    for name in ("disc", "ring", "heart", "letter_a"):
        t = preset_rgba(name, 32)
        assert t.shape == (32, 32, 4) and t.min() >= 0 and t.max() <= 1
        assert (t[..., :3] <= t[..., 3:4] + 1e-6).all() and t[..., 3].sum() > 50
    raw = np.zeros((8, 8, 4), np.uint8)
    raw[1, 2] = [255, 0, 0, 128]
    custom = NCATarget(kind="custom", size=8, rgba_b64=base64.b64encode(raw.tobytes()).decode())
    img = target_rgba(custom)
    assert img[1, 2, 0] == pytest.approx(128 / 255) and img[1, 2, 3] == pytest.approx(128 / 255)
    assert padded_target(custom, 3).shape == (14, 14, 4)
    with pytest.raises(ValidationError):
        NCATarget(kind="custom", size=9, rgba_b64=base64.b64encode(raw.tobytes()).decode())


@needs_jax
def test_perception_and_alive_mask_match_numpy() -> None:
    import jax.numpy as jnp

    from alife_lab.substrates.nca import model

    rng = np.random.default_rng(0)
    x = rng.random((1, 7, 9, 2)).astype(np.float32)
    p = np.asarray(model.perceive(jnp.asarray(x)))[0]
    pad = np.pad(x[0], ((1, 1), (1, 1), (0, 0)))
    sob = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8
    for c in range(2):
        win = np.lib.stride_tricks.sliding_window_view(pad[..., c], (3, 3))
        np.testing.assert_allclose(p[..., 3 * c], x[0, ..., c], atol=1e-6)
        np.testing.assert_allclose(p[..., 3 * c + 1], (win * sob).sum((-1, -2)), atol=1e-6)
        np.testing.assert_allclose(p[..., 3 * c + 2], (win * sob.T).sum((-1, -2)), atol=1e-6)
    a = np.zeros((1, 5, 5, 4), np.float32)
    a[0, 2, 2, 3] = 0.5
    alive = np.asarray(model.alive_mask(jnp.asarray(a), 0.1))[0, ..., 0]
    assert alive.sum() == 9 and alive[1:4, 1:4].all()


@needs_jax
def test_untrained_control_does_not_grow() -> None:
    from alife_lab.substrates.nca.nca_adapter import NCAAdapter

    sp = ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "u",
                "substrate": "nca",
                "seed": 0,
                "target": {"preset": "disc", "size": 12},
                "pad": 2,
                "dynamics": {"weights_kind": "untrained"},
            }
        }
    )
    a = NCAAdapter()
    s = a.initialize(sp)
    x0 = s.x.copy()
    for _ in range(30):
        s = a.step(s, sp)
    assert np.array_equal(s.x, x0)  # zero output layer: no update anywhere


@needs_jax
def test_training_is_deterministic_and_reduces_loss() -> None:
    from alife_lab.substrates.nca.train_adapter import NCATrainAdapter

    sp = train_spec()
    a = NCATrainAdapter()
    s1, s2 = a.initialize(sp), a.initialize(sp)
    losses = []
    for _ in range(40):
        s1 = a.step(s1, sp)
        losses.append(a.metrics(s1, sp)["loss"])
    for _ in range(40):
        s2 = a.step(s2, sp)
    assert a.state_distance(s1, s2)["exact_equal"]
    assert np.mean(losses[-5:]) < 0.7 * np.mean(losses[:5])


@needs_jax
def test_training_checkpoint_continues_bitwise_and_exports() -> None:
    from alife_lab.substrates.nca.nca_adapter import NCAAdapter
    from alife_lab.substrates.nca.train_adapter import NCATrainAdapter

    sp = train_spec(dynamics={**TINY["dynamics"], "experiment": "regenerating",
                              "damaged_per_batch": 2})  # fmt: skip
    a = NCATrainAdapter()
    s = a.initialize(sp)
    for _ in range(5):
        s = a.step(s, sp)
    ser = a.serialize_state(s)
    r = a.deserialize_state(decode_checkpoint(encode_checkpoint(ser["arrays"], ser["meta"])), sp)
    for _ in range(3):
        s, r = a.step(s, sp), a.step(r, sp)
    assert a.state_distance(s, r)["exact_equal"]

    exp = sp.experiment
    spec = nca_spec_from_training(exp, a.serialize_state(s), "run_0123456789abcdef", "0" * 64)  # type: ignore[arg-type]
    nsp = ExperimentSpec.model_validate(spec)
    assert nsp.experiment.dynamics.weights.source["optimizer_step"] == 8  # type: ignore[union-attr]
    n = NCAAdapter()
    st = n.initialize(nsp)
    for _ in range(10):
        st = n.step(st, nsp)
    st, info = n.apply_intervention(st, nsp, "damage", {"x": 8, "y": 8, "radius": 3})
    assert info["alpha_mass_after"] <= info["alpha_mass_before"]
    ser2 = n.serialize_state(st)
    r2 = n.deserialize_state(
        decode_checkpoint(encode_checkpoint(ser2["arrays"], ser2["meta"])), nsp
    )
    for _ in range(10):
        st, r2 = n.step(st, nsp), n.step(r2, nsp)
    assert n.state_distance(st, r2)["exact_equal"]
    assert set(n.metrics(st, nsp)) == {"alive_fraction", "alpha_mass", "target_mse"}


def test_nca_spec_guards() -> None:
    base = {"name": "n", "substrate": "nca", "seed": 0, "target": {"preset": "disc", "size": 12}}
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate({"experiment": base})  # trained needs weights
    raw = np.zeros(10, "<f4").tobytes()
    bad = {"data_b64": base64.b64encode(raw).decode(), "sha256": "0" * 64}
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate({"experiment": {**base, "dynamics": {"weights": bad}}})
    with pytest.raises(ValidationError):
        train_spec(dynamics={**TINY["dynamics"], "steps_min": 50, "steps_max": 10})


@needs_jax
def test_pretrained_model_regrows_after_damage_but_untrained_control_does_not() -> None:
    from pathlib import Path

    from alife_lab.service.catalog import load_presets
    from alife_lab.substrates.nca.nca_adapter import NCAAdapter

    presets = load_presets(Path(__file__).resolve().parents[2] / "configs")
    trained = presets["nca_heart24_pretrained"].spec
    raw = trained.model_dump(mode="json")
    raw["experiment"]["dynamics"].update(weights_kind="untrained", weights=None)
    control = ExperimentSpec.model_validate(raw)
    a = NCAAdapter()
    result = {}
    for name, sp in (("trained", trained), ("untrained", control)):
        s = a.initialize(sp)
        for _ in range(200):
            s = a.step(s, sp)
        grown = a.metrics(s, sp)["target_mse"]
        s, _ = a.apply_intervention(s, sp, "damage", {"x": 18, "y": 14, "radius": 6})
        damaged = a.metrics(s, sp)["target_mse"]
        for _ in range(50):
            s = a.step(s, sp)
        result[name] = (grown, damaged, a.metrics(s, sp)["target_mse"])
    g, d, r = result["trained"]
    assert g < 0.01 and d > 4 * g and r < 0.01  # grows, is damaged, regrows
    assert min(result["untrained"]) > 0.1  # control never approaches the target
