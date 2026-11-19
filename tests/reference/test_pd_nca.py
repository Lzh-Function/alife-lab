"""Petri Dish NCA (JAX): invariants of the competition step, per-perspective gradient
isolation, hidden-channel write masks, controls (frozen / uniform), determinism, checkpoints and
interventions. Skipped without JAX."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec

pytestmark = pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX missing")


def make(seed: int = 0, **dyn: Any) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "p",
                "substrate": "pd_nca",
                "seed": seed,
                "world": {"width": 20, "height": 20},
                "dynamics": {
                    "n_ncas": 3,
                    "cell_state_dim": 8,
                    "cell_hidden_dim": 3,
                    "model": {"hidden_dim": 8},
                    **dyn,
                },
            }
        }
    )


def adapter() -> Any:
    from alife_lab.substrates.pd_nca.adapter import PDNCAAdapter

    return PDNCAAdapter()


def test_step_invariants() -> None:
    sp = make()
    a = adapter()
    s = a.initialize(sp)
    for _ in range(15):
        s = a.step(s, sp)
        al = s.x[..., :4]
        np.testing.assert_allclose(al.sum(-1), 1.0, atol=1e-5)  # aliveness is a distribution
        assert al.min() >= 0 and np.abs(s.x).max() <= 1.0
    terr = s.territories(3)
    assert terr.sum() == pytest.approx(1.0, abs=1e-5)


def test_gradient_isolation_between_perspectives() -> None:
    import jax
    import jax.numpy as jnp

    from alife_lab.substrates.pd_nca import model
    from alife_lab.substrates.pd_nca.adapter import _shape

    sp = make()
    a = adapter()
    s = a.initialize(sp)
    sh = _shape(sp.experiment)
    params = {k: jnp.asarray(v) for k, v in s.params.items()}
    own_nonzero = 0
    for p in range(1, sh.m):

        def obj(pr: Any, p: int = p) -> Any:
            own = jnp.arange(sh.m) == p
            x = model.ca_step(pr, jnp.asarray(s.sun), jnp.asarray(s.x), sh, own)
            x = model.ca_step(pr, jnp.asarray(s.sun), x, sh, own)
            return jnp.arcsinh(jnp.sum(x[..., p]) + 1e-3)

        g = jax.grad(obj)(params)
        hd = sh.hidden_dim
        for j in range(sh.n):
            enc_j = np.asarray(g["enc"][..., j * hd : (j + 1) * hd])
            comp_j = np.asarray(g["comp"][j])
            if j == p - 1:
                own_nonzero += int(np.abs(comp_j).sum() > 0)  # zero only if this NCA died
            else:  # other NCAs' weights receive no gradient in this perspective
                assert not enc_j.any() and not comp_j.any()
    assert own_nonzero >= 1


def test_hidden_write_mask() -> None:
    from alife_lab.substrates.pd_nca import model
    from alife_lab.substrates.pd_nca.adapter import _shape

    sh = _shape(make(model={"hidden_dim": 8, "per_hid_upd": 0.34}).experiment)
    vis = model.hidden_visibility(sh)
    assert vis[:, : sh.s].all()
    assert vis[:, sh.s :].sum(1).tolist() == [1.0, 1.0, 1.0]  # int(0.34 * 3) = 1 channel each
    assert [int(np.argmax(r[sh.s :])) for r in vis] == [0, 1, 2]  # offset by NCA index


def test_frozen_keeps_weights_and_online_changes_them() -> None:
    a = adapter()
    for learning, changed in (("frozen", False), ("online", True)):
        sp = make(learning=learning)
        s = a.initialize(sp)
        w0 = {k: v.copy() for k, v in s.params.items()}
        for _ in range(5):
            s = a.step(s, sp)
        assert any(not np.array_equal(w0[k], s.params[k]) for k in w0) == changed


def test_uniform_control_has_zero_learning_signal() -> None:
    sp = make(competition="uniform")
    a = adapter()
    s = a.initialize(sp)
    for _ in range(5):
        s = a.step(s, sp)
    assert s.last["grad_norm"] == 0.0


def test_deterministic_checkpoint_and_interventions() -> None:
    sp = make(optimizer="adam", learning_rate=0.001)
    a = adapter()
    s1, s2 = a.initialize(sp), a.initialize(sp)
    for _ in range(6):
        s1, s2 = a.step(s1, sp), a.step(s2, sp)
    assert a.state_distance(s1, s2)["exact_equal"]
    ser = a.serialize_state(s1)
    r = a.deserialize_state(decode_checkpoint(encode_checkpoint(ser["arrays"], ser["meta"])), sp)
    for _ in range(4):
        s1, r = a.step(s1, sp), a.step(r, sp)
    assert a.state_distance(s1, r)["exact_equal"]
    s1, info = a.apply_intervention(s1, sp, "seed", {"x": 5, "y": 5, "radius": 1, "agent": 2})
    assert info["cells"] == 5 and s1.x[5, 5, 3] == 1.0
    s1, _ = a.apply_intervention(s1, sp, "erase", {"x": 5, "y": 5, "radius": 1})
    assert s1.x[5, 5, 0] == 1.0 and s1.x[5, 5, 1:].sum() == 0
    q = a.inspect(s1, sp, {"x": 5, "y": 5})
    assert len(q["interaction"]) == 4 and q["dominant"] == 0
    m = a.metrics(s1, sp)
    assert {"territory.sun", "territory.nca_0", "territory.nca_2"} <= set(m)
