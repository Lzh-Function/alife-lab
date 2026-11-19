"""Gray-Scott: scalar oracle (bitwise), stability guard, steady state, conservation under pure
diffusion, pattern formation vs the diffusion-only control, interventions, checkpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.gray_scott import model
from alife_lab.substrates.gray_scott.gray_scott import GrayScottAdapter

A = GrayScottAdapter()


def make(n: int = 64, **exp: Any) -> ExperimentSpec:
    world = {"width": n, "height": n, "length": 2.5 * n / 256, **exp.pop("world", {})}
    return ExperimentSpec.model_validate(
        {"experiment": {"name": "gs", "substrate": "gray_scott", "seed": 3, "world": world, **exp}}
    )


@pytest.mark.parametrize("reaction", [True, False])
def test_vectorised_step_equals_scalar_oracle_bitwise(reaction: bool) -> None:
    rng = np.random.default_rng(0)
    u, v = rng.random((9, 11)), rng.random((9, 11)) * 0.3
    kw: dict[str, Any] = {"Du": 2e-5, "Dv": 1e-5, "F": 0.04, "k": 0.06, "dt": 1.0,
                          "dx": 2.5 / 256, "reaction": reaction}  # fmt: skip
    nu, nv = model.step(u, v, **kw)
    ou, ov = model.step_oracle(u.tolist(), v.tolist(), **kw)
    assert np.array_equal(nu, np.array(ou)) and np.array_equal(nv, np.array(ov))


def test_laplacian_of_quadratic_and_constant() -> None:
    assert np.array_equal(model.laplacian(np.full((8, 8), 3.0), 0.5), np.zeros((8, 8)))
    a = np.zeros((7, 7))
    a[3, 3] = 1.0
    lap = model.laplacian(a, 1.0)
    assert lap[3, 3] == -4.0 and lap[2, 3] == lap[3, 4] == 1.0 and lap.sum() == 0.0


def test_spec_rejects_unstable_time_step() -> None:
    with pytest.raises(ValidationError, match="unstable"):
        make(dynamics={"dt": 2.0})  # dx^2 / (4 Du) = 1.19
    with pytest.raises(ValidationError):
        make(world={"width": 64, "height": 32})


def test_trivial_steady_state_is_exactly_invariant() -> None:
    sp = make(initialization={"preset": "uniform"})
    s = A.initialize(sp)
    for _ in range(50):
        s = A.step(s, sp)
    assert (s.u == 1.0).all() and (s.v == 0.0).all()


def test_pure_diffusion_conserves_totals() -> None:
    sp = make(dynamics={"reaction": False})
    s = A.initialize(sp)
    tu, tv, std0 = float(s.u.sum()), float(s.v.sum()), float(s.v.std())
    for _ in range(500):
        s = A.step(s, sp)
    assert abs(float(s.u.sum()) - tu) / tu < 1e-12
    assert abs(float(s.v.sum()) - tv) / tv < 1e-12
    assert float(s.v.std()) < 0.5 * std0  # perturbation spreads out instead of sharpening


def test_pattern_forms_with_reaction_but_not_in_control() -> None:
    def final(**dyn: Any) -> dict[str, float]:
        sp = make(n=128, dynamics={"F": 0.0545, "k": 0.062, **dyn})
        s = A.initialize(sp)
        for _ in range(4000):
            s = A.step(s, sp)
        return A.metrics(s, sp)

    coral, control = final(), final(reaction=False)
    assert coral["v_area_fraction"] > 0.1 and coral["v_std"] > 0.05
    assert control["v_area_fraction"] == 0.0 and control["v_std"] < 0.01


def test_seed_erase_inspect_and_checkpoint() -> None:
    sp = make()
    s = A.initialize(sp)
    for _ in range(25):
        s = A.step(s, sp)
    s, info = A.apply_intervention(s, sp, "seed", {"x": 0, "y": 0, "radius": 2})
    assert info["cells"] == 13 and s.v[63, 63] == 0.25  # wraps around
    s, _ = A.apply_intervention(s, sp, "erase", {"x": 32, "y": 32, "radius": 12})
    assert (s.u[32, 32], s.v[32, 32]) == (1.0, 0.0)
    q = A.inspect(s, sp, {"x": 0, "y": 0})
    ser = A.serialize_state(s)
    r = A.deserialize_state(decode_checkpoint(encode_checkpoint(ser["arrays"], ser["meta"])), sp)
    nxt = A.step(r, sp)
    assert nxt.v[0, 0] == pytest.approx(q["v"] + q["dv_dt"], abs=1e-15)
    s2 = A.deserialize_state(ser, sp)
    s3 = A.deserialize_state(ser, sp)
    for _ in range(30):
        s2, s3 = A.step(s2, sp), A.step(s3, sp)
    assert A.state_distance(s2, s3)["exact_equal"]
    assert len(A.render(s2)[1]) == 64 * 64
