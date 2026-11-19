"""AgentWorld: sensors, policy outputs, feeding never exceeds available food, energy
bookkeeping, mode semantics (evolution mutates, fixed copies, lifetime learns without
inheritance), controls, checkpoints and the inspector history."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.agent_world import model
from alife_lab.substrates.agent_world.adapter import AgentWorldAdapter

A = AgentWorldAdapter()


def make(seed: int = 0, count: int = 40, **dyn: Any) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "a",
                "substrate": "agent_world",
                "seed": seed,
                "world": {"width": 50.0, "height": 50.0, "resolution": 25},
                "initialization": {"count": count},
                "dynamics": dyn,
            }
        }
    )


def test_probabilities_and_sensors() -> None:
    sp = make()
    s = A.initialize(sp)
    x = model.sensors(s.agents, s.food, s.signal, sp.experiment)  # type: ignore[arg-type]
    assert x.shape == (40, model.N_IN) and (x[:, 7] == 1).all()
    _, p = model.policy(s.agents, x, sp.experiment.dynamics)  # type: ignore[union-attr]
    np.testing.assert_allclose(p.sum(1), 1.0)
    sp2 = make(communication=False)
    s2 = A.initialize(sp2)
    s2.signal[:] = 5.0
    x2 = model.sensors(s2.agents, s2.food, s2.signal, sp2.experiment)  # type: ignore[arg-type]
    assert not x2[:, 3:6].any()


def test_feeding_never_exceeds_food_and_food_stays_bounded() -> None:
    sp = make(count=300, food={"initial": 0.05, "bite": 1.0})
    s = A.initialize(sp)
    for _ in range(50):
        before = s.food.copy()
        s = A.step(s, sp)
        assert (s.food >= -1e-12).all() and (s.food <= 1.0 + 1e-12).all()
        del before


def test_evolution_mutates_fixed_copies_lifetime_learns_without_inheritance() -> None:
    def after(mode: str, n: int = 400) -> Any:
        sp = make(controller={"mode": mode}, body={"reproduce_threshold": 6.0})
        s = A.initialize(sp)
        w0 = {int(i): w for i, w in zip(s.agents.ids, s.agents.w1, strict=True)}
        for _ in range(n):
            s = A.step(s, sp)
        return s, w0

    for mode in ("fixed", "lifetime"):
        s, w0 = after(mode)
        kids = np.flatnonzero(s.agents.parent >= 0)
        assert kids.size > 0
        # genetic weights of every agent equal those of some founder (no mutation)
        founders = np.stack(list(w0.values()))
        for k in kids[:20]:
            assert any(np.array_equal(s.agents.w1[k], f) for f in founders)
    s, _ = after("lifetime")
    old = s.agents.age > 50
    assert np.linalg.norm(s.agents.dw2[old], axis=(1, 2)).max() > 0  # learned during life
    newborn = s.agents.age == 0
    assert not s.agents.dw2[newborn].any()  # learning is not inherited
    s, w0 = after("evolution")
    kids = np.flatnonzero(s.agents.parent >= 0)
    founders = np.stack(list(w0.values()))
    assert not any(np.array_equal(s.agents.w1[k], f) for k in kids[:20] for f in founders)


def test_controls_shuffle_and_random() -> None:
    sp = make(shuffle_policies=True)
    s = A.initialize(sp)
    ids, w = s.agents.ids.copy(), s.agents.w1.copy()
    s = A.step(s, sp)
    same = s.agents.ids[: ids.size] == ids
    assert not np.array_equal(s.agents.w1[: ids.size][same], w[same])  # weights moved
    spr = make(controller={"kind": "random"})
    sr = A.initialize(spr)
    x = model.sensors(sr.agents, sr.food, sr.signal, spr.experiment)  # type: ignore[arg-type]
    _, p = model.policy(sr.agents, x, spr.experiment.dynamics)  # type: ignore[union-attr]
    assert np.allclose(p, 0.25)


def test_checkpoint_continues_bitwise_and_inspector_history() -> None:
    sp = make(controller={"mode": "lifetime"})
    s = A.initialize(sp)
    for _ in range(60):
        s = A.step(s, sp)
    ser = A.serialize_state(s)
    r = A.deserialize_state(decode_checkpoint(encode_checkpoint(ser["arrays"], ser["meta"])), sp)
    for _ in range(60):
        s, r = A.step(s, sp), A.step(r, sp)
    assert A.state_distance(s, r)["exact_equal"]
    q = A.inspect(s, sp, {"x": 25, "y": 25})
    hist = q["agent"]["history"]
    assert len(hist["action"]) == min(32, q["agent"]["age"])
    assert all(abs(sum(p) - 1) < 1e-3 for p in hist["probs"])
    s, info = A.apply_intervention(s, sp, "remove", {"x": 25, "y": 25, "radius": 50})
    assert info["removed"] > 0 and s.agents.n == 0
    s, info = A.apply_intervention(s, sp, "add_food", {"x": 25, "y": 25, "radius": 5})
    assert info["cells"] > 0


def test_spec_guards() -> None:
    with pytest.raises(ValidationError):
        make(count=3000, max_population=100)
