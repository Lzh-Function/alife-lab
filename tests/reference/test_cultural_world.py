"""Reference, conservation, 3-layer separation, and causal tests for ``cultural_world``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from alife_lab.core.spec import CulturalWorldExperiment, ExperimentSpec
from alife_lab.service.catalog import load_presets
from alife_lab.substrates import get_adapter
from alife_lab.substrates.cultural_world.agents import make_offspring
from alife_lab.substrates.cultural_world.ecology import max_ledger_residual
from alife_lab.substrates.cultural_world.memory import SkillProcedure

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def _make_spec(
    *,
    seed: int = 42,
    count: int = 36,
    regime: str = "seasonal",
    controller_kind: str = "scripted",
    tx_regime: str = "custom",
    vertical: bool = True,
    horizontal: bool = True,
    oblique: bool = False,
    lifetime_learning: bool = True,
    allow_invention: bool = True,
    allow_improvement: bool = True,
    invention_prob: float = 0.2,
    error_probability: float = 0.0,
    max_age: int = 120,
    reproduce_threshold: float = 6.5,
) -> ExperimentSpec:
    return ExperimentSpec(
        schema_version=1,
        experiment=CulturalWorldExperiment(
            name="test_cultural",
            mode="exploratory",
            substrate="cultural_world",
            seed=seed,
            world={
                "width": 32.0,
                "height": 32.0,
                "resolution": 16,
                "environment_regime": regime,
                "regime_period": 80,
            },
            initialization={"count": count},
            dynamics={
                "genetic_evolution": True,
                "lifetime_learning": lifetime_learning,
                "transmission_regime": tx_regime,
                "transmission": {
                    "vertical": vertical,
                    "horizontal": horizontal,
                    "oblique": oblique,
                    "observation_range": 8.0,
                    "transmission_cost": 0.005,
                    "error_probability": error_probability,
                    "base_success_prob": 0.9,
                },
                "technology": {
                    "allow_invention": allow_invention,
                    "allow_improvement": allow_improvement,
                    "max_program_length": 4,
                    "max_skills_per_agent": 4,
                    "invention_prob": invention_prob,
                    "skill_energy_cost": 0.01,
                    "forbid_preinstalled_recipes": True,
                },
                "controller": {
                    "kind": controller_kind,
                    "hidden": 6,
                    "learning_rate": 0.05,
                    "mutation_sigma": 0.08,
                },
                "initial_energy": 4.5,
                "metabolism": 0.04,
                "move_cost": 0.01,
                "reproduce_threshold": reproduce_threshold,
                "max_age": max_age,
                "max_population": 200,
            },
        ),
    )


def test_schema_rejects_reference_and_dsl_whitelist_and_presets_load() -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec(
            schema_version=1,
            experiment={
                "name": "bad_ref",
                "mode": "reference",
                "substrate": "cultural_world",
                "seed": 1,
            },
        )

    with pytest.raises(ValueError, match="disallowed operation"):
        SkillProcedure(
            skill_id=1,
            parent_skill_id=-1,
            revision=0,
            inventor_id=0,
            origin_tick=1,
            depth=0,
            target_resource=1,
            ops=("locate", "eval_arbitrary_python"),
            phase=0.0,
            efficiency=1.0,
        )

    presets = load_presets(CONFIGS_DIR)
    cw_presets = [p for p in presets.values() if p.spec.experiment.substrate == "cultural_world"]
    assert len(cw_presets) >= 5
    for p in cw_presets:
        assert isinstance(p.spec.experiment, CulturalWorldExperiment)
        assert p.spec.experiment.dynamics.technology.forbid_preinstalled_recipes is True


@pytest.mark.parametrize("regime", ["stationary", "seasonal", "shifting_patches"])
def test_resource_ledger_conservation_and_nonnegativity(regime: str) -> None:
    adapter = get_adapter("cultural_world")
    spec = _make_spec(seed=5, regime=regime, controller_kind="mlp")
    state = adapter.initialize(spec)
    assert max_ledger_residual(state.eco, 0.0, 0.0) == 0.0

    for step_idx in range(100):
        state = adapter.step(state, spec)
        if step_idx == 40:
            state, info = adapter.apply_intervention(
                state,
                spec,
                "add_resource",
                {"x": 16.0, "y": 16.0, "radius": 6.0, "kind": "material_a", "amount": 0.3},
            )
            assert info["added"] > 0.0
        a = state.agents
        assert np.all(state.eco.food >= 0.0)
        assert np.all(state.eco.material_a >= 0.0)
        assert np.all(state.eco.material_b >= 0.0)
        if a.n > 0:
            assert np.all(a.inv_a >= 0.0)
            assert np.all(a.inv_b >= 0.0)
        residual = max_ledger_residual(
            state.eco,
            float(a.inv_a.sum()) if a.n else 0.0,
            float(a.inv_b.sum()) if a.n else 0.0,
        )
        assert residual <= 1e-9
        metrics = adapter.metrics(state, spec)
        assert metrics["resource_ledger_residual"] <= 1e-9


def test_strict_three_layer_separation_on_offspring() -> None:
    adapter = get_adapter("cultural_world")
    spec = _make_spec(seed=9, controller_kind="mlp", lifetime_learning=True, vertical=True)
    state = adapter.initialize(spec)
    for _ in range(50):
        state = adapter.step(state, spec)

    assert state.agents.n > 0
    assert float(np.max(np.abs(state.agents.dw2))) > 0.0

    # Spawn newborns from top-energy parents and verify zero leakage from lifetime/cultural layers.
    p_idx = np.arange(min(5, state.agents.n), dtype=np.int64)
    exp = spec.experiment
    assert isinstance(exp, CulturalWorldExperiment)
    children, _ = make_offspring(
        state.agents,
        p_idx,
        tick=state.tick,
        next_id=state.next_id,
        exp=exp,
        rng_birth=state.rngs["birth"],
    )
    assert np.all(children.dw2 == 0.0)
    assert np.all(children.rbase == 0.0)
    assert np.all(children.inv_a == 0.0)
    assert np.all(children.inv_b == 0.0)
    assert np.all(children.private_best_val == 0.0)
    assert all(len(s) == 0 for s in children.skills)


def test_transmission_channel_isolation_and_post_death_retention() -> None:
    adapter = get_adapter("cultural_world")

    # Group A: genetic only -> no invention, no lifetime dw2, no transmission
    spec_a = _make_spec(
        seed=12,
        controller_kind="mlp",
        lifetime_learning=False,
        tx_regime="none",
        allow_invention=False,
        allow_improvement=False,
    )
    st_a = adapter.initialize(spec_a)
    for _ in range(60):
        st_a = adapter.step(st_a, spec_a)
    m_a = adapter.metrics(st_a, spec_a)
    assert np.all(st_a.agents.dw2 == 0.0)
    assert m_a["mean_skills_per_agent"] == 0.0
    assert m_a["social_learning_fraction"] == 0.0
    assert m_a["skill_retention_after_death"] == 0.0

    # Group B: lifetime only -> inventions allowed, 0 transmission attempts
    spec_b = _make_spec(seed=12, tx_regime="none", invention_prob=0.25)
    st_b = adapter.initialize(spec_b)
    for _ in range(60):
        st_b = adapter.step(st_b, spec_b)
    assert st_b.tx_attempts == 0
    assert st_b.tx_successes == 0
    assert adapter.metrics(st_b, spec_b)["social_learning_fraction"] == 0.0
    assert adapter.metrics(st_b, spec_b)["skill_retention_after_death"] == 0.0

    # Group C: vertical only -> only "vertical" mode in log
    spec_c = _make_spec(seed=12, tx_regime="vertical_only", invention_prob=0.25, max_age=80)
    st_c = adapter.initialize(spec_c)
    for _ in range(140):
        st_c = adapter.step(st_c, spec_c)
    assert len(st_c.registry.transmissions) > 0
    assert all(ev["transmission_mode"] == "vertical" for ev in st_c.registry.transmissions)

    # Oblique only: no vertical, no horizontal -> only "oblique" mode in log (older non-parents)
    spec_o = _make_spec(
        seed=12,
        vertical=False,
        horizontal=False,
        oblique=True,
        invention_prob=0.25,
        max_age=120,
        reproduce_threshold=6.0,
    )
    st_o = adapter.initialize(spec_o)
    for _ in range(160):
        st_o = adapter.step(st_o, spec_o)
    assert len(st_o.registry.transmissions) > 0
    assert all(ev["transmission_mode"] == "oblique" for ev in st_o.registry.transmissions)

    # Multi-channel with mortality -> invented skills persist after original inventor dies
    spec_e = _make_spec(
        seed=12,
        vertical=True,
        horizontal=True,
        oblique=True,
        invention_prob=0.25,
        max_age=65,
        reproduce_threshold=6.2,
    )
    st_e = adapter.initialize(spec_e)
    for _ in range(180):
        st_e = adapter.step(st_e, spec_e)
    m_e = adapter.metrics(st_e, spec_e)
    assert st_e.births > 0
    assert st_e.deaths > 0
    assert m_e["social_learning_fraction"] > 0.0
    assert m_e["skill_retention_after_death"] > 0.0


def test_erase_culture_causal_intervention_preserves_non_cultural_layers() -> None:
    adapter = get_adapter("cultural_world")
    spec = _make_spec(seed=21, controller_kind="mlp", lifetime_learning=True, invention_prob=0.25)
    state = adapter.initialize(spec)
    for _ in range(55):
        state = adapter.step(state, spec)

    a = state.agents
    assert sum(len(s) for s in a.skills) > 0

    snap_arrays = {
        "ids": a.ids.copy(),
        "w1": a.w1.copy(),
        "b1": a.b1.copy(),
        "w2": a.w2.copy(),
        "b2": a.b2.copy(),
        "dw2": a.dw2.copy(),
        "energy": a.energy.copy(),
        "inv_a": a.inv_a.copy(),
        "inv_b": a.inv_b.copy(),
        "x": a.x.copy(),
        "y": a.y.copy(),
        "food": state.eco.food.copy(),
        "mat_a": state.eco.material_a.copy(),
        "mat_b": state.eco.material_b.copy(),
    }

    state, detail = adapter.apply_intervention(state, spec, "erase_culture", {})
    assert detail["erased_skill_slots"] > 0
    assert all(len(s) == 0 for s in state.agents.skills)

    # Every non-cultural array must remain bit-identical.
    for key in ("ids", "w1", "b1", "w2", "b2", "dw2", "energy", "inv_a", "inv_b", "x", "y"):
        assert np.array_equal(getattr(state.agents, key), snap_arrays[key]), f"{key} mutated!"
    assert np.array_equal(state.eco.food, snap_arrays["food"])
    assert np.array_equal(state.eco.material_a, snap_arrays["mat_a"])
    assert np.array_equal(state.eco.material_b, snap_arrays["mat_b"])


def test_deterministic_checkpoint_roundtrip_and_split_env_stream() -> None:
    adapter = get_adapter("cultural_world")
    spec_a = _make_spec(seed=33, tx_regime="none")
    spec_e = _make_spec(seed=33, tx_regime="all")

    # Shared initial resource realization despite different transmission regimes
    st_a0 = adapter.initialize(spec_a)
    st_e0 = adapter.initialize(spec_e)
    assert np.array_equal(st_a0.eco.food, st_e0.eco.food)
    assert np.array_equal(st_a0.eco.material_a, st_e0.eco.material_a)
    assert np.array_equal(st_a0.eco.material_b, st_e0.eco.material_b)

    # Uninterrupted vs serialized checkpoint restore (stepping original before deserializing
    # guarantees serialized payloads do not alias mutable arrays).
    uninterrupted = adapter.initialize(spec_e)
    for _ in range(30):
        uninterrupted = adapter.step(uninterrupted, spec_e)
    payload = adapter.serialize_state(uninterrupted)
    for _ in range(25):
        uninterrupted = adapter.step(uninterrupted, spec_e)

    restored = adapter.deserialize_state(payload, spec_e)
    for _ in range(25):
        restored = adapter.step(restored, spec_e)

    dist = adapter.state_distance(uninterrupted, restored)
    assert dist["exact_equal"] is True
    assert dist["value"] == 0
    assert adapter.metrics(restored, spec_e)["resource_ledger_residual"] <= 1e-9


def test_cultural_world_api_lifecycle(tmp_path: Path) -> None:
    """End-to-end controller + API inspect + erase_culture paired-fork test."""
    import time
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from alife_lab.api.app import create_app
    from alife_lab.execution.executors import LocalExecutor
    from alife_lab.service.settings import load_settings

    settings = replace(load_settings(), home=tmp_path / "home", runs_root=tmp_path / "runs")
    tok = "cw-test-token"
    app = create_app(settings, LocalExecutor(), tok, 8000)
    headers = {"X-ALife-Token": tok, "Origin": "http://127.0.0.1:8000"}

    def wait_paused(client: TestClient, run_id: str) -> dict:
        for _ in range(80):
            info = client.get(f"/api/v1/experiments/{run_id}").json()["data"]
            if info["state"] == "paused":
                return info
            time.sleep(0.05)
        raise AssertionError(f"run {run_id} did not reach paused state")

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        r = client.post(
            "/api/v1/experiments",
            json={
                "preset_id": "cultural_world_baseline",
                "profile_id": "cpu_dev",
                "ticks_per_second": 0,
                "auto_start": True,
                "start_paused": True,
                "overrides": {
                    "initialization": {"count": 40},
                    "dynamics": {
                        "controller": {"kind": "scripted"},
                        "technology": {"invention_prob": 0.25},
                    },
                    "runner": {"max_steps": 500, "checkpoint_interval": 25},
                },
            },
            headers={**headers, "Idempotency-Key": "cw-api-create"},
        )
        assert r.status_code == 201, r.text
        rid = r.json()["data"]["id"]
        wait_paused(client, rid)

        st_r = client.post(f"/api/v1/experiments/{rid}/step", json={"count": 25}, headers=headers)
        assert st_r.status_code == 200, st_r.text

        insp = client.get(f"/api/v1/experiments/{rid}/inspect?x=32&y=32")
        assert insp.status_code == 200, insp.text
        idata = insp.json()["data"]
        assert idata["substrate"] == "cultural_world"
        assert idata["transmission"] == {"vertical": True, "horizontal": True, "oblique": False}
        assert idata["cultural_agent"] is not None
        assert idata["ledger_residuals"]["food"] <= 1e-9

        pf = client.post(
            f"/api/v1/experiments/{rid}/paired-fork",
            json={"treatment_intervention": {"kind": "erase_culture", "payload": {}}},
            headers={**headers, "Idempotency-Key": "cw-api-pf"},
        )
        assert pf.status_code == 201, pf.text
        pdata = pf.json()["data"]
        assert [m["arm"] for m in pdata["members"]] == ["control", "treatment"]
        arm_ids = [m["run"]["id"] for m in pdata["members"]]
        for aid in arm_ids:
            wait_paused(client, aid)
            sr = client.post(f"/api/v1/experiments/{aid}/step", json={"count": 15}, headers=headers)
            assert sr.status_code == 200, sr.text

        rep = client.get(f"/api/v1/comparisons/{pdata['id']}/report")
        assert rep.status_code == 200, rep.text
        assert set(rep.json()["data"]["metrics"]) == set(arm_ids)
        ev_treat = client.get(f"/api/v1/experiments/{arm_ids[1]}/events").json()["data"]
        assert any(e["type"] == "intervention_applied" for e in ev_treat)
