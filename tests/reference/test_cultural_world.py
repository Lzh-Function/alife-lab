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


# --------------------------------------------------------------------------- #
# Stage 3 / P4-P5 tests: Compositional DSL, PlacedArtifacts, H6, E2/E3 Assays
# --------------------------------------------------------------------------- #


def test_p4_dsl_pipeline_order_sensitivity_and_safety() -> None:
    """Whitelisted DSL strictly rejects unsafe ops, rewards order, and composes DAG nodes."""
    from alife_lab.substrates.cultural_world.technology import (
        ALLOWED_OPS,
        compose_skills,
        evaluate_pipeline,
    )

    assert {"combine", "transform", "place_object", "use_object"} <= set(ALLOWED_OPS)
    for bad in ("eval", "exec", "__import__", "os.system", "import_mod"):
        with pytest.raises(ValueError, match="disallowed operation"):
            evaluate_pipeline(("process_a", bad, "extract"), 1)

    good_seq = ("locate", "process_a", "process_b", "combine", "transform", "extract", "store")
    bad_order = ("combine", "transform", "extract", "process_a", "process_b", "locate", "store")
    eff_good, comp_good, valid_good = evaluate_pipeline(good_seq, 3)
    eff_bad, comp_bad, valid_bad = evaluate_pipeline(bad_order, 3)
    assert comp_good is True and valid_good is True
    assert comp_bad is False and valid_bad is False
    assert eff_good > eff_bad + 1.0

    pa = SkillProcedure(
        skill_id=10,
        parent_skill_id=-1,
        revision=0,
        inventor_id=1,
        origin_tick=5,
        depth=0,
        target_resource=1,
        ops=("locate", "process_a", "extract"),
        phase=0.25,
        efficiency=1.40,
        root_efficiency=1.40,
    )
    pb = SkillProcedure(
        skill_id=11,
        parent_skill_id=-1,
        revision=0,
        inventor_id=2,
        origin_tick=7,
        depth=0,
        target_resource=2,
        ops=("gather", "process_b", "extract"),
        phase=0.35,
        efficiency=1.45,
        root_efficiency=1.45,
    )
    child = compose_skills(
        pa,
        pb,
        skill_id=12,
        composer_id=3,
        tick=20,
        env_phase=0.30,
        max_len=8,
        allow_externalization=True,
        rng_tech=np.random.default_rng(7),
    )
    assert child.parent_skill_id == 10
    assert child.secondary_parent_id == 11
    assert child.depth == 1
    assert child.target_resource == 3
    assert "combine" in child.ops
    assert child.efficiency > max(pa.efficiency, pb.efficiency)


def test_p4_artifacts_conservation_decay_and_destroy_objects() -> None:
    """PlacedArtifact material is strictly conserved across placement, decay, and destroy."""
    presets = load_presets(CONFIGS_DIR)
    raw = presets["cultural_world_compositional"].spec.model_dump(mode="json")
    raw["experiment"]["dynamics"]["technology"]["artifact_decay_ticks"] = 35
    raw["experiment"]["dynamics"]["technology"]["invention_prob"] = 0.08
    spec = ExperimentSpec.model_validate(raw)
    adapter = get_adapter("cultural_world")
    state = adapter.initialize(spec)
    max_art = 0
    for t in range(1, 220):
        state = adapter.step(state, spec)
        max_art = max(max_art, len(state.artifacts))
        if t % 20 == 0:
            res = adapter.metrics(state, spec)["resource_ledger_residual"]
            assert res <= 1e-9, f"ledger leak at t={t}: {res}"
    assert max_art >= 1, "expected externalized artifacts to be placed"
    assert state.externalized_uses >= 0

    if not state.artifacts:
        for _ in range(40):
            state = adapter.step(state, spec)
            if state.artifacts:
                break
    assert len(state.artifacts) >= 1
    wiped, info = adapter.apply_intervention(state, spec, "destroy_objects", {})
    assert info["destroyed_artifacts"] >= 1
    assert len(wiped.artifacts) == 0
    assert adapter.metrics(wiped, spec)["resource_ledger_residual"] <= 1e-9


def test_p4_checkpoint_roundtrip_artifacts_and_dual_parents() -> None:
    """V2 checkpoint serialization preserves PlacedArtifact objects and dual-parent DAG."""
    presets = load_presets(CONFIGS_DIR)
    spec = presets["cultural_world_compositional"].spec
    adapter = get_adapter("cultural_world")
    state = adapter.initialize(spec)
    for _ in range(180):
        state = adapter.step(state, spec)
    blob = adapter.serialize_state(state)
    restored = adapter.deserialize_state(blob, spec)
    uninterrupted = adapter.initialize(spec)
    for _ in range(180):
        uninterrupted = adapter.step(uninterrupted, spec)
    uninterrupted = adapter.step(uninterrupted, spec)
    restored = adapter.step(restored, spec)
    dist = adapter.state_distance(uninterrupted, restored)
    assert dist["exact_equal"] is True
    assert adapter.metrics(restored, spec)["resource_ledger_residual"] <= 1e-9
    # Verify secondary_parent_id survived roundtrip for composed procedures
    composed_before = [
        p.secondary_parent_id
        for p in state.registry.procedures.values()
        if p.secondary_parent_id >= 0
    ]
    if composed_before:
        assert any(
            p.secondary_parent_id >= 0
            for p in adapter.deserialize_state(blob, spec).registry.procedures.values()
        )


def test_p4_h6_externalized_recovery_after_culture_wipe() -> None:
    """H6: erasing carried culture keeps PlacedArtifact objects for async skill recovery."""
    presets = load_presets(CONFIGS_DIR)
    raw = presets["cultural_world_compositional"].spec.model_dump(mode="json")
    raw["experiment"]["seed"] = 1
    spec = ExperimentSpec.model_validate(raw)
    adapter = get_adapter("cultural_world")
    st = adapter.initialize(spec)
    for _ in range(250):
        st = adapter.step(st, spec)
    assert len(st.artifacts) > 0, "expected active artifacts before midpoint wipe"
    snap = adapter.serialize_state(st)

    st_keep = adapter.deserialize_state(snap, spec)
    st_keep, _ = adapter.apply_intervention(st_keep, spec, "erase_culture", {})
    st_wipe = adapter.deserialize_state(snap, spec)
    st_wipe, _ = adapter.apply_intervention(st_wipe, spec, "erase_culture", {})
    st_wipe, _ = adapter.apply_intervention(st_wipe, spec, "destroy_objects", {})
    for _ in range(80):
        st_keep = adapter.step(st_keep, spec)
        st_wipe = adapter.step(st_wipe, spec)

    m_keep = adapter.metrics(st_keep, spec)
    m_wipe = adapter.metrics(st_wipe, spec)
    assert m_keep["resource_ledger_residual"] <= 1e-9
    assert m_wipe["resource_ledger_residual"] <= 1e-9
    assert m_keep["externalized_learning_fraction"] > m_wipe["externalized_learning_fraction"]
    assert m_keep["improvement_depth"] >= m_wipe["improvement_depth"]


def test_p5_cumulative_culture_assay_passes_compositional_and_rejects_negative_control() -> None:
    """P5 5-criterion assay validates compositional E3 and rejects no_composition E1 cleanly."""
    from alife_lab.evolution.cultural_assays import judge, make_plan, run_cultural_assay

    presets = load_presets(CONFIGS_DIR)
    rep_pos = run_cultural_assay(presets["cultural_world_compositional"].spec, n_seeds=3, steps=450)
    assert rep_pos["judgement"]["status"] == "validated", rep_pos["judgement"]
    assert rep_pos["judgement"]["evidence_level"] == "E3", rep_pos["judgement"]

    rep_neg = run_cultural_assay(
        presets["cultural_world_no_composition"].spec, n_seeds=3, steps=450
    )
    assert rep_neg["judgement"]["status"] == "rejected", rep_neg["judgement"]
    assert rep_neg["judgement"]["evidence_level"] == "E1"
    assert rep_neg["judgement"]["verdicts"]["compositional_refinement"]["passed"] is False

    # Verify E2 vs E3 rule boundary directly (core 4 pass, causal fails -> E2)
    spec = presets["cultural_world_compositional"].spec
    plan = make_plan(spec.experiment, spec.model_dump(mode="json"), {}, n_seeds=3, steps=200)
    synthetic_results = {
        **rep_pos["results"],
        "erase_culture_causal": {
            "id": "erase_culture_causal",
            "intact_depth_60": 1.0,
            "wiped_depth_60": 2.0,
            "intact_final_utility": 1.0,
            "mute_final_utility": 1.5,
            "intact_final_energy": 10.0,
            "mute_final_energy": 12.0,
        },
    }
    e2_verdict = judge(plan, synthetic_results)
    assert e2_verdict["status"] == "validated"
    assert e2_verdict["evidence_level"] == "E2"


def test_p5_discovery_ledger_api_and_2level_sweeps(tmp_path: Path) -> None:
    """API registers cumulative_culture in Discovery Ledger and validates 2-level sweep specs."""
    import time
    from dataclasses import replace

    import yaml
    from fastapi.testclient import TestClient

    from alife_lab.api.app import create_app
    from alife_lab.core.sweep import SweepSpec
    from alife_lab.execution.executors import LocalExecutor
    from alife_lab.service.settings import load_settings
    from alife_lab.service.sweeps import SweepService

    settings = replace(load_settings(), home=tmp_path / "home", runs_root=tmp_path / "runs")
    tok = "cw-p5-token"
    app = create_app(settings, LocalExecutor(), tok, 8000)
    headers = {"X-ALife-Token": tok, "Origin": "http://127.0.0.1:8000"}

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        r = client.post(
            "/api/v1/experiments",
            json={
                "preset_id": "cultural_world_compositional",
                "profile_id": "cpu_dev",
                "ticks_per_second": 0,
                "auto_start": True,
                "start_paused": True,
            },
            headers={**headers, "Idempotency-Key": "cw-p5-create"},
        )
        assert r.status_code == 201, r.text
        rid = r.json()["data"]["id"]
        for _ in range(80):
            if client.get(f"/api/v1/experiments/{rid}").json()["data"]["state"] == "paused":
                break
            time.sleep(0.05)
        st = client.post(f"/api/v1/experiments/{rid}/step", json={"count": 30}, headers=headers)
        assert st.status_code == 200
        insp = client.get(f"/api/v1/experiments/{rid}/inspect?x=32&y=32").json()["data"]
        assert insp["externalized_enabled"] is True
        assert "active_artifacts" in insp

        d_res = client.post(
            "/api/v1/discoveries",
            json={
                "run_id": rid,
                "candidate": 0,
                "claim_kind": "cumulative_culture",
                "title": "Compositional P4/P5 check",
            },
            headers=headers,
        )
        assert d_res.status_code == 201, d_res.text
        d = d_res.json()["data"]
        assert d["claim_kind"] == "cumulative_culture"
        assert d["evidence_level"] == "E1"
        plan = d["assays"][0]["plan"]
        assert plan["max_evidence_level"] == "E3"
        assert [c["id"] for c in plan["checks"]] == [
            "post_inventor_retention",
            "compositional_refinement",
            "transmission_ablation",
            "erase_culture_causal",
            "heldout_transfer",
        ]
        rec = client.get(f"/api/v1/discoveries/{d['id']}/record").json()["data"]
        assert "Pre-registered 5-criterion" in rec["claim"]
        assert any("E3" in lim for lim in rec["limitations"])

    # Also validate new 2-level sweeps through SweepService.plan
    from alife_lab.service.controller import Controller

    ctrl = Controller(settings, LocalExecutor())
    try:
        presets = load_presets(CONFIGS_DIR)
        svc = SweepService(ctrl)
        for fname, n_runs in (
            ("cultural_world_p4_composition_ablation.yaml", 10),
            ("cultural_world_p5_phase_diagram.yaml", 27),
        ):
            raw_sw = yaml.safe_load((CONFIGS_DIR / "sweeps" / fname).read_text(encoding="utf-8"))
            plan_out = svc.plan(SweepSpec.model_validate(raw_sw))
            assert plan_out["runs"] == n_runs
            assert (
                presets["cultural_world_compositional"].spec.experiment.substrate
                == "cultural_world"
            )
    finally:
        ctrl.db.close()
