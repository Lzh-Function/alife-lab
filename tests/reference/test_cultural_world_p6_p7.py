"""Reference, conservation, causal isolation, API, and scaling tests for CulturalWorld P6–P7."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from alife_lab.api.app import create_app
from alife_lab.cli import main as cli_main
from alife_lab.core.spec import CulturalWorldExperiment, CulturalWorldSocial, ExperimentSpec
from alife_lab.evolution.cultural_social_assay import (
    run_cultural_scaling_benchmark,
    run_h4_exchange_assay,
)
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.catalog import load_presets
from alife_lab.service.settings import load_settings
from alife_lab.substrates import get_adapter
from alife_lab.substrates.cultural_world import codecs
from alife_lab.substrates.cultural_world.ecology import max_ledger_residual
from alife_lab.substrates.cultural_world.social import (
    apply_local_exchange,
    compute_social_metrics,
    extract_cultural_graph,
)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def _exchange_spec(seed: int = 3, **social_overrides: Any) -> ExperimentSpec:
    presets = load_presets(CONFIGS_DIR)
    raw = presets["cultural_world_exchange"].spec.model_dump(mode="json")
    raw["experiment"]["seed"] = seed
    raw["experiment"]["dynamics"]["social"].update(social_overrides)
    return ExperimentSpec.model_validate(raw)


def _no_exchange_spec(seed: int = 3) -> ExperimentSpec:
    presets = load_presets(CONFIGS_DIR)
    raw = presets["cultural_world_no_exchange_control"].spec.model_dump(mode="json")
    raw["experiment"]["seed"] = seed
    return ExperimentSpec.model_validate(raw)


def test_p6_defaults_presets_and_biome_segregation() -> None:
    """P6 social defaults preserve P0-P5, presets validate, and biomes segregate A vs B."""
    default_social = CulturalWorldSocial()
    assert default_social.allow_exchange is False
    assert default_social.interdependent_resources is False
    assert default_social.shuffle_roles_control is False

    presets = load_presets(CONFIGS_DIR)
    assert {"cultural_world_exchange", "cultural_world_no_exchange_control"}.issubset(
        presets.keys()
    )

    ex_spec = presets["cultural_world_exchange"].spec
    no_ex_spec = presets["cultural_world_no_exchange_control"].spec
    assert isinstance(ex_spec.experiment, CulturalWorldExperiment)
    assert isinstance(no_ex_spec.experiment, CulturalWorldExperiment)
    assert ex_spec.experiment.dynamics.social.allow_exchange is True
    assert ex_spec.experiment.dynamics.social.interdependent_resources is True
    assert ex_spec.experiment.dynamics.technology.allow_composition is True
    assert ex_spec.experiment.dynamics.technology.allow_externalization is True
    assert ex_spec.experiment.dynamics.technology.forbid_preinstalled_recipes is True
    assert no_ex_spec.experiment.dynamics.social.allow_exchange is False
    assert no_ex_spec.experiment.dynamics.social.interdependent_resources is True
    assert {
        "trade_count",
        "trade_volume",
        "specialization_index",
        "division_of_labor_index",
        "interdependence_index",
        "repeat_partner_rate",
    }.issubset(set(ex_spec.experiment.observations.metrics))

    adapter = get_adapter("cultural_world")
    state = adapter.initialize(ex_spec)
    r = ex_spec.experiment.world.resolution
    half = r // 2
    assert float(state.eco.material_a[:, :half].sum()) > float(state.eco.material_a[:, half:].sum())
    assert float(state.eco.material_b[:, half:].sum()) > float(state.eco.material_b[:, :half].sum())
    assert max_ledger_residual(state.eco, 0.0, 0.0) == 0.0


def test_p6_atomic_barter_mass_conservation_and_nonnegativity() -> None:
    """Every step and atomic barter preserves resource mass (<= 1e-9) and zero negative stock."""
    adapter = get_adapter("cultural_world")
    spec = _exchange_spec(seed=3)
    exp = spec.experiment
    assert isinstance(exp, CulturalWorldExperiment)
    state = adapter.initialize(spec)

    for step_idx in range(250):
        state = adapter.step(state, spec)
        a = state.agents
        assert np.all(state.eco.food >= 0.0)
        assert np.all(state.eco.material_a >= 0.0)
        assert np.all(state.eco.material_b >= 0.0)
        if a.n > 0:
            assert np.all(a.inv_a >= 0.0)
            assert np.all(a.inv_b >= 0.0)
        art_a = sum(art.mass_a for art in state.artifacts)
        art_b = sum(art.mass_b for art in state.artifacts)
        res = max_ledger_residual(
            state.eco,
            (float(a.inv_a.sum()) if a.n else 0.0) + art_a,
            (float(a.inv_b.sum()) if a.n else 0.0) + art_b,
        )
        assert res <= 1e-9, f"ledger leak at step {step_idx + 1}: {res}"
        assert adapter.metrics(state, spec)["resource_ledger_residual"] <= 1e-9

    assert state.trade_count > 0
    assert state.trade_volume > 0.0
    assert len(state.trade_events) > 0

    # Direct atomic barter verification on co-located complementary agents.
    a = state.agents
    assert a.n >= 2
    a.x[0], a.y[0] = 10.0, 10.0
    a.x[1], a.y[1] = 10.5, 10.5
    a.energy[0] = max(float(a.energy[0]), 3.0)
    a.energy[1] = max(float(a.energy[1]), 3.0)
    a.inv_a[0], a.inv_b[0] = 1.2, 0.0
    a.inv_a[1], a.inv_b[1] = 0.0, 1.2
    sum_a_before = float(a.inv_a.sum())
    sum_b_before = float(a.inv_b.sum())
    trades_now, vol_now = apply_local_exchange(state, exp)
    assert trades_now >= 1
    assert vol_now > 0.0
    assert abs(float(a.inv_a.sum()) - sum_a_before) <= 1e-12
    assert abs(float(a.inv_b.sum()) - sum_b_before) <= 1e-12
    assert np.all(a.inv_a >= 0.0)
    assert np.all(a.inv_b >= 0.0)

    # Atomic all-or-nothing rejection when counterparty cannot cover minimum lot.
    iso_spec = _exchange_spec(seed=3)
    iso_exp = iso_spec.experiment
    assert isinstance(iso_exp, CulturalWorldExperiment)
    iso = adapter.initialize(iso_spec)
    iso.agents = iso.agents.take(np.asarray([True, True] + [False] * (iso.agents.n - 2)))
    iso.agents.x[:] = 8.0
    iso.agents.y[:] = 8.0
    iso.agents.energy[:] = 3.0
    iso.agents.inv_a[:] = np.asarray([0.8, 0.0], dtype=np.float64)
    iso.agents.inv_b[:] = np.asarray([0.0, 0.02], dtype=np.float64)
    inv_a_snap = iso.agents.inv_a.copy()
    inv_b_snap = iso.agents.inv_b.copy()
    rej_trades, rej_vol = apply_local_exchange(iso, iso_exp)
    assert rej_trades == 0
    assert rej_vol == 0.0
    assert np.array_equal(iso.agents.inv_a, inv_a_snap)
    assert np.array_equal(iso.agents.inv_b, inv_b_snap)


def test_p6_forbid_exchange_causal_isolation_and_toggle() -> None:
    """``forbid_exchange`` freezes future barter while leaving all other layers bit-identical."""
    adapter = get_adapter("cultural_world")
    spec = _exchange_spec(seed=5)
    state = adapter.initialize(spec)
    for _ in range(220):
        state = adapter.step(state, spec)

    assert state.trade_count > 0
    assert sum(len(s) for s in state.agents.skills) > 0
    a = state.agents

    snap_arrays = {
        "ids": a.ids.copy(),
        "w1": a.w1.copy(),
        "b1": a.b1.copy(),
        "w2": a.w2.copy(),
        "b2": a.b2.copy(),
        "gene_lr": a.gene_lr.copy(),
        "gene_social": a.gene_social.copy(),
        "dw2": a.dw2.copy(),
        "rbase": a.rbase.copy(),
        "energy": a.energy.copy(),
        "inv_a": a.inv_a.copy(),
        "inv_b": a.inv_b.copy(),
        "x": a.x.copy(),
        "y": a.y.copy(),
        "heading": a.heading.copy(),
        "private_best_x": a.private_best_x.copy(),
        "private_best_y": a.private_best_y.copy(),
        "private_best_val": a.private_best_val.copy(),
        "food": state.eco.food.copy(),
        "material_a": state.eco.material_a.copy(),
        "material_b": state.eco.material_b.copy(),
    }
    skills_snap = copy.deepcopy(a.skills)
    artifacts_snap = copy.deepcopy(state.artifacts)
    trades_at_forbid = state.trade_count
    vol_at_forbid = state.trade_volume

    state, detail = adapter.apply_intervention(state, spec, "forbid_exchange", {"forbid": True})
    assert detail == {"exchange_enabled": False}
    assert state.exchange_override is False

    for key in (
        "ids",
        "w1",
        "b1",
        "w2",
        "b2",
        "gene_lr",
        "gene_social",
        "dw2",
        "rbase",
        "energy",
        "inv_a",
        "inv_b",
        "x",
        "y",
        "heading",
        "private_best_x",
        "private_best_y",
        "private_best_val",
    ):
        assert np.array_equal(getattr(state.agents, key), snap_arrays[key]), f"{key} mutated!"
    assert np.array_equal(state.eco.food, snap_arrays["food"])
    assert np.array_equal(state.eco.material_a, snap_arrays["material_a"])
    assert np.array_equal(state.eco.material_b, snap_arrays["material_b"])
    assert state.agents.skills == skills_snap
    assert state.artifacts == artifacts_snap

    # Stepping after forbid_exchange produces zero subsequent trades.
    for _ in range(60):
        state = adapter.step(state, spec)
    assert state.trade_count == trades_at_forbid
    assert state.trade_volume == vol_at_forbid
    insp_off = adapter.inspect(state, spec, {"x": 0.0, "y": 0.0})
    assert insp_off["exchange_enabled"] is False
    assert insp_off["social_summary"]["exchange_enabled"] is False

    # Empty payload {} toggles exchange back on and resumes trading.
    state, toggle_on = adapter.apply_intervention(state, spec, "forbid_exchange", {})
    assert toggle_on == {"exchange_enabled": True}
    for _ in range(60):
        state = adapter.step(state, spec)
    assert state.trade_count > trades_at_forbid

    state, toggle_off = adapter.apply_intervention(state, spec, "forbid_exchange", {})
    assert toggle_off == {"exchange_enabled": False}


def test_p6_checkpoint_v3_roundtrip_and_v2_backward_compat() -> None:
    """Checkpoint v3 round-trips social state bit-for-bit and steps identically."""
    assert codecs.SERIALIZE_VERSION == 3
    adapter = get_adapter("cultural_world")
    spec = _exchange_spec(seed=7)

    uninterrupted = adapter.initialize(spec)
    for _ in range(220):
        uninterrupted = adapter.step(uninterrupted, spec)
    assert uninterrupted.trade_count > 0
    assert len(uninterrupted.trade_events) > 0
    assert len(uninterrupted.partner_pairs) > 0

    blob = adapter.serialize_state(uninterrupted)
    assert blob["meta"]["serialize_version"] == 3

    # Step original first to prove serialized blob holds zero mutable aliases.
    for _ in range(30):
        uninterrupted = adapter.step(uninterrupted, spec)

    restored = adapter.deserialize_state(blob, spec)
    for _ in range(30):
        restored = adapter.step(restored, spec)

    dist = adapter.state_distance(uninterrupted, restored)
    assert dist["exact_equal"] is True
    assert dist["value"] == 0
    assert uninterrupted.trade_count == restored.trade_count
    assert uninterrupted.trade_volume == pytest.approx(restored.trade_volume, abs=0.0, rel=0.0)
    assert uninterrupted.repeat_trades == restored.repeat_trades
    assert uninterrupted.trade_events == restored.trade_events
    assert uninterrupted.partner_pairs == restored.partner_pairs
    assert uninterrupted.agent_role_uses == restored.agent_role_uses
    assert uninterrupted.agent_trades == restored.agent_trades
    assert uninterrupted.exchange_override == restored.exchange_override
    assert adapter.metrics(restored, spec)["resource_ledger_residual"] <= 1e-9

    # Verify exchange_override=False also round-trips and stays locked across steps.
    forbid_blob = copy.deepcopy(blob)
    forbid_state = adapter.deserialize_state(forbid_blob, spec)
    forbid_state, _ = adapter.apply_intervention(
        forbid_state, spec, "forbid_exchange", {"forbid": True}
    )
    f_blob = adapter.serialize_state(forbid_state)
    trades_locked = forbid_state.trade_count
    for _ in range(20):
        forbid_state = adapter.step(forbid_state, spec)
    f_restored = adapter.deserialize_state(f_blob, spec)
    assert f_restored.exchange_override is False
    for _ in range(20):
        f_restored = adapter.step(f_restored, spec)
    assert f_restored.trade_count == trades_locked
    assert adapter.state_distance(forbid_state, f_restored)["exact_equal"] is True

    # Backward compatibility: v2 payload lacking P6 metadata deserializes cleanly.
    v2_blob = copy.deepcopy(blob)
    v2_blob["meta"]["serialize_version"] = 2
    for key in (
        "exchange_override",
        "trade_events",
        "partner_pairs",
        "trade_count",
        "trade_volume",
        "repeat_trades",
        "agent_role_uses",
        "agent_trades",
    ):
        v2_blob["meta"].pop(key, None)
    v2_restored = adapter.deserialize_state(v2_blob, spec)
    assert v2_restored.trade_count == 0
    assert v2_restored.exchange_override is None
    v2_restored = adapter.step(v2_restored, spec)
    assert adapter.metrics(v2_restored, spec)["resource_ledger_residual"] <= 1e-9


def test_p6_specialization_vs_division_of_labor_and_cultural_graph() -> None:
    """Distinguish specialization from division of labor and validate DAG + Trade Network."""
    adapter = get_adapter("cultural_world")
    ex_spec = _exchange_spec(seed=2)
    no_ex_spec = _no_exchange_spec(seed=2)
    ex_exp = ex_spec.experiment
    assert isinstance(ex_exp, CulturalWorldExperiment)

    st_ex = adapter.initialize(ex_spec)
    st_no = adapter.initialize(no_ex_spec)
    for _ in range(260):
        st_ex = adapter.step(st_ex, ex_spec)
        st_no = adapter.step(st_no, no_ex_spec)

    m_ex = adapter.metrics(st_ex, ex_spec)
    m_no = adapter.metrics(st_no, no_ex_spec)

    # Unilateral biome specialization alone produces specialization_index > 0 but zero DoL.
    assert m_no["specialization_index"] > 0.4
    assert m_no["trade_count"] == 0.0
    assert m_no["division_of_labor_index"] == 0.0
    assert m_no["interdependence_index"] == 0.0

    # Active bilateral exchange + complementary specialization produces high DoL & interdependence.
    assert m_ex["trade_count"] > 0.0
    assert m_ex["specialization_index"] > 0.5
    assert m_ex["division_of_labor_index"] > 0.3
    assert m_ex["interdependence_index"] > 0.2

    # Synthetic unilateral-all-A population: specialization_index == 1.0, DoL == 0.0 even w/ trades.
    uni = adapter.deserialize_state(adapter.serialize_state(st_ex), ex_spec)
    uni.agent_role_uses = {int(aid): (10, 0, 0) for aid in uni.agents.ids.tolist()}
    m_uni = compute_social_metrics(uni, ex_exp)
    assert m_uni["specialization_index"] == pytest.approx(1.0)
    assert m_uni["division_of_labor_index"] == 0.0

    # Standard inspect omits cultural_graph; view=graph includes bounded DAG + Trade Network.
    std_insp = adapter.inspect(st_ex, ex_spec, {"x": 0.0, "y": 0.0})
    assert "cultural_graph" not in std_insp
    assert std_insp["exchange_enabled"] is True
    assert set(std_insp["social_summary"]) == {
        "trade_count",
        "trade_volume",
        "specialization_index",
        "division_of_labor_index",
        "interdependence_index",
        "repeat_partner_rate",
        "exchange_enabled",
    }
    assert std_insp["cultural_agent"] is not None
    assert std_insp["cultural_agent"]["role"] in {"A", "B", "dual", "generalist"}
    assert isinstance(std_insp["cultural_agent"]["trades"], int)

    graph_insp = adapter.inspect(st_ex, ex_spec, {"x": 0.0, "y": 0.0, "view": "graph"})
    assert "cultural_graph" in graph_insp
    graph = extract_cultural_graph(st_ex, ex_exp, max_nodes=30, max_edges=60)
    assert set(graph) == {"knowledge_dag", "trade_network"}

    dag = graph["knowledge_dag"]
    assert 0 < len(dag["nodes"]) <= 30
    assert len(dag["edges"]) <= 60
    dag_ids = {n["skill_id"] for n in dag["nodes"]}
    for node in dag["nodes"]:
        assert set(node) == {
            "skill_id",
            "parent_skill_id",
            "secondary_parent_id",
            "depth",
            "efficiency",
            "target_resource",
            "ops",
            "inventor_id",
            "inventor_dead",
            "alive_holders",
        }
        assert node["target_resource"] in {"material_a", "material_b", "dual_a_b"}
        assert isinstance(node["ops"], list)
        assert isinstance(node["inventor_dead"], bool)
    for edge in dag["edges"]:
        assert set(edge) == {"source", "target", "kind"}
        assert edge["kind"] in {"refine", "compose"}
        assert edge["source"] in dag_ids
        assert edge["target"] in dag_ids

    tnet = graph["trade_network"]
    assert 0 < len(tnet["nodes"]) <= 30
    assert len(tnet["edges"]) <= 60
    trade_ids = {n["agent_id"] for n in tnet["nodes"]}
    for tnode in tnet["nodes"]:
        assert set(tnode) == {"agent_id", "role", "energy", "inv_a", "inv_b", "trades"}
        assert tnode["role"] in {"A", "B", "dual", "generalist"}
    for tedge in tnet["edges"]:
        assert set(tedge) == {"source", "target", "count", "volume"}
        assert tedge["count"] >= 1
        assert tedge["volume"] > 0.0
        assert tedge["source"] != tedge["target"]
        assert tedge["source"] in trade_ids
        assert tedge["target"] in trade_ids


def test_p6_api_inspect_graph_and_forbid_exchange_endpoint(tmp_path: Path) -> None:
    """FastAPI inspect?view=graph and POST /interventions forbid_exchange work end-to-end."""
    settings = replace(
        load_settings(),
        home=tmp_path / "home",
        runs_root=tmp_path / "runs",
        configs_dir=CONFIGS_DIR,
        frontend_dist=tmp_path / "no_dist",
    )
    tok = "cw-p6-token"
    app = create_app(settings, LocalExecutor(), tok, 8000)
    headers = {"X-ALife-Token": tok, "Origin": "http://127.0.0.1:8000"}

    def wait_paused(client: TestClient, run_id: str) -> dict[str, Any]:
        for _ in range(100):
            info = client.get(f"/api/v1/experiments/{run_id}").json()["data"]
            if info["state"] == "paused":
                return dict(info)
            time.sleep(0.05)
        raise AssertionError(f"run {run_id} did not reach paused state")

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        create_r = client.post(
            "/api/v1/experiments",
            json={
                "preset_id": "cultural_world_exchange",
                "profile_id": "cpu_dev",
                "ticks_per_second": 0,
                "auto_start": True,
                "start_paused": True,
                "overrides": {
                    "runner": {"max_steps": 500, "checkpoint_interval": 100},
                },
            },
            headers={**headers, "Idempotency-Key": "cw-p6-create"},
        )
        assert create_r.status_code == 201, create_r.text
        rid = create_r.json()["data"]["id"]
        wait_paused(client, rid)

        st_r = client.post(f"/api/v1/experiments/{rid}/step", json={"count": 220}, headers=headers)
        assert st_r.status_code == 200, st_r.text

        insp_plain = client.get(f"/api/v1/experiments/{rid}/inspect?x=0&y=0")
        assert insp_plain.status_code == 200, insp_plain.text
        plain_data = insp_plain.json()["data"]
        assert plain_data["exchange_enabled"] is True
        assert "cultural_graph" not in plain_data
        assert plain_data["cultural_agent"]["role"] in {"A", "B", "dual", "generalist"}

        insp_graph = client.get(f"/api/v1/experiments/{rid}/inspect?x=0&y=0&view=graph")
        assert insp_graph.status_code == 200, insp_graph.text
        gdata = insp_graph.json()["data"]
        assert gdata["social_summary"]["trade_count"] > 0
        assert "knowledge_dag" in gdata["cultural_graph"]
        assert "trade_network" in gdata["cultural_graph"]
        assert len(gdata["cultural_graph"]["knowledge_dag"]["nodes"]) > 0
        assert len(gdata["cultural_graph"]["trade_network"]["nodes"]) > 0

        iv_r = client.post(
            f"/api/v1/experiments/{rid}/interventions",
            json={"kind": "forbid_exchange", "payload": {"forbid": True}},
            headers=headers,
        )
        assert iv_r.status_code == 200, iv_r.text

        after_iv = client.get(f"/api/v1/experiments/{rid}/inspect?x=0&y=0").json()["data"]
        assert after_iv["exchange_enabled"] is False
        locked_trades = after_iv["social_summary"]["trade_count"]

        st2 = client.post(f"/api/v1/experiments/{rid}/step", json={"count": 50}, headers=headers)
        assert st2.status_code == 200, st2.text
        after_step = client.get(f"/api/v1/experiments/{rid}/inspect?x=0&y=0").json()["data"]
        assert after_step["exchange_enabled"] is False
        assert after_step["social_summary"]["trade_count"] == locked_trades

        events = client.get(f"/api/v1/experiments/{rid}/events").json()["data"]
        assert any(
            ev["type"] == "intervention_applied"
            and ev.get("payload", {}).get("kind") == "forbid_exchange"
            for ev in events
        )


def test_p6_p7_assays_benchmarks_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """H4 assay, P7 scaling benchmark, and ``alife cultural social-assay/benchmark`` CLIs pass."""
    presets = load_presets(CONFIGS_DIR)
    ex_spec = presets["cultural_world_exchange"].spec

    assay = run_h4_exchange_assay(ex_spec, n_seeds=2, steps=200)
    assert set(assay) == {
        "preset_name",
        "seeds",
        "steps",
        "arms",
        "comparisons",
        "checks",
        "all_checks_passed",
        "wall_seconds",
    }
    assert set(assay["arms"]) == {
        "exchange_intact",
        "no_exchange_control",
        "role_shuffled_control",
        "forbid_exchange_midpoint",
    }
    assert assay["all_checks_passed"] is True, assay["checks"]
    for arm in assay["arms"].values():
        assert arm["max_ledger_residual"] <= 1e-9

    bench = run_cultural_scaling_benchmark(
        populations=(40, 80),
        max_skills_grid=(2, 4),
        steps=25,
        seed=1,
        include_gpu=False,
    )
    assert len(bench["grid"]) == 4
    assert bench["gpu_probe"]["available"] is False
    for row in bench["grid"]:
        assert set(row) == {
            "population",
            "max_skills",
            "steps",
            "wall_seconds",
            "ms_per_tick",
            "agent_ticks_per_sec",
            "peak_rss_mib",
            "max_ledger_residual",
            "hotspots_ms",
        }
        assert row["max_ledger_residual"] <= 1e-9
        assert set(row["hotspots_ms"]) == {
            "ecology",
            "sensors_policy",
            "skills_tech",
            "transmission",
            "exchange",
            "mortality_repro",
        }

    rc1 = cli_main(
        [
            "cultural",
            "social-assay",
            "cultural_world_exchange",
            "--seeds",
            "1",
            "--steps",
            "140",
            "--json",
        ]
    )
    assert rc1 == 0
    cli_assay = json.loads(capsys.readouterr().out)
    assert cli_assay["all_checks_passed"] is True
    assert cli_assay["arms"]["exchange_intact"]["mean_trade_count"] > 0.0

    rc2 = cli_main(
        ["cultural", "benchmark", "--populations", "40", "--skills", "2", "--steps", "20", "--json"]
    )
    assert rc2 == 0
    cli_bench = json.loads(capsys.readouterr().out)
    assert len(cli_bench["grid"]) == 1
    assert cli_bench["grid"][0]["max_ledger_residual"] <= 1e-9
