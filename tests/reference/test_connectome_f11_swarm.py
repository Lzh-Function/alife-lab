"""Phase F11 Stage 3 CPU-only reference & contract tests for collective fly swarm."""

from __future__ import annotations

import json

import pytest

from alife_lab.cli import main
from alife_lab.connectome.embodied import make_synthetic_fly_embodied_pack
from alife_lab.connectome.fly_swarm import (
    make_synthetic_male_aggression_pack,
    run_male_aggression_assay,
    run_ramdya_collective_assay,
)
from alife_lab.core.spec import ExperimentSpec, FlySwarmExperiment
from alife_lab.substrates.connectome.swarm_adapter import FlySwarmAdapter


def test_fly_swarm_spec_validation() -> None:
    spec = ExperimentSpec.model_validate(
        {
            "schema_version": 1,
            "experiment": {
                "name": "swarm_test",
                "mode": "exploratory",
                "substrate": "fly_swarm",
                "seed": 7,
                "dataset_id": "synthetic_fly_embodied",
                "num_flies": 12,
                "wiring": "reference",
            },
        }
    )
    assert isinstance(spec.experiment, FlySwarmExperiment)
    assert spec.experiment.num_flies == 12
    assert spec.experiment.dynamics.enable_touch is True
    assert spec.experiment.dynamics.odor_enabled is True

    with pytest.raises(ValueError, match="exploratory"):
        ExperimentSpec.model_validate(
            {
                "schema_version": 1,
                "experiment": {
                    "name": "bad_ref",
                    "mode": "reference",
                    "substrate": "fly_swarm",
                    "seed": 1,
                },
            }
        )


def test_ramdya_collective_assay_cpu_all_checks_pass() -> None:
    pack = make_synthetic_fly_embodied_pack()
    rep = run_ramdya_collective_assay(
        pack,
        group_sizes=(1, 8, 16),
        ticks=2000,
        seed=1,
        backend="numpy",
    )
    assert rep["dataset_id"] == "synthetic_fly_embodied"
    assert rep["max_concurrent_neurons"] == pack.cells.num_cells * 16
    assert rep["all_checks_passed"], f"Ramdya checks failed: {rep}"
    assert rep["checks"]["group_exceeds_lone_avoidance"]
    assert rep["checks"]["touch_silencing_reduces_avoidance"]
    assert rep["checks"]["shuffled_wiring_reduces_avoidance"]
    assert rep["checks"]["no_odor_near_chance"]
    assert rep["controls"]["touch_silenced"]["encounters_per_fly_s"] == 0.0


def test_fly_swarm_adapter_interventions_render_and_exact_checkpoint() -> None:
    spec = ExperimentSpec.model_validate(
        {
            "schema_version": 1,
            "experiment": {
                "name": "adapter_swarm",
                "mode": "exploratory",
                "substrate": "fly_swarm",
                "seed": 3,
                "dataset_id": "synthetic_fly_embodied",
                "num_flies": 8,
                "dynamics": {
                    "social_mode": "mixed_courtship_swarm",
                    "enable_touch": True,
                    "odor_enabled": True,
                },
            },
        }
    )
    adapter = FlySwarmAdapter()
    adapter.set_backend("numpy")
    state = adapter.initialize(spec)
    for _ in range(15):
        state = adapter.step(state, spec)
    assert state.tick == 15

    # Render binary frame header + exact 8 * 8 float32 payload
    header, payload = adapter.render(state)
    assert header["codec"] == "fly_swarm_v1"
    assert header["num_flies"] == 8
    assert len(payload) == 8 * 8 * 4

    # Live interventions toggle cleanly
    state, r1 = adapter.apply_intervention(state, spec, "toggle_odor", {})
    assert r1["odor_enabled"] is False
    state, r2 = adapter.apply_intervention(state, spec, "toggle_touch", {})
    assert r2["enable_touch"] is False
    state, r3 = adapter.apply_intervention(state, spec, "wake_all", {})
    assert r3["woke"] == 8
    state, r4 = adapter.apply_intervention(state, spec, "scatter", {})
    assert r4["scattered"] == 8

    # Inspect returns fly diagnostics
    info = adapter.inspect(state, spec, {"fly_index": 0})
    assert info["fly_index"] == 0
    assert "pose" in info

    # Exact checkpoint round-trip + bitwise deterministic continuation
    blob = adapter.serialize_state(state)
    state2 = adapter.deserialize_state(blob, spec)
    dist0 = adapter.state_distance(state, state2)
    assert dist0["exact_equal"] is True

    for _ in range(10):
        state = adapter.step(state, spec)
        state2 = adapter.step(state2, spec)
    dist1 = adapter.state_distance(state, state2)
    assert dist1["exact_equal"] is True


def test_cli_connectome_swarm_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "connectome",
            "swarm",
            "synthetic_fly_embodied",
            "--sizes",
            "1,16",
            "--ticks",
            "1800",
            "--backend",
            "numpy",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert rc == 0, f"CLI swarm failed: {data}"
    assert data["all_checks_passed"] is True
    assert len(data["group_size_curve"]) == 2


def test_male_aggression_assay_cpu_all_checks_pass() -> None:
    pack = make_synthetic_male_aggression_pack()
    rep = run_male_aggression_assay(
        pack,
        num_flies=8,
        ticks=1200,
        seed=1,
        backend="numpy",
    )
    assert rep["mode"] == "male_aggression_arena"
    assert rep["all_checks_passed"], f"Male aggression checks failed: {rep}"
    assert rep["conditions"]["intact_males"]["total_lunges"] >= 4
    assert rep["conditions"]["cva_blocked"]["mean_da1_pn_hz"] == 0.0


def test_cli_connectome_swarm_aggression_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "connectome",
            "swarm",
            "synthetic_male_aggression",
            "--mode",
            "aggression",
            "--sizes",
            "8",
            "--ticks",
            "1000",
            "--backend",
            "numpy",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert rc == 0, f"CLI aggression failed: {data}"
    assert data["all_checks_passed"] is True
    assert data["mode"] == "male_aggression_arena"
