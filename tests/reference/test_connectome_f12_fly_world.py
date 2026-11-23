"""Phase F12 CPU-only reference & contract tests for ``fly_world`` artificial life."""

from __future__ import annotations

import json

import pytest

from alife_lab.cli import main
from alife_lab.connectome.embodied import make_synthetic_fly_embodied_pack
from alife_lab.connectome.fly_world import (
    run_f12_signal_emergence_assay,
    verify_compressed_vs_spiking_fidelity,
)
from alife_lab.core.spec import ExperimentSpec, FlyWorldExperiment
from alife_lab.substrates.connectome.fly_world_adapter import FlyWorldAdapter


def test_fly_world_spec_validation() -> None:
    spec = ExperimentSpec.model_validate(
        {
            "schema_version": 1,
            "experiment": {
                "name": "fw_test",
                "mode": "exploratory",
                "substrate": "fly_world",
                "seed": 5,
                "dataset_id": "synthetic_fly_embodied",
                "initial_population": 16,
                "max_population": 32,
            },
        }
    )
    assert isinstance(spec.experiment, FlyWorldExperiment)
    assert spec.experiment.dynamics.brain_resolution == "cell_type_compressed"

    with pytest.raises(ValueError, match="exploratory"):
        ExperimentSpec.model_validate(
            {
                "schema_version": 1,
                "experiment": {
                    "name": "bad_ref",
                    "mode": "reference",
                    "substrate": "fly_world",
                    "seed": 1,
                },
            }
        )


def test_compressed_brain_fidelity_cpu() -> None:
    pack = make_synthetic_fly_embodied_pack()
    fid = verify_compressed_vs_spiking_fidelity(pack)
    assert fid["all_checks_passed"], f"Compressed fidelity failed: {fid}"
    assert fid["sugar_mn9_hz"] > 20.0
    assert fid["sugar_bitter_mn9_hz"] < fid["sugar_mn9_hz"] * 0.25
    assert fid["antennal_poke_dng84_hz"] > 20.0


def test_fly_world_adapter_interventions_render_and_exact_checkpoint() -> None:
    spec = ExperimentSpec.model_validate(
        {
            "schema_version": 1,
            "experiment": {
                "name": "fw_ckpt",
                "mode": "exploratory",
                "substrate": "fly_world",
                "seed": 2,
                "dataset_id": "synthetic_fly_embodied",
                "initial_population": 16,
                "max_population": 32,
            },
        }
    )
    adapter = FlyWorldAdapter()
    adapter.set_backend("numpy")
    st = adapter.initialize(spec)
    for _ in range(25):
        st = adapter.step(st, spec)
    assert st.tick == 25

    hdr, payload = adapter.render(st)
    assert hdr["codec"] == "fly_world_v1"
    assert hdr["num_flies"] == st.population
    assert len(payload) == st.population * 12 * 4
    assert len(hdr["signal_dictionary"]) == 2

    st, r1 = adapter.apply_intervention(st, spec, "toggle_signals", {})
    assert r1["signal_channel_enabled"] is False
    st, r2 = adapter.apply_intervention(st, spec, "spawn_predator", {})
    assert r2["predator_active"] is True
    st, r3 = adapter.apply_intervention(st, spec, "add_food", {})
    assert r3["food_count"] >= 4
    st, r4 = adapter.apply_intervention(st, spec, "toggle_mutations", {})
    assert r4["mutation_rate"] == 0.0

    info = adapter.inspect(st, spec, {"fly_index": 0})
    assert "fly_distance" in info and "morphology" in info

    blob = adapter.serialize_state(st)
    st2 = adapter.deserialize_state(blob, spec)
    assert adapter.state_distance(st, st2)["exact_equal"] is True
    for _ in range(15):
        st = adapter.step(st, spec)
        st2 = adapter.step(st2, spec)
    assert adapter.state_distance(st, st2)["exact_equal"] is True


def test_f12_signal_emergence_assay_and_cli_cpu(capsys: pytest.CaptureFixture[str]) -> None:
    pack = make_synthetic_fly_embodied_pack()
    rep = run_f12_signal_emergence_assay(
        pack,
        seeds=(1, 2),
        evo_ticks=600,
        eval_ticks=250,
        backend="numpy",
    )
    assert rep["all_checks_passed"], f"F12 emergence assay failed: {rep}"

    rc = main(
        [
            "connectome",
            "evolve",
            "synthetic_fly_embodied",
            "--seeds",
            "1",
            "--evo-ticks",
            "450",
            "--eval-ticks",
            "200",
            "--backend",
            "numpy",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert rc == 0, f"CLI evolve failed: {data}"
    assert data["all_checks_passed"] is True
