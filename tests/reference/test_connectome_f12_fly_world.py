"""Phase F12 CPU-only reference & contract tests for ``fly_world`` artificial life."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from alife_lab.cli import main
from alife_lab.connectome.embodied import make_synthetic_fly_embodied_pack
from alife_lab.connectome.fly_world import (
    GENOME_DIM,
    _NumpyCSR,
    replay_evolved_genome_in_full_spiking_brain,
    run_f12_signal_emergence_assay,
    verify_compressed_vs_spiking_fidelity,
)
from alife_lab.core.spec import ExperimentSpec, FlyEmbodiedExperiment, FlyWorldExperiment
from alife_lab.substrates.connectome.embodied_adapter import FlyEmbodiedAdapter
from alife_lab.substrates.connectome.fly_world_adapter import FlyWorldAdapter

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
needs_mujoco = pytest.mark.skipif(
    not HAS_MUJOCO, reason="MuJoCo not installed (uv sync --extra embodied)"
)


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
            "--replay-fine",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert rc == 0, f"CLI evolve failed: {data}"
    assert data["all_checks_passed"] is True
    assert data["fine_replay"] is not None
    assert data["fine_replay"]["phase"] == "F12d_full_brain_replay"
    assert data["fine_replay"]["qualitative_agreement_count"] >= 3


def test_numpy_csr_fallback_and_f12d_full_brain_replay_api(tmp_path: Path) -> None:
    """Verify SciPy-free _NumpyCSR matmul, F12d full-brain replay, and REST replay export."""
    # 1. Pure-NumPy CSR matmul (used in SciPy-free minimal CPU CI)
    mat = _NumpyCSR(
        (3, 3),
        np.array([0, 1, 2], dtype=np.int32),
        np.array([1, 2, 0], dtype=np.int32),
        np.array([2.0, -1.0, 0.5], dtype=np.float32),
    )
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    y = mat @ x
    assert np.allclose(y, [[6.0, 8.0], [-5.0, -6.0], [0.5, 1.0]])

    # 2. Direct full-brain spiking replay of an evolved descendant genome
    pack = make_synthetic_fly_embodied_pack()
    genome = np.zeros(GENOME_DIM, dtype=np.float32)
    genome[0] = 0.25  # enhanced sugar sensitivity
    genome[2] = 0.15  # slightly enhanced antennal touch
    genome[8] = 0.10  # body_scale
    rep = replay_evolved_genome_in_full_spiking_brain(
        pack, genome, backend="numpy", seed=1, ticks=200, fly_id=42, generation=5
    )
    assert rep["all_qualitative_agree"] is True
    assert rep["full_spiking_rates_hz"]["sugar_mn9"] > 20.0
    assert (
        rep["full_spiking_rates_hz"]["sugar_bitter_mn9"]
        < rep["full_spiking_rates_hz"]["sugar_mn9"] * 0.5
    )
    assert rep["full_spiking_rates_hz"]["antennal_poke_dng84"] > 20.0

    # 3. FlyEmbodiedExperiment spec validation with evolved_genome (CPU-only, no MuJoCo)
    spec = ExperimentSpec.model_validate(
        {
            "schema_version": 1,
            "experiment": {
                "name": "f12d_embodied_test",
                "mode": "exploratory",
                "substrate": "fly_embodied",
                "seed": 1,
                "dataset_id": "synthetic_fly_embodied",
                "dynamics": {
                    "evolved_genome": [float(v) for v in genome.tolist()],
                    "evolved_fly_id": 42,
                    "evolved_generation": 5,
                    "evolved_fly_distance": 0.123,
                },
            },
        }
    )
    assert isinstance(spec.experiment, FlyEmbodiedExperiment)
    assert spec.experiment.dynamics.evolved_fly_id == 42
    assert spec.experiment.dynamics.evolved_generation == 5

    # 4. Controller + REST API export_fly_world_replay_spec end-to-end
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from alife_lab.api.app import create_app
    from alife_lab.execution.executors import LocalExecutor
    from alife_lab.service.settings import load_settings

    settings = replace(load_settings(), home=tmp_path / "home", runs_root=tmp_path / "runs")
    tok = "f12d-test-token"
    app = create_app(settings, LocalExecutor(), tok, 8000)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        headers = {"X-ALife-Token": tok, "Origin": "http://127.0.0.1:8000"}
        fw_spec = {
            "schema_version": 1,
            "experiment": {
                "name": "f12_api_source",
                "mode": "exploratory",
                "substrate": "fly_world",
                "seed": 1,
                "dataset_id": "synthetic_fly_embodied",
                "initial_population": 8,
                "max_population": 16,
                "runner": {"max_steps": 15, "checkpoint_interval": 10},
            },
        }
        r = client.post(
            "/api/v1/experiments",
            json={
                "spec": fw_spec,
                "profile_id": "cpu_dev",
                "ticks_per_second": 0,
                "auto_start": True,
            },
            headers={**headers, "Idempotency-Key": "f12d-test-create"},
        )
        assert r.status_code == 201, r.text
        run_id = r.json()["data"]["id"]

        # Wait briefly for completion (15 ticks on synthetic takes < 0.3s)
        import time

        for _ in range(60):
            st_r = client.get(f"/api/v1/experiments/{run_id}").json()["data"]
            if st_r["state"] in ("completed", "failed"):
                break
            time.sleep(0.1)
        assert st_r["state"] == "completed", st_r

        exp_r = client.post(
            f"/api/v1/experiments/{run_id}/fly-world/replay",
            json={},
            headers=headers,
        )
        assert exp_r.status_code == 200, exp_r.text
        body = exp_r.json()["data"]
        assert body["spec"]["experiment"]["substrate"] == "fly_embodied"
        assert len(body["spec"]["experiment"]["dynamics"]["evolved_genome"]) == GENOME_DIM
        assert body["descendant"]["dataset_id"] == "synthetic_fly_embodied"


@needs_mujoco
def test_f12d_fly_embodied_adapter_replay_mujoco() -> None:
    """Verify F12d evolved_genome loads into FlyEmbodiedAdapter and renders descendant metadata."""
    genome = np.zeros(GENOME_DIM, dtype=np.float32)
    genome[0] = 0.25
    genome[2] = 0.15
    genome[8] = 0.10
    spec = ExperimentSpec.model_validate(
        {
            "schema_version": 1,
            "experiment": {
                "name": "f12d_embodied_test",
                "mode": "exploratory",
                "substrate": "fly_embodied",
                "seed": 1,
                "dataset_id": "synthetic_fly_embodied",
                "dynamics": {
                    "evolved_genome": [float(v) for v in genome.tolist()],
                    "evolved_fly_id": 42,
                    "evolved_generation": 5,
                    "evolved_fly_distance": 0.123,
                },
            },
        }
    )
    assert isinstance(spec.experiment, FlyEmbodiedExperiment)
    adapter = FlyEmbodiedAdapter()
    adapter.set_backend("numpy")
    w = adapter.initialize(spec)
    adapter.step(w, spec)
    hdr, _ = adapter.render(w)
    assert hdr["evolved_descendant"] is not None
    assert hdr["evolved_descendant"]["fly_id"] == 42
    assert hdr["evolved_descendant"]["generation"] == 5
    assert "F12d Full-Brain" in hdr["level_note"]
