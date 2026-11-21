"""Phase F3 reference & integration tests: systematic cell-type silencing screen and
connectome sensory input search (CMA-ES, MAP-Elites, Random, shuffled control).

Runs fully offline on CPU-only CI using ``synthetic_tiny``; when CuPy + GPU are available,
also asserts bitwise CPU == GPU equivalence for both the silencing screen and batched search.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alife_lab.cli import main as cli_main
from alife_lab.connectome.ablation import run_silencing_screen
from alife_lab.connectome.lif import LIFParams, make_synthetic_tiny_pack
from alife_lab.core.spec import ExperimentSpec, load_experiment_spec
from alife_lab.core.sweep import SweepSpec
from alife_lab.execution.executors import FakeExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from alife_lab.substrates.connectome import ConnectomeAdapter
from alife_lab.substrates.search.adapter import SearchAdapter
from tests.integration.test_controller import arun, wait_for

ROOT = Path(__file__).resolve().parents[2]


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def test_silenced_cell_types_in_spec_and_hash_invariance() -> None:
    base_raw = {
        "experiment": {
            "name": "tiny_hash_check",
            "mode": "exploratory",
            "substrate": "connectome",
            "seed": 1,
            "dataset_id": "synthetic_tiny",
            "wiring": "reference",
            "stimuli": [{"port": "sugar_grn", "rate_hz": 150.0}],
            "runner": {"max_steps": 100, "checkpoint_interval": 50},
        }
    }
    s_default = ExperimentSpec.model_validate(base_raw)
    base_explicit = {
        "experiment": {**base_raw["experiment"], "silenced_cell_types": []},
    }
    s_empty = ExperimentSpec.model_validate(base_explicit)
    assert s_default.digest() == s_empty.digest()
    assert "silenced_cell_types" not in s_default.model_dump(mode="json")["experiment"]

    # Silencing IN1 must abolish MN9 firing completely.
    s_silenced = ExperimentSpec.model_validate(
        {"experiment": {**base_raw["experiment"], "silenced_cell_types": ["IN1"]}}
    )
    assert s_silenced.digest() != s_default.digest()

    adapter = ConnectomeAdapter()
    w_intact = adapter.initialize(s_default)
    w_sil = adapter.initialize(s_silenced)
    for _ in range(100):
        adapter.step(w_intact, s_default)
        adapter.step(w_sil, s_silenced)
    assert adapter.metrics(w_intact, s_default)["mn9_rate_hz"] > 10.0
    assert adapter.metrics(w_sil, s_silenced)["mn9_rate_hz"] == 0.0

    # Unknown cell_type raises ValueError rather than silently doing nothing.
    s_bad = ExperimentSpec.model_validate(
        {"experiment": {**base_raw["experiment"], "silenced_cell_types": ["NO_SUCH_TYPE"]}}
    )
    with pytest.raises(ValueError, match="unknown silenced cell_type"):
        adapter.initialize(s_bad)


def test_silencing_screen_synthetic_tiny_reproducible_and_gpu_bitwise() -> None:
    pack = make_synthetic_tiny_pack()
    csr = pack.edges.to_csr(pack.cells.num_cells)
    params = LIFParams()

    r1 = run_silencing_screen(
        pack,
        csr,
        params,
        stimuli=[("sugar_grn", 150.0)],
        cell_types=["IN1", "LB1a"],
        target_port="mn9_proboscis_extension",
        seeds=(1, 2, 3, 4, 5),
        ticks=200,
        backend="numpy",
    )
    r2 = run_silencing_screen(
        pack,
        csr,
        params,
        stimuli=[("sugar_grn", 150.0)],
        cell_types=["IN1", "LB1a"],
        target_port="mn9_proboscis_extension",
        seeds=(1, 2, 3, 4, 5),
        ticks=200,
        backend="numpy",
    )
    assert r1["baseline"] == r2["baseline"]
    assert r1["results"] == r2["results"]
    assert r1["baseline"]["mean_hz"] > 20.0

    by_type = {row["cell_type"]: row for row in r1["results"]}
    # Silencing the relay interneuron must cause 100% suppression of MN9 with p < 0.05.
    assert by_type["IN1"]["mean_hz"] == 0.0
    assert by_type["IN1"]["suppression_pct"] == 100.0
    assert by_type["IN1"]["p_value"] is not None
    assert by_type["IN1"]["p_value"] < 0.05
    # Silencing the inactive bitter neuron must leave MN9 completely unchanged.
    assert by_type["LB1a"]["delta_hz"] == 0.0

    if _has_cupy():
        rgpu = run_silencing_screen(
            pack,
            csr,
            params,
            stimuli=[("sugar_grn", 150.0)],
            cell_types=["IN1", "LB1a"],
            target_port="mn9_proboscis_extension",
            seeds=(1, 2, 3, 4, 5),
            ticks=200,
            backend="cupy",
            batch_size=4,
        )
        assert rgpu["baseline"] == r1["baseline"]
        assert rgpu["results"] == r1["results"]


def test_connectome_sweep_silencing_via_controller(tmp_path: Path) -> None:
    import yaml

    settings = Settings(
        project_root=ROOT,
        home=tmp_path / "home",
        runs_root=tmp_path / "runs",
        configs_dir=ROOT / "configs",
        frontend_dist=ROOT / "frontend" / "dist",
        max_active_jobs=4,
        default_ticks_per_second=20.0,
    )
    raw = yaml.safe_load((ROOT / "configs/sweeps/connectome_silencing_tiny.yaml").read_text())
    # Shrink ticks for fast CI test while preserving all 4 conditions x 5 seeds.
    raw["sweep"]["overrides"]["runner"]["max_steps"] = 100
    raw["sweep"]["overrides"]["runner"]["checkpoint_interval"] = 100
    sweep_spec = SweepSpec.model_validate(raw)

    async def main() -> None:
        ctrl = Controller(settings, FakeExecutor())
        await ctrl.start()
        try:
            summary = await ctrl.sweeps.create(sweep_spec, author="test")
            sid = summary["id"]
            await wait_for(lambda: ctrl.sweeps.summary(sid)["state"] != "running", 30)
            assert ctrl.sweeps.summary(sid)["state"] == "completed"
            rep = ctrl.sweeps.report(sid)
            conds = {c["label"]: c["summary"]["mean"] for c in rep["conditions"]}
            assert conds["intact"] is not None and conds["intact"] > 10.0
            assert conds["silence_IN1"] == 0.0
            assert conds["silence_LB3c"] == 0.0
            assert conds["silence_LB1a"] == pytest.approx(conds["intact"])
        finally:
            await ctrl.shutdown()

    arun(main)


def test_connectome_input_search_reproducible_and_finds_sugar_over_bitter() -> None:
    spec = load_experiment_spec(ROOT / "configs/scenarios/search_connectome_tiny_cma.yaml")
    # Only stimuli.<idx>.rate_hz is searchable; attempting to search wiring or LIF params fails.
    bad = spec.model_dump(mode="json")
    bad["experiment"]["space"] = [{"path": "dynamics.w_syn_mv", "low": 0.1, "high": 1.0}]
    with pytest.raises(Exception, match="not searchable"):
        ExperimentSpec.model_validate(bad)

    adapter_a = SearchAdapter()
    adapter_a.set_backend("numpy")
    st_a = adapter_a.initialize(spec)
    for _ in range(4):
        adapter_a.step(st_a, spec)

    # Same spec + same seed must be bitwise reproducible.
    adapter_b = SearchAdapter()
    adapter_b.set_backend("numpy")
    st_b = adapter_b.initialize(spec)
    for _ in range(4):
        adapter_b.step(st_b, spec)
    assert np.array_equal(st_a.score, st_b.score)
    assert np.array_equal(st_a.u, st_b.u)

    # Best discovered stimulus must drive strong MN9 firing with high sugar & lower bitter.
    best_i = st_a.best
    assert best_i is not None
    assert st_a.score[best_i] > 30.0
    sugar_hz, bitter_hz = float(st_a.x[best_i, 0]), float(st_a.x[best_i, 1])
    assert sugar_hz > bitter_hz

    # Bitwise CPU == GPU match on batched search when CuPy is installed.
    if _has_cupy():
        adapter_gpu = SearchAdapter()
        adapter_gpu.set_backend("cupy")
        st_gpu = adapter_gpu.initialize(spec)
        for _ in range(4):
            adapter_gpu.step(st_gpu, spec)
        assert np.array_equal(st_gpu.score, st_a.score)
        assert np.array_equal(st_gpu.u, st_a.u)


def test_cli_connectome_screen_synthetic_tiny(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(
        [
            "connectome",
            "screen",
            "synthetic_tiny",
            "--stim",
            "sugar_grn:150",
            "--seeds",
            "1,2,3",
            "--ticks",
            "150",
            "--backend",
            "numpy",
            "--json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["dataset_id"] == "synthetic_tiny"
    assert data["baseline"]["mean_hz"] > 10.0
    assert any(r["cell_type"] == "IN1" and r["mean_hz"] == 0.0 for r in data["results"])
