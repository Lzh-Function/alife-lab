"""Stage 4 / S2-S4 species packs, cross-species comparison, and F13a-c chimera tests."""

from __future__ import annotations

import json
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from alife_lab.api.app import create_app
from alife_lab.cli import main as cli_main
from alife_lab.connectome.celegans import load_celegans_302, make_synthetic_celegans_302_pack
from alife_lab.connectome.chimera import (
    make_chimera_worm_fly_mb_pack,
    run_f13a_intraspecies_hybrid_assay,
    run_f13b_brain_transplant_experiment,
    run_f13c_mb_graft_assay,
)
from alife_lab.connectome.cross_species import (
    analyze_connectome_topology,
    compare_species_connectomes,
)
from alife_lab.connectome.registry import DatasetError, raw_dir, verify_dataset
from alife_lab.connectome.schema import (
    MODEL_GRADED,
    MODEL_SPIKING_LIF,
    SYNAPSE_GAP_JUNCTION,
)
from alife_lab.connectome.species_packs import (
    _safe_read_zip_csv,
    ciona_larva_reference_checks,
    fly_larva_l1_reference_checks,
    load_ciona_larva,
    load_fly_larva_l1,
    load_platynereis_3d,
    make_synthetic_ciona_pack,
    make_synthetic_fly_larva_pack,
    make_synthetic_platynereis_pack,
    platynereis_3d_reference_checks,
)
from alife_lab.core.spec import ConnectomeExperiment, ExperimentSpec
from alife_lab.execution.executors import FakeExecutor
from alife_lab.service.catalog import load_presets
from alife_lab.service.settings import load_settings
from alife_lab.substrates import get_adapter

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"


def _has_local(dataset_id: str) -> bool:
    ds_dir = load_settings().resolved_datasets_dir
    d = raw_dir(ds_dir, dataset_id)
    if not d.exists():
        return False
    try:
        rep = verify_dataset(dataset_id, ds_dir)
    except Exception:
        return False
    return bool(rep.get("ok"))


def test_synthetic_s2_s4_species_packs_pass_reference_checks() -> None:
    larva = make_synthetic_fly_larva_pack()
    assert all(c["ok"] for c in fly_larva_l1_reference_checks(larva))
    assert np.all(larva.cells.cell_model == MODEL_SPIKING_LIF)
    assert "larva_olfactory_sn" in larva.io_ports
    assert "larva_kc" in larva.io_ports

    platy = make_synthetic_platynereis_pack()
    assert all(c["ok"] for c in platynereis_3d_reference_checks(platy))
    assert np.all(platy.cells.cell_model == MODEL_GRADED)
    assert int((platy.edges.kind == SYNAPSE_GAP_JUNCTION).sum()) > 0
    assert "platy_photoreceptor_prc" in platy.io_ports

    ciona = make_synthetic_ciona_pack()
    assert all(c["ok"] for c in ciona_larva_reference_checks(ciona))
    assert np.all(ciona.cells.cell_model == MODEL_SPIKING_LIF)
    assert "ciona_photoreceptors" in ciona.io_ports
    assert "ciona_motor_mn" in ciona.io_ports


def test_zip_reader_rejects_path_traversal(tmp_path: Path) -> None:
    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.csv", "a,b\n1,2\n")
    with pytest.raises(DatasetError, match="unsafe member path"):
        _safe_read_zip_csv(bad_zip, "escape.csv")


def test_chimera_rejects_reference_mode() -> None:
    with pytest.raises(ValidationError, match="exploratory"):
        ConnectomeExperiment(
            name="bad_chimera_real",
            substrate="connectome",
            seed=1,
            dataset_id="chimera_worm_fly_mb",
            mode="reference",
        )
    with pytest.raises(ValidationError, match="exploratory"):
        ConnectomeExperiment(
            name="bad_chimera_synth",
            substrate="connectome",
            seed=1,
            dataset_id="synthetic_chimera_worm_fly_mb",
            mode="reference",
        )
    ok = ConnectomeExperiment(
        name="ok_chimera_synth",
        substrate="connectome",
        seed=1,
        dataset_id="synthetic_chimera_worm_fly_mb",
        mode="exploratory",
    )
    assert ok.mode == "exploratory"


@pytest.mark.parametrize(
    ("dataset_id", "mode", "expected_species_count"),
    [
        ("synthetic_fly_larva", "reference", 1),
        ("synthetic_platynereis", "reference", 1),
        ("synthetic_ciona", "reference", 1),
        ("synthetic_chimera_worm_fly_mb", "exploratory", 2),
    ],
)
def test_connectome_adapter_s2_s4_and_hybrid_chimera_checkpoint_roundtrip(
    dataset_id: str,
    mode: str,
    expected_species_count: int,
) -> None:
    adapter = get_adapter("connectome")
    spec = ExperimentSpec(
        experiment=ConnectomeExperiment(
            name=f"test_{dataset_id}",
            substrate="connectome",
            dataset_id=dataset_id,
            mode=mode,
            seed=7,
        ),
    )
    st = adapter.initialize(spec)
    for _ in range(5):
        st = adapter.step(st, spec)

    info = adapter.inspect(st, spec, {"cell_index": 0})
    assert info["dataset_id"] == dataset_id
    assert isinstance(info["species_name"], str) and len(info["species_name"]) > 0
    assert len(info["species_provenance"]) == expected_species_count
    assert abs(sum(info["species_provenance"].values()) - 1.0) < 0.01

    mets = adapter.metrics(st, spec)
    assert "mean_firing_rate_hz" in mets and np.isfinite(mets["mean_firing_rate_hz"])

    hdr, payload = adapter.render(st)
    assert hdr["codec"] == "connectome_v1"
    assert isinstance(payload, bytes)

    ckpt = adapter.serialize_state(st)
    restored = adapter.deserialize_state(ckpt, spec)
    dist = adapter.state_distance(st, restored)
    assert dist["exact_equal"] is True

    # Stepping both from checkpoint stays bitwise identical
    st_next = adapter.step(st, spec)
    restored_next = adapter.step(restored, spec)
    assert adapter.state_distance(st_next, restored_next)["exact_equal"] is True


def test_cross_species_comparison_offline() -> None:
    packs = {
        "synthetic_ciona": make_synthetic_ciona_pack(),
        "synthetic_celegans_302": make_synthetic_celegans_302_pack(),
        "synthetic_platynereis": make_synthetic_platynereis_pack(),
        "synthetic_fly_larva": make_synthetic_fly_larva_pack(),
    }
    report = compare_species_connectomes(packs, seed=1, ticks=40, run_physiology=True)
    assert len(report["datasets"]) == 4
    assert len(report["topology"]) == 4
    assert len(report["sensorimotor_probe"]) == 4
    for topo in report["topology"]:
        assert topo["num_cells"] > 0
        assert topo["num_edges"] > 0
        assert 0.0 <= topo["reciprocity"] <= 1.0
    for phys in report["sensorimotor_probe"]:
        assert "reference_motor_response" in phys
        assert "shuffled_motor_response" in phys


def test_f13a_intraspecies_hybrid_offline() -> None:
    a = make_synthetic_celegans_302_pack()
    b = make_synthetic_celegans_302_pack()
    rep = run_f13a_intraspecies_hybrid_assay(a, b, blend_ratio=0.5, seed=3, ticks=30)
    assert rep["chimera_kind"] == "F13a_intraspecies_hybrid"
    assert "NOT claims about natural biological animals" in rep["framing_note"]
    assert rep["shared_cell_types"] == 302
    assert len(rep["provenance_breakdown"]) == 2
    assert "reference_motor_response" in rep["hybrid_probe"]
    assert "shuffled_motor_response" in rep["hybrid_probe"]


@pytest.mark.parametrize("kind", ["ciona", "platynereis"])
def test_f13b_brain_transplant_offline(kind: str) -> None:
    donor = make_synthetic_ciona_pack() if kind == "ciona" else make_synthetic_platynereis_pack()
    rep = run_f13b_brain_transplant_experiment(donor, seed=1, bridge_trials=4, eval_steps=40)
    assert rep["chimera_kind"] == "F13b_brain_transplant"
    assert rep["intact_beats_silenced"] is True
    assert rep["arms"]["donor_silenced"]["distance_mm"] == 0.0
    assert rep["arms"]["donor_intact"]["distance_mm"] > 0.05
    prov = rep["provenance_breakdown"]
    assert abs(sum(prov.values()) - 1.0) < 0.01


def test_f13c_mb_graft_chimera_offline_passes_all_controls() -> None:
    pack = make_chimera_worm_fly_mb_pack(None, None, seed=1)
    assert len(pack.cells.species_names) == 2
    assert np.any(pack.cells.cell_model == MODEL_GRADED)
    assert np.any(pack.cells.cell_model == MODEL_SPIKING_LIF)
    topo = analyze_connectome_topology(pack)
    assert topo["num_cells"] > 302

    rep = run_f13c_mb_graft_assay(None, None, seed=1, train_trials=4)
    assert rep["chimera_kind"] == "F13c_mb_circuit_graft"
    assert rep["all_checks_passed"] is True, rep["checks"]
    assert rep["arms"]["paired_intact"]["delta_avoid_odor_a_mv"] > 1.0
    assert rep["arms"]["graft_silenced"]["delta_avoid_odor_a_mv"] == 0.0
    assert rep["arms"]["unpaired_control"]["delta_avoid_odor_a_mv"] < 0.5


def test_stage4_cli_subcommands_offline(capsys: pytest.CaptureFixture[str]) -> None:
    rc1 = cli_main(
        [
            "connectome",
            "compare-species",
            "--datasets",
            "synthetic_ciona,synthetic_celegans_302,synthetic_platynereis,synthetic_fly_larva",
            "--ticks",
            "20",
            "--json",
        ]
    )
    assert rc1 == 0
    out1 = json.loads(capsys.readouterr().out)
    assert len(out1["datasets"]) == 4

    rc2 = cli_main(
        [
            "connectome",
            "chimera",
            "--kind",
            "f13c",
            "--host",
            "synthetic_celegans_302",
            "--donor",
            "synthetic_fly_larva",
            "--trials",
            "3",
            "--json",
        ]
    )
    assert rc2 == 0
    out2 = json.loads(capsys.readouterr().out)
    assert out2["all_checks_passed"] is True


def test_stage4_presets_and_api_chimera_inspect(tmp_path: Path) -> None:
    presets = load_presets(CONFIGS_DIR)
    expected_presets = {
        "connectome_fly_larva_l1",
        "connectome_fly_larva_l1_shuffled",
        "connectome_platynereis_3d",
        "connectome_platynereis_3d_shuffled",
        "connectome_ciona_larva",
        "connectome_ciona_larva_shuffled",
        "connectome_chimera_worm_fly_mb",
        "connectome_chimera_worm_fly_mb_shuffled",
    }
    assert expected_presets.issubset(presets.keys())
    assert presets["connectome_chimera_worm_fly_mb"].spec.experiment.mode == "exploratory"

    settings = replace(
        load_settings(),
        runs_root=tmp_path / "runs",
        home=tmp_path / "home",
        configs_dir=CONFIGS_DIR,
        frontend_dist=tmp_path / "no_dist",
    )
    token = "stage4-token"
    app = create_app(settings, FakeExecutor(), token, port=8000, extra_hosts=("testserver",))
    with TestClient(app) as client:
        h = {"X-ALife-Token": token}
        synth_spec = ExperimentSpec(
            experiment=ConnectomeExperiment(
                name="api_chimera_test",
                substrate="connectome",
                dataset_id="synthetic_chimera_worm_fly_mb",
                mode="exploratory",
                seed=1,
            )
        ).model_dump(mode="json")
        r = client.post(
            "/api/v1/experiments",
            json={
                "spec": synth_spec,
                "profile_id": "cpu_dev",
                "ticks_per_second": 0,
                "auto_start": True,
                "start_paused": True,
            },
            headers=h,
        )
        assert r.status_code == 201, r.text
        rid = r.json()["data"]["id"]
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if client.get(f"/api/v1/experiments/{rid}").json()["data"]["state"] == "paused":
                break
            time.sleep(0.02)
        st_resp = client.post(f"/api/v1/experiments/{rid}/step", json={"count": 2}, headers=h)
        assert st_resp.status_code == 200, st_resp.text
        insp = client.get(f"/api/v1/experiments/{rid}/inspect?cell_index=0")
        assert insp.status_code == 200, insp.text
        payload = insp.json()["data"]
        assert payload["dataset_id"] == "synthetic_chimera_worm_fly_mb"
        assert len(payload["species_provenance"]) == 2
        client.post(f"/api/v1/experiments/{rid}/cancel", headers=h)


@pytest.mark.skipif(
    not (_has_local("fly_larva_l1") and _has_local("platynereis_3d") and _has_local("ciona_larva")),
    reason="Empirical S2-S4 connectome datasets not downloaded locally",
)
def test_real_s2_s4_datasets_and_f13c_graft() -> None:
    ds_dir = load_settings().resolved_datasets_dir

    larva = load_fly_larva_l1(ds_dir, verify_hashes=True)
    assert all(c["ok"] for c in fly_larva_l1_reference_checks(larva))

    platy = load_platynereis_3d(ds_dir, verify_hashes=True)
    assert all(c["ok"] for c in platynereis_3d_reference_checks(platy))

    ciona = load_ciona_larva(ds_dir, verify_hashes=True)
    assert all(c["ok"] for c in ciona_larva_reference_checks(ciona))

    if _has_local("celegans_302"):
        worm = load_celegans_302(ds_dir, verify_hashes=True)
        rep = run_f13c_mb_graft_assay(worm, larva, seed=1, train_trials=4)
        assert rep["all_checks_passed"] is True, rep["checks"]
