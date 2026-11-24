"""Stage 2 / S1 C. elegans 302-neuron graded + gap-junction & RFT worm verification tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from alife_lab.cli import main as cli_main
from alife_lab.connectome.celegans import (
    celegans_302_reference_checks,
    load_celegans_302,
    load_celegans_from_edgelist_csv,
    make_synthetic_celegans_302_pack,
    normalize_celegans_name,
)
from alife_lab.connectome.graded import (
    GradedCircuit,
    GradedParams,
    make_initial_graded_state,
    step_graded_substep_cpu,
    step_graded_tick_cpu,
    step_hybrid_tick_cpu,
)
from alife_lab.connectome.lif import LIFParams, make_initial_state
from alife_lab.connectome.registry import verify_dataset
from alife_lab.connectome.schema import (
    MODEL_GRADED,
    MODEL_SPIKING_LIF,
    POS_SOMA,
    ROLE_INTERNEURON,
    ROLE_MOTOR,
    ROLE_SENSORY,
    SYNAPSE_CHEMICAL,
    SYNAPSE_GAP_JUNCTION,
    CellTable,
    ConnectomePack,
    DatasetProvenance,
    EdgeTable,
    compute_pack_digest,
)
from alife_lab.connectome.worm_activity import (
    CELEGANS_ACTIVITY_DIR_NAME,
    CELEGANS_ACTIVITY_FILENAME,
    load_celegans_activity_recordings,
    run_worm_activity_benchmark,
)
from alife_lab.connectome.worm_body import (
    BODY_LENGTH_MM,
    N_JOINTS,
    make_worm_world,
    step_rft_kinematics,
    step_worm_tick,
)
from alife_lab.core.spec import ExperimentSpec, WormEmbodiedExperiment
from alife_lab.service.catalog import load_presets
from alife_lab.substrates import get_adapter

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def _simple_graded_pack(edges: EdgeTable, models: np.ndarray | None = None) -> ConnectomePack:
    n = 3
    m_arr = models if models is not None else np.full(n, MODEL_GRADED, dtype=np.uint8)
    cells = CellTable(
        source_id=np.array([1, 2, 3], dtype=np.int64),
        species_names=("test_graded",),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=("N0", "N1", "N2"),
        super_class=("sensory", "central", "motor"),
        cell_class=("S", "I", "M"),
        cell_sub_class=("", "", ""),
        region=("head", "head", "head"),
        side=("left", "midline", "right"),
        role=np.array([ROLE_SENSORY, ROLE_INTERNEURON, ROLE_MOTOR], dtype=np.uint8),
        cell_model=m_arr,
        neurotransmitter=("acetylcholine", "acetylcholine", "gaba"),
        position_nm=np.zeros((n, 3), dtype=np.float32),
        position_source=np.full(n, POS_SOMA, dtype=np.uint8),
    )
    prov = DatasetProvenance("test_graded", "test", "v1", "unit test", "MIT", ())
    dig = compute_pack_digest("test_graded", "test", cells, edges, {}, prov)
    return ConnectomePack("test_graded", "test", cells, edges, {}, prov, dig)


def test_graded_gap_junction_conservation_and_chemical_signs() -> None:
    # 1. Pure ohmic two-cell symmetric gap junction with zero leak preserves mean voltage
    edges = EdgeTable(
        pre=np.array([0, 1], dtype=np.int32),
        post=np.array([1, 0], dtype=np.int32),
        count=np.array([5, 5], dtype=np.int32),
        sign=np.array([1, 1], dtype=np.int8),
        kind=np.array([SYNAPSE_GAP_JUNCTION, SYNAPSE_GAP_JUNCTION], dtype=np.uint8),
    )
    pack = _simple_graded_pack(edges)
    circuit = GradedCircuit.from_csr(pack.edges.to_csr(3))
    params = GradedParams(c_m=1.0, g_leak=0.0, g_gap=0.05, dt_ms=0.2, substeps_per_tick=1)
    st = make_initial_graded_state(3, params)
    st.v[:] = np.array([-20.0, -50.0, -35.0], dtype=np.float32)

    for _ in range(80):
        step_graded_substep_cpu(st, circuit, params, gap_junctions_enabled=True)
        assert abs(float(st.v[0] + st.v[1]) - (-70.0)) < 1e-4
    assert abs(float(st.v[0]) - (-35.0)) < 0.05
    assert abs(float(st.v[1]) - (-35.0)) < 0.05

    # Silencing endpoint disconnects gap-junction current cleanly
    st2 = make_initial_graded_state(3, params)
    st2.v[:] = np.array([-20.0, -50.0, -35.0], dtype=np.float32)
    st2.silenced[0] = np.uint8(1)
    step_graded_tick_cpu(st2, circuit, params, gap_junctions_enabled=True)
    assert float(st2.v[1]) == -50.0

    # 2. Excitatory (+1) vs inhibitory (-1) chemical graded synapses
    edges_c = EdgeTable(
        pre=np.array([0, 2], dtype=np.int32),
        post=np.array([1, 1], dtype=np.int32),
        count=np.array([8, 8], dtype=np.int32),
        sign=np.array([1, -1], dtype=np.int8),
        kind=np.array([SYNAPSE_CHEMICAL, SYNAPSE_CHEMICAL], dtype=np.uint8),
    )
    p_c = _simple_graded_pack(edges_c)
    circ_c = GradedCircuit.from_csr(p_c.edges.to_csr(3))
    gp = GradedParams(dt_ms=0.25, substeps_per_tick=4)

    st_exc = make_initial_graded_state(3, gp)
    st_exc.v[0] = np.float32(-10.0)
    st_exc.v[2] = np.float32(-60.0)
    for _ in range(15):
        step_graded_tick_cpu(st_exc, circ_c, gp)
    assert float(st_exc.v[1]) > gp.e_leak_mv + 2.0
    assert float(st_exc.v[1]) <= gp.e_exc_mv

    st_inh = make_initial_graded_state(3, gp)
    st_inh.v[0] = np.float32(-60.0)
    st_inh.v[2] = np.float32(-10.0)
    for _ in range(15):
        step_graded_tick_cpu(st_inh, circ_c, gp)
    assert float(st_inh.v[1]) < gp.e_leak_mv - 2.0
    assert float(st_inh.v[1]) >= gp.e_inh_mv


def test_hybrid_spiking_lif_plus_graded_and_gap_network() -> None:
    edges = EdgeTable(
        pre=np.array([0, 1, 2], dtype=np.int32),
        post=np.array([1, 2, 1], dtype=np.int32),
        count=np.array([15, 6, 6], dtype=np.int32),
        sign=np.array([1, 1, 1], dtype=np.int8),
        kind=np.array(
            [SYNAPSE_CHEMICAL, SYNAPSE_GAP_JUNCTION, SYNAPSE_GAP_JUNCTION], dtype=np.uint8
        ),
    )
    models = np.array([MODEL_SPIKING_LIF, MODEL_GRADED, MODEL_GRADED], dtype=np.uint8)
    pack = _simple_graded_pack(edges, models=models)
    csr = pack.edges.to_csr(3)
    circ = GradedCircuit.from_csr(csr)
    lif_p = LIFParams(substeps_per_tick=1)
    gp = GradedParams(dt_ms=0.25, substeps_per_tick=4)
    st = make_initial_state(3, lif_p)
    st.v[:] = np.float32(gp.e_leak_mv)
    st.v[0] = np.float32(lif_p.v_0)
    st.stim_rate_hz[0] = np.float32(500.0)

    for _ in range(25):
        step_hybrid_tick_cpu(st, pack, csr, circ, lif_p, gp, seed=1)

    assert int(st.total_spikes[0]) > 0
    assert float(st.v[1]) > gp.e_leak_mv + 0.5
    assert float(st.v[2]) > gp.e_leak_mv + 0.2


def test_celegans_302_pack_checks_and_csv_loader(tmp_path: Path) -> None:
    pack = make_synthetic_celegans_302_pack()
    checks = celegans_302_reference_checks(pack)
    assert all(c["ok"] for c in checks)
    assert len(set(pack.cells.cell_type)) == 302

    csv_file = tmp_path / "herm_test.csv"
    csv_file.write_text(
        "Source,Target,Weight,Type\nALML,AVAL,10,chemical\nAVBL,DB1,6,EJ\n",
        encoding="utf-8",
    )
    loaded = load_celegans_from_edgelist_csv(csv_file, dataset_id="celegans_302")
    assert loaded.cells.num_cells == 302
    assert loaded.edges.num_edges == 3  # 1 chemical + 2 symmetric gap edges
    gap_rows = loaded.edges.kind == SYNAPSE_GAP_JUNCTION
    assert int(gap_rows.sum()) == 2

    # VNC zero-pad normalization, spaced headers, and bidirectional electrical deduplication
    assert normalize_celegans_name("VA01") == "VA1"
    assert normalize_celegans_name("DB01") == "DB1"
    assert normalize_celegans_name("DD06") == "DD6"
    assert normalize_celegans_name("VA12") == "VA12"
    assert normalize_celegans_name(" AVAL ") == "AVAL"

    spaced_csv = tmp_path / "herm_spaced.csv"
    spaced_csv.write_text(
        "Source, Target, Weight, Type\n"
        "ALML , AVAL , 10 , chemical\n"
        "AVBL , DB01 , 6 , electrical\n"
        "DB01 , AVBL , 6 , electrical\n",
        encoding="utf-8",
    )
    loaded_spaced = load_celegans_from_edgelist_csv(spaced_csv, dataset_id="celegans_302")
    assert loaded_spaced.edges.num_edges == 3
    gap_mask = loaded_spaced.edges.kind == SYNAPSE_GAP_JUNCTION
    assert int(gap_mask.sum()) == 2
    assert list(loaded_spaced.edges.count[gap_mask]) == [6, 6]


def test_rft_scallop_theorem_and_directional_traveling_waves() -> None:
    def run_wave(wave_dir: int) -> float:
        x, y, hd = 0.0, 0.0, 0.0
        dt = 0.01
        s = np.arange(N_JOINTS, dtype=np.float64)
        k = 2.0 * math.pi / 9.0
        w = 2.0 * math.pi * 1.5
        phi = np.sin(-wave_dir * k * s) * 0.35 if wave_dir != 0 else np.zeros(N_JOINTS)
        for step in range(200):
            t = step * dt
            if wave_dir == 0:
                phi_n = 0.35 * math.sin(w * t) * np.sin(k * s)
            else:
                phi_n = 0.35 * np.sin(w * t - wave_dir * k * s)
            phi_dot = (phi_n - phi) / dt
            res = step_rft_kinematics(x, y, hd, phi, phi_dot, dt, body_len=BODY_LENGTH_MM)
            x, y, hd = res[0], res[1], res[2]
            phi = phi_n
        return x

    fwd = run_wave(+1)
    rev = run_wave(-1)
    stand = run_wave(0)
    assert fwd > 0.20
    assert rev < -0.20
    assert abs(fwd + rev) < 1e-3
    assert abs(stand) < 1e-2


def test_worm_embodied_ablations_reversal_and_delayed_checkpoint() -> None:
    w_intact = make_worm_world(wiring="reference", seed=0, gap_junctions_enabled=True)
    w_nogap = make_worm_world(wiring="reference", seed=0, gap_junctions_enabled=False)
    w_shuf = make_worm_world(wiring="shuffled_control", seed=0, gap_junctions_enabled=True)
    for _ in range(180):
        step_worm_tick(w_intact)
        step_worm_tick(w_nogap)
        step_worm_tick(w_shuf)

    assert float(w_intact.lif.v[w_intact.pools.avb].mean()) > -20.0
    assert w_intact.signed_forward_mm > 0.05
    assert w_intact.distance_mm > 5.0 * w_nogap.distance_mm
    assert w_intact.distance_mm > 3.0 * w_shuf.distance_mm

    # Nose poke reversal trigger
    sf_before = w_intact.signed_forward_mm
    w_intact.poke_nose_ticks = 25
    for _ in range(30):
        step_worm_tick(w_intact)
    assert float(w_intact.lif.v[w_intact.pools.ava].mean()) > float(
        w_intact.lif.v[w_intact.pools.avb].mean()
    )
    assert w_intact.signed_forward_mm < sf_before - 0.01
    assert w_intact.reversals >= 1

    # Presets load & validate
    presets = load_presets(CONFIGS_DIR)
    for pid in (
        "connectome_celegans_302",
        "connectome_celegans_302_shuffled",
        "worm_embodied_crawl",
        "worm_embodied_no_gap_control",
        "worm_embodied_shuffled_control",
        "worm_embodied_avoidance",
    ):
        assert pid in presets

    # Delayed checkpoint roundtrip (non-aliasing array copies)
    adapter = get_adapter("worm_embodied")
    spec: ExperimentSpec = presets["worm_embodied_crawl"].spec
    assert isinstance(spec.experiment, WormEmbodiedExperiment)
    st = adapter.initialize(spec)
    for _ in range(40):
        st = adapter.step(st, spec)
    ckpt = adapter.serialize_state(st)
    for _ in range(30):
        st = adapter.step(st, spec)
    restored = adapter.deserialize_state(ckpt, spec)
    for _ in range(30):
        restored = adapter.step(restored, spec)
    assert adapter.state_distance(st, restored)["exact_equal"] is True


def test_worm_activity_benchmark_separates_controls() -> None:
    res = run_worm_activity_benchmark(seed=7, steps=80)
    arms = res["arms"]
    assert arms["reference"]["heldout_r2"] == 1.0
    assert arms["reference"]["heldout_r2"] > arms["no_gap_junctions"]["heldout_r2"]
    assert arms["no_gap_junctions"]["heldout_r2"] > arms["shuffled_control"]["heldout_r2"]


def test_s1_api_lifecycle_connectome_and_worm_embodied(tmp_path: Path) -> None:
    import time
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from alife_lab.api.app import create_app
    from alife_lab.execution.executors import LocalExecutor
    from alife_lab.service.settings import load_settings

    settings = replace(load_settings(), home=tmp_path / "home", runs_root=tmp_path / "runs")
    tok = "s1-test-token"
    app = create_app(settings, LocalExecutor(), tok, 8000)
    headers = {"X-ALife-Token": tok, "Origin": "http://127.0.0.1:8000"}

    def wait_paused(client: TestClient, run_id: str) -> dict:
        for _ in range(80):
            info = client.get(f"/api/v1/experiments/{run_id}").json()["data"]
            if info["state"] == "paused":
                return info
            time.sleep(0.05)
        raise AssertionError(f"run {run_id} did not pause")

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        meta = client.get("/api/v1/connectome/synthetic_celegans_302/meta")
        assert meta.status_code == 200, meta.text
        assert meta.json()["data"]["num_cells"] == 302

        geom = client.get("/api/v1/connectome/synthetic_celegans_302/geometry")
        assert geom.status_code == 200

        r1 = client.post(
            "/api/v1/experiments",
            json={
                "preset_id": "connectome_celegans_302",
                "profile_id": "cpu_dev",
                "ticks_per_second": 0,
                "auto_start": True,
                "start_paused": True,
            },
            headers={**headers, "Idempotency-Key": "s1-test-conn-create-0001"},
        )
        assert r1.status_code == 201, r1.text
        cid = r1.json()["data"]["id"]
        wait_paused(client, cid)
        assert (
            client.post(
                f"/api/v1/experiments/{cid}/step", json={"count": 10}, headers=headers
            ).status_code
            == 200
        )

        r2 = client.post(
            "/api/v1/experiments",
            json={
                "preset_id": "worm_embodied_crawl",
                "profile_id": "cpu_dev",
                "ticks_per_second": 0,
                "auto_start": True,
                "start_paused": True,
            },
            headers={**headers, "Idempotency-Key": "s1-test-worm-create-0001"},
        )
        assert r2.status_code == 201, r2.text
        wid = r2.json()["data"]["id"]
        wait_paused(client, wid)
        assert (
            client.post(
                f"/api/v1/experiments/{wid}/step", json={"count": 20}, headers=headers
            ).status_code
            == 200
        )

        insp = client.get(f"/api/v1/experiments/{wid}/inspect?x=0&y=0")
        assert insp.status_code == 200, insp.text
        assert insp.json()["data"]["substrate"] == "worm_embodied"
        assert (
            insp.json()["data"]["worm_pools_mv"]["avb"]
            > insp.json()["data"]["worm_pools_mv"]["ava"]
        )

        iv = client.post(
            f"/api/v1/experiments/{wid}/interventions",
            json={"kind": "poke_nose", "payload": {"duration_ticks": 20}},
            headers=headers,
        )
        assert iv.status_code == 200, iv.text

        pf = client.post(
            f"/api/v1/experiments/{wid}/paired-fork",
            json={
                "treatment_intervention": {
                    "kind": "toggle_gap_junctions",
                    "payload": {"enabled": False},
                }
            },
            headers={**headers, "Idempotency-Key": "s1-test-worm-paired-fork-0001"},
        )
        assert pf.status_code == 201, pf.text


def test_cli_fit_worm_synthetic_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(["connectome", "fit-worm", "--source", "synthetic", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    arms = out["arms"]
    assert (
        arms["reference"]["heldout_r2"]
        > arms["no_gap_junctions"]["heldout_r2"]
        > arms["shuffled_control"]["heldout_r2"]
    )


_DATASETS_DIR = Path(__file__).resolve().parents[2] / ".alife" / "datasets"
_HAS_REAL_CELEGANS_EM = (
    _DATASETS_DIR / "celegans_302" / "raw" / "herm_full_edgelist.csv"
).is_file()
_HAS_REAL_CELEGANS_ACT = (
    _DATASETS_DIR / CELEGANS_ACTIVITY_DIR_NAME / "raw" / CELEGANS_ACTIVITY_FILENAME
).is_file()


@pytest.mark.skipif(not _HAS_REAL_CELEGANS_EM, reason="local celegans_302 EM CSV not downloaded")
def test_real_celegans_302_empirical_pack() -> None:
    rep = verify_dataset("celegans_302", _DATASETS_DIR)
    assert rep["ok"] is True
    pack = load_celegans_302(_DATASETS_DIR, verify_hashes=True)
    checks = celegans_302_reference_checks(pack)
    assert all(c["ok"] for c in checks)
    assert pack.edges.num_edges == 5764


@pytest.mark.skipif(
    not _HAS_REAL_CELEGANS_ACT, reason="local worm_data_short.parquet not downloaded"
)
def test_real_celegans_activity_parquet_loader() -> None:
    recs = load_celegans_activity_recordings(
        _DATASETS_DIR, max_worms=2, min_matched_neurons=40, max_frames=200, verify_hash=True
    )
    assert len(recs) == 2
    for rec in recs:
        assert rec.traces.ndim == 2
        assert np.all(np.isfinite(rec.traces))
        assert int(rec.cell_indices.min()) >= 0
        assert int(rec.cell_indices.max()) < 302
