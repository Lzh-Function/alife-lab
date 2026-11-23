"""API tests for F2 connectome metadata, cell search, 3D geometry and live control."""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from alife_lab.api.app import create_app
from alife_lab.execution.executors import FakeExecutor
from alife_lab.service.settings import Settings

TOKEN = "fixture"


def _unpack_frame(raw: bytes) -> tuple[dict[str, Any], bytes]:
    hlen = struct.unpack(">I", raw[:4])[0]
    header = json.loads(raw[4 : 4 + hlen].decode("utf-8"))
    return header, raw[4 + hlen :]


def _client(tmp_path: Path, datasets_dir: Path | None = None) -> TestClient:
    root = Path(__file__).resolve().parents[2]
    settings = Settings(
        project_root=root,
        home=tmp_path / "home",
        runs_root=tmp_path / "runs",
        configs_dir=root / "configs",
        frontend_dist=root / "frontend" / "dist",
        max_active_jobs=4,
        default_ticks_per_second=20.0,
        datasets_dir=datasets_dir or (root / ".alife" / "datasets"),
    )
    app = create_app(settings, FakeExecutor(), TOKEN, port=8000)
    return TestClient(app, base_url="http://127.0.0.1:8000")


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    base = {
        "X-ALife-Token": TOKEN,
        "Origin": "http://127.0.0.1:8000",
    }
    if extra:
        base.update(extra)
    return base


def test_connectome_meta_search_and_geometry_synthetic_tiny(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        meta = client.get("/api/v1/connectome/synthetic_tiny/meta")
        assert meta.status_code == 200
        m = meta.json()["data"]
        assert m["dataset_id"] == "synthetic_tiny"
        assert m["num_cells"] == 4
        assert any(p["port_id"] == "sugar_grn" for p in m["ports"])
        assert any(p["port_id"] == "mn9_proboscis_extension" for p in m["ports"])

        # Search by role, query and port.
        res_all = client.get("/api/v1/connectome/synthetic_tiny/cells")
        assert res_all.status_code == 200
        assert res_all.json()["data"]["total"] == 4

        res_motor = client.get("/api/v1/connectome/synthetic_tiny/cells?role=motor")
        assert res_motor.status_code == 200
        items = res_motor.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["cell_type"] == "CB0701"

        res_port = client.get("/api/v1/connectome/synthetic_tiny/cells?port=sugar_grn")
        assert res_port.status_code == 200
        assert res_port.json()["data"]["total"] == 1

        # Geometry binary blob.
        geom = client.get("/api/v1/connectome/synthetic_tiny/geometry")
        assert geom.status_code == 200
        header, body = _unpack_frame(geom.content)
        assert header["codec"] == "connectome_geom_v1"
        assert header["num_cells"] == 4
        # 4 cells * (3 * int16 + uint8 role + uint8 region) = 4 * 8 = 32 bytes.
        assert len(body) == 32
        pos = np.frombuffer(body[:24], dtype="<i2").reshape(4, 3)
        assert np.all(np.abs(pos) <= 32000)

        # Invalid dataset_id rejected cleanly.
        bad = client.get("/api/v1/connectome/../../etc/meta")
        assert bad.status_code in (400, 404)
        missing = client.get("/api/v1/connectome/no_such_pack/meta")
        assert missing.status_code == 404


def test_connectome_preset_create_intervene_inspect_and_replay_frame(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        presets = client.get("/api/v1/presets").json()["data"]
        ids = {p["id"] for p in presets}
        assert {
            "connectome_tiny_circuit",
            "connectome_fly_sugar",
            "connectome_fly_sugar_bitter",
            "connectome_fly_grooming",
            "connectome_fly_shuffled_control",
        } <= ids

        # Create & run paused on CPU.
        created = client.post(
            "/api/v1/experiments",
            headers=_headers(),
            json={
                "preset_id": "connectome_tiny_circuit",
                "overrides": {"seed": 7, "runner": {"max_steps": 200, "checkpoint_interval": 20}},
                "profile_id": "cpu_dev",
                "ticks_per_second": 0,
                "auto_start": True,
                "start_paused": True,
            },
        )
        assert created.status_code == 201, created.text
        run_id = created.json()["data"]["id"]

        # Wait until worker is paused.
        for _ in range(100):
            st = client.get(f"/api/v1/experiments/{run_id}").json()["data"]["state"]
            if st == "paused":
                break
            time.sleep(0.05)
        assert st == "paused"

        # Step 25 ticks -> checkpoint at 20 and MN9 should fire.
        step = client.post(
            f"/api/v1/experiments/{run_id}/step",
            headers=_headers(),
            json={"count": 25},
        )
        assert step.status_code == 200
        for _ in range(100):
            cur = client.get(f"/api/v1/experiments/{run_id}").json()["data"]["clock"]["value"]
            if cur >= 25:
                break
            time.sleep(0.05)
        assert cur >= 25

        # Inspect by cell_index (2 = MN9 / CB0701) and by port.
        ins_cell = client.get(f"/api/v1/experiments/{run_id}/inspect?cell_index=2").json()["data"]
        assert ins_cell["substrate"] == "connectome"
        assert ins_cell["cell_type"] == "CB0701"
        assert ins_cell["total_spikes"] > 0

        ins_port = client.get(
            f"/api/v1/experiments/{run_id}/inspect?port=mn9_proboscis_extension"
        ).json()["data"]
        assert ins_port["port"] == "mn9_proboscis_extension"
        assert ins_port["total_spikes"] > 0

        # Intervene: silence sugar_grn and verify.
        intv = client.post(
            f"/api/v1/experiments/{run_id}/interventions",
            headers=_headers(),
            json={
                "kind": "silence",
                "payload": {"port": "sugar_grn", "silenced": True},
                "requested_clock": cur,
            },
        )
        assert intv.status_code == 200
        assert intv.json()["data"]["ack"]["status"] == "applied"

        # Checkpoint frame uses connectome_v1 codec.
        ckpts = client.get(f"/api/v1/experiments/{run_id}/checkpoints").json()["data"]
        assert len(ckpts) >= 1
        frame_resp = client.get(f"/api/v1/experiments/{run_id}/checkpoints/{ckpts[-1]['id']}/frame")
        assert frame_resp.status_code == 200
        fheader, fbody = _unpack_frame(frame_resp.content)
        assert fheader["codec"] == "connectome_v1"
        assert fheader["dataset_id"] == "synthetic_tiny"
        assert len(fheader["ports"]) >= 3
        assert len(fbody) == fheader["fired_count"] * 4


def test_connectome_flywire_search_and_geometry_if_present(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    ds_dir = root / ".alife" / "datasets"
    if not (ds_dir / "flywire_783" / "raw" / "Completeness_783.csv").is_file():
        pytest.skip("flywire_783 dataset not downloaded locally")

    with _client(tmp_path, datasets_dir=ds_dir) as client:
        meta = client.get("/api/v1/connectome/flywire_783/meta")
        assert meta.status_code == 200
        m = meta.json()["data"]
        assert m["num_cells"] == 138_639
        assert len(m["regions"]) >= 70

        # Search for MN9 (CB0701) by cell_type.
        res = client.get("/api/v1/connectome/flywire_783/cells?q=CB0701")
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["total"] == 2
        assert all(item["cell_type"] == "CB0701" for item in data["items"])

        # Geometry payload covers all 138,639 neurons.
        geom = client.get("/api/v1/connectome/flywire_783/geometry")
        assert geom.status_code == 200
        header, body = _unpack_frame(geom.content)
        assert header["num_cells"] == 138_639
        assert len(body) == 138_639 * 8
