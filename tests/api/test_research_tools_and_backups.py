"""API, security, and end-to-end subprocess tests for GUI-CUI parity (ADR 0054)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from alife_lab.api.app import create_app
from alife_lab.execution.executors import FakeExecutor
from alife_lab.service.research_tools import RESEARCH_TOOLS, build_tool_argv
from alife_lab.service.settings import load_settings

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "parity-test-token"
H = {"X-ALife-Token": TOKEN}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = replace(
        load_settings(),
        runs_root=tmp_path / "runs",
        home=tmp_path / "home",
        datasets_dir=tmp_path / "datasets",
        backups_dir=tmp_path / "backups",
        configs_dir=ROOT / "configs",
        frontend_dist=tmp_path / "no_dist",
    )
    app = create_app(settings, FakeExecutor(), TOKEN, port=8000, extra_hosts=("testserver",))
    with TestClient(app) as c:
        yield c


def _wait_job(client: TestClient, job_id: str, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/v1/tools/research/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        job: dict[str, Any] = resp.json()["data"]
        if job["status"] != "running":
            return job
        time.sleep(0.15)
    raise AssertionError(f"research job {job_id} did not finish within {timeout}s")


def test_build_tool_argv_allowlist_and_injection_guards() -> None:
    assert len(RESEARCH_TOOLS) == 21
    spec, argv, norm = build_tool_argv(
        "connectome_compare_species",
        {"datasets": "synthetic_ciona,synthetic_platynereis", "ticks": 30, "seed": 2},
    )
    assert spec.id == "connectome_compare_species"
    assert argv[:2] == ["connectome", "compare-species"]
    assert "--json" in argv
    assert norm["ticks"] == 30

    with pytest.raises(ValueError, match="unknown research tool"):
        build_tool_argv("arbitrary_shell", {})
    with pytest.raises(ValueError, match="unexpected parameter"):
        build_tool_argv("connectome_compare_species", {"evil_flag": "1"})
    with pytest.raises(ValueError, match="disallowed characters"):
        build_tool_argv("connectome_compare_species", {"datasets": "ciona; rm -rf /"})
    with pytest.raises(ValueError, match="disallowed characters"):
        build_tool_argv("connectome_compare_species", {"datasets": "$(whoami)"})
    with pytest.raises(ValueError, match="must be one of"):
        build_tool_argv("connectome_chimera", {"kind": "invalid_kind"})
    with pytest.raises(ValueError, match="must be <="):
        build_tool_argv("connectome_compare_species", {"ticks": 999_999})


def test_connectome_dataset_endpoints(client: TestClient) -> None:
    ds_resp = client.get("/api/v1/connectome/datasets")
    assert ds_resp.status_code == 200
    datasets = ds_resp.json()["data"]
    ids = {d["dataset_id"] for d in datasets}
    assert {
        "flywire_783",
        "celegans_302",
        "fly_larva_l1",
        "platynereis_3d",
        "ciona_larva",
    }.issubset(ids)
    assert {"synthetic_ciona", "synthetic_platynereis", "synthetic_fly_larva"}.issubset(ids)

    sum_resp = client.get("/api/v1/connectome/datasets/synthetic_ciona/summary")
    assert sum_resp.status_code == 200, sum_resp.text
    sdata = sum_resp.json()["data"]
    assert sdata["dataset_id"] == "synthetic_ciona"
    assert sdata["all_checks_ok"] is True

    # CSRF token required for mutations
    assert client.post("/api/v1/connectome/datasets/ciona_larva/verify").status_code == 403
    bad_ver = client.post("/api/v1/connectome/datasets/nope/verify", headers=H)
    assert bad_ver.status_code == 400


def test_research_jobs_e2e_compare_species_chimera_and_phase_diagram(client: TestClient) -> None:
    cat = client.get("/api/v1/tools/research/catalog")
    assert cat.status_code == 200
    tools = cat.json()["data"]
    assert len(tools) == 21

    # CSRF token required
    assert (
        client.post(
            "/api/v1/tools/research/jobs",
            json={"tool_id": "connectome_compare_species", "params": {}},
        ).status_code
        == 403
    )

    # 1. Cross-species comparison (offline synthetic)
    r1 = client.post(
        "/api/v1/tools/research/jobs",
        json={
            "tool_id": "connectome_compare_species",
            "params": {
                "datasets": "synthetic_ciona,synthetic_platynereis",
                "ticks": 20,
                "seed": 1,
            },
        },
        headers=H,
    )
    assert r1.status_code == 201, r1.text
    j1 = _wait_job(client, r1.json()["data"]["id"])
    assert j1["status"] == "completed", j1
    assert len(j1["result"]["topology"]) == 2
    assert len(j1["result"]["sensorimotor_probe"]) == 2

    # 2. F13c chimera assay (offline synthetic)
    r2 = client.post(
        "/api/v1/tools/research/jobs",
        json={
            "tool_id": "connectome_chimera",
            "params": {
                "kind": "f13c",
                "host": "synthetic_celegans_302",
                "donor": "synthetic_fly_larva",
                "trials": 3,
                "seed": 1,
            },
        },
        headers=H,
    )
    assert r2.status_code == 201, r2.text
    j2 = _wait_job(client, r2.json()["data"]["id"])
    assert j2["status"] == "completed", j2
    assert j2["result"]["all_checks_passed"] is True

    # 3. Cultural 2D phase diagram (small fast grid)
    r3 = client.post(
        "/api/v1/tools/research/jobs",
        json={
            "tool_id": "cultural_phase_diagram",
            "params": {
                "preset": "cultural_world_compositional",
                "periods": "80,200",
                "costs": "0.0,0.04",
                "seeds": "0",
                "steps": 60,
            },
        },
        headers=H,
    )
    assert r3.status_code == 201, r3.text
    j3 = _wait_job(client, r3.json()["data"]["id"])
    assert j3["status"] == "completed", j3
    assert len(j3["result"]["cells"]) == 4

    # Job list includes all 3 jobs
    listed = client.get("/api/v1/tools/research/jobs").json()["data"]
    assert len(listed) >= 3


def test_backup_create_verify_and_test_restore_api(client: TestClient) -> None:
    # CSRF required
    assert client.post("/api/v1/backups").status_code == 403

    # Create a small run first so the DB has realistic content
    cr = client.post(
        "/api/v1/experiments",
        json={"preset_id": "conway_glider", "profile_id": "cpu_dev"},
        headers=H,
    )
    assert cr.status_code == 201

    bk = client.post("/api/v1/backups", headers=H)
    assert bk.status_code == 201, bk.text
    bk_data = bk.json()["data"]
    bid = bk_data["backup_id"]

    listed = client.get("/api/v1/backups").json()["data"]
    assert any(item["backup_id"] == bid for item in listed["items"])

    ver = client.post(f"/api/v1/backups/{bid}/verify", headers=H)
    assert ver.status_code == 200, ver.text
    assert ver.json()["data"]["ok"] is True

    rest = client.post(f"/api/v1/backups/{bid}/restore", headers=H)
    assert rest.status_code == 200, rest.text
    assert rest.json()["data"]["runs"] == 1

    # Invalid ID and path traversal rejected
    bad_id = client.post("/api/v1/backups/bk_bad..id/verify", headers=H)
    assert bad_id.status_code == 400
    bad = client.post("/api/v1/backups/..%2Fetc/verify", headers=H)
    assert bad.status_code in (400, 404, 405)
