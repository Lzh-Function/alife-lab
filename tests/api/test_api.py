"""REST / WebSocket API and local security policy (FakeExecutor)."""

from __future__ import annotations

import json
import struct
import time
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from alife_lab.api.app import create_app
from alife_lab.execution.executors import FakeExecutor
from alife_lab.service.settings import load_settings

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "fixture"
H = {"X-ALife-Token": TOKEN}
ORIGIN = "http://testserver"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = replace(
        load_settings(),
        runs_root=tmp_path / "runs",
        home=tmp_path / "home",
        configs_dir=ROOT / "configs",
        frontend_dist=tmp_path / "no_dist",
    )
    app = create_app(settings, FakeExecutor(), TOKEN, port=8000, extra_hosts=("testserver",))
    with TestClient(app) as c:
        yield c


def data(resp: Any) -> Any:
    body = resp.json()
    assert body["api_version"] == "v1"
    return body["data"]


def wait_state(client: TestClient, rid: str, state: str, timeout: float = 10) -> dict[str, Any]:
    end = time.time() + timeout
    while time.time() < end:
        run = data(client.get(f"/api/v1/experiments/{rid}"))
        if run["state"] == state:
            return run
        time.sleep(0.02)
    raise AssertionError(f"state {state} not reached")


def test_catalog_endpoints(client: TestClient) -> None:
    assert data(client.get("/api/v1/health"))["status"] == "ok"
    subs = data(client.get("/api/v1/substrates"))
    assert subs[0]["slug"] == "conway"
    presets = {p["id"]: p for p in data(client.get("/api/v1/presets"))}
    assert "conway_glider" in presets
    schema = data(client.get("/api/v1/substrates/conway/schema"))
    assert "properties" in schema["experiment_spec_schema"]
    profiles = {p["id"]: p for p in data(client.get("/api/v1/execution-profiles"))}
    assert profiles["cluster_batch"]["implemented"] is False
    caps = data(client.get("/api/v1/system/capabilities"))
    assert caps["api_contract_version"] == 1
    assert client.get("/api/v1/nope").status_code == 404


def test_security_policy(client: TestClient) -> None:
    body = {"preset_id": "conway_glider", "profile_id": "cpu_dev"}
    assert client.post("/api/v1/experiments", json=body).status_code == 403
    assert (
        client.post("/api/v1/experiments", json=body, headers={"X-ALife-Token": "x"}).status_code
        == 403
    )
    r = client.post(
        "/api/v1/experiments", json=body, headers={**H, "Origin": "http://evil.example"}
    )
    assert r.status_code == 403
    r = client.get("/api/v1/health", headers={"Host": "evil.example:8000"})
    assert r.status_code == 421  # DNS-rebinding defence
    big = {"preset_id": "conway_glider", "overrides": {"name": "x" * (3 * 1024 * 1024)}}
    assert client.post("/api/v1/experiments", json=big, headers=H).status_code == 413
    url = "/api/v1/stream/experiments/run_0000000000000000"
    with pytest.raises(Exception), client.websocket_connect(url):  # noqa: B017
        pass


def test_validation_errors(client: TestClient) -> None:
    r = client.post("/api/v1/experiments", json={"preset_id": "nope"}, headers=H)
    assert r.status_code == 400
    r = client.post(
        "/api/v1/experiments",
        json={"preset_id": "conway_glider", "overrides": {"world": {"width": 5}}},
        headers=H,
    )
    assert r.status_code == 400 and "width" in r.json()["error"]["message"]
    r = client.post(
        "/api/v1/experiments", json={"preset_id": "conway_glider", "bogus": 1}, headers=H
    )
    assert r.status_code == 422
    r = client.post(
        "/api/v1/experiments/validate",
        json={"preset_id": "conway_glider", "profile_id": "cluster_batch"},
        headers=H,
    )
    assert data(r)["report"]["ok"] is False
    assert client.get("/api/v1/experiments/run_0000000000000000").status_code == 404


def test_create_is_idempotent(client: TestClient) -> None:
    body = {"preset_id": "conway_glider", "profile_id": "cpu_dev"}
    key = {"Idempotency-Key": "create-" + uuid.uuid4().hex}
    a = data(client.post("/api/v1/experiments", json=body, headers={**H, **key}))
    b = data(client.post("/api/v1/experiments", json=body, headers={**H, **key}))
    assert a["id"] == b["id"] and b["idempotent_replay"] is True
    r = client.post(
        "/api/v1/experiments", json={**body, "ticks_per_second": 3}, headers={**H, **key}
    )
    assert r.status_code == 409
    assert len(data(client.get("/api/v1/experiments"))) == 1


def _unpack(frame: bytes) -> tuple[dict[str, Any], bytes]:
    (n,) = struct.unpack(">I", frame[:4])
    return json.loads(frame[4 : 4 + n]), frame[4 + n :]


def test_live_run_over_api_and_websocket(client: TestClient) -> None:
    body = {
        "preset_id": "conway_glider",
        "profile_id": "cpu_dev",
        "ticks_per_second": 0,
        "auto_start": True,
        "start_paused": True,
    }
    run = data(client.post("/api/v1/experiments", json=body, headers=H))
    rid = run["id"]
    wait_state(client, rid, "paused")
    ws_url = f"/api/v1/stream/experiments/{rid}?token={TOKEN}"
    with client.websocket_connect(ws_url, headers={"Origin": ORIGIN}) as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["run"]["id"] == rid
        res = data(client.post(f"/api/v1/experiments/{rid}/step", json={"count": 4}, headers=H))
        assert res["ack"]["applied_clock"] == 4
        got_preview = got_metrics = False
        for _ in range(50):
            msg = ws.receive()
            if msg.get("bytes"):
                header, payload = _unpack(msg["bytes"])
                assert header["codec"] == "bitpack1"
                assert len(payload) == (header["width"] * header["height"] + 7) // 8
                got_preview = True
            elif msg.get("text"):
                parsed = json.loads(msg["text"])
                if parsed["type"] == "metrics":
                    got_metrics = True
            if got_preview and got_metrics:
                break
        assert got_preview and got_metrics
    res = data(
        client.post(
            f"/api/v1/experiments/{rid}/interventions",
            json={
                "kind": "brush",
                "payload": {"shape": "cells", "cells": [[0, 0]], "value": 1},
                "requested_clock": 4,
            },
            headers=H,
        )
    )
    assert res["status"] == "applied" and res["run"]["effective_mode"] == "exploratory"
    res = data(client.post(f"/api/v1/experiments/{rid}/checkpoints", headers=H))
    assert res["status"] == "applied"
    ckpts = data(client.get(f"/api/v1/experiments/{rid}/checkpoints"))
    frame = client.get(f"/api/v1/experiments/{rid}/checkpoints/{ckpts[-1]['id']}/frame")
    header, _ = _unpack(frame.content)
    assert header["tick"] == 4
    metrics = data(client.get(f"/api/v1/experiments/{rid}/metrics"))
    assert metrics["rows"][-1]["tick"] == 4
    events = data(client.get(f"/api/v1/experiments/{rid}/events"))
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    prov = data(client.get(f"/api/v1/experiments/{rid}/provenance"))
    assert prov["science"]["experiment_spec_sha256"] == run["spec_hash"]
    assert data(client.post(f"/api/v1/experiments/{rid}/verify", headers=H))["ok"] is True
    # step while running is a conflict
    data(client.post(f"/api/v1/experiments/{rid}/resume", headers=H))
    wait_state(client, rid, "running")
    assert client.post(f"/api/v1/experiments/{rid}/step", headers=H).status_code == 409
    data(client.post(f"/api/v1/experiments/{rid}/cancel", headers=H))
    wait_state(client, rid, "cancelled")
