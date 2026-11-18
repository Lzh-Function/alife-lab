"""Unclean controller stop (SIGKILL of `alife serve`): workers exit, nothing is lost or
silently adopted, and the run can be restored and replay-verified after restart."""

from __future__ import annotations

import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Server:
    def __init__(self, data: Path, port: int) -> None:
        self.data, self.port = data, port
        self.base = f"http://127.0.0.1:{port}"
        env = {
            **os.environ,
            "ALIFE_HOME": str(data / "home"),
            "ALIFE_RUNS_DIR": str(data / "runs"),
        }
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "alife_lab.cli",
                "serve",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env,
        )
        for _ in range(300):
            try:
                if httpx.get(f"{self.base}/api/v1/health", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.05)
        else:
            raise RuntimeError("server did not start")
        token = (data / "home" / "session_token").read_text().strip()
        self.client = httpx.Client(base_url=self.base, headers={"X-ALife-Token": token}, timeout=60)

    def call(self, method: str, path: str, **kw: Any) -> Any:
        r = self.client.request(method, f"/api/v1{path}", **kw)
        r.raise_for_status()
        return r.json()["data"]

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.client.close()


@pytest.fixture
def data(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path


def wait_until(pred: Any, timeout: float = 20) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.05)
    raise TimeoutError


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie child of the killed server is not "alive"
        with open(f"/proc/{pid}/status") as fh:
            return "zombie" not in fh.read()
    except OSError:
        return False


@pytest.mark.integration
def test_controller_sigkill_recovery(data: Path) -> None:
    port = free_port()
    server = Server(data, port)
    body = {
        "preset_id": "conway_random_soup",
        "profile_id": "cpu_dev",
        "ticks_per_second": 400,
        "auto_start": True,
        "overrides": {"runner": {"max_steps": 1_000_000, "checkpoint_interval": 100}},
    }
    rid = server.call("POST", "/experiments", json=body)["id"]
    wait_until(lambda: server.call("GET", f"/experiments/{rid}")["clock"]["value"] > 350)
    db = data / "home" / "alife.sqlite3"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    worker_pid = conn.execute("SELECT worker_pid FROM jobs WHERE run_id=?", (rid,)).fetchone()[0]
    conn.close()
    # Simulate an interrupted publication left behind by the crash.
    planted = data / "runs" / rid / "checkpoints" / "e001_t000000999999.npz.staging"
    planted.write_bytes(b"partial")

    server.proc.kill()  # SIGKILL: no shutdown checkpoint, no cleanup
    server.proc.wait(timeout=10)
    server.client.close()
    wait_until(lambda: not pid_alive(worker_pid), timeout=10)  # worker notices EOF and exits

    server = Server(data, port)
    try:
        run = server.call("GET", f"/experiments/{rid}")
        assert run["state"] == "interrupted"
        caps = server.call("GET", "/system/capabilities")
        integrity = caps["startup_integrity"]
        assert integrity["db_quick_check"] == "ok"
        assert integrity["quarantined_staging_files"] == [
            f"{rid}/checkpoints/e001_t000000999999.npz.staging"
        ]
        assert not planted.exists()
        assert list((data / "runs" / rid / "quarantine").iterdir())
        types = [e["type"] for e in server.call("GET", f"/experiments/{rid}/events")]
        assert "run_interrupted" in types and "staging_quarantined" in types
        # Unindexed checkpoints (if the kill raced a publication) are reported, never used.
        for item in integrity["unindexed_checkpoints"]:
            assert item.startswith(f"{rid}/checkpoints/")
        latest = run["latest_checkpoint"]
        assert latest["simulation_clock"] % 100 == 0 and latest["simulation_clock"] >= 300

        server.call("POST", f"/experiments/{rid}/restore", json={"start_paused": True})
        wait_until(lambda: server.call("GET", f"/experiments/{rid}")["state"] == "paused")
        assert (
            server.call("GET", f"/experiments/{rid}")["clock"]["value"]
            == latest["simulation_clock"]
        )
        server.call("POST", f"/experiments/{rid}/step", json={"count": 150})
        server.call("POST", f"/experiments/{rid}/checkpoints", json={})
        assert server.call("POST", f"/experiments/{rid}/verify")["ok"]
        report = server.call("POST", f"/experiments/{rid}/replay/verify")
        assert report["status"] == "verified", report["segments"]
        metrics = server.call("GET", f"/experiments/{rid}/metrics?max_points=20000")
        ticks = [r["tick"] for r in metrics["rows"]]
        assert ticks == list(range(ticks[-1] + 1))  # no gaps, no duplicates
    finally:
        server.stop()
