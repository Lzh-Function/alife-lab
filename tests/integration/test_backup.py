"""Backup -> verify -> restore into an empty location -> runs are verifiable and resumable."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from alife_lab.execution.executors import FakeExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from alife_lab.storage.backup import BackupError, create_backup, restore_backup, verify_backup
from tests.integration.test_controller import arun, spec, state_of, wait_for

BRUSH = {"kind": "brush", "payload": {"shape": "circle", "x": 9, "y": 9, "radius": 2, "value": 1}}


def test_backup_verify_restore_roundtrip(settings: Settings, tmp_path: Path) -> None:
    dest = tmp_path / "backups"
    run_ids: dict[str, str] = {}

    async def build_and_backup() -> Path:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            conway = c.create_run(
                spec("random", max_steps=10_000, interval=25, density=0.3), "cpu_dev", 0
            )["id"]
            await c.start_run(conway, start_paused=True)
            await wait_for(lambda: state_of(c, conway) == "paused")
            await c.control(conway, "step", {"count": 30})
            await c.control(conway, "intervene", BRUSH)
            await c.control(conway, "step", {"count": 40})
            child = c.fork_run(conway, label="child")["id"]
            boids_spec = c.presets["boids_noisy"].spec
            boids = c.create_run(boids_spec, "cpu_dev", 0)["id"]
            await c.start_run(boids, start_paused=True)
            await wait_for(lambda: state_of(c, boids) == "paused")
            await c.control(boids, "step", {"count": 450})
            await c.cancel_run(boids)
            run_ids.update(conway=conway, child=child, boids=boids)
            # Backup while the conway run is still live (paused worker attached).
            result = create_backup(settings.db_path, settings.runs_root, dest)
            assert result["live_runs"] == [conway]
            assert result["runs"] == 3
            return Path(result["backup"])
        finally:
            await c.shutdown()

    backup = arun_value(build_and_backup)
    assert backup.name.startswith("bk_") and not list(dest.glob("*.partial"))
    report = verify_backup(backup)
    assert report["ok"], report
    assert not any("session_token" in f or "lock" in f for f in report["unlisted_files"])

    # Restore into empty locations and use them as a fresh installation.
    restored = replace(settings, home=tmp_path / "r_home", runs_root=tmp_path / "r_runs")
    out = restore_backup(backup, restored.home, restored.runs_root)
    assert out["runs"] == 3

    async def use_restored() -> None:
        c = Controller(restored, FakeExecutor())
        await c.start()
        try:
            assert {r["id"] for r in c.list_runs()} == set(run_ids.values())
            assert c.run_summary(run_ids["conway"])["state"] == "interrupted"
            assert c.run_summary(run_ids["child"])["parent_run_id"] == run_ids["conway"]
            for rid in run_ids.values():
                assert c.verify_run(rid)["ok"]
            for rid in (run_ids["conway"], run_ids["boids"]):
                assert (await c.verify_replay(rid))["status"] == "verified"
            assert [i["status"] for i in c.list_interventions(run_ids["conway"])] == ["applied"]
            await c.restore_run(run_ids["conway"])
            await wait_for(lambda: state_of(c, run_ids["conway"]) == "paused")
            await c.control(run_ids["conway"], "step", {"count": 5})
        finally:
            await c.shutdown()

    arun(use_restored)


def test_backup_safety_checks(settings: Settings, tmp_path: Path) -> None:
    async def build() -> Path:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            rid = c.create_run(spec(), "cpu_dev", 0)["id"]
            await c.start_run(rid, start_paused=True)
            await wait_for(lambda: state_of(c, rid) == "paused")
            await c.control(rid, "step", {"count": 3})
        finally:
            await c.shutdown()
        with pytest.raises(BackupError):  # destination inside the runs directory
            create_backup(settings.db_path, settings.runs_root, settings.runs_root / "bk")
        return Path(create_backup(settings.db_path, settings.runs_root, tmp_path / "b")["backup"])

    backup = arun_value(build)
    # Tampering with any file is detected.
    tampered = tmp_path / "tampered"
    shutil.copytree(backup, tampered)
    ckpt = next((tampered / "runs").rglob("*.npz"))
    raw = bytearray(ckpt.read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # flip one byte
    ckpt.write_bytes(bytes(raw))
    bad = verify_backup(tampered)
    assert not bad["ok"] and any("checksum mismatch" in p for p in bad["problems"])
    with pytest.raises(BackupError):
        restore_backup(tampered, tmp_path / "h1", tmp_path / "r1")
    # Incomplete backup (no manifest) is never valid.
    incomplete = tmp_path / "incomplete"
    shutil.copytree(backup, incomplete)
    (incomplete / "backup_manifest.json").unlink()
    assert not verify_backup(incomplete)["ok"]
    # Never restore over existing data.
    with pytest.raises(BackupError):
        restore_backup(backup, settings.home, tmp_path / "r2")


def arun_value(fn):  # type: ignore[no-untyped-def]
    import asyncio

    return asyncio.run(fn())
