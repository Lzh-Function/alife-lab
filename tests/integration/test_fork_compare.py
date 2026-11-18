"""Fork / paired fork / comparison, rollback bookkeeping and DB migration."""

from __future__ import annotations

import os
import signal
import sqlite3
from pathlib import Path

import pytest

from alife_lab.execution.executors import FakeExecutor, LocalExecutor
from alife_lab.service.controller import Controller, ControllerError, NotFound
from alife_lab.service.settings import Settings
from alife_lab.storage.db import MIGRATIONS, Database
from tests.integration.test_controller import arun, spec, state_of, wait_for

BRUSH = {"kind": "brush", "payload": {"shape": "circle", "x": 20, "y": 20, "radius": 3, "value": 1}}


async def paused_parent(c: Controller, ticks: int = 30) -> str:
    rid = c.create_run(
        spec("random", max_steps=100_000, interval=50, density=0.35), "cpu_dev", ticks_per_second=0
    )["id"]
    await c.start_run(rid, start_paused=True)
    await wait_for(lambda: state_of(c, rid) == "paused")
    await c.control(rid, "step", {"count": ticks})
    return rid


async def wait_paused(c: Controller, *rids: str) -> None:
    await wait_for(lambda: all(state_of(c, r) == "paused" for r in rids))


def test_paired_fork_negative_control_and_treatment(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            parent = await paused_parent(c, 30)
            # Negative control: two arms without intervention must stay identical.
            null = await c.paired_fork(parent, "now", treatment_intervention=None)
            a, b = (m["run"]["id"] for m in null["members"])
            await wait_paused(c, a, b)
            # Parent continues too: a fork is a bit-exact continuation of its parent.
            for rid in (a, b, parent):
                await c.control(rid, "step", {"count": 170})
            report = c.comparison_report(null["id"])
            div = report["divergence"]
            assert [p["tick"] for p in div["points"]] == [30, 50, 100, 150, 200]
            assert all(p["hamming_to_first"] == [0, 0] for p in div["points"])
            parent_vs_fork = c.create_comparison([parent, a], name="parent vs fork")
            div2 = c.comparison_report(parent_vs_fork["id"])["divergence"]
            assert div2["common_ticks"] >= 3
            assert all(p["hamming_to_first"] == [0, 0] for p in div2["points"])

            # Free worker slots (ALIFE_MAX_ACTIVE_JOBS defaults to 4; extra runs stay queued).
            for rid in (a, b):
                await c.cancel_run(rid)
            # Treatment: a brush at the fork tick makes the arms diverge.
            treated = await c.paired_fork(parent, "now", treatment_intervention=BRUSH)
            ctl_id, trt_id = (m["run"]["id"] for m in treated["members"])
            await wait_paused(c, ctl_id, trt_id)
            fork_tick = c.run_summary(ctl_id)["clock"]["value"]
            assert fork_tick == 200
            for rid in (ctl_id, trt_id):
                await c.control(rid, "step", {"count": 100})
            div3 = c.comparison_report(treated["id"])["divergence"]
            later = [p for p in div3["points"] if p["tick"] > fork_tick]
            assert later and all(p["hamming_to_first"][1] > 0 for p in later)
            interventions = c.list_interventions(trt_id)
            assert len(interventions) == 1
            assert interventions[0]["status"] == "applied"
            assert interventions[0]["applied_clock"] == fork_tick
            assert c.run_summary(trt_id)["effective_mode"] == "exploratory"
            assert c.run_summary(ctl_id)["effective_mode"] == "reference"
            summary = c.run_summary(trt_id)
            assert summary["parent_run_id"] == parent
            assert summary["label"] == f"treatment@t{fork_tick}"
            assert set(c.run_summary(parent)["children"]) >= {a, b, ctl_id, trt_id}
            manifest = c.provenance(trt_id)
            assert manifest["lineage"]["fork_tick"] == fork_tick
            assert manifest["lineage"]["parent_run_id"] == parent
            # Child metric history starts at the fork tick.
            ticks = [r["tick"] for r in c.metrics(trt_id)["rows"]]
            assert ticks[0] == fork_tick and ticks[-1] == fork_tick + 100
            assert c.verify_run(trt_id)["ok"]
        finally:
            await c.shutdown()

    arun(main)


def test_fork_constraints_and_spec_change(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            parent = await paused_parent(c, 10)
            await c.control(parent, "checkpoint")
            with pytest.raises(ControllerError):
                c.fork_run(parent, spec_overrides={"world": {"width": 64}})
            with pytest.raises(ControllerError):
                c.fork_run(parent, spec_overrides={"runner": {"max_steps": 5}})
            with pytest.raises(ControllerError):  # reference mode forbids other rules
                c.fork_run(parent, spec_overrides={"dynamics": {"rule": "B36/S23"}})
            with pytest.raises(NotFound):
                c.fork_run(parent, checkpoint_id="ckpt_missing")
            child = c.fork_run(
                parent,
                spec_overrides={"mode": "exploratory", "dynamics": {"rule": "B36/S23"}},
                label="highlife branch",
            )
            assert child["spec_hash"] != c.run_summary(parent)["spec_hash"]
            assert child["experiment_id"] != c.run_summary(parent)["experiment_id"]
            await c.start_run(child["id"], start_paused=True)
            await wait_paused(c, child["id"])
            res = await c.control(child["id"], "step", {"count": 5})
            assert res["ack"]["applied_clock"] == 15
            same = c.fork_run(parent)
            assert same["experiment_id"] == c.run_summary(parent)["experiment_id"]
        finally:
            await c.shutdown()

    arun(main)


def test_brush_then_shutdown_at_checkpoint_tick_is_preserved(settings: Settings) -> None:
    """Regression: an intervention after a checkpoint at the same tick must be saved."""

    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        rid = await paused_parent(c, 50)  # periodic checkpoint at tick 50
        res = await c.control(rid, "intervene", BRUSH)
        alive_after = res["ack"]["result"]["alive_after"]
        res = await c.control(rid, "checkpoint")  # same tick, new state -> unique file name
        assert res["status"] == "applied"
        await c.control(rid, "intervene", {**BRUSH, "payload": {**BRUSH["payload"], "x": 5}})
        await c.shutdown()
        c2 = Controller(settings, FakeExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_paused(c2, rid)
            rows = c2.metrics(rid)["rows"]
            assert rows[-1]["tick"] == 50
            assert rows[-1]["values"]["alive_cell_count"] > alive_after
            assert all(i["status"] == "applied" for i in c2.list_interventions(rid))
        finally:
            await c2.shutdown()

    arun(main)


@pytest.mark.integration
def test_rewound_interventions_are_marked_rolled_back(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        try:
            rid = await paused_parent(c, 10)  # last checkpoint: tick 0
            await c.control(rid, "intervene", BRUSH)
            pid = c.db.one("SELECT worker_pid FROM jobs WHERE run_id=?", (rid,))["worker_pid"]
            os.kill(pid, signal.SIGKILL)
            await wait_for(lambda: state_of(c, rid) == "interrupted")
            await c.restore_run(rid)
            await wait_paused(c, rid)
            assert c.run_summary(rid)["clock"]["value"] == 0
            assert [i["status"] for i in c.list_interventions(rid)] == ["rolled_back"]
            assert "interventions_rolled_back" in [e["type"] for e in c.list_events(rid)]
            # Replay must ignore the rolled-back intervention and still match checkpoints.
            await c.control(rid, "step", {"count": 60})
            await c.control(rid, "checkpoint")
            report = await c.verify_replay(rid)
            assert report["ok"] and report["segments_checked"] >= 2
        finally:
            await c.shutdown()

    arun(main)


def test_migration_from_v1_keeps_backup(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at REAL)")
    conn.executescript(MIGRATIONS[0][1])
    conn.execute("INSERT INTO schema_migrations VALUES (1, 0)")
    conn.commit()
    conn.close()
    db = Database(path)
    try:
        assert db.schema_version == len(MIGRATIONS)
        backups = list(tmp_path.glob("db.sqlite3.pre-v*.bak"))
        assert len(backups) == 1
        old = sqlite3.connect(backups[0])
        assert old.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 1
        old.close()
    finally:
        db.close()
