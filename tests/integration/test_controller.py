"""Controller integration tests (FakeExecutor and real LocalExecutor worker processes)."""

from __future__ import annotations

import asyncio
import os
import signal
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alife_lab.artifacts.store import decode_checkpoint
from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import FakeExecutor, LocalExecutor
from alife_lab.service.controller import Conflict, Controller, ControllerLocked
from alife_lab.service.settings import Settings, load_settings

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return replace(
        load_settings(),
        runs_root=tmp_path / "runs",
        home=tmp_path / "home",
        configs_dir=ROOT / "configs",
    )


def spec(
    preset: str = "glider", max_steps: int = 1000, interval: int = 100, **init: Any
) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "itest",
                "substrate": "conway",
                "seed": 7,
                "world": {"width": 48, "height": 40, "boundary": "periodic"},
                "initialization": {"preset": preset, **init},
                "runner": {"max_steps": max_steps, "checkpoint_interval": interval},
                "observations": {"metrics": ["alive_cell_count", "glider_displacement"]},
            }
        }
    )


async def wait_for(pred: Callable[[], bool], timeout: float = 20.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not pred():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.02)


def state_of(c: Controller, run_id: str) -> str:
    return str(c.run_summary(run_id)["state"])


def final_grid(c: Controller, run_id: str) -> np.ndarray:
    latest = c.latest_checkpoint(run_id)
    assert latest is not None
    data = c.store.resolve(run_id, latest["logical_path"]).read_bytes()
    return decode_checkpoint(data)["arrays"]["grid"]


def arun(coro_fn: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(coro_fn())


def test_interactive_lifecycle_with_fake_executor(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            run = c.create_run(spec(), "cpu_dev", ticks_per_second=0)
            rid = run["id"]
            assert run["state"] == "pending"
            await c.start_run(rid, start_paused=True)
            await wait_for(lambda: state_of(c, rid) == "paused")
            res = await c.control(rid, "step", {"count": 4})
            assert res["status"] == "applied"
            assert res["ack"]["applied_clock"] == 4
            # Stepping while paused is allowed; play/pause round trip.
            await c.control(rid, "play")
            await wait_for(lambda: state_of(c, rid) == "running")
            res = await c.control(rid, "pause")
            assert res["status"] == "applied"
            await wait_for(lambda: state_of(c, rid) == "paused")
            tick = c.run_summary(rid)["clock"]["value"]
            # Brush intervention: logged with requested and applied clocks.
            cid = str(uuid.uuid4())
            res = await c.control(
                rid,
                "intervene",
                {
                    "kind": "brush",
                    "payload": {"shape": "circle", "x": 5, "y": 5, "radius": 2, "value": 1},
                },
                command_id=cid,
                requested_clock=tick,
            )
            assert res["status"] == "applied"
            again = await c.control(
                rid, "intervene", {"kind": "brush", "payload": {}}, command_id=cid
            )
            assert again["duplicate"] is True  # idempotent: not applied twice
            interventions = c.list_interventions(rid)
            assert len(interventions) == 1
            assert interventions[0]["applied_clock"] == tick
            assert interventions[0]["requested_clock"] == tick
            summary = c.run_summary(rid)
            assert summary["effective_mode"] == "exploratory"
            assert summary["declared_mode"] == "reference"
            types = [e["type"] for e in c.list_events(rid)]
            assert "intervention_applied" in types and "mode_changed" in types
            # Manual checkpoint.
            res = await c.control(rid, "checkpoint")
            assert res["status"] == "applied"
            assert c.latest_checkpoint(rid)["reason"] == "manual"
            assert c.verify_run(rid)["ok"]
            # Invalid commands.
            with pytest.raises(Conflict):
                await c.control(rid, "play", expected_revision=-5)
            res = await c.control(
                rid,
                "intervene",
                {"kind": "brush", "payload": {"shape": "circle", "x": 999, "y": 0}},
            )
            assert res["status"] == "rejected"
            await c.cancel_run(rid)
            await wait_for(lambda: state_of(c, rid) == "cancelled")
        finally:
            await c.shutdown()

    arun(main)


def test_single_controller_lock(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        try:
            with pytest.raises(ControllerLocked):
                Controller(settings, FakeExecutor())
        finally:
            await c.shutdown()

    arun(main)


def test_profile_and_role_validation(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor(gpu_count=0))
        try:
            _, report = c.validate(spec(), "remote_compute")
            assert not report.ok and "not implemented" in report.errors[0]
            _, report = c.validate(spec(), "local_interactive")
            assert not report.ok  # profile requires gpu_count=1, fake target has none
            _, report = c.validate(spec(), "cpu_dev")
            assert report.ok and report.resolved_backend == "numpy"
        finally:
            await c.shutdown()

    arun(main)


@pytest.mark.integration
def test_restart_and_restore_is_bitwise_identical(settings: Settings) -> None:
    """500 ticks + controller restart + restore + 500 ticks == 1000 continuous ticks."""
    sp = spec("random", max_steps=1000, interval=250, density=0.35)

    async def continuous() -> np.ndarray:
        s = replace(settings, home=settings.home / "a", runs_root=settings.runs_root / "a")
        c = Controller(s, LocalExecutor())
        await c.start()
        try:
            rid = c.create_run(sp, "cpu_dev", ticks_per_second=0)["id"]
            await c.start_run(rid)
            await wait_for(lambda: state_of(c, rid) == "completed", 60)
            m = c.metrics(rid)
            assert m["total_points"] == 1001
            return final_grid(c, rid), m["rows"][-1]["values"]
        finally:
            await c.shutdown()

    async def interrupted() -> tuple[np.ndarray, dict[str, Any]]:
        s = replace(settings, home=settings.home / "b", runs_root=settings.runs_root / "b")
        c = Controller(s, LocalExecutor())
        await c.start()
        rid = c.create_run(sp, "cpu_dev", ticks_per_second=0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 437})
        await c.shutdown()  # graceful: checkpoint at tick 437, then exit

        c2 = Controller(s, LocalExecutor())
        await c2.start()
        try:
            summary = c2.run_summary(rid)
            assert summary["state"] == "interrupted"
            assert summary["latest_checkpoint"]["simulation_clock"] == 437
            assert summary["latest_checkpoint"]["reason"] == "shutdown"
            await c2.restore_run(rid, start_paused=False)
            await wait_for(lambda: state_of(c2, rid) == "completed", 60)
            m = c2.metrics(rid)
            assert m["total_points"] == 1001
            jobs = c2.db.all(
                "SELECT epoch, start_tick FROM jobs WHERE run_id=? ORDER BY epoch", (rid,)
            )
            assert [(j["epoch"], j["start_tick"]) for j in jobs] == [(1, 0), (2, 437)]
            return final_grid(c2, rid), m["rows"][-1]["values"]
        finally:
            await c2.shutdown()

    async def main() -> None:
        grid_a, metrics_a = await continuous()
        grid_b, metrics_b = await interrupted()
        np.testing.assert_array_equal(grid_a, grid_b)
        assert metrics_a == metrics_b

    arun(main)


@pytest.mark.integration
def test_worker_kill_marks_interrupted_and_restores(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        try:
            rid = c.create_run(
                spec(max_steps=100_000, interval=50), "cpu_dev", ticks_per_second=500
            )["id"]
            await c.start_run(rid)
            await wait_for(lambda: c.run_summary(rid)["clock"]["value"] > 120, 30)
            pid = c.db.one("SELECT worker_pid FROM jobs WHERE run_id=?", (rid,))["worker_pid"]
            os.kill(pid, signal.SIGKILL)
            await wait_for(lambda: state_of(c, rid) == "interrupted", 20)
            summary = c.run_summary(rid)
            assert summary["finish_reason"] == "worker_lost"
            ckpt_tick = summary["latest_checkpoint"]["simulation_clock"]
            assert ckpt_tick % 50 == 0 and ckpt_tick >= 100
            await c.restore_run(rid, start_paused=True)
            await wait_for(lambda: state_of(c, rid) == "paused", 30)
            assert c.run_summary(rid)["clock"]["value"] == ckpt_tick
            await c.control(rid, "step", {"count": 3})
            m = c.metrics(rid)
            ticks = [r["tick"] for r in m["rows"]]
            # Every tick exactly once: rows of the killed epoch beyond the checkpoint are
            # either never flushed or excluded as superseded.
            assert ticks == list(range(ckpt_tick + 4))
            assert all(r["values"]["alive_cell_count"] == 5 for r in m["rows"])
            assert c.verify_run(rid)["ok"]
        finally:
            await c.shutdown()

    arun(main)
