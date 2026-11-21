"""Embodied creature through the platform: worker, push intervention, restart, replay."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for


@pytest.mark.integration
def test_creature_restart_replay(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        raw = c.presets["creature_quadruped_wave"].spec
        rid = c.create_run(raw, "cpu_dev", 0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 30})
        push = {"kind": "push", "payload": {"vx": 1.5}}
        assert (await c.control(rid, "intervene", push))["ack"]["status"] == "applied"
        await c.control(rid, "step", {"count": 20})
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 30)
            await c2.control(rid, "step", {"count": 500})  # crosses the checkpoint at 500
            await wait_for(lambda: c2.run_summary(rid)["clock"]["value"] >= 550, 60)
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            past = await c2.inspect(rid, {"x": 0, "y": 0}, tick=40)
            assert past["source"]["method"] == "deterministic_replay" and len(past["legs"]) == 4
        finally:
            await c2.shutdown()

    arun(main)
