"""Lenia through the shared platform: worker process, paint intervention, restart,
exact replay verification across a periodic checkpoint, replay frames and past inspection."""

from __future__ import annotations

import pytest

from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for

PAINT = {"kind": "paint", "payload": {"x": 10, "y": 10, "radius": 4, "value": 0.8}}


@pytest.mark.integration
def test_lenia_restart_replay_inspect(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        rid = c.create_run(c.presets["lenia_orbium"].spec, "cpu_dev", 0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 200})
        res = await c.control(rid, "intervene", PAINT)
        assert res["ack"]["result"]["cells"] == 49
        info = await c.inspect(rid, {"x": 10, "y": 10})
        assert info["value"] == 0.8 and info["substrate"] == "lenia"
        await c.control(rid, "step", {"count": 50})
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 30)
            await c2.control(rid, "step", {"count": 800})  # crosses the checkpoint at 1000
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            frames = await c2.replay_frames(rid, 199, 3)
            assert [f["codec"] for f in frames] == ["field_u8"] * 3
            past = await c2.inspect(rid, {"x": 10, "y": 10}, tick=201)
            assert past["source"]["method"] == "deterministic_replay"
            assert 0 < past["value"] <= 1
        finally:
            await c2.shutdown()

    arun(main)
