"""Physarum through the shared platform: worker process, attract intervention, restart, replay
verification across a periodic checkpoint and past inspection."""

from __future__ import annotations

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for

ATTRACT = {"kind": "attract", "payload": {"x": 10, "y": 10, "radius": 3, "amount": 40}}


@pytest.mark.integration
def test_physarum_restart_replay_inspect(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        raw = c.presets["physarum_network"].spec.model_dump(mode="json")
        raw["experiment"]["world"].update(width=64, height=64)
        raw["experiment"]["runner"]["checkpoint_interval"] = 100
        rid = c.create_run(ExperimentSpec.model_validate(raw), "cpu_dev", 0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 40})
        res = await c.control(rid, "intervene", ATTRACT)
        assert res["ack"]["result"]["cells"] == 29
        await c.control(rid, "step", {"count": 20})
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 30)
            await c2.control(rid, "step", {"count": 80})  # crosses the checkpoint at 100
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            past = await c2.inspect(rid, {"x": 10, "y": 10}, tick=41)
            assert past["source"]["method"] == "deterministic_replay"
            assert past["substrate"] == "physarum" and past["nearest_agent"] is not None
        finally:
            await c2.shutdown()

    arun(main)
