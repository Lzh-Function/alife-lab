"""Multichannel Lenia through the platform: worker, channel intervention, restart, replay."""

from __future__ import annotations

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for


@pytest.mark.integration
def test_lenia_mc_restart_replay(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        raw = c.presets["lenia_mc_swarm"].spec.model_dump(mode="json")
        raw["experiment"]["world"]["size"] = 64
        raw["experiment"]["runner"] = {"max_steps": 1000, "checkpoint_interval": 30}
        rid = c.create_run(ExperimentSpec.model_validate(raw), "cpu_dev", 0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 20})
        paint = {"kind": "paint", "payload": {"x": 5, "y": 5, "radius": 2, "channel": 2}}
        assert (await c.control(rid, "intervene", paint))["ack"]["status"] == "applied"
        await c.control(rid, "step", {"count": 5})
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 30)
            await c2.control(rid, "step", {"count": 10})  # crosses the checkpoint at 30
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            past = await c2.inspect(rid, {"x": 32, "y": 32}, tick=10)
            assert past["source"]["method"] == "deterministic_replay" and len(past["values"]) == 3
        finally:
            await c2.shutdown()

    arun(main)
