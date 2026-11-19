"""Reaction network through the shared platform: worker process, species injection and
reaction knock-out interventions, restart, replay verification and past inspection."""

from __future__ import annotations

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for


@pytest.mark.integration
def test_reaction_network_restart_replay_inspect(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        raw = c.presets["reaction_network_cycle"].spec.model_dump(mode="json")
        raw["experiment"]["runner"]["checkpoint_interval"] = 100
        rid = c.create_run(ExperimentSpec.model_validate(raw), "cpu_dev", 0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 40})
        add = await c.control(rid, "intervene", {"kind": "add", "payload": {"y": 2, "count": 30}})
        assert add["ack"]["result"]["species"] == "I2"
        ko = {"kind": "disable_reaction", "payload": {"reaction": "rep3"}}
        assert (await c.control(rid, "intervene", ko))["ack"]["result"]["enabled"] is False
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
            past = await c2.inspect(rid, {"x": 0, "y": 2}, tick=41)
            assert past["source"]["method"] == "deterministic_replay"
            assert past["selected_species"]["name"] == "I2"
            rep3 = next(r for r in past["reactions"] if r["id"] == "rep3")
            assert rep3["enabled"] is False
        finally:
            await c2.shutdown()

    arun(main)
